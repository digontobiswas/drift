"""Numerics that stay correct under fp16 autocast.

Training runs under ``torch.autocast`` with fp16 (V100s have no bf16), where the largest
finite value is 65504. Three patterns in this codebase were safe in fp32 and broke in fp16:

1. **Multiplying by an exact zero to discard a branch.** ``mask * a + (1 - mask) * b``
   is meant to return ``a`` where ``mask == 1``. If ``b`` overflowed to ``Inf`` it returns
   ``NaN`` instead, because ``0 * Inf = NaN`` -- whereas in fp32 the same ``b`` is merely
   large and ``0 * b = 0``. The branch that was supposed to be ignored poisons the result.
   :func:`select` picks with ``torch.where`` instead, which never multiplies.

2. **"Graph-connected zero" as ``x.sum() * 0.0``.** Used so an empty loss term still has a
   ``grad_fn``. If any element of ``x`` is ``Inf`` the sum is ``Inf`` and the "zero" is
   ``NaN``, which then makes the *total* loss non-finite even though the term was meant to
   contribute nothing. :func:`graph_zero` cannot become non-finite.

3. **Intermediate overflow in geometry.** E.g. un-projecting a pixel ``u * depth`` reaches
   1598 * 72.4 = 115,695 before being rotated back down to metres. The result fits fp16;
   the intermediate does not. :func:`full_precision` runs such a block in fp32.

4. **A convolution feeding a GroupNorm.** GroupNorm divides out any overall scale of its
   input, so the conv before it is scale-invariant: nothing in the loss constrains how large
   its weights get, and under Adam their norm only ever grows. In fp32 that is harmless --
   the norm divides it away. In fp16 the conv's raw output eventually passes 65504 and is
   Inf before the norm ever sees it. This is what stopped every full training run: the
   CMLI LiDAR generator's middle conv crept up to 99% of the fp16 limit by epoch 5 and then
   overflowed on nearly every batch. :class:`NormFedConv3d` removes the scale before the
   conv and lets the norm discard it, so the result is unchanged and cannot overflow.

These helpers stop an overflow from spreading, remove the overflows that were only ever
intermediate, and remove the scale drift behind the one that was not.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "NormFedConv2d",
    "NormFedConv3d",
    "full_precision",
    "graph_zero",
    "protect_norm_fed_convs",
    "select",
]

# Largest magnitude a NormFedConv may emit: a quarter of fp16's 65504, so the GroupNorm that
# follows (run in fp32 by autocast) never receives anything near the limit.
NORM_FED_TARGET = 16384.0


@contextmanager
def full_precision(device_type: str) -> Iterator[None]:
    """Disable autocast for the enclosed block, so ops run in their inputs' dtype.

    Inputs must be cast with ``.float()`` inside the block -- disabling autocast does not
    convert tensors that are already fp16.

    Args:
        device_type: ``tensor.device.type`` of the tensors involved (``"cuda"``/``"cpu"``).
    """
    with torch.autocast(device_type=device_type, enabled=False):
        yield


def select(mask: Tensor, if_true: Tensor, if_false: Tensor) -> Tensor:
    """``if_true`` where ``mask`` is set, else ``if_false`` -- without multiplying.

    Equivalent to ``mask * if_true + (1 - mask) * if_false`` for a binary ``mask``, except
    that a non-finite value in the branch *not* selected cannot leak into the result.
    Gradients flow only to the selected branch, exactly as with the multiplicative form.

    Args:
        mask: Broadcastable to the branches. Bool, or float/int in ``{0, 1}`` (anything
            ``> 0.5`` counts as set). Must be binary: a soft weight is not a selection.
        if_true: Value where ``mask`` is set.
        if_false: Value where it is not.
    """
    cond = mask if mask.dtype == torch.bool else mask > 0.5
    out_dtype = torch.promote_types(if_true.dtype, if_false.dtype)
    return torch.where(cond, if_true.to(out_dtype), if_false.to(out_dtype))


def graph_zero(*tensors: Tensor) -> Tensor:
    """A scalar ``0`` that is connected to the autograd graph of ``tensors``.

    Replaces ``x.sum() * 0.0``, which is ``NaN`` whenever ``x`` holds an ``Inf``. The sum is
    passed through ``nan_to_num`` first, so the product is ``0`` for any input, and its
    gradient with respect to every element is ``0`` as well.

    Args:
        tensors: One or more tensors; floating-point ones are connected, others ignored.

    Returns:
        A 0-dim fp32 tensor equal to zero.
    """
    total = None
    for t in tensors:
        if not torch.is_tensor(t) or not t.is_floating_point():
            continue
        term = torch.nan_to_num(t.float().sum(), nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        total = term if total is None else total + term
    if total is None:
        raise ValueError("graph_zero needs at least one floating-point tensor.")
    return total


def _norm_fed_scale(conv: nn.Module, x: Tensor) -> Tensor:
    """A detached scalar ``s >= 1`` such that ``conv(x) / s`` provably stays within the target.

    By Cauchy-Schwarz each output is at most ``||w_c|| * ||x_patch|| + |b_c|``, and
    ``||x_patch|| <= sqrt(fan_in) * max|x|``. The bound is rigorous, so the scaled output
    cannot overflow whatever the data -- not merely on the batches seen so far. ``s`` is 1
    (no change at all) until the bound actually reaches the target, which with PyTorch's
    default initialisation is far from the case at the start of training.
    """
    with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
        w = conv.weight
        fan_in = w[0].numel()
        w_norm = w.detach().float().flatten(1).norm(dim=1)  # (out_channels,)
        x_max = torch.linalg.vector_norm(x.detach(), ord=float("inf")).float()
        bound = w_norm * (math.sqrt(fan_in) * x_max)
        if conv.bias is not None:
            bound = bound + conv.bias.detach().float().abs()
        return (bound.max() / NORM_FED_TARGET).clamp(min=1.0)


class _NormFedMixin:
    """Forward for a conv whose output goes straight into a GroupNorm.

    Computes ``conv(x; w/s, b/s) = conv(x; w, b) / s`` and relies on the norm to discard the
    ``1/s``. Because GroupNorm is invariant to a positive scale of its input (up to its
    ``eps``, which becomes ``s^2 * eps`` -- negligible next to the variance of an output that
    was large enough to need scaling), the composed function is unchanged. So are the
    gradients: the norm's derivative is homogeneous of degree -1, which cancels the ``1/s``
    exactly, and ``s`` is detached. Training dynamics are therefore those of the plain conv,
    minus the overflow.

    ONLY valid when the very next operation on the output is a GroupNorm (or another
    scale-invariant normalisation). Never use it before a BatchNorm in eval mode, a
    residual add, or an activation.
    """

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        s = _norm_fed_scale(self, x)
        bias = self.bias / s if self.bias is not None else None
        return self._conv_forward(x, self.weight / s, bias)


class NormFedConv2d(_NormFedMixin, nn.Conv2d):
    """``nn.Conv2d`` for use directly before a GroupNorm; see :class:`_NormFedMixin`."""


class NormFedConv3d(_NormFedMixin, nn.Conv3d):
    """``nn.Conv3d`` for use directly before a GroupNorm; see :class:`_NormFedMixin`."""


_NORM_FED = {nn.Conv2d: NormFedConv2d, nn.Conv3d: NormFedConv3d}


def protect_norm_fed_convs(model: nn.Module) -> List[str]:
    """Convert every conv that feeds directly into a GroupNorm inside an ``nn.Sequential``.

    Sequential is the one place where "directly feeds" can be read off the structure: it
    calls its children in order, so ``seq[i + 1](seq[i](x))`` is guaranteed. A conv inside a
    custom ``forward`` cannot be verified this way and must be converted where it is defined
    (as ``drift.models.e4a._ConvBnAct`` does).

    Conversion swaps the instance's class in place: no parameter, buffer or ``state_dict``
    key changes, so checkpoints written before this existed load unchanged -- and, since the
    function is unchanged too, behave the same.

    Args:
        model: Module to convert in place.

    Returns:
        Qualified names of the convs converted, for logging.
    """
    converted: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for (child_name, child), (_, nxt) in zip(children, children[1:]):
            if type(child) in _NORM_FED and isinstance(nxt, nn.GroupNorm):
                child.__class__ = _NORM_FED[type(child)]
                converted.append(f"{name}.{child_name}" if name else child_name)
    return converted
