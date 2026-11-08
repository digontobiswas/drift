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

Note what none of these can fix: an activation whose *final* value exceeds 65504 is
``Inf`` in fp16 no matter where it is computed, and only normalisation or a smaller scale
upstream prevents it. These helpers stop an overflow from spreading, and remove the
overflows that were only ever intermediate.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import Tensor

__all__ = ["full_precision", "graph_zero", "select"]


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
