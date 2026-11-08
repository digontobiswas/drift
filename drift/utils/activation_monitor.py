"""Find where a forward pass first goes non-finite, and how close fp16 is to its limit.

A non-finite *loss* says only that something, somewhere, overflowed. Under fp16 autocast
the culprit is a module whose output exceeded 65504 (or a ``0 * Inf`` downstream of one),
and the loss is merely where it surfaced. :class:`ActivationMonitor` registers a forward
hook on every submodule and records, in execution order, whether each output was finite
and its largest finite magnitude. On a bad batch :meth:`first_non_finite` names the
earliest module to produce a non-finite value -- the origin, not the symptom -- and
:meth:`headroom` lists the modules running closest to the fp16 ceiling over the run, which
is the early warning for overflows that have not happened yet.

Diagnostic only: it reads every activation, so it slows training noticeably. Enable it for
a short run (``tools/train.py --detect-nonfinite --max-batches 40``), not a full one.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ["ActivationMonitor", "FP16_MAX"]

FP16_MAX = 65504.0


def _iter_tensors(obj, depth: int = 0) -> Iterator[Tensor]:
    """Yield floating-point tensors from a module output (nested containers included)."""
    if depth > 4:
        return
    if torch.is_tensor(obj):
        if obj.is_floating_point() and obj.numel() > 0:
            yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item, depth + 1)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item, depth + 1)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _iter_tensors(getattr(obj, f.name), depth + 1)


def _stats(tensors: List[Tensor]) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[torch.dtype]]:
    """(all-finite flag, max |value|, dtype of the first tensor) -- still on device.

    One reduction per tensor (an inf-norm, which is Inf or NaN exactly when the tensor
    holds a non-finite value), with autocast off so it runs in the tensor's own dtype and
    allocates no full-size temporary: on a 16 GB card already near its limit, a diagnostic
    that doubles activation memory would OOM before it found anything. Results stay on the
    device so recording costs no host synchronisation; they are read back only when a
    report is asked for.
    """
    if not tensors:
        return None, None, None
    amax = None
    with torch.autocast(device_type=tensors[0].device.type, enabled=False):
        for t in tensors:
            a = torch.linalg.vector_norm(t.detach(), ord=float("inf")).float()
            amax = a if amax is None else torch.maximum(amax, a)
    return torch.isfinite(amax), amax, tensors[0].dtype


@dataclasses.dataclass
class _Record:
    name: str
    out_dtype: Optional[torch.dtype]
    out_finite: Optional[Tensor]
    out_amax: Optional[Tensor]
    in_finite: Optional[Tensor]
    in_amax: Optional[Tensor]


class ActivationMonitor:
    """Forward hooks on every submodule of ``model``, recording finiteness and magnitude.

    Args:
        model: The model to watch. Hooks go on every named submodule except the root.

    Usage::

        mon = ActivationMonitor(model)
        for batch in loader:
            mon.start_batch()
            loss = model(batch) ...
            if not torch.isfinite(loss):
                print(mon.describe_first_non_finite())
        print(mon.describe_headroom())
        mon.remove()
    """

    def __init__(self, model: nn.Module) -> None:
        self._records: List[_Record] = []
        self._peak: Dict[str, Tensor] = {}
        self._peak_dtype: Dict[str, torch.dtype] = {}
        self._handles = []
        for name, module in model.named_modules():
            if not name:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs, output) -> None:
            out_f, out_a, out_dt = _stats(list(_iter_tensors(output)))
            if out_f is None:
                return
            in_f, in_a, _ = _stats(list(_iter_tensors(inputs)))
            self._records.append(_Record(name, out_dt, out_f, out_a, in_f, in_a))
            # Peaks track finite magnitudes only, so one Inf does not hide the rest.
            finite_a = torch.where(out_f, out_a, torch.zeros_like(out_a))
            prev = self._peak.get(name)
            self._peak[name] = finite_a if prev is None else torch.maximum(prev, finite_a)
            self._peak_dtype[name] = out_dt
        return hook

    def start_batch(self) -> None:
        """Forget the previous batch's per-module records (peaks are kept)."""
        self._records = []

    def first_non_finite(self) -> Optional[_Record]:
        """The earliest module, in execution order, whose output was not all finite."""
        for rec in self._records:
            if rec.out_finite is not None and not bool(rec.out_finite):
                return rec
        return None

    def describe_first_non_finite(self) -> str:
        rec = self.first_non_finite()
        if rec is None:
            return ("[nonfinite] every module output was finite; the non-finite value arose "
                    "in loss arithmetic outside any nn.Module (see the per-term values).")
        in_state = "n/a" if rec.in_finite is None else ("finite" if bool(rec.in_finite) else "NON-FINITE")
        in_amax = "n/a" if rec.in_amax is None else f"{float(rec.in_amax):.4g}"
        verdict = (
            "its input was already non-finite, so look at what feeds it"
            if rec.in_finite is not None and not bool(rec.in_finite)
            else "its inputs were finite, so the overflow happened inside this module"
        )
        return (
            f"[nonfinite] first non-finite output: {rec.name} (dtype {rec.out_dtype}); "
            f"input {in_state}, max|input| = {in_amax}, max |output| = "
            f"{float(rec.out_amax):.4g} -- {verdict}."
        )

    def headroom(self, top_k: int = 15) -> List[Tuple[str, float, torch.dtype]]:
        """Modules with the largest peak |output| seen so far, fp16 outputs first."""
        rows = [(n, float(v), self._peak_dtype[n]) for n, v in self._peak.items()]
        rows.sort(key=lambda r: (r[2] != torch.float16, -r[1]))
        return rows[:top_k]

    def describe_headroom(self, top_k: int = 15) -> str:
        lines = [f"[headroom] largest activations seen (fp16 max is {FP16_MAX:g}):"]
        for name, peak, dtype in self.headroom(top_k):
            pct = f"{100.0 * peak / FP16_MAX:6.1f}% of fp16 max" if dtype == torch.float16 else "   (not fp16)"
            lines.append(f"[headroom]   {peak:12.4g}  {pct}  {name}")
        return "\n".join(lines)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
