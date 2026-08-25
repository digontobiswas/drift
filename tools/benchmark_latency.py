#!/usr/bin/env python3
"""DRIFT inference latency / memory / FLOPs benchmark. Backs the paper's efficiency claim.

Measures, for one forward pass of the full DRIFT pipeline at a given config's ``(T_p, T_o)``:

- **Latency**: mean / std / p95 wall-clock milliseconds over ``--repeats`` timed runs, after
  ``--warmup`` untimed warm-up runs. ``torch.cuda.synchronize()`` brackets every timed call
  when running on CUDA (required -- CUDA kernel launches are asynchronous, so an un-synced
  ``time.perf_counter()`` would just measure Python dispatch overhead, not device work).
- **Per-horizon latency**: the same total forward latency divided by ``T_o``. DRIFT's
  ``DecoupledForecaster``/``Refiner`` are **not** autoregressive across horizons -- all
  ``T_o`` future frames are produced in one forward call (spec §2.12/§2.13) -- so there is
  no literal "time to reach horizon k" to measure without invasive internal instrumentation.
  This amortized ms/frame is the standard way efficiency is reported for non-autoregressive
  forecasters and is what makes DRIFT's latency roughly flat as ``T_o`` grows, unlike an
  autoregressive baseline's -- see the module-level stage breakdown below for *where* the
  time actually goes.
- **Stage breakdown**: wall-clock time attributed to each top-level submodule (camera/lidar
  encoders, CMLI, CVQG+fusion, Observer, Forecaster, Refiner, heads) via forward hooks, summed
  over the timed repeats -- this is what a paper-table's "latency" row is usually decomposed
  into for a systems/ablation discussion.
- **Peak memory**: ``torch.cuda.max_memory_allocated()`` on CUDA; on CPU, the process's peak
  RSS via ``resource.getrusage`` (an approximation -- it is whole-process peak RSS including
  Python/interpreter overhead, not an isolated measurement of this forward call alone; noted
  in the output).
- **Parameter count**: total and trainable, in millions.
- **FLOPs**: via ``fvcore`` (preferred) or ``thop`` if either is importable; otherwise this
  row is reported as "unavailable" with a clear one-line explanation, never a silent omission
  or a crash (neither package is a hard dependency -- spec §7 forbids new hard dependencies).

Examples:
    CPU smoke test with the tiny config::

        python tools/benchmark_latency.py --config tiny --device cpu --repeats 10 --warmup 3

    Full-size config on GPU, more repeats for a tighter p95::

        python tools/benchmark_latency.py --config cam4docc_2s --device cuda --repeats 50 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from configs import get_config, list_configs
from configs.base import DriftConfig
from drift.data.cam4docc_dataset import SyntheticOccDataset
from drift.data.collate import collate_fn
from drift.models.drift import DRIFT
from drift.utils.seed import seed_everything

# Top-level submodules timed individually for the stage breakdown. Attribute names on `DRIFT`;
# a name absent under a given config/ablation (e.g. `lidar_encoder` under `camera_only`, or
# `forecaster` under `no_instance_path`, which instead uses `static_path`/`dense_flow_fallback`)
# is silently skipped rather than raising, since which modules exist is config-dependent by
# design (spec §6 ablation presets).
_STAGE_MODULES = [
    "camera_encoder", "lidar_encoder", "cmli", "cvqg", "cross_modal_fusion", "observer",
    "forecaster", "static_path", "dense_flow_fallback", "refiner",
    "occ_head", "flow_head", "uncertainty_head",
]


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """Parse ``tools/benchmark_latency.py`` command-line arguments."""
    p = argparse.ArgumentParser(description="Benchmark DRIFT inference latency/memory/FLOPs.")
    p.add_argument("--config", type=str, default="tiny", choices=list_configs())
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--repeats", type=int, default=30, help="Number of timed forward passes.")
    p.add_argument("--warmup", type=int, default=5, help="Number of untimed warm-up forward passes.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", type=str, default=None)
    return p.parse_args(argv)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[benchmark] CUDA requested but not available; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def _hardware_string(device: torch.device) -> str:
    """One-line description of the exact hardware this run measured, for the paper table."""
    cpu = platform.processor() or platform.machine()
    parts = [f"platform={platform.platform()}", f"python={platform.python_version()}", f"torch={torch.__version__}"]
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        cap = torch.cuda.get_device_capability(idx)
        parts.append(f"gpu={name} (sm_{cap[0]}{cap[1]})")
    else:
        parts.append(f"cpu={cpu}")
        try:
            import os

            n_threads = torch.get_num_threads()
            n_cpus = os.cpu_count()
            parts.append(f"torch_threads={n_threads}/{n_cpus}")
        except Exception:  # pragma: no cover - best-effort only
            pass
    return ", ".join(parts)


def build_batch(cfg: DriftConfig, device: torch.device, batch_size: int, seed: int) -> Dict[str, Any]:
    """Build one collated synthetic batch matching ``cfg.model``'s shapes, moved to ``device``.

    Benchmarking always uses ``SyntheticOccDataset`` (no real nuScenes/Cam4DOcc data is
    available in this environment, and latency/memory/FLOPs do not depend on real vs.
    synthetic *content* -- only on tensor shapes, which this dataset reproduces exactly).
    """
    m, d = cfg.model, cfg.data
    ds = SyntheticOccDataset(
        num_samples=batch_size, T_p=m.T_p, T_f=m.T_f, T_o=m.T_o, N_cam=m.N_cam,
        H_img=d.H_img, W_img=d.W_img, num_classes=m.num_classes, latent_size=m.latent_size,
        occ_size=m.occ_size, point_cloud_range=m.point_cloud_range, in_channels=d.in_channels,
        num_points_range=d.num_points_range, num_boxes_range=d.num_boxes_range,
        modality_dropout_p=0.0, seed=seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif hasattr(v, "to") and callable(v.to):
            out[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            out[k] = [[t.to(device) if torch.is_tensor(t) else t for t in s] for s in v]
        elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], "to"):
            out[k] = [s.to(device) for s in v]
        else:
            out[k] = v
    return out


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _StageTimer:
    """Accumulates per-submodule wall-clock time across repeated forward passes via hooks.

    A pre-hook stamps the entry time; a post-hook (bracketed by ``torch.cuda.synchronize`` when
    on CUDA) computes and accumulates the elapsed time. Nested/reentrant module calls are not
    expected among ``_STAGE_MODULES`` (they are DRIFT's direct top-level children, called
    exactly once each per forward), so no re-entrancy guard is needed.
    """

    def __init__(self, model: DRIFT, device: torch.device) -> None:
        self.device = device
        self.totals: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}
        self._handles: List[Any] = []
        for name in _STAGE_MODULES:
            mod = getattr(model, name, None)
            if mod is None or not isinstance(mod, torch.nn.Module):
                continue
            self.totals[name] = 0.0
            self._handles.append(mod.register_forward_pre_hook(self._make_pre(name)))
            self._handles.append(mod.register_forward_hook(self._make_post(name)))

    def _make_pre(self, name: str):
        def hook(module, inputs):  # noqa: ANN001 - torch hook signature
            _sync(self.device)
            self._starts[name] = time.perf_counter()

        return hook

    def _make_post(self, name: str):
        def hook(module, inputs, output):  # noqa: ANN001 - torch hook signature
            _sync(self.device)
            self.totals[name] += time.perf_counter() - self._starts[name]

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


def measure_latency(
    model: DRIFT, batch: Dict[str, Any], device: torch.device, repeats: int, warmup: int,
) -> Dict[str, Any]:
    """Time ``repeats`` forward passes (after ``warmup`` untimed ones) and collect stage totals.

    Args:
        model: The model to benchmark (already ``.eval()``'d and moved to ``device``).
        batch: One collated DRIFT batch, already moved to ``device``.
        device: The device being benchmarked.
        repeats: Number of timed forward passes.
        warmup: Number of untimed forward passes run first (lets CPU caches / CUDA kernels
            / cuDNN autotuning settle before timing begins).

    Returns:
        Dict with ``times_ms`` (list, length ``repeats``), ``mean_ms``, ``std_ms``, ``p95_ms``,
        and ``stage_ms`` (mean per-repeat ms spent in each top-level submodule).
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(batch)
        _sync(device)

        stage_timer = _StageTimer(model, device)
        times_ms: List[float] = []
        try:
            for _ in range(repeats):
                _sync(device)
                t0 = time.perf_counter()
                model(batch)
                _sync(device)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
        finally:
            stage_timer.remove()

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0
    sorted_times = sorted(times_ms)
    p95_idx = min(len(sorted_times) - 1, int(round(0.95 * (len(sorted_times) - 1))))
    p95_ms = sorted_times[p95_idx]

    stage_ms = {name: (total / repeats) * 1000.0 for name, total in stage_timer.totals.items()}

    return {"times_ms": times_ms, "mean_ms": mean_ms, "std_ms": std_ms, "p95_ms": p95_ms, "stage_ms": stage_ms}


def measure_peak_memory(model: DRIFT, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Run one forward pass and report peak memory, in MB.

    On CUDA: ``torch.cuda.max_memory_allocated()`` since the last ``reset_peak_memory_stats``,
    an exact measurement of tensor memory attributable to this forward call. On CPU: whole-
    process peak RSS via ``resource.getrusage`` -- necessarily an approximation (it includes
    interpreter/import overhead accumulated before this function ran), documented as such in
    the returned dict's ``note``.
    """
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(batch)
        torch.cuda.synchronize(device)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        return {"peak_mb": peak_mb, "method": "torch.cuda.max_memory_allocated", "note": None}

    import resource

    with torch.no_grad():
        model(batch)
    ru_maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux
    peak_mb = ru_maxrss_kb / 1024.0
    return {
        "peak_mb": peak_mb, "method": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "note": "approximate whole-process peak RSS, not isolated to this forward call",
    }


def count_parameters(model: DRIFT) -> Dict[str, float]:
    """Total and trainable parameter counts, in millions."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_m": total / 1e6, "trainable_m": trainable / 1e6}


def try_compute_flops(model: DRIFT, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attempt a FLOPs count via ``fvcore`` (preferred) or ``thop``. Returns ``None`` if neither
    is importable, or if the trace itself fails (e.g. an op fvcore/thop can't hook)  -- either
    way a one-line explanation is printed, never a silent omission or a crash. Spec §7 forbids
    adding either as a hard dependency, so both imports are optional (``try/except``).
    """
    model.eval()
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        pass
    else:
        try:
            with torch.no_grad():
                flops = FlopCountAnalysis(model, (batch,))
                flops.unsupported_ops_warnings(False)
                flops.uncalled_modules_warnings(False)
                total = flops.total()
            return {"backend": "fvcore", "gflops": total / 1e9}
        except Exception as e:  # pragma: no cover - depends on optional dependency internals
            print(f"[benchmark] fvcore FLOPs counting failed ({type(e).__name__}: {e}); skipping FLOPs.")
            return None

    try:
        from thop import profile
    except ImportError:
        print(
            "[benchmark] Neither `fvcore` nor `thop` is importable; FLOPs will be reported as "
            "unavailable. Install one (`pip install fvcore` or `pip install thop`) for a FLOPs "
            "count -- neither is a hard dependency of DRIFT (spec §7)."
        )
        return None

    try:
        with torch.no_grad():
            macs, _params = profile(model, inputs=(batch,), verbose=False)
        return {"backend": "thop", "gflops": 2.0 * macs / 1e9}  # thop reports MACs; FLOPs ~= 2*MACs
    except Exception as e:  # pragma: no cover - depends on optional dependency internals
        print(f"[benchmark] thop FLOPs counting failed ({type(e).__name__}: {e}); skipping FLOPs.")
        return None


def format_report(
    cfg: DriftConfig, device: torch.device, hardware: str, latency: Dict[str, Any],
    memory: Dict[str, Any], params: Dict[str, float], flops: Optional[Dict[str, Any]],
    repeats: int, warmup: int, batch_size: int,
) -> str:
    """Render every measurement as a clean, paper-table-ready text report."""
    lines: List[str] = []
    W = 78
    lines.append("=" * W)
    lines.append("DRIFT latency / memory / FLOPs benchmark")
    lines.append(f"hardware: {hardware}")
    lines.append(f"config={cfg.name}  device={device}  batch_size={batch_size}  repeats={repeats}  warmup={warmup}")
    lines.append(
        f"T_p={cfg.model.T_p} T_o={cfg.model.T_o} N_cam={cfg.model.N_cam} "
        f"latent_size={tuple(cfg.model.latent_size)} num_classes={cfg.model.num_classes}"
    )
    lines.append("=" * W)

    lines.append("")
    lines.append("Latency (full forward pass, all T_o horizons in one call)")
    lines.append("-" * W)
    lines.append(f"{'mean (ms)':<20}{latency['mean_ms']:>14.3f}")
    lines.append(f"{'std (ms)':<20}{latency['std_ms']:>14.3f}")
    lines.append(f"{'p95 (ms)':<20}{latency['p95_ms']:>14.3f}")
    lines.append(f"{'per-horizon (ms)':<20}{latency['mean_ms'] / cfg.model.T_o:>14.3f}"
                 f"   (= mean / T_o={cfg.model.T_o}; forecaster is non-autoregressive, see module docstring)")
    lines.append(f"{'throughput (Hz)':<20}{1000.0 / latency['mean_ms']:>14.3f}")

    lines.append("")
    lines.append("Stage breakdown (mean ms/forward, via forward hooks)")
    lines.append("-" * W)
    for name, ms in latency["stage_ms"].items():
        pct = 100.0 * ms / latency["mean_ms"] if latency["mean_ms"] > 0 else 0.0
        lines.append(f"{name:<24}{ms:>12.3f} ms{pct:>10.1f} %")

    lines.append("")
    lines.append("Memory / parameters / FLOPs")
    lines.append("-" * W)
    mem_note = f"  ({memory['note']})" if memory["note"] else ""
    lines.append(f"{'peak memory (MB)':<24}{memory['peak_mb']:>14.1f}{mem_note}")
    lines.append(f"{'params, total (M)':<24}{params['total_m']:>14.3f}")
    lines.append(f"{'params, trainable (M)':<24}{params['trainable_m']:>14.3f}")
    if flops is not None:
        lines.append(f"{'FLOPs (G), backend='+flops['backend']:<24}{flops['gflops']:>14.3f}")
    else:
        lines.append(f"{'FLOPs':<24}{'unavailable':>14} (fvcore/thop not importable or trace failed; see log above)")
    lines.append("=" * W)
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    cfg = get_config(args.config)
    seed_everything(args.seed)

    device = _resolve_device(args.device)
    hardware = _hardware_string(device)
    print(f"[benchmark] config={cfg.name} device={device}")
    print(f"[benchmark] hardware: {hardware}")

    model = DRIFT(cfg.model, cfg.loss).to(device)
    model.eval()

    batch = build_batch(cfg, device, batch_size=args.batch_size, seed=args.seed)

    print(f"[benchmark] warming up ({args.warmup} runs) and timing ({args.repeats} runs)...")
    latency = measure_latency(model, batch, device, repeats=args.repeats, warmup=args.warmup)

    print("[benchmark] measuring peak memory...")
    memory = measure_peak_memory(model, batch, device)

    params = count_parameters(model)

    print("[benchmark] attempting FLOPs count (fvcore, then thop)...")
    flops = try_compute_flops(model, batch)

    report = format_report(
        cfg, device, hardware, latency, memory, params, flops,
        repeats=args.repeats, warmup=args.warmup, batch_size=args.batch_size,
    )
    print()
    print(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": cfg.name, "device": str(device), "hardware": hardware,
            "batch_size": args.batch_size, "repeats": args.repeats, "warmup": args.warmup,
            "T_p": cfg.model.T_p, "T_o": cfg.model.T_o,
            "latency": {k: v for k, v in latency.items() if k != "times_ms"},
            "latency_raw_ms": latency["times_ms"],
            "memory": memory, "params": params, "flops": flops,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[benchmark] wrote JSON results to {out_path}")


if __name__ == "__main__":
    main()
