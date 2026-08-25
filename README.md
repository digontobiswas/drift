# DRIFT

**D**ecoupled **R**obust **I**nstance-**F**low **T**emporal occupancy world model.

DRIFT is a LiDAR–camera fusion world model for 4D semantic occupancy forecasting: given a
short window of past LiDAR sweeps + multi-view images, it predicts future per-voxel semantic
occupancy, per-voxel flow, and per-voxel predictive uncertainty over a horizon of several
future frames, and degrades gracefully when a sensor modality is missing or unreliable at
inference time. The full module contract lives in [`docs/DESIGN_SPEC.md`](docs/DESIGN_SPEC.md)
— that document is the single source of truth for shapes, names, and interfaces; this README
is the practical entry point (install, run, and an honest account of what has and hasn't been
validated).

## 0. Novelty summary

DRIFT is a clean-room reimplementation (no copied code) that reuses/modifies components from
three prior works, plus three genuinely new pieces. Full rationale for each MODIFIED/DIFFERENT
row is in `docs/DESIGN_SPEC.md` §0; the short version is reproduced here.

| Component | Origin | Status |
|---|---|---|
| E4A (4D UNet aggregation) | OccProphet (ICLR 2025, [arXiv:2502.15180](https://arxiv.org/abs/2502.15180)) | **REUSED** — baseline component |
| TAF (Tripling-Attention Fusion) | OccProphet | **REUSED** |
| Conditional Forecaster (hypernetwork) | OccProphet | **REUSED**, repurposed |
| Refiner (E4A on [past, future]) | OccProphet | **REUSED** |
| Coarse Voxel Query Generator | Doracamom (TCSVT 2026, [arXiv:2501.15394](https://arxiv.org/abs/2501.15394)) | **MODIFIED** — real 3D LiDAR volume, not height-broadcast |
| Cross-Modal BEV-Voxel Fusion | Doracamom | **MODIFIED** — two-sided gate (LiDAR is not a weak prior) |
| Decoupled static/dynamic forecasting | DFIT-OccWorld ([arXiv:2412.13772](https://arxiv.org/abs/2412.13772)) | **DIFFERENT** — instance-query, not dense voxel flow |
| **Instance-query dynamic path** | — | **★ NOVEL** |
| **CMLI (Cross-Modal Latent Imagination)** | — | **★ NOVEL** |
| **Per-voxel uncertainty head** | — | **★ NOVEL** |

Evaluation follows the [Cam4DOcc](https://arxiv.org/abs/2311.17663) benchmark protocol
(pooled confusion-matrix IoU, cumulative per-horizon buckets, backward centroid-offset flow
GT — see `drift/metrics/iou.py` and `docs/DESIGN_SPEC.md` §5).

**Why the LiDAR substitution is not a trivial sensor swap.** Doracamom broadcasts its radar
BEV vector to every height (`unsqueeze(-1).repeat(..., Z)`) in two places, because 4D radar
has almost no elevation resolution — doing the same with LiDAR would discard its single
strongest signal. DRIFT replaces both broadcasts with a genuine 3D volume from a
sparse/pillar 3D encoder.

**Why instance-query dynamics fit the GT better than dense flow.** Cam4DOcc's flow GT is not
a per-voxel displacement field — it is a *backward centroid-offset* field: every voxel of
instance *k* at frame *t* stores `centroid_k(t-1) - own_index(t)`. The GT is therefore already
instance-centric; predicting it with a dense flow field spends capacity re-deriving a
structure that's known a priori. DRIFT's instance-query path predicts the per-instance
translation directly and renders it — sparser and better matched to the supervision.

## 1. Install

```bash
python -m pip install -r requirements.txt
```

No `mmcv`/`mmdet3d`/`spconv` dependency — the whole model runs on plain PyTorch, CPU included
(spec §7). `fvcore` or `thop` are optional, only used by `tools/benchmark_latency.py` for a
FLOPs count; neither is required for training, evaluation, or the robustness study.

```bash
python -m pip install fvcore   # optional, for a FLOPs count in tools/benchmark_latency.py
```

Requires Python 3.10+, PyTorch 2.x. Verify the install:

```bash
python -m pytest tests/ -q   # 48 passed
```

## 2. Data prep

Two datasets share the exact same sample contract (`docs/DESIGN_SPEC.md` §4):

- **`SyntheticOccDataset`** (`drift/data/cam4docc_dataset.py`) generates shape- and
  semantically-plausible samples on the fly, no external data or download required. This is
  what every command below uses by default (`--dataset synthetic`, the default for the `tiny`
  config) and what every module in this repo has actually been exercised against.
- **`Cam4DOccDataset`** reads the derived, protocol-compatible dataset built by
  **`tools/prepare_nuscenes.py`** from a raw nuScenes download: a JSON `ann_file` indexing
  per-sample `.npz` files (calibration, ego-motion, sparse occupancy/instance GT, latent-
  resolution flow). Images and point clouds are *not* copied — the index stores paths back
  into the read-only nuScenes tree and the dataset decodes them lazily, which keeps the
  derived dataset around 12 MB per sample instead of ~500 MB.

### Building the derived dataset from raw nuScenes

```bash
python tools/prepare_nuscenes.py \
    --nuscenes-root /path/to/nuscenes-trainval \   # read-only source
    --out-root      /path/to/derived \             # new, writable; must be outside the source
    --split val --protocol gmo
```

`--protocol gmo` is the Cam4DOcc benchmark protocol: 3 classes (0 free, 1 general static
occupancy, 2 general movable object). It needs only the base `v1.0-trainval` archives — no
separate nuScenes-lidarseg download. Use the `cam4docc_gmo` preset with it.

The script is **resumable** (existing `.npz` files are skipped), shardable across a job array
(`--shard` / `--num-shards`, merged afterwards with `tools/merge_shards.py`), and refuses to
write anywhere inside `--nuscenes-root`.

To point any tool at real data instead of synthetic:

```bash
--dataset cam4docc --data-root /path/to/derived --ann-file drift_train.json  # or drift_val.json
```

Or set `DRIFT_DATA_ROOT` / `DRIFT_ANN_FILE` / `DRIFT_CKPT_DIR` once in the environment and
every preset picks them up — this is how the Slurm scripts drive the whole ablation grid
without editing a config file.

### Running on an HPC cluster

**[`docs/HPC_GUIDE.md`](docs/HPC_GUIDE.md)** is the end-to-end runbook: environment setup
(including the no-`ssl`-module and no-internet-on-compute-nodes cases), preprocessing as a
job array, a 10-minute smoke test that catches the failures which otherwise surface six hours
into a 24-hour run, the full ablation grid, and result collection.

```bash
bash scripts/setup_env.sh          # login node: venv + dependencies
bash scripts/fetch_weights.sh      # login node: pretrained weights (compute nodes are offline)
sbatch --export=ALL,SPLIT=val slurm/00_preprocess.slurm
sbatch slurm/01_smoke.slurm        # verify before committing GPU-days
sbatch slurm/03_ablations.slurm    # all 7 configs
sbatch slurm/04_eval.slurm
python tools/collect_results.py --results-dir $DRIFT_RESULTS_DIR --out results.md
```

## 3. Commands

Every tool shares the same `--config <preset>` selection (§4 below) and CPU/CUDA device
handling; `--device` auto-falls-back to CPU with a printed warning if CUDA is requested but
unavailable. All four commands below are copy-paste runnable as shown — they are exactly what
was used to produce the smoke-test output quoted in this repo's implementation log, on this
CPU-only, no-GPU, no-nuScenes environment.

### Train

```bash
# CPU smoke run on synthetic data (a couple of iterations)
python tools/train.py --config tiny --max-iters 3 --device cpu

# Single-GPU on the default preset, real data
python tools/train.py --config cam4docc_2s --data-root /data/cam4docc \
    --ann-file train.json --dataset cam4docc

# Multi-GPU (4 ranks) via torchrun
torchrun --nproc_per_node=4 tools/train.py --config cam4docc_2s --distributed
```

### Evaluate

Reports the Cam4DOcc-style IoU table (per-class, `IOU_mean`, `IoU_c`, `IoU_f`, cumulative
per-horizon buckets), flow EPE/angular/magnitude error in metres, and ECE.

```bash
# CPU smoke test, random-init model (no checkpoint needed)
python tools/eval.py --config tiny --random-init --device cpu

# Trained checkpoint on the val split, dump results to JSON
python tools/eval.py --config cam4docc_2s --checkpoint work_dirs/drift/latest.pth \
    --dataset cam4docc --data-root /data/cam4docc --ann-file val.json \
    --json-out results/eval_cam4docc_2s.json
```

### Benchmark latency / memory / FLOPs

```bash
python tools/benchmark_latency.py --config tiny --device cpu --repeats 30 --warmup 5
```

Reports mean/std/p95 latency, an amortized per-horizon (ms/frame) figure, a per-submodule
stage breakdown, peak memory, parameter count, and FLOPs (if `fvcore`/`thop` is installed —
degrades to a clearly-labeled "unavailable" otherwise, never a crash).

### Robustness (sensor dropout)

```bash
# Full default scenario grid, random-init smoke test
python tools/run_robustness.py --config tiny --random-init --device cpu --num-batches 4

# Trained checkpoint, a chosen subset of scenarios, JSON out
python tools/run_robustness.py --config cam4docc_2s --checkpoint work_dirs/drift/latest.pth \
    --dataset cam4docc --data-root /data/cam4docc --ann-file val.json \
    --scenarios clean,lidar_only,camera_only,moderate_dropout,severe_dropout \
    --json-out results/robustness_cam4docc_2s.json
```

Runs the model under `clean` / `lidar_only` / `camera_only` / `moderate_dropout` (0.3, 0.3) /
`severe_dropout` (0.7, 0.7) — selectable via `--scenarios` — and reports absolute IoU plus
percent retained vs. the both-present baseline. This is the direct measurement of what CMLI
buys: graceful degradation instead of a cliff.

## 4. Configs and ablation presets

Plain Python dataclasses, `configs/base.py` + `configs/drift_fusion_nuscenes.py`, no `mmcv`
dependency (spec §6). `python -c "from configs import list_configs; print(list_configs())"`:

| Preset | Purpose |
|---|---|
| `cam4docc_2s` | **Default.** `T_p=3, T_f=4, T_o=6`, +2.0s horizon. Cam4DOcc/OccProphet-comparable protocol. |
| `extended_3s` | `T_p=3, T_f=6, T_o=8`, +3.0s horizon. **Not** directly comparable to Cam4DOcc numbers — see [Known limitations](#5-known-limitations). |
| `no_cmli` | Ablation: CMLI disabled — a missing modality is zeroed, not imagined. |
| `no_instance_path` | Ablation: the ★NOVEL instance-query dynamic path replaced by a dense-flow-field baseline (`drift.models.drift._DenseFlowDynamicPath`), DFIT-OccWorld-style. |
| `no_uncertainty` | Ablation: the ★NOVEL per-voxel uncertainty head disabled. |
| `camera_only` | Ablation: no LiDAR encoder — pure camera perception. |
| `lidar_only` | Ablation: no camera encoder — pure LiDAR perception. |
| `fusion_sum` | Ablation: CVQG uses plain summation instead of the gated-fusion default. |
| `tiny` | CI / smoke-test config: `latent_size=(16,16,4)`, `num_queries=8`, CPU, `SyntheticOccDataset`. Not a paper preset — this is what every command in this README was actually run with. |

Every preset above `cam4docc_2s()` other than `tiny` inherits from it and overrides only the
one axis it varies, so the resulting ablation table (§4.1) is apples-to-apples.

### 4.1 Results / ablation table (fill in after training on real data)

Numbers below are placeholders — this repo has been verified end-to-end on synthetic data
only (see [Known limitations](#5-known-limitations)); no run against real nuScenes/Cam4DOcc
data has produced numbers yet. Fill in after running `tools/eval.py` against a trained
checkpoint on the `cam4docc_2s` val split (the only protocol-comparable preset — do not report
`extended_3s` numbers in this table).

| Config | `IOU_mean` | `IoU_c` | `IoU_f` | Flow EPE (m) | ECE | Params (M) | Latency (ms) |
|---|---|---|---|---|---|---|---|
| `cam4docc_2s` (full model) | | | | | | | |
| `no_cmli` | | | | | | | |
| `no_instance_path` | | | | | | | |
| `no_uncertainty` | | | | | | | |
| `camera_only` | | | | | | | |
| `lidar_only` | | | | | | | |
| `fusion_sum` | | | | | | | |

Robustness (fill in from `tools/run_robustness.py --config cam4docc_2s ...`):

| Scenario | `lidar_dropout` | `cam_dropout` | `IoU_c` | `IoU_f` | % retained |
|---|---|---|---|---|---|
| `clean` | 0.0 | 0.0 | | | 100.0% |
| `lidar_only` | 0.0 | 1.0 | | | |
| `camera_only` | 1.0 | 0.0 | | | |
| `moderate_dropout` | 0.3 | 0.3 | | | |
| `severe_dropout` | 0.7 | 0.7 | | | |

## 5. Known limitations

This implementation pass verified every module against `tiny` config + `SyntheticOccDataset`
on CPU (no GPU, no real nuScenes data were available in this environment): 48 unit/integration
tests pass, and `tools/train.py`, `tools/eval.py`, `tools/benchmark_latency.py`, and
`tools/run_robustness.py` all run end-to-end producing real numbers on synthetic data. The
following are honest, known gaps rather than bugs — read before citing any number this repo
produces:

1. **The +3.0s horizon is not Cam4DOcc-comparable.** nuScenes keyframes are 2 Hz (0.5s per
   step). Cam4DOcc's published protocol spans only +2.0s (`T_f=4`), which is exactly the
   `cam4docc_2s` preset — that is the only preset whose numbers are comparable to published
   Cam4DOcc/OccProphet results. `extended_3s` (`T_f=6`, +3.0s) is a real, useful config for
   probing how DRIFT's forecast quality decays further out, but its numbers must **never** be
   reported alongside or compared against published Cam4DOcc/OccProphet numbers — they are a
   different protocol. See `docs/DESIGN_SPEC.md` §1 for the source of this constraint.

2. **A duplicated `mat2pose_vec`.** `drift/data/ego_motion.py` owns the canonical
   `mat2pose_vec` (4×4 → 6-DoF pose vector), which `drift/data/collate.py` and the rest of the
   data pipeline import and use. `drift/models/observer.py` independently carries its own
   private copy, `_mat2pose_vec`, used only internally to build `Observer`'s ego-motion
   conditioning channels. This is harmless duplication (both compute the same quantity from
   the same convention, and neither is on a hot path) but is worth unifying — `Observer`
   should import the canonical implementation instead of maintaining a second one that could
   silently drift out of sync with it.

3. **The real-nuScenes path is now built, but has still never seen a real download.**
   `tools/prepare_nuscenes.py` builds the derived dataset from raw nuScenes, and
   `Cam4DOccDataset` reads it. Both halves are exercised in CI: `tests/test_preprocessing.py`
   pins the geometry (quaternion convention, oriented-box rasterization against analytic
   volume, sparse encode/decode round-trip through the dataset's own decoder), and the loader
   has been driven end-to-end through `train.py` and `eval.py` on fabricated files written in
   exactly the on-disk format the preprocessing script emits. What has **not** happened is a
   run against an actual nuScenes download — so anything that depends on real metadata
   (category coverage, `box_velocity` edge cases, scenes with unusual sensor timing) is still
   unvalidated. `slurm/01_smoke.slurm` exists precisely to surface that class of problem in
   ten minutes rather than six hours into a training run; read its class-balance output before
   trusting anything downstream.


4. **LSS splatting uses `scatter_add_`, not the optimized CUDA `bev_pool`.**
   `drift/models/encoders/camera_encoder.py`'s depth-splat lift is implemented with a plain,
   readable `scatter_add_` (marked `# TODO(perf)` at the call site, per spec §2.1), rather than
   the cumulative-sum-trick CUDA `bev_pool` kernel BEVDet/OccProphet-style implementations
   use. This is deliberately the clearer, more auditable implementation and is correct, but it
   is also slower — the latency numbers `tools/benchmark_latency.py` reports (and any paper
   efficiency claim built on them) will improve, likely substantially on GPU, once that swap
   is made. The stage-breakdown output of `tools/benchmark_latency.py` already isolates
   `camera_encoder`'s share of total latency, so the size of that potential improvement is
   directly measurable from its output today.

### Other notes from this implementation pass

- A second, more serious bug surfaced only once the loader ran at batch size > 1:
  `DRIFT.loss` sliced `batch['gt_boxes'][:1]` for the present-frame detection auxiliary.
  `gt_boxes` is `List[B][T_o]` — batch-major — so that expression sliced the *batch* down to
  one element while `instance_loss` still looped over all `B` states, raising `IndexError` for
  every batch size above 1. Every test in the suite ran `B=1`, where the two slicings happen to
  coincide, so it stayed invisible until the cam4docc loader was driven at `B=2`. Fixed to
  `[boxes[:1] for boxes in gt_boxes]`, with `tests/test_forward_backward.py::TestBatchSizeGreaterThanOne`
  (which deliberately uses `T_o != B`) pinning the contract so it cannot regress.

- `tools/train.py` had one small, now-fixed bug: `--max-iters N` (the exact pattern used in
  that script's own docstring example, and in every smoke-test command in this README) stopped
  training *before* the epoch-end checkpoint save, so a debug/CI run produced zero checkpoint
  files — silently defeating both the documented smoke-test use case and `--resume`. Fixed by
  reordering the save-then-break so an early stop still saves a checkpoint.
