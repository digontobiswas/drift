# Running DRIFT on the HPC cluster

End-to-end runbook: clone → environment → preprocess → smoke test → train → results.

The workflow assumes the local → GitHub → HPC loop:

```
laptop (VS Code)  --git push-->  GitHub  --git pull-->  HPC working dir (on $SCRATCH)  --sbatch-->  compute nodes
```

Everything cluster-specific lives in **one** file: [`slurm/env.sh`](../slurm/env.sh).
No config or job script hardcodes a path.

---

## 0. Ground rules for this cluster

| Rule | Why it matters here |
|---|---|
| `python` is 2.7 — always use `python3` | The `apps/python/3.10.13` module does not repoint `python` |
| Work and submit from your scratch directory, not `$HOME` | Cluster policy; `$HOME` is also quota-tight |
| Compute nodes have **no internet** | Pretrained weights and pip must be handled on the **login node** first |
| The nuScenes tree is **read-only** | It is shared with another running project — nothing here writes into it |
| `screen` sessions are login-node-local | Note which login node you started one on |

---

## 1. Clone and configure

Clone into the scratch directory you work in — the same one holding
`nuscenes-trainval`:

```bash
cd /scratch/scratch26/$USER          # your working directory
git clone https://github.com/digontobiswas/drift.git
cd drift
```

`slurm/env.sh` resolves every path from its own location, so cloning into your
working directory is usually all the configuration needed. With the repo at
`<workdir>/drift`, it derives:

```bash
DRIFT_REPO=<workdir>/drift                          # where this file lives
NUSCENES_ROOT=<workdir>/nuscenes-trainval           # read-only source
DRIFT_DATA_ROOT=<workdir>/drift_data/cam4docc_gmo   # derived GT (new, writable)
DRIFT_CKPT_DIR=<workdir>/drift_runs
DRIFT_RESULTS_DIR=<workdir>/drift_results
TORCH_HOME=<workdir>/.torch_cache
```

Any of these can be overridden by exporting it before sourcing. If
`NUSCENES_ROOT` does not exist, sourcing fails immediately with the path it
tried — rather than 40 minutes into a job.

The derived dataset goes beside the repo, never inside `$NUSCENES_ROOT`.
`tools/prepare_nuscenes.py` refuses to start if you point `--out-root` inside the
nuScenes tree, so that mistake cannot happen silently.

---

## 2. Environment (login node, once)

```bash
bash scripts/setup_env.sh
```

This loads the Python module, creates `drift_env`, and installs everything.

**If it reports that Python has no `ssl` module**, pip cannot reach PyPI at all.
The script tries two fixes automatically (`LD_LIBRARY_PATH`, then another
`python3` on the system). If both fail, build the wheels on your laptop:

```bash
# on your laptop, in the repo
bash scripts/make_wheelhouse.sh
scp -r wheelhouse biswabandhurj@login02:/scratch/scratch26/biswabandhurj/drift/
# back on the cluster
bash scripts/setup_env.sh          # picks up wheelhouse/ automatically
```

Then pre-fetch the ImageNet weights — **this step is not optional**:

```bash
bash scripts/fetch_weights.sh
```

Without it, a training job on a compute node either hangs on a network call
until walltime or silently falls back to random initialization, which would
quietly invalidate every number in the ablation table.

---

## 3. Preprocess nuScenes

`Cam4DOccDataset` reads *derived* ground truth, not raw nuScenes.
`tools/prepare_nuscenes.py` builds it.

What it produces, per sample, is one compressed `.npz` holding calibration,
ego-motion, sparse occupancy/instance GT and latent-resolution flow.
**Images and point clouds are not copied** — the annotation JSON stores absolute
paths back into the read-only nuScenes tree and the dataset decodes them lazily.
That keeps the derived dataset ~12 MB/sample instead of ~500 MB.

Check the disk cost before committing to it:

```bash
source slurm/env.sh
python tools/prepare_nuscenes.py --nuscenes-root "$NUSCENES_ROOT" \
    --out-root "$DRIFT_DATA_ROOT" --split val --dry-run
```

Rough totals: **val ≈ 70 GB**, **train ≈ 340 GB**. Check your quota
(`lfs quota -u $USER /scratch` or equivalent) first.

Then run it — val first, so you can smoke-test while train is still building:

```bash
sbatch --export=ALL,SPLIT=val   slurm/00_preprocess.slurm
sbatch --export=ALL,SPLIT=train slurm/00_preprocess.slurm
```

Both are 16-task CPU-only array jobs and are **resumable** — a task killed at
walltime can just be resubmitted; existing `.npz` files are skipped.

To try a small subset first: `--export=ALL,SPLIT=val,SCENE_LIMIT=20`.

When all array tasks for a split finish, merge the shard indices:

```bash
python tools/merge_shards.py --out-root $DRIFT_DATA_ROOT --split val
python tools/merge_shards.py --out-root $DRIFT_DATA_ROOT --split train
```

---

## 4. Smoke test (do this before any long job)

```bash
sbatch slurm/01_smoke.slurm
```

~10 minutes on one GPU, and it answers the three questions that otherwise
surface six hours into a 24-hour run:

1. Does the venv import torch with a working CUDA runtime on a compute node?
2. Does a real preprocessed sample load with correct shapes and class balance?
3. Does forward+backward fit in GPU memory at this batch size?

Read `logs/smoke_<jobid>.out` before going further. In particular check the
`gt_occ` class balance block — if GMO (class 2) is 0% on every sample, the
preprocessing found no movable objects and something is wrong upstream.

---

## 5. Train

One config:

```bash
sbatch slurm/02_train.slurm                                    # cam4docc_gmo
sbatch --export=ALL,CONFIG=camera_only slurm/02_train.slurm
```

The whole ablation grid (7 configs, one array task each):

```bash
sbatch slurm/03_ablations.slurm
```

Both auto-resume from `$DRIFT_CKPT_DIR/<config>/latest.pth`, so requeueing
after a walltime kill costs nothing. Multi-GPU is handled with `torchrun`
whenever more than one GPU is allocated.

**Learning rate**: the schedule is now warmup + cosine decay
(`TrainConfig.lr_scheduler = "cosine"`, 500 warmup steps, decaying to 1% of base
LR). Constant LR is still available with `lr_scheduler = "constant"` if you want
the old behaviour for a controlled comparison.

---

## 6. Results

```bash
sbatch slurm/04_eval.slurm                                       # all 7 configs
sbatch slurm/05_robustness.slurm                                 # full model
sbatch --export=ALL,CONFIG=no_cmli slurm/05_robustness.slurm     # CMLI comparison
```

Then turn the JSONs into the paper's tables:

```bash
python tools/collect_results.py --results-dir $DRIFT_RESULTS_DIR --out results.md
```

Configs that have not been evaluated yet appear as `—` with an explicit
"incomplete grid" note, so a half-finished grid never reads as a complete one.

---

## 7. Debugging loop

The realistic cadence is submit → read logs → fix → resubmit, many times:

```bash
squeue -u $USER                       # what is running
tail -f logs/train_<jobid>.out        # live progress
sacct -j <jobid> --format=JobID,State,ExitCode,MaxRSS,Elapsed
```

Common failures on this cluster:

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` | venv not activated in job | `source slurm/env.sh` at the top of the script |
| Job hangs at startup, no output | Trying to download weights | Run `scripts/fetch_weights.sh` on the login node |
| `CUDA out of memory` | Batch size too large | `--export=ALL,BATCH_SIZE=1`, or reduce `num_workers` |
| `annotation file not found` | Preprocessing incomplete | Run `tools/merge_shards.py` for that split |
| `AttributeError: np.bool` | numpy ≥ 1.24 with old third-party code | Already pinned `<2.0`; replace the alias with plain `bool` |
| Job rejected at submit | Submitted from `$HOME` | Clone and submit from your scratch working directory |

---

## 8. Iterating on code

Edit locally in VS Code → `git push` → on the cluster:

```bash
cd /scratch/scratch26/$USER/drift && git pull && sbatch slurm/...
```

`.gitignore` keeps the venv, checkpoints, logs and derived data out of git, so a
`git pull` never disturbs a running job's outputs.
