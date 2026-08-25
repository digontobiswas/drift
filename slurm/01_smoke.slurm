#!/bin/bash
#SBATCH --job-name=drift_smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Run this BEFORE any long training job. It answers, in ~10 minutes, the three
# questions that otherwise surface 6 hours into a 24-hour run:
#
#   1. Does the venv import torch with a working CUDA runtime on a compute node?
#   2. Does Cam4DOccDataset actually load a real preprocessed sample -- correct
#      shapes, images found on the read-only nuScenes tree, GT decoded?
#   3. Does a forward+backward step complete inside GPU memory at this batch size?
#
#     cd $SCRATCH/drift && sbatch slurm/01_smoke.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIG="${CONFIG:-cam4docc_gmo}"
ANN="${DRIFT_VAL_ANN_FILE}"

echo "=== 1/4  environment ==="
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"gpu memory     : {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB total")
else:
    raise SystemExit("!! no CUDA on this compute node -- check --gres=gpu:1 and the cuda module")
PY

echo
echo "=== 2/4  unit tests (synthetic, no data dependency) ==="
python -m pytest tests/ -q

echo
echo "=== 3/4  real-data loading ==="
python - <<PY
import os
from drift.data.cam4docc_dataset import Cam4DOccDataset

ds = Cam4DOccDataset(
    data_root=os.environ["DRIFT_DATA_ROOT"],
    ann_file="$ANN",
    img_hw=(256, 704),
)
print(f"dataset length : {len(ds)}")
s = ds[0]
print("--- sample 0 shapes ---")
for k, v in s.items():
    if hasattr(v, "shape"):
        print(f"  {k:12}: {tuple(v.shape)} {v.dtype}")
    elif isinstance(v, list):
        head = v[0]
        extra = f" first={tuple(head.shape)}" if hasattr(head, "shape") else ""
        print(f"  {k:12}: list[{len(v)}]{extra}")
    elif isinstance(v, dict):
        print(f"  {k:12}: " + ", ".join(f"{kk}{tuple(vv.shape)}" for kk, vv in v.items()))

occ = s["gt_occ"]
import torch
counts = torch.bincount(occ.reshape(-1), minlength=3).tolist()
total = occ.numel()
print("--- gt_occ class balance (0=free 1=GSO 2=GMO) ---")
for i, c in enumerate(counts):
    print(f"  class {i}: {c:>12,}  ({100*c/total:5.2f}%)")
if counts[2] == 0:
    print("  !! no GMO voxels in this sample -- fine for one frame, alarming if every sample is like this")
PY

echo
echo "=== 4/4  one forward+backward on GPU ==="
python tools/train.py \
    --config "$CONFIG" \
    --ann-file "$ANN" \
    --max-iters 3 \
    --batch-size "${BATCH_SIZE:-1}" \
    --num-workers 2 \
    --log-interval 1 \
    --device cuda \
    --ckpt-dir "$DRIFT_CKPT_DIR/_smoke"

echo
echo "=== smoke test passed -- safe to submit 02_train.slurm / 03_ablations.slurm ==="
