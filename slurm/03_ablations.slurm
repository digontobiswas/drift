#!/bin/bash
#SBATCH --job-name=drift_abl
#SBATCH --output=logs/abl_%A_%a.out
#SBATCH --error=logs/abl_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --array=0-6
#
# The full ablation grid from docs/PROBLEM_STATEMENT_AND_EXPERIMENTS.md §8:
# the full model plus the six single-axis ablations, one array task each, all
# sharing identical hyperparameters so the resulting table is apples-to-apples.
#
#     cd $SCRATCH/drift && sbatch slurm/03_ablations.slurm
#
# Run a subset with e.g. --array=0,1,4 . Each task auto-resumes from its own
# latest.pth, so requeueing after a walltime kill costs nothing.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIGS=(
    cam4docc_gmo        # 0  full model
    no_cmli             # 1  - cross-modal latent imagination
    no_instance_path    # 2  - instance queries (dense flow instead)
    no_uncertainty      # 3  - per-voxel uncertainty head
    camera_only         # 4  no LiDAR encoder
    lidar_only          # 5  no camera encoder
    fusion_sum          # 6  sum fusion instead of gated
)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"

EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"

echo "[abl] task ${SLURM_ARRAY_TASK_ID:-0} -> config=$CONFIG"

CKPT="$DRIFT_CKPT_DIR/$CONFIG/latest.pth"
RESUME_ARG=()
[[ -f "$CKPT" ]] && RESUME_ARG=(--resume "$CKPT") && echo "[abl] auto-resuming from $CKPT"

LAUNCH=(python tools/train.py)
if [[ "$NGPU" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU" tools/train.py --distributed)
fi

"${LAUNCH[@]}" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU ))" \
    --device cuda \
    --log-interval 50 \
    "${RESUME_ARG[@]}"

echo "[abl] $CONFIG done -> $DRIFT_CKPT_DIR/$CONFIG"
echo "[abl] evaluate the whole grid afterwards with: sbatch slurm/04_eval.slurm"
