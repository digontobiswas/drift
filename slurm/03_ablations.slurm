#!/bin/bash
#SBATCH --job-name=drift_abl
#SBATCH --partition=gpu
#SBATCH --output=logs/abl_%A_%a.out
#SBATCH --error=logs/abl_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --signal=USR1@600
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

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"

echo "[abl] task ${SLURM_ARRAY_TASK_ID:-0} -> config=$CONFIG"

CKPT="$DRIFT_CKPT_DIR/$CONFIG/latest.pth"

# Baseline for the requeue decision below: "made progress" vs "died immediately".
read -r START_EPOCH START_STEP < <(python tools/ckpt_info.py "$CKPT")
echo "[abl] starting from epoch=$START_EPOCH step=$START_STEP"

LAUNCH=(python tools/train.py)
if [[ "$NGPU" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU" tools/train.py --distributed)
fi

# The command line is built as ONE array, with --resume appended only when there is something
# to resume from. The previous shape kept the resume flag in its own array and expanded it as
# "${RESUME_ARG[@]}" -- harmless on a modern shell, fatal here: the compute nodes run
# RHEL7-era bash 4.2, where expanding an EMPTY array under `set -u` raises "unbound variable"
# instead of expanding to nothing (bash 4.4 fixed this). Every ablation starts with no
# checkpoint, so all six array tasks died on that line inside a minute, before training began.
# One always-non-empty array sidesteps the version difference without the cryptic
# ${ARR[@]+"${ARR[@]}"} guard.
TRAIN_CMD=(
    "${LAUNCH[@]}"
    --config "$CONFIG"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU ))"
    --device cuda
    --log-interval 50
)
if [[ -f "$CKPT" ]]; then
    TRAIN_CMD+=(--resume "$CKPT")
    echo "[abl] auto-resuming from $CKPT"
else
    echo "[abl] no checkpoint at $CKPT -- starting $CONFIG from scratch"
fi

TRAIN_RC=0
"${TRAIN_CMD[@]}" || TRAIN_RC=$?

echo "[abl] $CONFIG exited with status $TRAIN_RC -> $DRIFT_CKPT_DIR/$CONFIG"

# --- self-requeue, per array task -------------------------------------------
# Each config resubmits ONLY ITSELF (--array=<this task>), so one config needing a
# fourth 72-hour chunk does not restart the six that already finished.
read -r END_EPOCH END_STEP < <(python tools/ckpt_info.py "$CKPT")
echo "[abl] reached epoch=$END_EPOCH step=$END_STEP (target $EPOCHS epochs)"

if [[ "${DRIFT_AUTO_REQUEUE:-1}" != "1" ]]; then
    echo "[abl] auto-requeue disabled (DRIFT_AUTO_REQUEUE=$DRIFT_AUTO_REQUEUE)"
elif [[ "$END_EPOCH" -ge "$EPOCHS" ]]; then
    echo "[abl] $CONFIG complete -- all $EPOCHS epochs done"
elif [[ "$END_STEP" -le "$START_STEP" ]]; then
    echo "[abl] !! $CONFIG made no progress (step $START_STEP -> $END_STEP); NOT resubmitting." >&2
    echo "[abl]    Inspect logs/abl_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.err" >&2
else
    echo "[abl] $CONFIG incomplete but progressing -- submitting follow-up for this task only"
    sbatch --array="${SLURM_ARRAY_TASK_ID:-0}" \
        --export=ALL,EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE" \
        "${SLURM_SUBMIT_DIR:-$PWD}/slurm/03_ablations.slurm"
fi

echo "[abl] evaluate the whole grid afterwards with: sbatch slurm/04_eval.slurm"
