#!/bin/bash
#SBATCH --job-name=drift_train
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --signal=USR1@600
#
# Train ONE config. Multi-GPU on a single node via torchrun.
#
#     cd $SCRATCH/drift
#     sbatch slurm/02_train.slurm                                   # cam4docc_gmo
#     sbatch --export=ALL,CONFIG=camera_only slurm/02_train.slurm
#
# SURVIVING THE 72-HOUR WALLTIME CAP
# ----------------------------------
# A full run is ~4 days but the gpu partition allows 72 hours, so every run is
# interrupted at least once. That is handled end to end, with no manual step:
#
#   1. `--signal=USR1@600` makes Slurm send SIGUSR1 ten minutes before the hard
#      kill. tools/train.py catches it, finishes the step in flight, writes a
#      checkpoint at that exact position, and exits cleanly.
#   2. On startup this script auto-resumes from `latest.pth` if one exists.
#   3. When the run ends without having reached the final epoch, it submits its
#      own successor, which picks up from that checkpoint.
#
# So `sbatch slurm/02_train.slurm` once is enough; the chain continues on its own
# until all EPOCHS are done. Set DRIFT_AUTO_REQUEUE=0 to disable step 3.
#
# To resume from a specific checkpoint instead, set RESUME explicitly:
#     sbatch --export=ALL,RESUME=$DRIFT_CKPT_DIR/cam4docc_gmo/epoch_3.pth slurm/02_train.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIG="${CONFIG:-cam4docc_gmo}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-1}"       # per GPU
NGPU="${NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

echo "[train] config=$CONFIG epochs=$EPOCHS gpus=$NGPU batch/gpu=$BATCH_SIZE"

# Auto-resume: if a checkpoint for this config already exists and RESUME was not
# set explicitly, pick it up. Makes requeue-after-walltime a no-op.
CKPT="$DRIFT_CKPT_DIR/$CONFIG/latest.pth"
if [[ -z "${RESUME:-}" && -f "$CKPT" ]]; then
    RESUME="$CKPT"
    echo "[train] auto-resuming from $RESUME"
fi

# Record where we start, so the requeue logic below can tell "interrupted with
# progress" (resubmit) from "died immediately" (do NOT resubmit -- that would spawn
# an endless chain of failing jobs burning the queue).
read -r START_EPOCH START_STEP < <(python tools/ckpt_info.py "$CKPT")
echo "[train] starting from epoch=$START_EPOCH step=$START_STEP"

LAUNCH=(python tools/train.py)
if [[ "$NGPU" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU" tools/train.py --distributed)
fi

# `set -e` would abort before the requeue logic if training exits non-zero, so the
# exit status is captured rather than allowed to kill the script.
TRAIN_RC=0
"${LAUNCH[@]}" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU ))" \
    --device cuda \
    --log-interval 50 \
    ${RESUME:+--resume "$RESUME"} || TRAIN_RC=$?

echo "[train] $CONFIG exited with status $TRAIN_RC. Checkpoints in $DRIFT_CKPT_DIR/$CONFIG"

# --- self-requeue ------------------------------------------------------------
read -r END_EPOCH END_STEP < <(python tools/ckpt_info.py "$CKPT")
echo "[train] reached epoch=$END_EPOCH step=$END_STEP (target $EPOCHS epochs)"

if [[ "${DRIFT_AUTO_REQUEUE:-1}" != "1" ]]; then
    echo "[train] auto-requeue disabled (DRIFT_AUTO_REQUEUE=$DRIFT_AUTO_REQUEUE)"
elif [[ "$END_EPOCH" -ge "$EPOCHS" ]]; then
    echo "[train] training complete -- all $EPOCHS epochs done, not resubmitting"
elif [[ "$END_STEP" -le "$START_STEP" ]]; then
    # No forward progress: whatever went wrong will go wrong again immediately.
    echo "[train] !! no progress this run (step $START_STEP -> $END_STEP); NOT resubmitting." >&2
    echo "[train]    Inspect logs/train_${SLURM_JOB_ID}.err before submitting again." >&2
else
    echo "[train] incomplete but progressing -- submitting the follow-up job"
    sbatch --export=ALL,CONFIG="$CONFIG",EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE" \
        "${SLURM_SUBMIT_DIR:-$PWD}/slurm/02_train.slurm"
fi
