#!/bin/bash
#SBATCH --job-name=drift_train
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --signal=USR1@600
#
# Train ONE config.
#
#     cd $SCRATCH/drift
#     sbatch slurm/02_train.slurm                                   # cam4docc_gmo
#     sbatch --export=ALL,CONFIG=camera_only slurm/02_train.slurm
#
# WHY ONE GPU BY DEFAULT
# ----------------------
# The partition is 22 two-GPU nodes and they sit at one GPU busy, one free almost
# permanently: a 2-GPU request waits for a node where BOTH free up at once, which
# took over ten hours with an otherwise empty queue, while a 1-GPU request started
# in under a minute against the same cluster state. Two GPUs is a ~2x speedup that
# has to be paid for in queue time that has run far higher than that. Ask for two
# with:
#
#     sbatch --gres=gpu:2 --cpus-per-task=16 --mem=96G \
#            --export=ALL,NGPU=2 slurm/02_train.slurm
#
# NGPU must be exported to match: it is what selects torchrun over plain python.
# Checkpoints carry the GPU count they were written under and the resume position
# is rescaled, so switching back and forth mid-run is safe.
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
# Overridable so `NUM_WORKERS=0` can be tested without editing this file. Worker
# processes are the standing suspect for the SIGSEGVs that have not been explained:
# every crash lands in the autograd engine while eight queue-feeder threads are alive
# alongside it, and 0 workers removes them entirely at the cost of slower loading.
NUM_WORKERS="${NUM_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-8} / NGPU ))}"

echo "[train] config=$CONFIG epochs=$EPOCHS gpus=$NGPU batch/gpu=$BATCH_SIZE workers=$NUM_WORKERS"

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
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    --log-interval 50 \
    ${RESUME:+--resume "$RESUME"} || TRAIN_RC=$?

echo "[train] $CONFIG exited with status $TRAIN_RC. Checkpoints in $DRIFT_CKPT_DIR/$CONFIG"

# --- self-requeue ------------------------------------------------------------
#
# A run that banked no progress used to end the chain outright, on the reasoning that
# whatever killed it instantly would kill its successor too. That reasoning does not
# hold for the failure this actually hits: the SIGSEGVs are stochastic, striking
# anywhere from the first iteration to several thousand in, so a run that died before
# its first checkpoint is usually just unlucky rather than doomed. Ending the chain
# there meant the training sat idle until someone noticed and resubmitted by hand --
# which, with a crash every few thousand steps, was most of the elapsed time.
#
# So a stall is now tolerated a few times in a row, and only a genuinely stuck run --
# DRIFT_MAX_STALLS consecutive attempts with nothing to show -- stops the chain. The
# counter resets on any real progress, so an occasional unlucky run never accumulates
# toward the limit.
#
# The limit was 3, which turned out to be far too tight for the way this actually fails.
# A stalled run costs about seven minutes of queue and startup and nothing else, whereas
# hitting the limit stops training silently -- discovered hours later, by which point the
# GPU has sat idle the whole time. The two outcomes are not remotely symmetric, so the
# tolerance is set where a bad patch is survivable: eight stalls is under an hour of
# wasted scheduling, against a chain that no longer dies overnight unattended.
STALLS="${DRIFT_STALL_COUNT:-0}"
MAX_STALLS="${DRIFT_MAX_STALLS:-8}"

read -r END_EPOCH END_STEP < <(python tools/ckpt_info.py "$CKPT")
echo "[train] reached epoch=$END_EPOCH step=$END_STEP (target $EPOCHS epochs)"

requeue() {  # $1 = stall count to carry into the next job
    sbatch --export=ALL,CONFIG="$CONFIG",EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE",\
NUM_WORKERS="$NUM_WORKERS",NGPU="$NGPU",DRIFT_STALL_COUNT="$1" \
        "${SLURM_SUBMIT_DIR:-$PWD}/slurm/02_train.slurm"
}

if [[ "${DRIFT_AUTO_REQUEUE:-1}" != "1" ]]; then
    echo "[train] auto-requeue disabled (DRIFT_AUTO_REQUEUE=$DRIFT_AUTO_REQUEUE)"
elif [[ "$END_EPOCH" -ge "$EPOCHS" ]]; then
    echo "[train] training complete -- all $EPOCHS epochs done, not resubmitting"
elif [[ "$END_STEP" -le "$START_STEP" ]]; then
    STALLS=$(( STALLS + 1 ))
    if [[ "$STALLS" -ge "$MAX_STALLS" ]]; then
        echo "[train] !! $STALLS consecutive runs banked no progress; stopping the chain." >&2
        echo "[train]    This is no longer bad luck -- inspect logs/train_${SLURM_JOB_ID}.err." >&2
        echo "[train]    Restart with: sbatch slurm/02_train.slurm" >&2
    else
        echo "[train] no progress this run (step $START_STEP -> $END_STEP), stall $STALLS/$MAX_STALLS -- retrying"
        requeue "$STALLS"
    fi
else
    echo "[train] incomplete but progressing -- submitting the follow-up job"
    requeue 0
fi
