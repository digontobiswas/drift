#!/bin/bash
#SBATCH --job-name=drift_abl
#SBATCH --partition=gpu
#SBATCH --output=logs/abl_%A_%a.out
#SBATCH --error=logs/abl_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --signal=B:USR1@600
#SBATCH --array=1-6
#
# The six single-axis ablations of the full model, one array task each:
#
#     cd $SCRATCH/drift && sbatch slurm/03_ablations.slurm
#
# THE FULL-MODEL ROW IS slurm/02_train.slurm, NOT TASK 0
# -------------------------------------------------------
# An ablation table compares like with like, so every task here runs under exactly the
# conditions of the main 02_train run: 1 GPU, batch 1, 12 epochs, the same steps per epoch
# and so the same LR schedule. That makes the main run the full-model row as-is. The
# previous version of this script asked for 2 GPUs per task; under DDP that halves the
# steps per epoch, and the resulting table showed every ablation beating the full model --
# a training-budget artefact, not a result. The resources above must stay identical to
# 02_train.slurm's (tests/test_slurm_scripts.py checks this).
#
# Task 0 (cam4docc_gmo) would train into the same checkpoint directory as the main run and
# is refused unless ABL_INCLUDE_FULL=1 is exported -- only for a grid run in its own
# DRIFT_CKPT_DIR, with no main run alive.
#
# More tasks than free GPUs is fine: the QOS caps a user at 4 GPUs, and tasks beyond that
# simply pend as QOSMaxGRESPerUser until one frees. Each task auto-resumes from its own
# latest.pth, crashes and walltime kills included, and resubmits only itself.
#
# Mail on each link's end/failure, as in 02_train.slurm:
#     sbatch --mail-type=END,FAIL --mail-user=you@example.com \
#            --export=ALL,DRIFT_MAIL=you@example.com slurm/03_ablations.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIGS=(
    cam4docc_gmo        # 0  full model (normally the 02_train run; see above)
    no_cmli             # 1  - cross-modal latent imagination
    no_instance_path    # 2  - instance queries (dense flow instead)
    no_uncertainty      # 3  - per-voxel uncertainty head
    camera_only         # 4  no LiDAR encoder
    lidar_only          # 5  no camera encoder
    fusion_sum          # 6  sum fusion instead of gated
)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG="${CONFIGS[$TASK]}"

if [[ "$TASK" == "0" && "${ABL_INCLUDE_FULL:-0}" != "1" ]]; then
    echo "[abl] !! task 0 (cam4docc_gmo) shares its checkpoint directory with the main" >&2
    echo "[abl]    02_train run, whose result already is the full-model row. Refusing." >&2
    echo "[abl]    Set ABL_INCLUDE_FULL=1 only with a separate DRIFT_CKPT_DIR." >&2
    exit 1
fi

EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NGPU="$(python -c 'import torch; print(torch.cuda.device_count())')"

echo "[abl] task $TASK -> config=$CONFIG epochs=$EPOCHS gpus=$NGPU batch/gpu=$BATCH_SIZE"

CKPT="$DRIFT_CKPT_DIR/$CONFIG/latest.pth"

# Baseline for the requeue decision below: "made progress" vs "died immediately".
read -r START_EPOCH START_STEP < <(python tools/ckpt_info.py "$CKPT")
echo "[abl] starting from epoch=$START_EPOCH step=$START_STEP"

LAUNCH=(python tools/train.py)
if [[ "$NGPU" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NGPU" tools/train.py --distributed)
fi

# The command line is built as ONE array, with --resume appended only when there is something
# to resume from. Compute nodes run RHEL7-era bash 4.2, where expanding an EMPTY array under
# `set -u` raises "unbound variable" (bash 4.4 fixed this); an earlier separate RESUME_ARG
# array killed all six tasks inside a minute. One always-non-empty array avoids it.
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
# Extra train.py flags for one config, carried to its follow-up jobs by --export=ALL. Only
# for settings that leave the arithmetic unchanged, so the ablation stays comparable -- e.g.
# lidar_only, which segfaulted on the same sample eight restarts in a row and ran through it
# with gradient checkpointing and data-loader workers off:
#     ( export ABL_EXTRA_ARGS="--no-grad-checkpoint --num-workers 0" CUDA_LAUNCH_BLOCKING=1
#       sbatch --array=5 slurm/03_ablations.slurm )
# Appended last, so a repeated flag (--num-workers) overrides the default above. Split into
# its own array only inside this guard: an empty array expanded under `set -u` is an error
# on the compute nodes' bash 4.2.
if [[ -n "${ABL_EXTRA_ARGS:-}" ]]; then
    read -r -a ABL_EXTRA <<< "$ABL_EXTRA_ARGS"
    TRAIN_CMD+=("${ABL_EXTRA[@]}")
    echo "[abl] extra train.py flags: ${ABL_EXTRA_ARGS}"
fi
echo "[abl] CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"

# Through run_forwarding_usr1 (slurm/forward_signal.sh), so the walltime warning reaches
# train.py and this script survives to submit the follow-up job.
TRAIN_RC=0
run_forwarding_usr1 "${TRAIN_CMD[@]}" || TRAIN_RC=$?

echo "[abl] $CONFIG exited with status $TRAIN_RC -> $DRIFT_CKPT_DIR/$CONFIG"

# --- self-requeue, per array task -------------------------------------------
# Each config resubmits ONLY ITSELF (--array=<this task>), so one config needing another
# 72-hour chunk does not restart the others.
#
# A run that banked no progress is retried, up to DRIFT_MAX_STALLS in a row, exactly as
# 02_train.slurm does. This used to stop the chain on the first such run, but the crashes
# here are stochastic: the main run restarted ~20 times in its first five days, several
# times dying within minutes, i.e. before its first 50-step checkpoint. Under the old rule
# each of those would have ended an ablation for good, silently.
STALLS="${DRIFT_STALL_COUNT:-0}"
MAX_STALLS="${DRIFT_MAX_STALLS:-8}"

read -r END_EPOCH END_STEP < <(python tools/ckpt_info.py "$CKPT")
echo "[abl] reached epoch=$END_EPOCH step=$END_STEP (target $EPOCHS epochs)"

requeue() {  # $1 = stall count to carry into the next job
    # shellcheck disable=SC2086  # intentionally unquoted: expands to nothing or to two flags
    sbatch ${DRIFT_MAIL:+--mail-type=END,FAIL --mail-user=$DRIFT_MAIL} \
        --array="$TASK" \
        --export=ALL,EPOCHS="$EPOCHS",BATCH_SIZE="$BATCH_SIZE",DRIFT_STALL_COUNT="$1" \
        "${SLURM_SUBMIT_DIR:-$PWD}/slurm/03_ablations.slurm"
}

if [[ "${DRIFT_AUTO_REQUEUE:-1}" != "1" ]]; then
    echo "[abl] auto-requeue disabled (DRIFT_AUTO_REQUEUE=$DRIFT_AUTO_REQUEUE)"
elif [[ "$END_EPOCH" -ge "$EPOCHS" ]]; then
    echo "[abl] $CONFIG complete -- all $EPOCHS epochs done"
elif [[ "$END_STEP" -le "$START_STEP" ]]; then
    STALLS=$(( STALLS + 1 ))
    if [[ "$STALLS" -ge "$MAX_STALLS" ]]; then
        echo "[abl] !! $CONFIG: $STALLS consecutive runs banked no progress; stopping its chain." >&2
        echo "[abl]    Inspect logs/abl_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}.err" >&2
        echo "[abl]    Restart with: sbatch --array=$TASK slurm/03_ablations.slurm" >&2
        exit 1
    fi
    echo "[abl] $CONFIG made no progress (step $START_STEP -> $END_STEP), stall $STALLS/$MAX_STALLS -- retrying"
    requeue "$STALLS"
else
    echo "[abl] $CONFIG incomplete but progressing -- submitting follow-up for this task only"
    requeue 0
fi

echo "[abl] evaluate the whole grid afterwards with: sbatch slurm/04_eval.slurm"
