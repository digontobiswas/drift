#!/bin/bash
#SBATCH --job-name=drift_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#
# Train ONE config. Multi-GPU on a single node via torchrun.
#
#     cd $SCRATCH/drift
#     sbatch slurm/02_train.slurm                                   # cam4docc_gmo
#     sbatch --export=ALL,CONFIG=camera_only slurm/02_train.slurm
#
# To resume after a walltime kill, point RESUME at the checkpoint:
#     sbatch --export=ALL,RESUME=$DRIFT_CKPT_DIR/cam4docc_gmo/latest.pth slurm/02_train.slurm

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
    ${RESUME:+--resume "$RESUME"}

echo "[train] $CONFIG finished. Checkpoints in $DRIFT_CKPT_DIR/$CONFIG"
