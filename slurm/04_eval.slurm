#!/bin/bash
#SBATCH --job-name=drift_eval
#SBATCH --partition=gpu
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-6
#
# Evaluate every trained ablation on the val split and dump one JSON per config
# into $DRIFT_RESULTS_DIR. Those JSONs are what fills the empty tables in
# README §4.1 and the ablation table in PROBLEM_STATEMENT_AND_EXPERIMENTS §8.
#
#     cd $SCRATCH/drift && sbatch slurm/04_eval.slurm
#
# Then collect them into markdown tables:
#     python tools/collect_results.py --results-dir $DRIFT_RESULTS_DIR

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIGS=(cam4docc_gmo no_cmli no_instance_path no_uncertainty camera_only lidar_only fusion_sum)
CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"
CKPT="${CKPT:-$DRIFT_CKPT_DIR/$CONFIG/latest.pth}"

if [[ ! -f "$CKPT" ]]; then
    echo "[eval] no checkpoint at $CKPT -- has $CONFIG finished training? skipping." >&2
    exit 0
fi

echo "[eval] config=$CONFIG checkpoint=$CKPT"

python tools/eval.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --ann-file "$DRIFT_VAL_ANN_FILE" \
    --batch-size "${BATCH_SIZE:-1}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
    --device cuda \
    --json-out "$DRIFT_RESULTS_DIR/eval_${CONFIG}.json"

echo "[eval] wrote $DRIFT_RESULTS_DIR/eval_${CONFIG}.json"
