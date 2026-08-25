#!/bin/bash
#SBATCH --job-name=drift_robust
#SBATCH --output=logs/robust_%j.out
#SBATCH --error=logs/robust_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Sensor-degradation study (PROBLEM_STATEMENT_AND_EXPERIMENTS §7): the same
# checkpoint evaluated under five dropout scenarios, plus the latency/memory
# profile the Pareto plot needs.
#
# Run it for the full model and for no_cmli -- the gap between those two curves
# IS the CMLI claim, so both are required, not optional.
#
#     cd $SCRATCH/drift
#     sbatch slurm/05_robustness.slurm
#     sbatch --export=ALL,CONFIG=no_cmli slurm/05_robustness.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIG="${CONFIG:-cam4docc_gmo}"
CKPT="${CKPT:-$DRIFT_CKPT_DIR/$CONFIG/latest.pth}"

if [[ ! -f "$CKPT" ]]; then
    echo "[robust] no checkpoint at $CKPT" >&2
    exit 1
fi

echo "=== robustness: $CONFIG ==="
python tools/run_robustness.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --ann-file "$DRIFT_VAL_ANN_FILE" \
    --batch-size "${BATCH_SIZE:-1}" \
    --num-workers "${SLURM_CPUS_PER_TASK:-4}" \
    --device cuda \
    --json-out "$DRIFT_RESULTS_DIR/robustness_${CONFIG}.json"

echo
echo "=== latency / memory / FLOPs: $CONFIG (GPU) ==="
python tools/benchmark_latency.py \
    --config "$CONFIG" \
    --device cuda \
    --repeats 50 --warmup 10 \
    --json-out "$DRIFT_RESULTS_DIR/latency_${CONFIG}.json"

echo "[robust] results in $DRIFT_RESULTS_DIR"
