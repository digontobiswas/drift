#!/bin/bash
#SBATCH --job-name=drift_report
#SBATCH --partition=gpu
#SBATCH --output=logs/report_%j.out
#SBATCH --error=logs/report_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Everything that turns finished evaluations into a deliverable: CSVs, the markdown
# tables, the quantitative figures, and the qualitative scene renders. Run it after
# slurm/04_eval.slurm (and 05_robustness.slurm, if that has been run) have written
# their JSONs.
#
#     cd $SCRATCH/drift && sbatch slurm/06_report.slurm
#
# Everything lands under $DRIFT_RESULTS_DIR:
#     csv/        machine-readable tables, one row per observation
#     figures/    the paper's quantitative figures, PNG + PDF
#     scenes/     one input-to-forecast page per sampled val scene
#     results.md  the markdown tables
#
# The GPU is needed only by the scene renders, which run the model. If the checkpoint
# is not there yet, that step is skipped and the CSVs and figures are still produced.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

CONFIG="${CONFIG:-cam4docc_gmo}"
CKPT="${CKPT:-$DRIFT_CKPT_DIR/$CONFIG/latest.pth}"
NUM_SCENES="${NUM_SCENES:-8}"

echo "[report] config=$CONFIG results=$DRIFT_RESULTS_DIR"

# --- 1. Tidy CSVs ---------------------------------------------------------
python tools/results_to_csv.py --results-dir "$DRIFT_RESULTS_DIR"

# --- 2. Markdown tables for the paper -------------------------------------
python tools/collect_results.py \
    --results-dir "$DRIFT_RESULTS_DIR" \
    --out "$DRIFT_RESULTS_DIR/results.md"

# --- 3. Quantitative figures ----------------------------------------------
python tools/plot_results.py \
    --csv-dir "$DRIFT_RESULTS_DIR/csv" \
    --out-dir "$DRIFT_RESULTS_DIR/figures"

# --- 4. Qualitative scene renders -----------------------------------------
# Needs the checkpoint and the val split; skipped rather than failed if the run has
# not produced a checkpoint yet, so steps 1-3 still deliver on a partial grid.
if [[ -f "$CKPT" ]]; then
    python tools/visualize_scenes.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --ann-file "$DRIFT_VAL_ANN_FILE" \
        --num-scenes "$NUM_SCENES" \
        --device cuda \
        --out-dir "$DRIFT_RESULTS_DIR/scenes"
else
    echo "[report] no checkpoint at $CKPT -- skipping the scene renders." >&2
fi

echo "[report] done. Everything is under $DRIFT_RESULTS_DIR:"
ls -la "$DRIFT_RESULTS_DIR"
