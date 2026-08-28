#!/bin/bash
#SBATCH --job-name=drift_prep
#SBATCH --partition=shared
#SBATCH --output=logs/prep_%A_%a.out
#SBATCH --error=logs/prep_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-15
#
# Build the derived Cam4DOcc-protocol ground truth from raw nuScenes.
# CPU only -- no GPU is requested or needed.
#
# Submit FROM $SCRATCH (cluster policy):
#     cd $SCRATCH/drift
#     sbatch --export=ALL,SPLIT=val   slurm/00_preprocess.slurm
#     sbatch --export=ALL,SPLIT=train slurm/00_preprocess.slurm
#
# The script is resumable: a sample whose .npz already exists is skipped, so a
# task that hits walltime can simply be resubmitted.
#
# After ALL array tasks finish, merge the shard indices:
#     python tools/merge_shards.py --out-root $DRIFT_DATA_ROOT --split val

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source slurm/env.sh

SPLIT="${SPLIT:-val}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"

echo "[prep] split=$SPLIT shard=$SHARD/$NUM_SHARDS"
echo "[prep] reading  : $NUSCENES_ROOT (read-only)"
echo "[prep] writing  : $DRIFT_DATA_ROOT"

python tools/prepare_nuscenes.py \
    --nuscenes-root "$NUSCENES_ROOT" \
    --out-root      "$DRIFT_DATA_ROOT" \
    --split         "$SPLIT" \
    --protocol      gmo \
    --T-p 3 --T-o 6 \
    --shard "$SHARD" --num-shards "$NUM_SHARDS" \
    ${SCENE_LIMIT:+--scene-limit "$SCENE_LIMIT"}

echo "[prep] shard $SHARD done"
