# Shared environment for every DRIFT Slurm job. Sourced, never executed.
#
#   source "$(dirname "$0")/env.sh"
#
# This is the ONLY file that should contain cluster-specific paths. Edit it once
# after cloning; every job script and every config preset reads from here.

# --- Paths ----------------------------------------------------------------

# The repo is wherever THIS file lives, resolved at source time. Nothing is
# assumed about $HOME vs $SCRATCH: clone anywhere and the paths follow.
_ENV_SH="${BASH_SOURCE[0]:-$0}"
export DRIFT_REPO="${DRIFT_REPO:-$(cd "$(dirname "$_ENV_SH")/.." && pwd)}"

# Everything the project writes goes beside the repo, so a scratch-only workflow
# needs no $HOME at all. Override any of these by exporting before sourcing.
DRIFT_WORK_BASE="${DRIFT_WORK_BASE:-$(dirname "$DRIFT_REPO")}"

# Raw nuScenes. READ-ONLY: shared with another project, so nothing here ever
# writes inside it. Defaults to a sibling of the repo.
export NUSCENES_ROOT="${NUSCENES_ROOT:-$DRIFT_WORK_BASE/nuscenes-trainval}"

# Where the *derived* dataset (ground truth built by tools/prepare_nuscenes.py)
# is written. Must be outside NUSCENES_ROOT.
export DRIFT_DATA_ROOT="${DRIFT_DATA_ROOT:-$DRIFT_WORK_BASE/drift_data/cam4docc_gmo}"

# Train/val annotation indices, relative to DRIFT_DATA_ROOT.
export DRIFT_ANN_FILE="${DRIFT_ANN_FILE:-drift_train.json}"
export DRIFT_VAL_ANN_FILE="${DRIFT_VAL_ANN_FILE:-drift_val.json}"

# Checkpoints and results. configs/_apply_env_paths appends the preset name.
export DRIFT_CKPT_DIR="${DRIFT_CKPT_DIR:-$DRIFT_WORK_BASE/drift_runs}"
export DRIFT_RESULTS_DIR="${DRIFT_RESULTS_DIR:-$DRIFT_WORK_BASE/drift_results}"

# Pretrained weights, pre-fetched on the login node by scripts/fetch_weights.sh.
# Compute nodes have no internet; without this they cannot load ImageNet weights.
# Kept beside the work base rather than in $HOME, which is often quota-tight.
export TORCH_HOME="${TORCH_HOME:-$DRIFT_WORK_BASE/.torch_cache}"

# Fail loudly and early rather than 40 minutes into a job.
if [[ ! -d "$NUSCENES_ROOT" ]]; then
    echo "!! NUSCENES_ROOT does not exist: $NUSCENES_ROOT" >&2
    echo "   Set it explicitly, e.g.  export NUSCENES_ROOT=/scratch/.../nuscenes-trainval" >&2
    exit 1
fi

# --- Modules and virtualenv ----------------------------------------------

DRIFT_PY_MODULE="${DRIFT_PY_MODULE:-apps/python/3.10.13/modulefile}"
DRIFT_CUDA_MODULE="${DRIFT_CUDA_MODULE:-compiler/cuda/12.2}"
DRIFT_ENV_DIR="${DRIFT_ENV_DIR:-$DRIFT_REPO/drift_env}"

if command -v module >/dev/null 2>&1; then
    module load "$DRIFT_PY_MODULE"  2>/dev/null || true
    module load "$DRIFT_CUDA_MODULE" 2>/dev/null || true
fi

if [[ -f "$DRIFT_ENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$DRIFT_ENV_DIR/bin/activate"
else
    echo "!! virtualenv not found at $DRIFT_ENV_DIR -- run scripts/setup_env.sh on the login node" >&2
    exit 1
fi

# If setup_env.sh needed an LD_LIBRARY_PATH fix for the ssl module, mirror it here.
export LD_LIBRARY_PATH="${DRIFT_EXTRA_LD_PATH:-}${DRIFT_EXTRA_LD_PATH:+:}${LD_LIBRARY_PATH:-}"

# --- Runtime hygiene ------------------------------------------------------

export PYTHONUNBUFFERED=1                  # so .out logs stream instead of buffering
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# The gpu partition's V100s have 16 GiB and this model runs near that ceiling even at
# batch_size=1, so allocator fragmentation matters.
#
# NOT expandable_segments: this platform's CUDA build rejects it outright --
#   "Warning: expandable_segments not supported on this platform"
# -- so setting it did nothing at all while appearing to help. max_split_size_mb is
# the fallback that does work here: it stops the allocator from carving large free
# blocks into small ones it can never recombine, which is the fragmentation mode that
# bites when a few ~500 MB volumes are allocated and freed every iteration.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export HF_HUB_OFFLINE=1                    # never attempt a network fetch on a compute node
export TORCH_HUB_OFFLINE=1
export PYTHONPATH="$DRIFT_REPO:${PYTHONPATH:-}"

mkdir -p "$DRIFT_CKPT_DIR" "$DRIFT_RESULTS_DIR" "$DRIFT_REPO/logs"

echo "--- DRIFT job environment ---"
echo "  host          : $(hostname)"
echo "  job           : ${SLURM_JOB_ID:-<interactive>} ${SLURM_ARRAY_TASK_ID:+array task $SLURM_ARRAY_TASK_ID}"
echo "  repo          : $DRIFT_REPO"
echo "  work base     : $DRIFT_WORK_BASE"
echo "  nuscenes (ro) : $NUSCENES_ROOT"
echo "  data root     : $DRIFT_DATA_ROOT"
echo "  ckpt dir      : $DRIFT_CKPT_DIR"
echo "  python        : $(python --version 2>&1)"
echo "  torch_home    : $TORCH_HOME"
echo "-----------------------------"
