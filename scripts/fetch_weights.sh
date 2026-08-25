#!/usr/bin/env bash
# Pre-download every network-fetched asset into the local torch hub cache.
#
# MUST be run on a LOGIN NODE. Compute nodes have no internet, so a training job
# that first touches the network at `CameraEncoder(pretrained=True)` will either
# hang until walltime or silently fall back to random init -- which quietly
# invalidates every number in the ablation table.
#
#   bash scripts/fetch_weights.sh
#
# Cache location follows $TORCH_HOME (default ~/.cache/torch). If $HOME has a
# small quota, set TORCH_HOME to a scratch path and export the same value in
# your Slurm scripts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DRIFT_ENV_DIR:-$REPO_ROOT/drift_env}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

if [[ -d "$ENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
fi

echo "TORCH_HOME=$TORCH_HOME"
mkdir -p "$TORCH_HOME/hub/checkpoints"

python - <<'PY'
import os
import ssl
import torch
import urllib.request

# On some HPC login nodes, the system CA bundle is outdated. Workaround:
# create an unverified SSL context for torch hub downloads only.
# This is pragmatic for a login-node-only weight fetch (compute nodes
# are offline anyway and never make network calls).
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass  # python < 3.10 doesn't have this attribute
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("fetching torchvision backbone weights ...")
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

# configs/base.py: CameraEncoderConfig.backbone is resnet50 for the real presets,
# resnet18 for `tiny`. Fetch both so every preset runs offline.
for name, fn, w in (
    ("resnet50", resnet50, ResNet50_Weights.IMAGENET1K_V1),
    ("resnet18", resnet18, ResNet18_Weights.IMAGENET1K_V1),
):
    fn(weights=w)
    print(f"  {name}: ok")

print(f"\ncached under {os.environ['TORCH_HOME']}/hub/checkpoints:")
for f in sorted(os.listdir(os.path.join(os.environ["TORCH_HOME"], "hub", "checkpoints"))):
    print("  ", f)
PY

cat <<EOF

=== done ===
Export this in every Slurm script so compute nodes read the cache instead of
trying to download:

    export TORCH_HOME=$TORCH_HOME
EOF
