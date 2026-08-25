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
import torchvision.models as tvm

# configs/base.py: CameraEncoderConfig.backbone is resnet50 for the real presets,
# resnet18 for `tiny`. Fetch both so every preset runs offline.
#
# Resolve the weights enum exactly the way drift/models/encoders/camera_encoder.py
# does -- `tvm.get_model_weights(name).DEFAULT`. Do NOT hardcode IMAGENET1K_V1
# here: for resnet50, DEFAULT is IMAGENET1K_V2 (resnet50-11ad3fa6.pth), so
# pinning V1 (resnet50-0676ba61.pth) cached a file the model never asks for. The
# compute node then tried to download the V2 file, failed (no internet), and
# silently fell back to random init -- which quietly invalidates every ablation
# number. Mirroring the model's own resolution keeps the two from drifting apart.
for name in ("resnet50", "resnet18"):
    weights = tvm.get_model_weights(name).DEFAULT
    tvm.get_model(name, weights=weights)
    print(f"  {name}: ok ({weights})")

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
