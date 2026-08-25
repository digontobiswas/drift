#!/usr/bin/env bash
# Build an offline wheelhouse on a machine WITH internet, for a cluster whose
# Python cannot reach PyPI (no ssl module).
#
# Run this on your laptop, then copy the folder across:
#
#   bash scripts/make_wheelhouse.sh
#   scp -r wheelhouse <user>@<login-node>:~/drift-project/drift/
#
# On the cluster, scripts/setup_env.sh picks it up automatically.
#
# The wheels are downloaded for linux-x86_64 / CPython 3.10 regardless of the
# machine running this script, so building on Windows or macOS is fine.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$REPO_ROOT/wheelhouse}"
PYVER="${DRIFT_TARGET_PYVER:-310}"
PLATFORM="${DRIFT_TARGET_PLATFORM:-manylinux2014_x86_64}"

mkdir -p "$DEST"
echo "building wheelhouse for cp$PYVER / $PLATFORM in $DEST"

# PyTorch comes from its own index (the PyPI build has no CUDA support).
python -m pip download \
    --dest "$DEST" \
    --index-url https://download.pytorch.org/whl/cu121 \
    --platform "$PLATFORM" \
    --python-version "$PYVER" \
    --only-binary=:all: \
    torch torchvision

python -m pip download \
    --dest "$DEST" \
    --platform "$PLATFORM" \
    --python-version "$PYVER" \
    --only-binary=:all: \
    -r "$REPO_ROOT/requirements.txt"

echo
echo "$(find "$DEST" -name '*.whl' | wc -l) wheels, $(du -sh "$DEST" | cut -f1) total"
echo "copy it to the cluster:  scp -r $DEST <user>@<login-node>:<repo>/"
