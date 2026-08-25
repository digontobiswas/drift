#!/usr/bin/env bash
# Create and populate the drift_env virtualenv on an HPC login node.
#
#   bash scripts/setup_env.sh
#
# Handles the three things that actually break on PARAM-class clusters:
#   1. `python` is 2.7 even after `module load` -- we always use `python3`.
#   2. The module Python may be built without the `ssl` extension, which makes
#      pip unable to reach PyPI at all. We detect that and try, in order:
#         a. adding the module's own lib dir to LD_LIBRARY_PATH
#         b. any other python3 on the system that *does* have ssl
#         c. an offline wheelhouse (see scripts/make_wheelhouse.sh)
#   3. Compute nodes have no internet, so everything network-touching -- pip
#      and the pretrained ResNet weights -- must happen here, on the login node.
#
# Run this ONCE per cluster account. Slurm jobs then just source the venv.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DRIFT_ENV_DIR:-$REPO_ROOT/drift_env}"
PY_MODULE="${DRIFT_PY_MODULE:-apps/python/3.10.13/modulefile}"
WHEELHOUSE="${DRIFT_WHEELHOUSE:-$REPO_ROOT/wheelhouse}"

echo "=== DRIFT environment setup ==="
echo "repo:        $REPO_ROOT"
echo "venv:        $ENV_DIR"

# --- 1. Load the Python module -------------------------------------------
if command -v module >/dev/null 2>&1; then
    module load "$PY_MODULE" 2>/dev/null || echo "!! could not load $PY_MODULE (continuing with system python3)"
fi

PY_BIN="$(command -v python3 || true)"
if [[ -z "$PY_BIN" ]]; then
    echo "!! no python3 found. Run 'module avail python' and set DRIFT_PY_MODULE." >&2
    exit 1
fi
echo "python3:     $PY_BIN ($("$PY_BIN" --version 2>&1))"

# --- 2. Does this interpreter have working SSL? ---------------------------
has_ssl() { "$1" -c "import ssl" >/dev/null 2>&1; }

if ! has_ssl "$PY_BIN"; then
    echo "!! $PY_BIN has no ssl module -- pip cannot reach PyPI. Trying fixes..."

    # (a) The interpreter's own prefix often ships libssl/libcrypto that just
    #     are not on the loader path.
    PY_PREFIX="$("$PY_BIN" -c 'import sys; print(sys.base_prefix)')"
    for cand in "$PY_PREFIX/lib" "$PY_PREFIX/lib64" /usr/lib64 /usr/lib; do
        [[ -d "$cand" ]] || continue
        export LD_LIBRARY_PATH="$cand:${LD_LIBRARY_PATH:-}"
        if has_ssl "$PY_BIN"; then
            echo "   fixed by LD_LIBRARY_PATH=$cand"
            echo "   (add this to your Slurm scripts too)"
            break
        fi
    done
fi

if ! has_ssl "$PY_BIN"; then
    # (b) Any other python3 on the system with a working ssl.
    for cand in /usr/bin/python3.11 /usr/bin/python3.10 /usr/bin/python3.9 /usr/bin/python3; do
        if [[ -x "$cand" ]] && has_ssl "$cand"; then
            echo "   using $cand instead ($("$cand" --version 2>&1)) -- it has working ssl"
            PY_BIN="$cand"
            break
        fi
    done
fi

SSL_OK=0
has_ssl "$PY_BIN" && SSL_OK=1

# --- 3. Create the venv ---------------------------------------------------
if [[ ! -d "$ENV_DIR" ]]; then
    "$PY_BIN" -m venv "$ENV_DIR"
    echo "created venv at $ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
echo "venv python: $(python --version 2>&1)"

# --- 4. Install requirements ---------------------------------------------
PIP_ARGS=(--trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host download.pytorch.org)

TORCH_INDEX="${DRIFT_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

if [[ "$SSL_OK" == "1" ]]; then
    python -m pip install --upgrade pip "${PIP_ARGS[@]}"

    # Order matters. `--index-url` *replaces* PyPI rather than adding to it, and the
    # PyTorch index carries only torch's own packages. Installing torchvision against
    # it first makes pip try to build Pillow from source (no matching wheel there),
    # whose build then fails looking for pybind11 -- which also only lives on PyPI.
    # So: satisfy the PyPI-side dependencies first, then torch sees them as already
    # installed and never reaches for a source build.
    #
    # Pillow is pinned below 11.0 and forced wheel-only (--only-binary): Pillow 11+
    # dropped manylinux2014 wheels in favor of manylinux_2_28 (glibc >= 2.28), which
    # older HPC login nodes (glibc 2.17) can't use -- pip would silently fall back to
    # a from-source build, which then fails because the system gcc there predates
    # C99-by-default and can't compile Pillow's C sources. --only-binary makes that
    # failure an immediate, readable "no matching distribution" instead.
    echo "--- installing PyPI dependencies first (numpy, Pillow, ...) ---"
    if ! python -m pip install "${PIP_ARGS[@]}" --only-binary=:all: "numpy>=1.24,<2.0" "Pillow>=10.0,<11.0"; then
        echo "!! No prebuilt wheel for numpy/Pillow on this platform (--only-binary refused a source build)." >&2
        echo "   This usually means the glibc here is older than manylinux2014 (glibc 2.17) expects, which" >&2
        echo "   would be unusual. Check 'ldd --version' and consider scripts/make_wheelhouse.sh instead." >&2
        exit 1
    fi

    echo "--- installing PyTorch from $TORCH_INDEX ---"
    if ! python -m pip install "${PIP_ARGS[@]}" --index-url "$TORCH_INDEX" torch torchvision; then
        echo "!! CUDA-build install failed; retrying with PyPI as a fallback index." >&2
        echo "   (this may land a CPU-only torch -- check the verification output below)" >&2
        python -m pip install "${PIP_ARGS[@]}" --extra-index-url "$TORCH_INDEX" torch torchvision
    fi

    echo "--- installing the rest ---"
    python -m pip install "${PIP_ARGS[@]}" -r "$REPO_ROOT/requirements.txt"
elif [[ -d "$WHEELHOUSE" ]]; then
    echo "--- no SSL: installing offline from $WHEELHOUSE ---"
    python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REPO_ROOT/requirements.txt"
else
    cat >&2 <<EOF

!! This Python cannot reach PyPI (no ssl module) and no wheelhouse was found at
   $WHEELHOUSE

   Fix it from a machine that DOES have internet (your laptop, same OS/arch):

       pip download -r requirements.txt -d wheelhouse \\
           --platform manylinux2014_x86_64 --python-version 310 \\
           --only-binary=:all:
       scp -r wheelhouse <user>@<login-node>:$REPO_ROOT/

   then re-run this script.
EOF
    exit 1
fi

# --- 5. Verify ------------------------------------------------------------
echo "--- verifying ---"
python - <<'PY'
import sys

problems = []
print("python  :", sys.version.split()[0])

import numpy, torch
print("numpy   :", numpy.__version__)
if int(numpy.__version__.split(".")[0]) >= 2:
    problems.append("numpy is 2.x -- nuscenes-devkit needs <2.0; run: pip install 'numpy<2.0'")

print("torch   :", torch.__version__, "| cuda build:", torch.version.cuda)
if torch.version.cuda is None:
    problems.append(
        "torch has NO CUDA support (CPU-only build). Training will not use the GPU.\n"
        "         Reinstall with: pip install --index-url https://download.pytorch.org/whl/cu121 "
        "--force-reinstall torch torchvision"
    )
print("cuda available here:", torch.cuda.is_available(), "(False on a login node is normal)")

for mod in ("torchvision", "PIL", "nuscenes"):
    try:
        __import__(mod)
        print(f"{mod:8}: ok")
    except ImportError as exc:
        print(f"{mod:8}: MISSING ({exc})")
        problems.append(f"{mod} did not install")

if problems:
    print("\n!! problems found:")
    for p in problems:
        print("   -", p)
    sys.exit(1)
print("\nall good.")
PY

cat <<EOF

=== done ===
Activate in every Slurm script with:

    module load $PY_MODULE
    source $ENV_DIR/bin/activate

Next, on this login node (compute nodes have no internet):

    bash scripts/fetch_weights.sh
EOF
