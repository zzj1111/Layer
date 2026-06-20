#!/usr/bin/env bash
# One-shot installer for verl-agent + ALFworld + GIGPO training environment.
#
# Verified locally against:
#   - Python 3.12.0 (conda env named 'verl')
#   - CUDA toolkit 12.8
#   - NVIDIA H200/H100 (compute capability 9.0)
#   - torch 2.8.0+cu128, vllm 0.11.0, ray 2.52.1, transformers 4.56.1
#   - alfworld 0.4.2, textworld 1.7.0
#
# Usage on a fresh server:
#   bash install/setup_verl_agent_env.sh [ENV_NAME] [PYTHON_VERSION]
#
# Examples:
#   bash install/setup_verl_agent_env.sh                  # default env=verl-agent, py=3.12
#   bash install/setup_verl_agent_env.sh my-env 3.10      # custom env name + py
#
# What this does:
#   1. Create / activate the named conda env
#   2. pip install vllm + flash-attn + verl (editable, this repo)
#   3. pip install ALFworld + textworld + gymnasium + stable-baselines3
#   4. Sanity-check every package can import
#
# What this DOES NOT do (handle separately):
#   - Download ALFworld game data (./scripts/extract_alfworld_data.sh handles
#     this from the bundled tarball in data/alfworld_tw/)
#   - Download Qwen base model (use huggingface_hub.snapshot_download)
set -euo pipefail

ENV_NAME="${1:-verl-agent}"
PY_VER="${2:-3.12}"

log() { echo "[$(date '+%H:%M:%S')] $*" ; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found in PATH"; exit 1; } ; }

log "==> precheck"
need conda
need git
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log "    repo dir: $REPO_DIR"
log "    env name: $ENV_NAME"
log "    python:   $PY_VER"

# ---- conda env -----------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "==> conda env '$ENV_NAME' already exists, reusing"
else
    log "==> creating conda env '$ENV_NAME' (python=$PY_VER)"
    conda create -n "$ENV_NAME" "python==$PY_VER" -y
fi

# Source conda for `conda activate` (login shells get it, scripts often don't)
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
log "    python: $(which python)  ($(python --version 2>&1))"

# ---- verl-core deps ------------------------------------------------------
log "==> installing verl core deps (vllm, flash-attn, this repo as -e)"
pip install --quiet --upgrade pip
pip install --quiet vllm==0.11.0
pip install --quiet flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install --quiet -e "$REPO_DIR"

# ---- ALFworld stack ------------------------------------------------------
log "==> installing ALFworld stack"
pip install --quiet gymnasium==0.29.1
pip install --quiet stable-baselines3==2.6.0
pip install --quiet alfworld   # pulls textworld + fast-downward-textworld

# textworld needs `python` (not `python3`) on PATH for fast-downward-textworld
# build. conda activate fixes this — verify.
if ! command -v python >/dev/null 2>&1; then
    log "WARNING: 'python' not on PATH; alfworld build may fail. Re-run inside activated env."
fi

# ---- sanity check --------------------------------------------------------
log "==> sanity check (importing packages)"
python <<'PYEOF'
import importlib.metadata as md
required = {
    "torch":              "core",
    "vllm":               "rollout engine",
    "ray":                "distributed",
    "wandb":              "logging",
    "transformers":       "tokenizer/model",
    "flash_attn":         "attn kernels",
    "alfworld":           "env package",
    "textworld":          "alfworld dep",
    "gymnasium":          "env API",
    "stable_baselines3":  "alfworld dep",
}
ok = []
miss = []
for pkg, purpose in required.items():
    try:
        m = __import__(pkg)
        try:
            v = md.version(pkg)
        except md.PackageNotFoundError:
            v = getattr(m, "__version__", "?")
        ok.append(f"  ✓ {pkg:20s} {v:15s} ({purpose})")
    except Exception as e:
        miss.append(f"  ✗ {pkg:20s} MISSING — {e}  ({purpose})")
print("\n".join(ok))
if miss:
    print("\n".join(miss))
    raise SystemExit(1)
print()
print("✓ all required packages present")
PYEOF

# ---- alfworld data check -------------------------------------------------
log "==> ALFworld data check"
ALF_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
if [[ -d "$ALF_DATA/json_2.1.1" ]]; then
    log "    ALFworld data found at $ALF_DATA  ✓"
else
    log "    ALFworld data NOT at $ALF_DATA"
    if [[ -f "$REPO_DIR/scripts/extract_alfworld_data.sh" ]]; then
        log "    extracting bundled tarball:"
        log "    bash $REPO_DIR/scripts/extract_alfworld_data.sh"
    else
        log "    OR run: alfworld-download   (needs internet)"
    fi
fi

log "==> DONE"
echo
echo "Activate with:"
echo "  conda activate $ENV_NAME"
echo
echo "Then launch training:"
echo "  cd $REPO_DIR"
echo "  SIZE=1.5B GPUS=0,1,2,3 bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --full --no-tmux"
