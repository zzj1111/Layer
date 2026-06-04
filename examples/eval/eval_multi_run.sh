#!/usr/bin/env bash
# Multi-seed eval: run eval_all_ckpts.py N independent times with temperature>0,
# each into its own save_root/runK subdir. Then call aggregate_multi_run.py.
#
# Each run uses time-based seed (vllm SamplingParams seed=time.time_ns()) so 8
# runs at temperature=0.6 give 8 different sample traces.
#
# Usage:
#   bash eval_multi_run.sh <ckpt_path_or_exp_dir> [N_RUNS=8]
#
# Override via env:
#   GPU_LIST, TEMPLATE, MAX_TOKENS, MAX_MODEL_LEN, TEMPERATURE, TOP_P,
#   N_SAMPLES, TASKS, SAVE_ROOT, WANDB_PROJECT (unused; we pass --no_wandb)
set -euo pipefail

CKPT_PATH="${1:?need ckpt path or exp dir}"
N_RUNS="${2:-8}"

# Defaults tuned for 1.5B Qwen-Math RL eval (override as needed).
TEMPLATE="${TEMPLATE:-qwen_math}"
MAX_TOKENS="${MAX_TOKENS:-3000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
N_SAMPLES="${N_SAMPLES:-1}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench,aime25}"
PY="${PY:-/mnt/data1/zha00175/miniconda/envs/verl/bin/python}"

# save_root_base: where the run1/, run2/, ... subdirs go.
EXP_TAG=$(basename "$CKPT_PATH")
SAVE_ROOT_BASE="${SAVE_ROOT_BASE:-$HOME/eval_results_multi/${EXP_TAG}_T${TEMPERATURE}}"
mkdir -p "$SAVE_ROOT_BASE"

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

echo "[multi-run] N_RUNS=$N_RUNS  T=$TEMPERATURE top_p=$TOP_P  n_samples=$N_SAMPLES"
echo "[multi-run] ckpt path : $CKPT_PATH"
echo "[multi-run] save base : $SAVE_ROOT_BASE"
echo "[multi-run] gpu list  : $GPU_LIST"
echo "[multi-run] template  : $TEMPLATE  max_tokens=$MAX_TOKENS"
echo ""

for i in $(seq 1 "$N_RUNS"); do
    SAVE_ROOT="$SAVE_ROOT_BASE/run$i"
    echo "=================================================="
    echo "[run $i/$N_RUNS] save_root=$SAVE_ROOT"
    echo "=================================================="
    GPU_LIST=$GPU_LIST \
    PY=$PY \
    "$PY" "$REPO_DIR/examples/eval/eval_all_ckpts.py" \
        "$CKPT_PATH" \
        --template "$TEMPLATE" \
        --max_tokens "$MAX_TOKENS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --n_samples "$N_SAMPLES" \
        --tasks "$TASKS" \
        --save_root "$SAVE_ROOT" \
        --no_wandb
done

echo ""
echo "=================================================="
echo "[aggregate] computing mean±std across $N_RUNS runs"
echo "=================================================="
"$PY" "$REPO_DIR/examples/eval/aggregate_multi_run.py" "$SAVE_ROOT_BASE"
