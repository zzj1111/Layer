#!/usr/bin/env bash
# ==============================================================================
# Paper-exact AIME evaluation (Skywork-OR1 / DeepSeek-R1-Distill setup)
# ==============================================================================
# Reproduces the math eval protocol of "Skywork Open Reasoner 1" (arXiv:2505.22312)
# and the DeepSeek-R1-Distill recommended settings:
#   benchmarks : AIME 2024 + AIME 2025   (task keys "aime","aime25")
#   metric     : Avg@32  (N_SAMPLES independent samples/problem, mean accuracy)
#   sampling   : temperature 0.6, top_p 1.0, top_k -1   (== Skywork or1_scripts/eval/eval_7b.sh)
#   length     : 32K generation (max_tokens 32768)
#   template   : r1d_box -> R1-Distill chat template, NO system prompt, with Skywork's
#                appended instruction " Let's think step by step and output the final
#                answer within \boxed{}." (verbatim from or1_data/eval/aime*.parquet);
#                boxed answer graded by oat_math_grader
#
# Runs data-parallel: one vLLM instance per GPU in GPU_LIST, then merges shards.
#
# Usage:
#   bash examples/eval/run_paper_eval_aime.sh <MODEL_DIR_or_HF_ID> [TAG]
#
# Example (base model on the 4 free GPUs):
#   GPU_LIST=0,1,6,7 bash examples/eval/run_paper_eval_aime.sh \
#     /mnt/data1/zha00175/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60 \
#     DeepSeek-R1-Distill-Qwen-7B-base
#
# Override anything via env: N_SAMPLES (Avg@K), TEMPERATURE, TOP_P, MAX_TOKENS,
# GPU_LIST, PY, SAVE_ROOT.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJ_DIR"   # evaluate_model.py + dataset paths are relative to the repo root

MODEL="${1:?need model path or HF id (a dir containing config.json)}"
TAG="${2:-$(basename "$MODEL")}"

# ---- paper-exact defaults (all overridable via env) ----
export TASKS="${TASKS:-[\"aime\",\"aime25\"]}"   # AIME24 (key "aime") + AIME25
export TEMPLATE="${TEMPLATE:-r1d_box}"           # R1-Distill template + Skywork boxed instruction
export TEMPERATURE="${TEMPERATURE:-0.6}"
export TOP_P="${TOP_P:-1.0}"                      # Skywork eval_7b.sh uses top_p=1.0
export N_SAMPLES="${N_SAMPLES:-32}"              # Avg@32
export MAX_TOKENS="${MAX_TOKENS:-32768}"         # 32K generation
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"   # 32K + prompt headroom
export GPU_LIST="${GPU_LIST:-0,1,6,7}"
export DATASET="${DATASET:-data/evaluation_suite_v2}"
export PY="${PY:-/mnt/data1/zha00175/miniconda/envs/verl/bin/python}"
export SAVE_ROOT="${SAVE_ROOT:-$HOME/eval_results/paper_aime}"
# keep HF fully offline if a complete local cache exists
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[paper-eval] model=$MODEL  tag=$TAG"
echo "             AIME24+AIME25  Avg@$N_SAMPLES  temp=$TEMPERATURE top_p=$TOP_P  max_tokens=$MAX_TOKENS  gpus=$GPU_LIST"
exec bash examples/eval/run_eval_dp8.sh "$MODEL" "$TAG"
