#!/usr/bin/env bash
# Data-parallel eval: 8 independent vllm instances, one per GPU, stride-sharded prompts.
# Usage:
#   bash examples/eval/run_eval_dp8.sh <CKPT_PATH> [TAG]
# CKPT_PATH must contain config.json (i.e. the actor/huggingface dir for verl ckpts).
set -euo pipefail

CKPT="${1:?need ckpt path (the dir containing config.json)}"
TAG="${2:-$(basename $(dirname $(dirname "$CKPT")))}"   # exp name
# GPU_LIST overrides N_GPU: "2,3,6,7" -> 4 shards on those GPUs.
# N_GPU alone defaults to "0,1,...,N_GPU-1".
if [ -n "${GPU_LIST:-}" ]; then
    IFS=',' read -r -a GPU_ARR <<< "$GPU_LIST"
else
    N_GPU="${N_GPU:-8}"
    GPU_ARR=($(seq 0 $((N_GPU - 1))))
fi
N_SHARD=${#GPU_ARR[@]}
TEMPLATE="${TEMPLATE:-r1d}"
MAX_TOKENS="${MAX_TOKENS:-32000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
TASKS="${TASKS:-[\"aime\",\"amc\",\"math\",\"minerva\",\"olympiad_bench\",\"aime25\"]}"
DATASET="${DATASET:-data/evaluation_suite_v2}"
PY="${PY:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda/bin/python}"
SAVE_DIR="${SAVE_DIR:-/tmp/eval_results/${TAG}}"
mkdir -p "${SAVE_DIR}"
LOG_DIR="${SAVE_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "[DP-${N_SHARD}] eval $CKPT"
echo "  tag=$TAG  template=$TEMPLATE  max_tokens=$MAX_TOKENS  gpus=${GPU_ARR[*]}  save=$SAVE_DIR"

PIDS=()
for i in $(seq 0 $((N_SHARD - 1))); do
    GPU_ID="${GPU_ARR[$i]}"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    nohup "$PY" examples/eval/evaluate_model.py \
        --model_name "$CKPT" \
        --template "$TEMPLATE" \
        --dataset_name "$DATASET" \
        --tasks "$TASKS" \
        --temperature 0 --top_p 1 \
        --max_tokens "$MAX_TOKENS" --max_model_len "$MAX_MODEL_LEN" \
        --n_samples 1 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --shard "${i}/${N_SHARD}" \
        --save_dir "$SAVE_DIR" \
        > "${LOG_DIR}/shard${i}_gpu${GPU_ID}.log" 2>&1 &
    PIDS+=($!)
    echo "  shard ${i} on GPU ${GPU_ID}  pid=${PIDS[-1]}"
done

echo "[wait] all ${N_SHARD} shards running, monitoring..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=$((FAIL + 1))
done
[ $FAIL -gt 0 ] && { echo "[ERROR] $FAIL shards failed, check ${LOG_DIR}/"; exit 1; }

echo "[merge] aggregating shards -> ${SAVE_DIR}/final.json"
"$PY" examples/eval/merge_eval_shards.py "$SAVE_DIR"
