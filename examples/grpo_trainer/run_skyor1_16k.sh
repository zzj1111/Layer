#!/bin/bash
# ==============================================================================
# Skywork-OR1 (MAGIC) RL training — DeepSeek-R1-Distill-Qwen-7B  @ 16K context
# ==============================================================================
# Paper: "Skywork Open Reasoner 1 Technical Report"  (arXiv:2505.22312)
#        Recipe name: MAGIC = Multi-stage Adaptive entropy scheduling for
#        GRPO In Convergence.
#
# This is the **16K stage** of Skywork's multi-stage context schedule
# (8K -> 16K -> 32K).  By default it starts from the base distill model; pass
# --init-from <ckpt_hf_dir> to continue from your own 8K-stage checkpoint.
#
# ------------------------------------------------------------------------------
# Skywork-OR1 "MAGIC" recipe  ->  verl flag mapping
# ------------------------------------------------------------------------------
#   GRPO, KEEP group-std advantage norm   algorithm.adv_estimator=grpo
#                                          algorithm.norm_adv_by_std_in_grpo=True   <- NOTE: True (unlike Dr.GRPO)
#   No KL (neither in-reward nor loss)     algorithm.use_kl_in_reward=False
#                                          actor.use_kl_loss=False  kl_loss_coef=0
#   Symmetric clip 0.2                     actor.clip_ratio_low=0.2  clip_ratio_high=0.2
#   Token-level policy loss                actor.loss_agg_mode=token-mean
#   Constant LR (paper 1e-6)               actor.optim.lr=$LR  warmup=0   (DEFAULT here 5e-6; LR=1e-6 for paper)
#   Group size (rollouts/prompt) = 16      actor_rollout_ref.rollout.n=16
#   Sampling temperature 1.0               rollout.temperature=1.0 top_p=1.0 top_k=-1
#   Rejection sampling (keep groups with   algorithm.filter_groups.enable=True
#     non-zero advantage)                  algorithm.filter_groups.metric=seq_reward
#                                          algorithm.filter_groups.max_num_gen_batches=10
#   0/1 rule reward (boxed), no overlong   reward_model.reward_manager=dapo
#     shaping                              reward_model.overlong_buffer.enable=False
#   16K context                            data.max_response_length=16384
#   Eval = AIME24 + AIME25, Avg@K          data.val_files=[aime24,aime25]
#     (paper uses Avg@32; default 8 here)  val_kwargs.{n=8,do_sample=True,temperature=0.6,top_p=1.0}
#                                          -> verl reports val-core/aime24 and /aime25 separately.
#                                          NOTE eval runs at the 16K training context (paper evals at 32K),
#                                          so very long AIME solutions may truncate; raise MAX_RESPONSE for
#                                          a closer-to-paper eval at higher memory cost.
#
# NOTE on checkpoints/resume: we save FULL FSDP training state
#   (save_contents=[model,optimizer,extra,hf_model]) so trainer.resume_mode=auto + --resume actually
#   restores the policy/optimizer/RNG.  Saving only "hf_model" (as the older Dr.GRPO scripts do) makes
#   --resume silently restart from base weights — verl has no hf_model *load* branch.  Each kept step is
#   therefore larger (sharded model + optimizer); max_actor_ckpt_to_keep=1 bounds disk to one.
#
# IMPLEMENTED VIA the DAPO recipe trainer (recipe.dapo.main_dapo):
#   The stock verl.trainer.main_ppo loop does NOT implement filter_groups
#   (rejection / dynamic sampling); only recipe/dapo/dapo_ray_trainer.py does.
#   Skywork's rejection sampling is a core part of MAGIC, so we run through the
#   DAPO trainer.  Reward routing is identical (data_source=math_drgrpo ->
#   Oat boxed grader, called through default_compute_score).  Set FILTER=false
#   to disable rejection sampling and get plain GRPO behaviour.
#
# APPROXIMATION (the one place we can't match the paper exactly):
#   * Adaptive entropy control (target entropy 0.2): stock verl has no adaptive
#     entropy controller, only a FIXED entropy_coeff.  We expose ENTROPY_COEFF
#     (default 0, matching DAPO).  To chase the paper's "keep entropy >= 0.2"
#     behaviour, set e.g. ENTROPY_COEFF=0.001, or port Skywork's controller.
#
# verl patches assumed present in this repo (same as the Dr.GRPO scripts):
#   * data_source=math_drgrpo -> Oat boxed_reward_fn scorer
#   * actor.train_layer_ids=<ids> -> freeze all but these layers (layer-wise RL)
#
# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------
# Full RL (8x B200, single node):
#   bash run_skyor1_16k.sh
#
# 16K stage continuing from your own 8K-stage checkpoint:
#   bash run_skyor1_16k.sh --init-from /path/to/8k_ckpt/global_step_540/actor/huggingface
#
# Layer-wise RL on one layer (your research line):
#   bash run_skyor1_16k.sh --layer 14
#
# Sweep across settings (sequential on one node):
#   bash run_skyor1_16k.sh --layers "full 0 6 12 18 27"
#
# Multi-node parallel — same --layers everywhere, each node a --part:
#   node1: ... --layers "full 0..27" --part 1/4
#   node2: ... --layers "full 0..27" --part 2/4   (etc.)
#
# Resume / extend a previous run (same settings, more epochs):
#   EPOCHS=8 bash run_skyor1_16k.sh --resume
#
# Override hyperparameters via env:
#   BATCH_SIZE=256 MINI_BATCH=128 ROLLOUT_N=16 LR=1e-6 ENTROPY_COEFF=0.001 \
#     bash run_skyor1_16k.sh
# ==============================================================================

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

# ===== defaults =====
MODEL_PATH="${MODEL_PATH:-/code-fsx/hongpaul-sandbox/code/temp/DeepSeek-R1-Distill-Qwen-7B}"   # local pre-downloaded weights
MODEL_TAG="${MODEL_TAG:-R1Distill7B}"   # short tag -> exp/ckpt names stay <64 chars (wandb tag limit)
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"   # 8x B200 server
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/skywork_or1_math}"   # paper's Skywork-OR1-RL-Data (math), 7B-difficulty-filtered (48k)
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test.parquet}"   # kept for the --data-dir convention; NOT used for val
# ----- validation = the paper's benchmarks: AIME 2024 + AIME 2025, scored Avg@K -----
AIME24_FILE="${AIME24_FILE:-$PROJ_DIR/data/aime24/test.parquet}"
AIME25_FILE="${AIME25_FILE:-$PROJ_DIR/data/aime25/test.parquet}"
VAL_FILES_HYDRA="${VAL_FILES_HYDRA:-['$AIME24_FILE','$AIME25_FILE']}"   # hydra list; verl reports per-benchmark
VAL_N="${VAL_N:-8}"   # Avg@K: independent samples/problem at eval (paper uses 32; 8 here)
CKPT_ROOT="${CKPT_ROOT:-/checkpoints/hongpaul-sandbox/rl-opt}"
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
PYTHON_BIN="${PYTHON_BIN:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda/bin/python}"
WANDB_API_KEY="${WANDB_API_KEY:-}"   # supply via env if WANDB_MODE=online; not needed for offline
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
WANDB_PROJECT="${WANDB_PROJECT:-Skywork-OR1-R1Distill7B}"
WANDB_MODE="${WANDB_MODE:-offline}"   # remote node, no internet -> offline; runs land under $WANDB_DIR/wandb/
WANDB_DIR="${WANDB_DIR:-/checkpoints/hongpaul-sandbox/rl-opt/wandb_offline}"
NO_TMUX=false; EXTRA_ARGS=()
LAYER="${LAYER:-}"      # single-experiment layer id(s); "" = full RL
LAYERS="${LAYERS:-}"    # sweep list, e.g. "full 0 6 12" or "full 0..27"
PART_K=1; PART_N=1      # --part K/N stride slicing for multi-node
RESUME=false            # --resume: reuse existing matching ckpt dir + verl auto-resume from latest step
INIT_FROM="${INIT_FROM:-}"   # --init-from: HF dir to warm-start the 16K stage (e.g. an 8K-stage checkpoint)

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPUS="$2"; shift 2 ;;
        --model)      MODEL_PATH="$2"; shift 2 ;;
        --model-tag)  MODEL_TAG="$2"; shift 2 ;;
        --ckpt-root)  CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; TRAIN_FILE="$DATA_DIR/train.parquet"; TEST_FILE="$DATA_DIR/test.parquet"; shift 2 ;;
        --init-from)  INIT_FROM="$2"; shift 2 ;;   # warm-start weights for the 16K stage
        --layer)      LAYER="$2"; shift 2 ;;
        --layers)     LAYERS="$2"; shift 2 ;;
        --part)       IFS='/' read -r PART_K PART_N <<< "$2"; shift 2 ;;
        --resume)     RESUME=true; shift ;;
        --no-tmux)    NO_TMUX=true; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# --init-from overrides the model weights we start from (keeps MODEL_TAG for naming).
if [[ -n "$INIT_FROM" ]]; then MODEL_PATH="$INIT_FROM"; fi

if [[ ! "$PART_K" =~ ^[0-9]+$ ]] || [[ ! "$PART_N" =~ ^[0-9]+$ ]] || (( PART_K < 1 )) || (( PART_N < 1 )) || (( PART_K > PART_N )); then
    echo "ERROR: --part must be 'K/N' with 1 <= K <= N (got '$PART_K/$PART_N')"; exit 1
fi

# Expand "A..B" ranges inside the sweep list (e.g. "full 0..27" -> full 0 1 ... 27)
expand_layers() {
    local out=()
    for tok in $1; do
        if [[ "$tok" =~ ^([0-9]+)\.\.([0-9]+)$ ]]; then
            for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do out+=("$i"); done
        else
            out+=("$tok")
        fi
    done
    echo "${out[*]}"
}

# ===== tmux auto-launch =====
if [[ -z "${TMUX:-}" ]] && [[ "$NO_TMUX" == "false" ]]; then
    PART_TAG=""; (( PART_N > 1 )) && PART_TAG="_p${PART_K}of${PART_N}"
    TAG=""
    [[ -n "$LAYERS" ]] && TAG="_sweep"
    [[ -z "$LAYERS" && -n "$LAYER" ]] && TAG="_L$(echo "$LAYER" | tr ',' '-')"
    TMUX_SESSION="skyor1_$(basename $MODEL_TAG | tr '.' '_')${TAG}${PART_TAG}_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL_PATH") --model-tag $(printf '%q' "$MODEL_TAG") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    [[ -n "$INIT_FROM" ]] && FULL_ARGS="$FULL_ARGS --init-from $(printf '%q' "$INIT_FROM")"
    [[ -n "$LAYER"  ]] && FULL_ARGS="$FULL_ARGS --layer $(printf '%q' "$LAYER")"
    [[ -n "$LAYERS" ]] && FULL_ARGS="$FULL_ARGS --layers $(printf '%q' "$LAYERS")"
    (( PART_N > 1 )) && FULL_ARGS="$FULL_ARGS --part ${PART_K}/${PART_N}"
    [[ "$RESUME" == "true" ]] && FULL_ARGS="$FULL_ARGS --resume"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    # Forward overridable env vars into the inner tmux shell (tmux doesn't inherit outer env).
    ENV_INJECT="LR=${LR:-} EPOCHS=${EPOCHS:-} MAX_STEPS=${MAX_STEPS:-} BATCH_SIZE=${BATCH_SIZE:-} MINI_BATCH=${MINI_BATCH:-} MICRO_BATCH=${MICRO_BATCH:-} LOG_PROB_MICRO_BATCH=${LOG_PROB_MICRO_BATCH:-} ROLLOUT_N=${ROLLOUT_N:-} ENTROPY_COEFF=${ENTROPY_COEFF:-} FILTER=${FILTER:-} FILTER_METRIC=${FILTER_METRIC:-} MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-} GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-} USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-} PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-} MAX_RESPONSE=${MAX_RESPONSE:-} GPU_MEM_UTIL=${GPU_MEM_UTIL:-} VAL_N=${VAL_N:-} SAVE_FREQ=${SAVE_FREQ:-} TEST_FREQ=${TEST_FREQ:-}"
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && cd $PROJ_DIR && $ENV_INJECT bash $SCRIPT_DIR/$SCRIPT_NAME $FULL_ARGS; exec bash"
    echo "Tmux '$TMUX_SESSION' started.  Attach: tmux attach -t $TMUX_SESSION"
    exit 0
fi

# Running inline (--no-tmux, or already inside a tmux/shell): if you have ALREADY activated your own
# python env (CONDA_PREFIX or VIRTUAL_ENV set, e.g. you ran `source .../bin/activate`), we respect it
# and do nothing. Only when no env is active do we fall back to activating the configured conda env so
# ray/vllm workers get a sane environment.
if [[ -z "${CONDA_PREFIX:-}" ]] && [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f "$CONDA_INIT" ]]; then
    echo "[env] no active python env; activating configured conda env: $CONDA_ENV_PATH"
    # shellcheck disable=SC1090
    source "$CONDA_INIT" && conda activate "$CONDA_ENV_PATH"
else
    echo "[env] using already-active env: ${CONDA_PREFIX:-${VIRTUAL_ENV:-<none>}}  (python: $(command -v python 2>/dev/null))"
fi

# ===== preflight =====
if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: train file missing: $TRAIN_FILE"
    echo "  build the paper's dataset with:"
    echo "    python examples/data_preprocess/skywork_or1.py --local_save_dir $DATA_DIR"
    echo "  (on a box with internet; copy data/skywork_or1_math/ to this server if needed)"
    exit 1
fi
for vf in "$AIME24_FILE" "$AIME25_FILE"; do
    if [[ ! -f "$vf" ]]; then
        echo "ERROR: val (AIME) file missing: $vf"
        echo "  build it with: python examples/data_preprocess/aime.py"
        exit 1
    fi
done
if [[ "$MODEL_PATH" == /* ]] && [[ ! -d "$MODEL_PATH" ]]; then
    echo "WARNING: local model path not found ($MODEL_PATH); will rely on HF download by name"
fi
mkdir -p "$CKPT_ROOT" "$WANDB_DIR"

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DATE=$(date +%m%d_%H%M)

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_API_KEY WANDB_ENTITY WANDB_MODE WANDB_DIR
export HF_HOME="${HF_HOME:-/code/hongpaul-sandbox/temp/OPT-RL/hf_cache}"

# ===== Skywork-OR1 (MAGIC) hyperparameters =====
# LR default 5e-6 (per request; the paper uses constant 1e-6). Override anytime with LR=1e-6.
if [[ -n "$LAYER" || -n "$LAYERS" ]]; then
    LR="${LR:-5e-6}"
else
    LR="${LR:-5e-6}"        # full RL: 5e-6 (paper uses 1e-6 -> set LR=1e-6 to match the paper)
fi
ROLLOUT_N="${ROLLOUT_N:-8}"         # group size per prompt (paper uses 16; default 8 per request)
MAX_RESPONSE="${MAX_RESPONSE:-16384}"   # always 16K (16*1024); not running a separate 8K stage
MAX_PROMPT=2048                     # Skywork math prompts are short (p99~315 tok, max~1.8k); 2048 -> 0 dropped
CLIP_RATIO_LOW=0.2                  # paper: symmetric clip 0.2
CLIP_RATIO_HIGH=0.2
CLIP_RATIO_C=10.0                   # dual-clip lower bound (inactive at adv>=0; matches DAPO default)
PPO_INNER_EPOCH=1                   # one proximal update epoch
# Paper uses an ADAPTIVE entropy controller (target entropy 0.2; Skywork's 7b_16k.sh:
# max_ent_coef=0.005, min=0, delta=0.0001, start=0.0). Stock verl has only a FIXED coeff, so the
# DEFAULT of 0 means NO entropy regularization at all. For a rough (non-adaptive) approximation set
# ENTROPY_COEFF=0.001. True adaptive control requires porting Skywork's controller into the actor.
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"     # prompts per training step (matches paper's 256)
MINI_BATCH="${MINI_BATCH:-128}"     # = BATCH/2 -> 2 gradient updates per rollout batch (paper: 256/128)
MICRO_BATCH="${MICRO_BATCH:-2}"     # sequences/GPU/backward. NOTE: counts SEQUENCES not tokens — at 16K each
                                    # unit is huge. If you OOM, prefer USE_DYNAMIC_BSZ=true over raising this.
LOG_PROB_MICRO_BATCH="${LOG_PROB_MICRO_BATCH:-16}"   # forward-only (no grad); larger ok
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"   # true -> cap tokens/microbatch instead of #sequences (16K-robust)
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-24576}"   # tokens/microbatch when USE_DYNAMIC_BSZ=true (>= prompt+some resp)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"   # vLLM KV-cache fraction; raise to 0.8-0.85 for faster generation (main speed knob)
EPOCHS="${EPOCHS:-1}"               # one pass over the training data (used only when MAX_STEPS is unset)
# MAX_STEPS: target an EXACT number of gradient updates. RECOMMENDED over EPOCHS because rejection
# sampling consumes >1 gen batch per step, so an "epoch" yields an unpredictable (~half) step count.
# When set, we run with a high epoch cap and let trainer.total_training_steps stop the run cleanly at
# exactly MAX_STEPS via is_last_step -> the SAME clean save path as periodic saves (no fragile
# end-of-data fall-through). Leave empty to fall back to EPOCHS.
MAX_STEPS="${MAX_STEPS:-}"

# ----- rejection / dynamic sampling (paper: keep only non-zero-advantage groups) -----
FILTER="${FILTER:-true}"
FILTER_METRIC="${FILTER_METRIC:-seq_reward}"   # group std of total reward; 0/1 reward -> drops all-correct/all-wrong
# If fewer than train_batch_size prompts survive filtering within this many gen rounds, the DAPO loop
# RAISES ValueError and stops. Raise this (or set 0 = endless trials) if early 16K batches are too hard.
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$BATCH_SIZE}"  # prompts generated per gen round (= dataloader batch); >BATCH_SIZE oversamples

STEPS_PER_EPOCH=$($PYTHON_BIN -c "import pandas as pd; print(max(1, len(pd.read_parquet('$TRAIN_FILE')) // $GEN_BATCH_SIZE))")
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
SAVE_FREQ="${SAVE_FREQ:-20}"        # save a checkpoint every 20 steps
TEST_FREQ="${TEST_FREQ:-25}"

# -----------------------------------------------------------------------------
# run_one  <setting>   where <setting> is "full" or layer id(s) like "14" / "0,14,27"
# -----------------------------------------------------------------------------
run_one() {
    local setting="${1:-full}"
    local LAYER_ARGS=()
    local LAYER_NAME="full"
    if [[ -n "$setting" && "$setting" != "full" ]]; then
        LAYER_ARGS=(+actor_rollout_ref.actor.train_layer_ids="$setting")
        LAYER_NAME="L$(echo "$setting" | tr ',' '-')"
    fi

    local FILTER_TAG=""; [[ "$FILTER" == "true" ]] && FILTER_TAG="_rs"   # rs = rejection sampling
    # ckpt folder name. Lives under $CKPT_ROOT/$WANDB_PROJECT/ (e.g. .../Skywork-OR1-R1Distill7B/),
    # which already encodes the recipe+model -> drop the redundant "SkyOR1_" and constant "r16384_".
    local EXP_NAME="${DATE}_${MODEL_TAG}_${LAYER_NAME}_n${ROLLOUT_N}_lr${LR}${FILTER_TAG}"

    if [[ "$RESUME" == "true" ]]; then
        local suffix="_${MODEL_TAG}_${LAYER_NAME}_n${ROLLOUT_N}_lr${LR}${FILTER_TAG}"
        local found
        found=$(ls -dt "${CKPT_ROOT}/${WANDB_PROJECT}/"*"${suffix}" 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            EXP_NAME=$(basename "$found")
            echo "[resume] reusing existing exp: $EXP_NAME"
        else
            echo "[resume] no prior exp matching *${suffix}; starting fresh as $EXP_NAME"
        fi
    fi

    local CKPTS_DIR="${CKPT_ROOT}/${WANDB_PROJECT}/${EXP_NAME}"
    mkdir -p "$CKPTS_DIR"
    local LOG_FILE="$CKPTS_DIR/train.log"
    export VERL_DEFAULT_LOCAL_DIR="$CKPTS_DIR"

    # rejection-sampling args (only when FILTER=true)
    local FILTER_ARGS=()
    if [[ "$FILTER" == "true" ]]; then
        FILTER_ARGS=(
            algorithm.filter_groups.enable=True
            "algorithm.filter_groups.metric=$FILTER_METRIC"
            algorithm.filter_groups.max_num_gen_batches=$MAX_NUM_GEN_BATCHES
            data.gen_batch_size=$GEN_BATCH_SIZE
        )
    else
        FILTER_ARGS=(algorithm.filter_groups.enable=False)
    fi

    # batch-sizing args: token-budgeted dynamic bsz (16K-robust) OR fixed micro-batch (#sequences)
    local BSZ_ARGS=()
    if [[ "$USE_DYNAMIC_BSZ" == "true" ]]; then
        BSZ_ARGS=(
            actor_rollout_ref.actor.use_dynamic_bsz=True
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN
            actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN
            actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN
        )
    else
        BSZ_ARGS=(
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH
        )
    fi

    # step-budget args: target an exact #updates (MAX_STEPS, robust) or fall back to EPOCHS
    local STEP_ARGS=()
    if [[ -n "$MAX_STEPS" ]]; then
        # high epoch cap so the data cycles; total_training_steps stops cleanly at MAX_STEPS
        STEP_ARGS=(trainer.total_epochs=1000 trainer.total_training_steps=$MAX_STEPS)
    else
        STEP_ARGS=(trainer.total_epochs=$EPOCHS)
    fi

    cat <<EOF
============================================================
  Skywork-OR1 (MAGIC) 16K  —  $EXP_NAME
  Model         : $MODEL_PATH
  GPUs          : $GPUS  (${NGPUS} GPUs)
  batch/mini    : $BATCH_SIZE / $MINI_BATCH    rollout n: $ROLLOUT_N    gen_batch: $GEN_BATCH_SIZE
  bsz mode      : dynamic=$USE_DYNAMIC_BSZ (max_token=$PPO_MAX_TOKEN_LEN)  micro=$MICRO_BATCH  logp_micro=$LOG_PROB_MICRO_BATCH
  lr            : $LR (constant)   clip: $CLIP_RATIO_LOW/$CLIP_RATIO_HIGH   max_resp: $MAX_RESPONSE
  loss_agg_mode : token-mean   norm_adv_by_std=True   entropy_coeff: $ENTROPY_COEFF
  rejection samp: FILTER=$FILTER  metric=$FILTER_METRIC  max_gen_batches=$MAX_NUM_GEN_BATCHES
  val (eval)    : AIME24 + AIME25  Avg@$VAL_N  (sample temp 0.6 top_p 1.0)  test_freq=$TEST_FREQ
  layer training: ${setting:-full (all params)}
  steps         : ${MAX_STEPS:+MAX_STEPS=$MAX_STEPS (exact, clean stop)}${MAX_STEPS:-EPOCHS=$EPOCHS -> ~$TOTAL_STEPS nominal (rejection sampling -> real updates ~half)}   save_freq: $SAVE_FREQ
  trainer       : recipe.dapo.main_dapo (filter_groups-capable)
  ckpts         : $CKPTS_DIR
============================================================
EOF

    $PYTHON_BIN -m recipe.dapo.main_dapo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=True \
        algorithm.use_kl_in_reward=False \
        "data.train_files='$TRAIN_FILE'" \
        "data.val_files=$VAL_FILES_HYDRA" \
        data.train_batch_size=$BATCH_SIZE \
        data.max_prompt_length=$MAX_PROMPT \
        data.max_response_length=$MAX_RESPONSE \
        data.filter_overlong_prompts=True \
        "data.truncation='error'" \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=$LR \
        "actor_rollout_ref.actor.optim.betas=[0.9,0.95]" \
        actor_rollout_ref.actor.optim.weight_decay=0.0 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0" \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
        actor_rollout_ref.actor.ppo_epochs=$PPO_INNER_EPOCH \
        actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
        actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
        actor_rollout_ref.actor.clip_ratio_c=$CLIP_RATIO_C \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
        "actor_rollout_ref.actor.loss_agg_mode='token-mean'" \
        "actor_rollout_ref.actor.checkpoint.save_contents='[\"model\",\"optimizer\",\"extra\",\"hf_model\"]'" \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
        actor_rollout_ref.rollout.agent.num_workers=$NGPUS \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT + MAX_RESPONSE)) \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        reward_model.reward_manager=dapo \
        reward_model.overlong_buffer.enable=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="$WANDB_PROJECT" \
        "trainer.experiment_name='$EXP_NAME'" \
        "trainer.default_local_dir='$CKPTS_DIR'" \
        trainer.n_gpus_per_node=$NGPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.max_actor_ckpt_to_keep=1 \
        trainer.max_critic_ckpt_to_keep=1 \
        trainer.test_freq=$TEST_FREQ \
        "${STEP_ARGS[@]}" \
        trainer.resume_mode=auto \
        "${BSZ_ARGS[@]}" \
        "${FILTER_ARGS[@]}" \
        ${LAYER_ARGS[@]+"${LAYER_ARGS[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

# ===== dispatch =====
if [[ -z "$LAYERS" ]]; then
    run_one "${LAYER:-full}"
else
    read -ra SWEEP <<< "$(expand_layers "$LAYERS")"
    MY=()
    for ((i = PART_K - 1; i < ${#SWEEP[@]}; i += PART_N)); do MY+=("${SWEEP[$i]}"); done
    echo "============================================================"
    echo "  Sweep (${#SWEEP[@]} settings): ${SWEEP[*]}"
    if (( PART_N > 1 )); then
        echo "  --part $PART_K/$PART_N -> this node runs ${#MY[@]}: ${MY[*]:-<none>}"
    else
        echo "  single node -> all ${#MY[@]}: ${MY[*]}"
    fi
    echo "============================================================"
    if (( ${#MY[@]} == 0 )); then echo "  [skip] nothing assigned to part $PART_K/$PART_N"; exit 0; fi
    for s in "${MY[@]}"; do
        run_one "$s"
        echo ""; echo "  [done] setting=$s  (part $PART_K/$PART_N)"; echo ""
    done
fi
