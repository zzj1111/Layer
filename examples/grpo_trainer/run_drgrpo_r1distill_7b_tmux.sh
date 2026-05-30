#!/bin/bash
# ==============================================================================
# Dr. GRPO training — DeepSeek-R1-Distill-Qwen-7B   (8x B200 server)
# ==============================================================================
# Paper: "Understanding R1-Zero-Like Training: A Critical Perspective"
#        (Dr. GRPO, arXiv:2503.20783, Table 6 + Sec 3)
#
# 7B-specific defaults (different from the 1.5B script):
#   - Model    : deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#   - Data     : math_drgrpo_lvl3to5   (5586 train, matches paper 7B SOTA recipe)
#   - max_resp : 16000  (R1-Distill needs long reasoning; default 3k clips >50%)
#   - LR       : 1e-6   (paper Table 6; safer for already-SFT'd distill model)
#   - EPOCHS   : 5
#   - MICRO    : 4 per GPU (7B + 16K seq needs lower micro than 1.5B)
#   - WANDB    : offline mode; runs saved under $WANDB_DIR/wandb/
#
# verl patches required (already applied in this repo):
#   1. algorithm.norm_adv_by_std_in_grpo=False    # remove /std on advantage
#   2. actor.loss_agg_mode=drgrpo                  # 1/MAX_TOKENS (core_algos.py)
#   3. data_source=math_drgrpo                     # Oat boxed_reward_fn scorer
#   4. actor.train_layer_ids=<ids>                 # freeze all but these layers
#
# ------------------------------------------------------------------------------
# SINGLE experiment (full RL, 8x B200):
#   bash run_drgrpo_r1distill_7b_tmux.sh
#
# Layer-wise RL on one layer:
#   bash run_drgrpo_r1distill_7b_tmux.sh --layer 14
#
# SWEEP across settings (sequential on one node):
#   bash run_drgrpo_r1distill_7b_tmux.sh --layers "full 0 6 12 18 27"
#
# MULTI-NODE parallel — same --layers on every node, each gets a --part:
#   node1:  ... --layers "0..27" --part 1/4
#   ... (same as 1.5B; supports "A..B" range expansion)
#
# RESUME / continue training from a previous run (same settings, extend epochs):
#   EPOCHS=10 bash run_drgrpo_r1distill_7b_tmux.sh --resume
# ==============================================================================

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJ_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
SCRIPT_NAME=$(basename "$0")

# ===== defaults =====
MODEL_PATH="${MODEL_PATH:-/code-fsx/hongpaul-sandbox/code/temp/DeepSeek-R1-Distill-Qwen-7B}"   # local pre-downloaded weights on server
MODEL_TAG="${MODEL_TAG:-DeepSeek-R1-Distill-Qwen-7B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"   # 8x B200 server
DATA_DIR="${DATA_DIR:-$PROJ_DIR/data/math_drgrpo_lvl3to5}"   # paper 7B SOTA recipe
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test.parquet}"
CKPT_ROOT="${CKPT_ROOT:-/checkpoints/hongpaul-sandbox/rl-opt}"   # all ckpts here; override with --ckpt-root
CONDA_INIT="${CONDA_INIT:-/code/hongpaul-sandbox/cuda/miniconda3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda}"
PYTHON_BIN="${PYTHON_BIN:-/code/hongpaul-sandbox/cuda/miniconda3/envs/cuda/bin/python}"
WANDB_API_KEY="${WANDB_API_KEY:-b8f38344ec7231ee89baa74ef7209dd5a43df6b2}"
WANDB_ENTITY="${WANDB_ENTITY:-mhong-university-of-minnesota}"
WANDB_PROJECT="${WANDB_PROJECT:-DrGRPO-R1Distill7B}"
WANDB_MODE="${WANDB_MODE:-offline}"   # remote node, no internet → offline; runs land under $WANDB_DIR/wandb/
WANDB_DIR="${WANDB_DIR:-/checkpoints/hongpaul-sandbox/rl-opt/wandb_offline}"   # where wandb stores offline runs
NO_TMUX=false; EXTRA_ARGS=()
LAYER="${LAYER:-}"      # single-experiment layer id(s); "" = full RL
LAYERS="${LAYERS:-}"    # sweep list, e.g. "full 0 6 12" or "full 0..27"
PART_K=1; PART_N=1      # --part K/N stride slicing for multi-node
RESUME=false            # --resume: reuse existing matching ckpt dir + verl auto-resume from latest step

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)       GPUS="$2"; shift 2 ;;
        --model)      MODEL_PATH="$2"; shift 2 ;;
        --model-tag)  MODEL_TAG="$2"; shift 2 ;;
        --ckpt-root)  CKPT_ROOT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; TRAIN_FILE="$DATA_DIR/train.parquet"; TEST_FILE="$DATA_DIR/test.parquet"; shift 2 ;;
        --layer)      LAYER="$2"; shift 2 ;;     # single experiment, this layer
        --layers)     LAYERS="$2"; shift 2 ;;    # sweep of settings (space-separated)
        --part)       IFS='/' read -r PART_K PART_N <<< "$2"; shift 2 ;;
        --resume)     RESUME=true; shift ;;        # reuse latest matching ckpt dir, verl auto-resumes from last step
        --no-tmux)    NO_TMUX=true; shift ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ ! "$PART_K" =~ ^[0-9]+$ ]] || [[ ! "$PART_N" =~ ^[0-9]+$ ]] || (( PART_K < 1 )) || (( PART_N < 1 )) || (( PART_K > PART_N )); then
    echo "ERROR: --part must be 'K/N' with 1 ≤ K ≤ N (got '$PART_K/$PART_N')"; exit 1
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
    # tmux silently converts '.' to '_' in session names, so we pre-sanitize
    # so the echoed name matches what `tmux attach -t <name>` will accept.
    TMUX_SESSION="drgrpo_$(basename $MODEL_TAG | tr '.' '_')${TAG}${PART_TAG}_$(date +%m%d_%H%M)"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL_PATH") --model-tag $(printf '%q' "$MODEL_TAG") --ckpt-root $(printf '%q' "$CKPT_ROOT") --data-dir $(printf '%q' "$DATA_DIR")"
    [[ -n "$LAYER"  ]] && FULL_ARGS="$FULL_ARGS --layer $(printf '%q' "$LAYER")"
    [[ -n "$LAYERS" ]] && FULL_ARGS="$FULL_ARGS --layers $(printf '%q' "$LAYERS")"
    (( PART_N > 1 )) && FULL_ARGS="$FULL_ARGS --part ${PART_K}/${PART_N}"
    [[ "$RESUME" == "true" ]] && FULL_ARGS="$FULL_ARGS --resume"
    for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do FULL_ARGS="$FULL_ARGS $(printf '%q' "$arg")"; done
    # Forward overridable env vars (LR, EPOCHS) into the inner tmux shell so
    # `LR=1e-6 bash run...sh` works (tmux doesn't inherit outer env by default).
    ENV_INJECT="LR=${LR:-} EPOCHS=${EPOCHS:-} BATCH_SIZE=${BATCH_SIZE:-} MINI_BATCH=${MINI_BATCH:-} MICRO_BATCH=${MICRO_BATCH:-} LOG_PROB_MICRO_BATCH=${LOG_PROB_MICRO_BATCH:-} ROLLOUT_N=${ROLLOUT_N:-}"
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && cd $PROJ_DIR && $ENV_INJECT bash $SCRIPT_DIR/$SCRIPT_NAME $FULL_ARGS; exec bash"
    echo "Tmux '$TMUX_SESSION' started.  Attach: tmux attach -t $TMUX_SESSION"
    exit 0
fi

# ===== preflight =====
if [[ ! -f "$TRAIN_FILE" ]] || [[ ! -f "$TEST_FILE" ]]; then
    echo "ERROR: dataset files missing under $DATA_DIR"
    echo "  build with: python examples/data_preprocess/math_drgrpo.py --local_save_dir $DATA_DIR"
    exit 1
fi
# MODEL_PATH may be an HF repo name (e.g. "Qwen/Qwen2.5-Math-1.5B") or a local dir.
# Only warn when it looks like a local path (starts with '/') but doesn't exist.
if [[ "$MODEL_PATH" == /* ]] && [[ ! -d "$MODEL_PATH" ]]; then
    echo "WARNING: local model path not found ($MODEL_PATH); will rely on HF download by name"
fi
mkdir -p "$CKPT_ROOT" "$WANDB_DIR"

NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DATE=$(date +%m%d_%H%M)

export CUDA_VISIBLE_DEVICES=$GPUS
export WANDB_API_KEY WANDB_ENTITY WANDB_MODE WANDB_DIR
export HF_HOME="${HF_HOME:-/code/hongpaul-sandbox/temp/OPT-RL/hf_cache}"   # shared model cache across nodes

# ===== Dr. GRPO hyperparams (paper Table 6) =====
# LR: full RL on R1-Distill -> 1e-6 (safer for already-SFT'd model, matches paper Table 6).
# Layer-wise RL -> 5e-6 (only one layer trainable, larger LR moves faster without destabilizing the rest).
if [[ -n "$LAYER" || -n "$LAYERS" ]]; then
    LR="${LR:-5e-6}"
else
    LR="${LR:-1e-6}"
fi
ROLLOUT_N="${ROLLOUT_N:-8}"   # paper Table 6: 8 responses per question
MAX_RESPONSE=16000       # R1-Distill needs long reasoning; default 3k clipped >50%
MAX_PROMPT=1024          # MATH questions are short (<512 typically)
CLIP_RATIO=0.2           # paper: 0.2 (symmetric)
PPO_INNER_EPOCH=1        # paper: inner proximal update epoch = 1
BATCH_SIZE="${BATCH_SIZE:-256}"     # data.train_batch_size (prompts); 256 saturates 8x B200 well
MINI_BATCH="${MINI_BATCH:-$BATCH_SIZE}"   # default = BATCH_SIZE (single mini-batch = single update per step, inner_epoch=1)
MICRO_BATCH="${MICRO_BATCH:-4}"     # 7B + 16K seq: 4 is safe; 8 OOM'd on B200 (activations roughly double per step)
LOG_PROB_MICRO_BATCH="${LOG_PROB_MICRO_BATCH:-32}"   # forward-only (no grad) — can be larger than MICRO_BATCH
EPOCHS="${EPOCHS:-5}"    # paper official num_prompt_epoch=20; 5 = ~218 steps/epoch × 5 = 1090 steps

STEPS_PER_EPOCH=$($PYTHON_BIN -c "import pandas as pd; print(max(1, len(pd.read_parquet('$TRAIN_FILE')) // $BATCH_SIZE))")
TOTAL_STEPS=$((STEPS_PER_EPOCH * EPOCHS))
SAVE_FREQ=$STEPS_PER_EPOCH   # save once per epoch

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

    local EXP_NAME="${DATE}_DrGRPO_${MODEL_TAG}_${LAYER_NAME}_n${ROLLOUT_N}_r${MAX_RESPONSE}_lr${LR}"

    # --resume: find an existing matching experiment dir (any DATE) and reuse it,
    # so verl trainer.resume_mode=auto picks up the latest global_step_* checkpoint.
    if [[ "$RESUME" == "true" ]]; then
        local suffix="_DrGRPO_${MODEL_TAG}_${LAYER_NAME}_n${ROLLOUT_N}_r${MAX_RESPONSE}_lr${LR}"
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

    cat <<EOF
============================================================
  Dr. GRPO  —  $EXP_NAME
  Model         : $MODEL_PATH
  GPUs          : $GPUS  (${NGPUS} GPUs)
  batch/mini/μ  : $BATCH_SIZE / $MINI_BATCH / $MICRO_BATCH    rollout n: $ROLLOUT_N
  lr            : $LR (constant)   clip: $CLIP_RATIO   max_resp: $MAX_RESPONSE
  loss_agg_mode : drgrpo   norm_adv_by_std=False
  layer training: ${setting:-full (all params)}
  epochs        : $EPOCHS   total_steps: $TOTAL_STEPS   save_freq: $SAVE_FREQ
  ckpts         : $CKPTS_DIR
============================================================
EOF

    $PYTHON_BIN -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.norm_adv_by_std_in_grpo=False \
        algorithm.use_kl_in_reward=False \
        "data.train_files='$TRAIN_FILE'" \
        "data.val_files='$TEST_FILE'" \
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
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
        actor_rollout_ref.actor.ppo_epochs=$PPO_INNER_EPOCH \
        actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.0 \
        actor_rollout_ref.actor.entropy_coeff=0.0 \
        "actor_rollout_ref.actor.loss_agg_mode='drgrpo'" \
        "actor_rollout_ref.actor.checkpoint.save_contents='[\"hf_model\"]'" \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.agent.num_workers=$NGPUS \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH \
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
        trainer.test_freq=16 \
        trainer.total_epochs=$EPOCHS \
        trainer.resume_mode=auto \
        ${LAYER_ARGS[@]+"${LAYER_ARGS[@]}"} \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
}

# ===== dispatch =====
if [[ -z "$LAYERS" ]]; then
    # single experiment: --layer X (or full)
    run_one "${LAYER:-full}"
else
    # sweep: stride-slice across nodes via --part K/N
    read -ra SWEEP <<< "$(expand_layers "$LAYERS")"
    MY=()
    for ((i = PART_K - 1; i < ${#SWEEP[@]}; i += PART_N)); do MY+=("${SWEEP[$i]}"); done
    echo "============================================================"
    echo "  Sweep (${#SWEEP[@]} settings): ${SWEEP[*]}"
    if (( PART_N > 1 )); then
        echo "  --part $PART_K/$PART_N → this node runs ${#MY[@]}: ${MY[*]:-<none>}"
    else
        echo "  single node → all ${#MY[@]}: ${MY[*]}"
    fi
    echo "============================================================"
    if (( ${#MY[@]} == 0 )); then echo "  [skip] nothing assigned to part $PART_K/$PART_N"; exit 0; fi
    for s in "${MY[@]}"; do
        run_one "$s"
        echo ""; echo "  [done] setting=$s  (part $PART_K/$PART_N)"; echo ""
    done
fi
