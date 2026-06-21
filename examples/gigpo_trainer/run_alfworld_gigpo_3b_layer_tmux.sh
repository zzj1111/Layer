#!/usr/bin/env bash
# GIGPO + ALFworld + Qwen2.5-3B-Instruct + layer-selective training (verl-agent fork)
#
# Layer training:
#   --layer 14                 train only layer 14 (Qwen2.5-3B has 36 layers, 0..35)
#   --layer "0,17,35"          train layers 0, 17, 35
#   --layer "first,middle,last,lm_head"   semantic shortcuts
#   --layers "5 13 21"         sweep: run 3 experiments sequentially (L5 → L13 → L21)
#   --layers "0..35"           sweep all 36 layers
#   --layers "full 0 14"       sweep full RL + L0 + L14
#   --full                     full-param RL (no freeze)  [default if --layer not set]
#
# Resume:
#   EPOCHS=20 bash run_alfworld_gigpo_3b_layer_tmux.sh --layer 14 --resume
#     reuses existing matching ckpt dir and continues training.
#
# Examples:
#   bash run_alfworld_gigpo_3b_layer_tmux.sh                       # full RL, fresh
#   bash run_alfworld_gigpo_3b_layer_tmux.sh --layer 17            # layer 17 only
#   bash run_alfworld_gigpo_3b_layer_tmux.sh --layers "0..35" --part 1/4   # node 1 of 4 sweep
#
# Server example (3B, 8 cards, venv, custom paths):
#   cd /code-fsx/.../verl_agent_layer
#   source /scratch/.../verl-alf/bin/activate
#   export PYTHON_BIN="$(command -v python)"
#   export MODEL_SIZE=3B
#   export MODEL_PATH=/code-fsx/.../Qwen2.5-3B-Instruct      # direct dir, has config.json
#   export CKPT_ROOT=/checkpoints/.../alf
#   export WANDB_MODE=offline WANDB_DIR=$CKPT_ROOT/wandb_offline
#   export GPUS=0,1,2,3,4,5,6,7
#   bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --no-tmux
set -euo pipefail

# ---- defaults ------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"               # default 8 cards; override locally

# ---- model selector (1.5B is paper default; 3B is our extension) ---------
MODEL_SIZE="${MODEL_SIZE:-1.5B}"                    # 1.5B (paper) or 3B
                                                    # NOTE: do NOT name this SIZE — conda
                                                    # binutils activation script clobbers it.
case "$MODEL_SIZE" in
    1.5B)
        DEFAULT_HF_REPO="Qwen/Qwen2.5-1.5B-Instruct"
        DEFAULT_TAG="Qwen2.5-1.5B-Instruct"
        DEFAULT_HF_CACHE="/home/zha00175/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
        ;;
    3B)
        DEFAULT_HF_REPO="Qwen/Qwen2.5-3B-Instruct"
        DEFAULT_TAG="Qwen2.5-3B-Instruct"
        DEFAULT_HF_CACHE="/mnt/data1/zha00175/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
        ;;
    *)
        echo "SIZE must be 1.5B or 3B (got: $MODEL_SIZE)" >&2; exit 1 ;;
esac
MODEL_PATH="${MODEL_PATH:-$DEFAULT_HF_CACHE}"
# Resolution:
#   1. If MODEL_PATH/config.json exists, use MODEL_PATH directly (server case:
#      user pre-downloaded to /scratch/.../Qwen2.5-3B-Instruct).
#   2. If MODEL_PATH is the HF cache "snapshots" dir, descend into first
#      hash subdir (local case: $HOME/.cache/huggingface/.../snapshots).
#   3. Fall back to HF repo name (will trigger online download).
if [[ -f "$MODEL_PATH/config.json" ]]; then
    :    # already a valid model dir
elif [[ -d "$MODEL_PATH" ]]; then
    snap=$(ls -d "$MODEL_PATH"/*/ 2>/dev/null | head -1 || true)
    if [[ -n "$snap" && -f "${snap%/}/config.json" ]]; then
        MODEL_PATH="${snap%/}"
    else
        MODEL_PATH="$DEFAULT_HF_REPO"
    fi
else
    MODEL_PATH="$DEFAULT_HF_REPO"
fi
MODEL_TAG="${MODEL_TAG:-$DEFAULT_TAG}"

ENGINE="${ENGINE:-vllm}"
# CKPT_ROOT default is local dev path; servers MUST export to a persistent
# location (e.g. /checkpoints/.../alf).
CKPT_ROOT="${CKPT_ROOT:-/mnt/data1/zha00175/ckpts_alfworld}"
WANDB_PROJECT="${WANDB_PROJECT:-verl_agent_alfworld}"
WANDB_MODE="${WANDB_MODE:-offline}"
# WANDB_DIR (offline run location). If unset, wandb defaults to ./wandb/
WANDB_DIR="${WANDB_DIR:-$CKPT_ROOT/wandb_offline}"
mkdir -p "$WANDB_DIR" 2>/dev/null || true
export WANDB_DIR WANDB_MODE WANDB_PROJECT

EPOCHS="${EPOCHS:-150}"
TRAIN_BATCH="${TRAIN_BATCH:-16}"
VAL_BATCH="${VAL_BATCH:-128}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_RESP="${MAX_RESP:-512}"
GIGPO_MODE="${GIGPO_MODE:-mean_std_norm}"
MAX_STEPS="${MAX_STEPS:-50}"

# Batch defaults: 1.5B uses paper defaults (mini=256, micro=32); 3B halves due to VRAM
if [[ "$MODEL_SIZE" == "1.5B" ]]; then
    MINI_BATCH="${MINI_BATCH:-256}"
    MICRO_BATCH="${MICRO_BATCH:-32}"
    LOG_PROB_MICRO="${LOG_PROB_MICRO:-32}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
else
    MINI_BATCH="${MINI_BATCH:-128}"
    MICRO_BATCH="${MICRO_BATCH:-16}"
    LOG_PROB_MICRO="${LOG_PROB_MICRO:-16}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
fi

NUM_CPUS_PER_ENV="${NUM_CPUS_PER_ENV:-0.1}"

# rollout TP: default 1 (paper uses 2, but 1 lets us scale group_size with
# rollout_dp = NGPUS instead of NGPUS/2). Override with ROLLOUT_TP env.
ROLLOUT_TP="${ROLLOUT_TP:-1}"

# Checkpointing: paper sets save_freq=-1 (never save). We default to 1 (save
# every epoch) so training is resumable. max_actor_ckpt_to_keep limits disk
# usage — only the latest N actors keep their FSDP+HF weights; older steps
# keep only data.pt (a few KB) so resume from any step still works.
SAVE_FREQ="${SAVE_FREQ:-1}"
MAX_ACTOR_CKPT="${MAX_ACTOR_CKPT:-3}"
TEST_FREQ="${TEST_FREQ:-5}"

# Python env. Two supported patterns:
#   (a) conda env  — set CONDA_INIT + CONDA_ENV_PATH (script will activate
#       inside tmux). PYTHON_BIN auto-derives from CONDA_ENV_PATH.
#   (b) venv  — export PYTHON_BIN directly and run with --no-tmux. Conda vars
#       still need to be set to SOMETHING for the tmux-launch fallback, but
#       won't be touched if --no-tmux is used.
CONDA_INIT="${CONDA_INIT:-/mnt/data1/zha00175/miniconda/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/data1/zha00175/miniconda/envs/verl}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_PATH/bin/python}"

LAYER=""
LAYERS=""
RESUME=false
FULL_FLAG=false
PART=""
NO_TMUX=false

# ---- parse args ----------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer)     LAYER="$2"; shift 2 ;;
        --layers)    LAYERS="$2"; shift 2 ;;
        --full)      FULL_FLAG=true; shift ;;
        --part)      PART="$2"; shift 2 ;;
        --resume)    RESUME=true; shift ;;
        --gpus)      GPUS="$2"; shift 2 ;;
        --model)     MODEL_PATH="$2"; shift 2 ;;
        --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
        --no-tmux)   NO_TMUX=true; shift ;;
        -h|--help)   sed -n '1,30p' "$0"; exit 0 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- conditional defaults ------------------------------------------------
# Layer mode tends to need slightly higher LR (mirrors our drgrpo scripts)
if [[ -n "$LAYER" || -n "$LAYERS" ]] && [ "$FULL_FLAG" = "false" ]; then
    LR="${LR:-3e-6}"
else
    LR="${LR:-1e-6}"
fi

# ---- sweep expansion -----------------------------------------------------
expand_layers() {
    local inp="$1"
    # "0..35" -> "0 1 ... 35"
    inp=$(echo "$inp" | awk '{
        n=split($0, a, " ");
        for (i=1;i<=n;i++) {
            if (a[i] ~ /^[0-9]+\.\.[0-9]+$/) {
                split(a[i], r, "..");
                for (j=r[1]; j<=r[2]; j++) printf "%d ", j;
            } else {
                printf "%s ", a[i];
            }
        }
    }')
    echo "$inp"
}

# Apply --part stride to sweep list
apply_part() {
    local list="$1"
    local part="$2"
    [ -z "$part" ] && { echo "$list"; return; }
    local k=${part%/*}; local n=${part#*/}
    local arr=($list); local out=""
    for (( i=0; i<${#arr[@]}; i++ )); do
        if (( (i % n) == (k - 1) )); then
            out+="${arr[i]} "
        fi
    done
    echo "$out"
}

# ---- tmux re-exec --------------------------------------------------------
DATE="${DATE:-$(date '+%m%d_%H%M')}"
if [ "$NO_TMUX" = "false" ] && [ -z "${VERL_NO_TMUX:-}" ]; then
    TAG=""
    [[ -n "$LAYERS" ]] && TAG="_sweep"
    [[ -z "$LAYERS" && -n "$LAYER" ]] && TAG="_L$(echo "$LAYER" | tr ',' '-')"
    PART_TAG=""
    [[ -n "$PART" ]] && PART_TAG="_part$(echo "$PART" | tr '/' '_')"
    TMUX_SESSION="alfworld_gigpo_3b${TAG}${PART_TAG}_${DATE}"
    PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && cd .. && pwd)"
    SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
    SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
    FULL_ARGS="--no-tmux --gpus $(printf '%q' "$GPUS") --model $(printf '%q' "$MODEL_PATH") --ckpt-root $(printf '%q' "$CKPT_ROOT")"
    [[ -n "$LAYER"  ]] && FULL_ARGS="$FULL_ARGS --layer $(printf '%q' "$LAYER")"
    [[ -n "$LAYERS" ]] && FULL_ARGS="$FULL_ARGS --layers $(printf '%q' "$LAYERS")"
    [[ "$FULL_FLAG" == "true" ]] && FULL_ARGS="$FULL_ARGS --full"
    [[ -n "$PART"   ]] && FULL_ARGS="$FULL_ARGS --part $(printf '%q' "$PART")"
    [[ "$RESUME" == "true" ]] && FULL_ARGS="$FULL_ARGS --resume"
    ENV_INJECT="MODEL_SIZE=$MODEL_SIZE LR=$LR EPOCHS=$EPOCHS GROUP_SIZE=$GROUP_SIZE TRAIN_BATCH=$TRAIN_BATCH MAX_RESP=$MAX_RESP MAX_PROMPT=$MAX_PROMPT MAX_STEPS=$MAX_STEPS MINI_BATCH=$MINI_BATCH MICRO_BATCH=$MICRO_BATCH LOG_PROB_MICRO=$LOG_PROB_MICRO GPU_MEM_UTIL=$GPU_MEM_UTIL SAVE_FREQ=$SAVE_FREQ MAX_ACTOR_CKPT=$MAX_ACTOR_CKPT TEST_FREQ=$TEST_FREQ"
    tmux new-session -d -s "$TMUX_SESSION" \
        "source $CONDA_INIT && conda activate $CONDA_ENV_PATH && cd $PROJ_DIR && $ENV_INJECT bash $SCRIPT_DIR/$SCRIPT_NAME $FULL_ARGS; exec bash"
    echo "Tmux '$TMUX_SESSION' started.  Attach: tmux attach -t $TMUX_SESSION"
    exit 0
fi

# ---- env setup -----------------------------------------------------------
export CUDA_VISIBLE_DEVICES=$GPUS
NGPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE
mkdir -p "$CKPT_ROOT"

# ---- one-time data prep --------------------------------------------------
DATA_DIR="$HOME/data/verl-agent/text"
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "[prep] generating verl-agent text dataset under $DATA_DIR"
    $PYTHON_BIN -m examples.data_preprocess.prepare \
        --mode text --train_data_size "$TRAIN_BATCH" --val_data_size "$VAL_BATCH"
fi

# ---- run_one ------------------------------------------------------------
# setting: "full" or layer id(s) like "14" / "0,14,27"
run_one() {
    local setting="${1:-full}"
    local LAYER_ARGS=()
    local LAYER_NAME="full"
    if [[ -n "$setting" && "$setting" != "full" ]]; then
        LAYER_ARGS=(+actor_rollout_ref.actor.train_layer_ids="$setting")
        LAYER_NAME="L$(echo "$setting" | tr ',' '-')"
    fi

    local EXP_NAME="${DATE}_gigpo_${MODEL_TAG}_${LAYER_NAME}_g${GROUP_SIZE}_b${TRAIN_BATCH}_lr${LR}"
    if [[ "$RESUME" == "true" ]]; then
        local suffix="_gigpo_${MODEL_TAG}_${LAYER_NAME}_g${GROUP_SIZE}_b${TRAIN_BATCH}_lr${LR}"
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

    cat <<EOF
============================================================
  GIGPO + ALFworld 16K  —  $EXP_NAME
  Model         : $MODEL_PATH  ($MODEL_SIZE)
  GPUs          : $GPUS  (${NGPUS} GPUs)
  group_size    : $GROUP_SIZE   train_batch: $TRAIN_BATCH   val_batch: $VAL_BATCH
  mini/micro    : $MINI_BATCH / $MICRO_BATCH per gpu   log_prob_micro: $LOG_PROB_MICRO
  lr            : $LR (constant)   max_resp: $MAX_RESP   max_steps: $MAX_STEPS
  algorithm     : GIGPO (mode=$GIGPO_MODE)   rollout TP: $ROLLOUT_TP
  layer training: ${setting:-full (all params)}
  total_epochs  : $EPOCHS
  save_freq     : $SAVE_FREQ epoch(s)   keep latest: $MAX_ACTOR_CKPT actor(s)   test_freq: $TEST_FREQ
  ckpts         : $CKPTS_DIR
============================================================
EOF

    # rollout TP: ROLLOUT_TP is set at top-of-file defaults; this is a no-op
    # placeholder so the line numbering / comment stays meaningful.
    local _unused=0

    $PYTHON_BIN -m verl.trainer.main_ppo \
        algorithm.adv_estimator=gigpo \
        data.train_files="$DATA_DIR/train.parquet" \
        data.val_files="$DATA_DIR/test.parquet" \
        data.train_batch_size=$TRAIN_BATCH \
        data.val_batch_size=$VAL_BATCH \
        data.max_prompt_length=$MAX_PROMPT \
        data.max_response_length=$MAX_RESP \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr=$LR \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        algorithm.gamma=0.95 \
        algorithm.gigpo.step_advantage_w=1.0 \
        algorithm.gigpo.mode=$GIGPO_MODE \
        env.env_name=alfworld/AlfredTWEnv \
        env.seed=0 \
        env.max_steps=$MAX_STEPS \
        env.rollout.n=$GROUP_SIZE \
        env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV \
        ${LAYER_ARGS[@]+"${LAYER_ARGS[@]}"} \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name="$WANDB_PROJECT" \
        "trainer.experiment_name='$EXP_NAME'" \
        "trainer.default_local_dir='$CKPTS_DIR'" \
        trainer.n_gpus_per_node=$NGPUS \
        trainer.nnodes=1 \
        trainer.save_freq=$SAVE_FREQ \
        trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT \
        trainer.test_freq=$TEST_FREQ \
        trainer.total_epochs=$EPOCHS \
        trainer.resume_mode=auto \
        trainer.val_before_train=True
}

# ---- driver --------------------------------------------------------------
if [[ -z "$LAYERS" ]]; then
    run_one "${LAYER:-full}"
else
    expanded=$(expand_layers "$LAYERS")
    chosen=$(apply_part "$expanded" "$PART")
    echo "Sweep: $chosen"
    for s in $chosen; do
        run_one "$s"
    done
fi
