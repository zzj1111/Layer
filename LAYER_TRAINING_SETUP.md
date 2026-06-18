# Layer-Selective GIGPO + ALFworld + Qwen2.5-3B-Instruct

Adds **layer-selective RL training** on top of verl-agent's GIGPO + ALFworld pipeline.
Branch: `alfworld-gigpo-layer`.

## What's new vs upstream verl-agent

| File | Change |
|---|---|
| `verl/workers/fsdp_workers.py` | Inserts `actor.train_layer_ids` hook BEFORE FSDP wrap. Freezes all params except selected layers; flips `use_orig_params=True` so FSDP doesn't pack frozen+trainable into one `FlatParameter`. |
| `examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh` | New launcher: Qwen2.5-3B-Instruct, GIGPO, ALFworld, with `--layer N` / `--layers "..."` / `--full` / `--resume` / sweep across nodes via `--part k/N`. |

## Install (one-time, per machine)

ALFworld + verl-agent need their own conda env per the upstream README.

```bash
# 1) verl-agent base env (Python 3.12 per upstream)
conda create -n verl-agent python=3.12 -y
conda activate verl-agent
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

# 2) verl itself (from this repo, after cloning)
cd /home/zha00175/verl_agent_layer
git checkout alfworld-gigpo-layer
pip install -e .

# 3) ALFworld deps (Inform7 auto-downloaded, no Java needed)
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
alfworld-download -f       # writes ~/.cache/alfworld/{json_2.1.1,detectors,...}
alfworld-play-tw           # optional sanity-play
```

## Quick start (4-GPU local)

```bash
# Layer 14 only, fresh
bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --layer 14

# Full-param RL, fresh
bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --full

# Sweep layers 0,17,35 sequentially (one tmux session)
bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --layers "0 17 35"

# Sweep all 36 layers, node 1 of 4 (interleaved)
bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --layers "0..35" --part 1/4

# Resume layer 14 to higher epoch count
EPOCHS=200 bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --layer 14 --resume
```

## Layer-spec syntax

```
--layer 14                 single layer
--layer "0,14,27"          multiple layers (comma-separated)
--layer "first,middle,last,lm_head"   semantic shortcuts
       (first=0, middle=18, last=35 for Qwen2.5-3B; lm_head, embed, norm also accepted)
--layers "5 13 21"         sweep mode (space-separated, runs sequentially)
--layers "0..35"           range expansion (sweep all 36 layers)
--layers "full 5 13 21"    sweep including full RL as baseline
--full                     explicit full-param RL (no freeze)
```

## What gets frozen

`actor.train_layer_ids=14` → only `model.layers.14.*` has `requires_grad=True`.
All other params (`model.embed_tokens`, `model.layers.{0..13,15..35}.*`,
`model.norm`, `lm_head`) are frozen.

Banner print on rank 0:
```
[layer_freeze] total_layers=36, trainable_layers=[14], extra=[],
               frozen_params=435, trainable_params=12
```

## Defaults (override via env)

| Var | Default | Note |
|---|---|---|
| `GPUS` | `0,1,2,3` | Local 4xH200 |
| `MODEL_PATH` | `~/.cache/.../Qwen2.5-3B-Instruct` snapshot, fallback `Qwen/Qwen2.5-3B-Instruct` | Auto-resolves HF cache |
| `EPOCHS` | `150` | Total epochs |
| `TRAIN_BATCH` | `16` | Prompts per training step |
| `VAL_BATCH` | `128` | Val episodes |
| `GROUP_SIZE` | `8` | GIGPO rollouts per prompt |
| `MAX_RESP` | `512` | tokens per turn |
| `MAX_STEPS` | `50` | env steps per episode |
| `LR` (layer) | `3e-6` | Higher for layer-only |
| `LR` (full) | `1e-6` | Lower for full-param |
| `WANDB_MODE` | `offline` | |
| `CKPT_ROOT` | `/mnt/data1/zha00175/ckpts_alfworld` | local default |

## Cluster (8-GPU server)

```bash
GPUS=0,1,2,3,4,5,6,7 \
CKPT_ROOT=/checkpoints/<user>/alfworld \
WANDB_MODE=offline \
CONDA_INIT=/scratch/.../code/cuda/bin/activate \
CONDA_ENV_PATH=/scratch/.../code/cuda \
PYTHON_BIN=/scratch/.../code/cuda/bin/python \
MODEL_PATH=/scratch/.../Qwen2.5-3B-Instruct \
bash examples/gigpo_trainer/run_alfworld_gigpo_3b_layer_tmux.sh --layer 14
```

## Caveats

- **3B is the "valley"**: upstream reports 1.5B → 86.7%, 7B → 90.8%, but **3B ~51-65%** with vanilla GIGPO. Use for ablation, not as a SOTA target.
- **Mixed-precision FSDP + use_orig_params=True**: works, but slightly more comm than packed FlatParameter mode. Acceptable for layer training; for full-param runs (`--full`) it falls back to upstream `use_orig_params=False`.
- **Resume requires same world size**: ckpt FSDP shards are `model_world_size_N_rank_*.pt`. If trained on 4 GPUs, must resume on 4 GPUs.
- **Layer-spec validation is best-effort**: `--layer 99` on a 36-layer model silently freezes everything (0 trainable params). Banner output catches this; check before letting it run for hours.

## Resume-search semantics

`--resume` builds a suffix from `_gigpo_<MODEL_TAG>_<LAYER_NAME>_g<GROUP>_b<BATCH>_lr<LR>` and globs `$CKPT_ROOT/$WANDB_PROJECT/*<suffix>`. Most-recent match (by mtime) becomes the reused dir; verl then auto-loads from its `latest_checkpointed_iteration.txt`. If no match found, falls through to fresh start with current `DATE` prefix (and prints the rejected suffix for debugging).
