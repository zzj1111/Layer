# Install — verl-agent + ALFworld + GIGPO

Tested on **Linux + NVIDIA H100/H200 + CUDA 12.8**.

## One-shot install (recommended)

```bash
bash install/setup_verl_agent_env.sh
# →  creates conda env 'verl-agent' (python 3.12), installs vllm + flash-attn +
#    verl (this repo, editable) + alfworld + textworld + gymnasium + sb3
# →  runs an import sanity check at the end
```

Custom env name + python version:
```bash
bash install/setup_verl_agent_env.sh my-env 3.10
```

## Manual install (if the script breaks)

```bash
# 1. create + activate env
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

# 2. core (vllm needs --no-build-isolation, can't go in requirements.txt)
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .                          # this repo, editable

# 3. ALFworld extras
pip install -r install/requirements-extras.txt

# 4. sanity
python -c "import torch, vllm, ray, transformers, alfworld, textworld, gymnasium, stable_baselines3; print('ok')"
```

## ALFworld game data

The ALFworld dataset is bundled in this repo at `data/alfworld/` (split into
two ~64MB `tar.gz.part-*` chunks). Extract once after install:

```bash
bash scripts/extract_alfworld_data.sh
# → installs to $ALFWORLD_DATA (default: ~/.cache/alfworld/)
```

If you have internet, the official command also works:
```bash
alfworld-download
```

## Qwen base model

Download once on a node with internet access, then symlink/scp to offline nodes:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')   # paper default
snapshot_download('Qwen/Qwen2.5-3B-Instruct')     # our extension
"
```

The launcher script (`run_alfworld_gigpo_3b_layer_tmux.sh`) auto-resolves the
local HF cache snapshot, or you can override `MODEL_PATH=<local/dir>`.

## Verify the install end-to-end

A 30-second CPU-only smoke test that imports the freeze-hook + a tiny model:
```bash
python tests/test_train_layer_ids_freeze.py
# Expected: "✓ all freeze tests passed"
```

## Known issues

- **`fast-downward-textworld` wheel build fails with `'python' not found`** —
  conda activate sets `python` on PATH; if you ran pip outside the env it
  defaults to `python3`. Solution: `conda activate <env>` then re-run pip.
- **vllm + flash-attn must be installed BEFORE `-e .`** (or via this script).
  Mixing them into a single `pip install -r` breaks because flash-attn needs
  `--no-build-isolation`.
- **ALFworld data is NOT in `pip install alfworld`** — you have to fetch it
  separately. We bundle it in this repo; see above.
- **No JDK required.** Old AlfWorld READMEs mention it; the modern release
  ships Inform7 as a precompiled binary.
