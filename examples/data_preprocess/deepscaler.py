"""Preprocess agentica-org/DeepScaleR-Preview-Dataset to verl GRPO parquet format.

Output schema matches data/math_drgrpo_*/ exactly so the existing
oat_math_grader reward function works without any verl changes:
    columns:       data_source, prompt, ability, reward_model, extra_info
    data_source:   "math_drgrpo"
    prompt:        [system, user]   # system prompt = standard "boxed answer" instruction
    reward_model:  {"style": "rule", "ground_truth": <answer string>}

OFFLINE mode (server with no HF access):
    1) Locally (or on a machine with internet) run:
          hf-cli download agentica-org/DeepScaleR-Preview-Dataset \\
              --repo-type dataset --local-dir /tmp/deepscaler_raw/
       OR (if you already have it cached somewhere):
          ls $HF_HOME/hub/datasets--agentica-org--DeepScaleR-Preview-Dataset/snapshots/*/deepscaler.json
    2) Run this script with --local_json <path-to-deepscaler.json>
    3) Copy the resulting data/deepscaler/ dir to the server (or commit to repo)

Usage:
    # Online (downloads from HF):
    python examples/data_preprocess/deepscaler.py \\
        --local_save_dir data/deepscaler

    # Offline (uses a pre-downloaded json file):
    python examples/data_preprocess/deepscaler.py \\
        --local_json /tmp/deepscaler_raw/deepscaler.json \\
        --local_save_dir data/deepscaler

    # Subsample for quick iteration:
    python examples/data_preprocess/deepscaler.py \\
        --local_json /tmp/deepscaler_raw/deepscaler.json \\
        --max_samples 5000
"""
import argparse
import json
import os
import sys

import pandas as pd


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def load_raw(local_json: str | None):
    """Load deepscaler.json from local path, or download from HF (online only)."""
    if local_json:
        if not os.path.isfile(local_json):
            sys.exit(f"--local_json file not found: {local_json}")
        with open(local_json, encoding="utf-8") as f:
            return json.load(f), local_json

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub not installed AND --local_json not given. "
                 "Install hf_hub or pre-download deepscaler.json and pass --local_json.")
    print("Downloading deepscaler.json from HuggingFace (agentica-org/DeepScaleR-Preview-Dataset) ...")
    path = hf_hub_download(
        repo_id="agentica-org/DeepScaleR-Preview-Dataset",
        filename="deepscaler.json",
        repo_type="dataset",
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--local_json", default=None,
                   help="Path to a pre-downloaded deepscaler.json (required on servers without HF access)")
    p.add_argument("--local_save_dir", default="data/deepscaler",
                   help="Output dir for train.parquet + test.parquet (default: data/deepscaler)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Subsample N problems from the raw set (0 = use all)")
    p.add_argument("--train_ratio", type=float, default=0.99,
                   help="Train split fraction (default 0.99, so ~400 held-out for val)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_source", default="math_drgrpo",
                   help="data_source field for reward routing (default math_drgrpo "
                        "→ verl uses oat_math_grader; change to 'math' for math_dapo path)")
    args = p.parse_args()

    raw_data, raw_path = load_raw(args.local_json)
    print(f"Loaded {len(raw_data)} problems from {raw_path}")

    # Convert raw → verl schema (matches math_drgrpo exactly)
    records = []
    for item in raw_data:
        # raw item has: problem, answer, solution
        records.append({
            "data_source": args.data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(item["answer"])},
            "extra_info": {"solution": item.get("solution", "")},
        })
    df = pd.DataFrame(records)

    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
        print(f"Subsampled to {args.max_samples} (seed={args.seed})")

    # Shuffle deterministically, split
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    split_idx = int(len(df) * args.train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # Dump a JSON example so a human can spot-check
    example = train_df.iloc[0].to_dict()
    with open(os.path.join(out_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"✓ wrote {train_path}   ({len(train_df)} rows)")
    print(f"✓ wrote {test_path}    ({len(test_df)} rows)")
    print(f"✓ data_source = {args.data_source!r}")
    print(f"  → uses verl's oat_math_grader (same reward as math_drgrpo_lvl3to5)")
    print()
    print("Now train with:")
    print(f"  bash examples/grpo_trainer/run_drgrpo_qwen25math_1.5b_tmux.sh \\")
    print(f"      --data-dir {out_dir} \\")
    print(f"      --layer 14")


if __name__ == "__main__":
    main()
