"""Preprocess Skywork/Skywork-OR1-RL-Data (math split) to verl GRPO parquet format.

This is the dataset used by the Skywork-OR1 technical report (arXiv:2505.22312).
The math split has 105,055 verifiable problems (NuminaMath-1.5, DeepScaleR, STILL-3,
Omni-MATH, AIME 1983-2023, ...), each annotated with a per-model offline difficulty
(0..16 = number of failures out of 16 samples) for the DeepSeek-R1-Distill-Qwen
{1.5B, 7B, 32B} models.

The raw rows are ALREADY in verl RLHF format, but two things must be fixed for THIS
repo's reward path, and the paper's OFFLINE FILTERING is applied:
  1. reward_model.ground_truth is a JSON-encoded list string (e.g. '["15625"]');
     we decode it to the plain answer string the oat_math_grader expects.
  2. data_source is a Skywork-internal label (e.g. "train-math-numinamath1.5_olympiads")
     that verl's reward dispatcher does NOT route; we remap it to "math_drgrpo" so
     verl uses oat_math_grader (the same 0/1 boxed reward as data/math_drgrpo_*/).
  3. Offline difficulty filtering (paper Sec. on MAGIC data): drop problems the target
     model solves every time (difficulty 0) or never (difficulty 16). For the 7B this
     keeps 48,371 / 105,055 rows. Disable with --no_filter.

Skywork's RL training prompt is just the bare problem (no system prompt, no instruction);
R1-Distill is relied on to box its answer. We reproduce that EXACTLY by default (paper-faithful).
Pass --add_system to instead prepend a boxed system prompt (not what the paper does).

Output schema matches data/math_drgrpo_*/ exactly:
    columns:      data_source, prompt, ability, reward_model, extra_info
    data_source:  "math_drgrpo"
    prompt:       [user]  (raw problem, paper-exact; or [system, user] with --add_system)
    reward_model: {"style": "rule", "ground_truth": <answer string>}

OFFLINE note (training server has no HF access): run this on a machine WITH internet
(it downloads one ~19MB parquet over plain HTTPS, no huggingface_hub needed), then copy
the resulting data/skywork_or1_math/ dir to the server (or commit it).

Usage:
    # Online (downloads the math parquet from HF):
    python examples/data_preprocess/skywork_or1.py --local_save_dir data/skywork_or1_math

    # Offline (use a pre-downloaded parquet):
    python examples/data_preprocess/skywork_or1.py \\
        --local_parquet /tmp/skywork_or1_raw/math.parquet \\
        --local_save_dir data/skywork_or1_math

    # No difficulty filtering (use all 105k):
    python examples/data_preprocess/skywork_or1.py --no_filter
"""
import argparse
import json
import os
import sys
import urllib.request
from typing import Any

import pandas as pd

MATH_PARQUET_URL = (
    "https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/"
    "resolve/main/data/math-00000-of-00001.parquet"
)
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
# Default difficulty filter targets the 7B model and the paper's offline filtering
# (drop always-solved=0 and never-solved=16; keep the [1, 15] band).
DEFAULT_DIFF_MODEL = "DeepSeek-R1-Distill-Qwen-7B"


def load_math_df(local_parquet: str | None) -> tuple[pd.DataFrame, str]:
    """Load the Skywork-OR1 math parquet from a local file or download it over HTTPS."""
    if local_parquet:
        if not os.path.isfile(local_parquet):
            sys.exit(f"--local_parquet file not found: {local_parquet}")
        return pd.read_parquet(local_parquet), local_parquet

    cache = os.path.join("/tmp/skywork_or1_raw", "math-00000-of-00001.parquet")
    if not os.path.isfile(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        print(f"Downloading math split from HF ...\n  {MATH_PARQUET_URL}")
        urllib.request.urlretrieve(MATH_PARQUET_URL, cache)  # noqa: S310 (trusted host)
    return pd.read_parquet(cache), cache


def decode_ground_truth(raw_gt: Any) -> str | list[str]:
    """Skywork stores ground_truth as a JSON list string, e.g. '["15625"]'. Decode it.

    Returns the single answer string when the list has one element (the common case),
    otherwise the list (oat_math_grader accepts either a str or a list of str).
    """
    if isinstance(raw_gt, str):
        try:
            decoded = json.loads(raw_gt)
        except json.JSONDecodeError:
            return raw_gt  # already a plain answer string
        if isinstance(decoded, list):
            decoded = [str(x) for x in decoded]
            return decoded[0] if len(decoded) == 1 else decoded
        return str(decoded)
    if isinstance(raw_gt, (list, tuple)):
        items = [str(x) for x in raw_gt]
        return items[0] if len(items) == 1 else items
    return str(raw_gt)


def build_prompt(question: str, add_system: bool) -> list[dict[str, str]]:
    user = {"role": "user", "content": question}
    if add_system:
        return [{"role": "system", "content": SYSTEM_PROMPT}, user]
    return [user]


def difficulty_of(extra_info: Any, model: str) -> int | None:
    if not isinstance(extra_info, dict):
        return None
    md = extra_info.get("model_difficulty")
    if not isinstance(md, dict):
        return None
    return md.get(model)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--local_parquet", default=None,
                   help="Pre-downloaded math-*.parquet (required on servers without HF access)")
    p.add_argument("--local_save_dir", default="data/skywork_or1_math",
                   help="Output dir for train.parquet + test.parquet")
    p.add_argument("--data_source", default="math_drgrpo",
                   help="data_source for reward routing (math_drgrpo -> oat_math_grader)")
    p.add_argument("--add_system", action="store_true",
                   help="Prepend a boxed system prompt (default OFF = raw Skywork prompt, paper-exact)")
    # offline difficulty filtering (paper recipe)
    p.add_argument("--difficulty_model", default=DEFAULT_DIFF_MODEL,
                   help="model key in extra_info.model_difficulty to filter on")
    p.add_argument("--min_difficulty", type=int, default=1,
                   help="keep rows with difficulty >= this (1 drops always-solved=0)")
    p.add_argument("--max_difficulty", type=int, default=15,
                   help="keep rows with difficulty <= this (15 drops never-solved=16)")
    p.add_argument("--no_filter", action="store_true",
                   help="disable difficulty filtering (use all rows)")
    # validation split
    p.add_argument("--test_parquet", default=None,
                   help="parquet to copy as the val set (default: repo MATH-500 if present, "
                        "else a held-out slice)")
    p.add_argument("--holdout_size", type=int, default=500,
                   help="held-out val rows when no MATH-500 / --test_parquet is available")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw_df, raw_path = load_math_df(args.local_parquet)
    print(f"Loaded {len(raw_df)} math problems from {raw_path}")

    # Offline difficulty filtering ------------------------------------------------
    if not args.no_filter:
        diff = raw_df["extra_info"].map(lambda e: difficulty_of(e, args.difficulty_model))
        keep = diff.between(args.min_difficulty, args.max_difficulty)  # NaN -> False
        dropped_easy = int((diff == 0).sum())
        dropped_hard = int((diff == 16).sum())
        raw_df = raw_df[keep].reset_index(drop=True)
        print(f"Offline filter on {args.difficulty_model}: keep difficulty in "
              f"[{args.min_difficulty}, {args.max_difficulty}] -> {len(raw_df)} rows "
              f"(dropped always-solved={dropped_easy}, never-solved={dropped_hard})")

    # Convert raw -> verl schema --------------------------------------------------
    add_system = args.add_system
    records = []
    for i, row in enumerate(raw_df.itertuples(index=False)):
        prompt_msgs = list(row.prompt)
        question = next((m["content"] for m in prompt_msgs if m["role"] == "user"),
                        prompt_msgs[-1]["content"])
        gt = decode_ground_truth(row.reward_model.get("ground_truth"))
        extra = dict(row.extra_info) if isinstance(row.extra_info, dict) else {}
        extra["split"] = "train"
        records.append({
            "data_source": args.data_source,
            "prompt": build_prompt(question, add_system),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": extra,
        })
    train_df = pd.DataFrame(records)

    out_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Validation set --------------------------------------------------------------
    repo_math500 = "data/math_drgrpo_lvl3to5/test.parquet"
    if args.test_parquet and os.path.isfile(args.test_parquet):
        test_df = pd.read_parquet(args.test_parquet)
        test_src = args.test_parquet
    elif os.path.isfile(repo_math500):
        test_df = pd.read_parquet(repo_math500)  # MATH-500, already data_source=math_drgrpo
        test_src = repo_math500 + " (MATH-500)"
    else:
        train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        n = min(args.holdout_size, max(1, len(train_df) // 10))
        test_df = train_df.iloc[:n].reset_index(drop=True)
        train_df = train_df.iloc[n:].reset_index(drop=True)
        test_src = f"held-out slice ({n} rows)"

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "test.parquet")
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    with open(os.path.join(out_dir, "train_example.json"), "w", encoding="utf-8") as f:
        json.dump(train_df.iloc[0].to_dict(), f, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"✓ wrote {train_path}   ({len(train_df)} rows)")
    print(f"✓ wrote {test_path}    ({len(test_df)} rows)  [val from: {test_src}]")
    print(f"✓ data_source = {args.data_source!r}  (oat_math_grader, 0/1 boxed reward)")
    print(f"✓ prompt = {'[system, user]' if add_system else '[user] (raw Skywork)'}")
    print()
    print("Train with:")
    print(f"  bash examples/grpo_trainer/run_skywork_or1_r1distill_7b_16k_tmux.sh --data-dir {out_dir}")


if __name__ == "__main__":
    main()
