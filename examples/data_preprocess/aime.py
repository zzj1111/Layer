"""Build AIME 2024 + AIME 2025 evaluation sets in verl parquet format.

Source: the curated HF-Arrow datasets already in this repo under
data/evaluation_suite_v2/{aime,aime25} (aime = AIME 2024, 30 problems each).
No internet / huggingface_hub required — read straight from the Arrow shards.

Output (one dir per benchmark so verl reports them separately):
    data/aime24/test.parquet   data_source="aime24"
    data/aime25/test.parquet   data_source="aime25"

data_source "aime24"/"aime25" start with "aime", so verl routes them to the
boxed math grader (verl.utils.reward_score.math_dapo) — a 0/1 \\boxed{} match,
consistent with the training reward. Answers are normalized to canonical ints
("025" -> "25") so a model that boxes "25" still matches.

Schema matches the training data (so the same chat template / system prompt apply):
    columns:      data_source, prompt, ability, reward_model, extra_info
    prompt:       [system (boxed instruction), user (problem)]
    reward_model: {"style": "rule", "ground_truth": "<int answer>"}

Usage:
    python examples/data_preprocess/aime.py                       # default suite dir + data/aime24, data/aime25
    python examples/data_preprocess/aime.py --no_system           # raw prompt (no boxed instruction)
"""
import argparse
import glob
import json
import os
from typing import Any

import pandas as pd
import pyarrow as pa

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
# (suite subdir, output dir, data_source) — "aime" in the suite is AIME 2024.
BENCHMARKS = [
    ("aime", "data/aime24", "aime24"),
    ("aime25", "data/aime25", "aime25"),
]


def read_arrow_dir(path: str) -> list[dict[str, Any]]:
    """Read a HF save_to_disk Arrow dataset dir without the `datasets` library."""
    try:  # prefer datasets if it happens to be installed (handles multi-shard cleanly)
        from datasets import load_from_disk

        return load_from_disk(path).to_list()
    except Exception:
        pass
    rows: list[dict[str, Any]] = []
    for shard in sorted(glob.glob(os.path.join(path, "*.arrow"))):
        src = pa.memory_map(shard, "r")
        table = None
        for opener in (pa.ipc.open_stream, pa.ipc.open_file):
            try:
                src.seek(0)
                table = opener(src).read_all()
                break
            except Exception:
                continue
        if table is None:
            raise RuntimeError(f"cannot read Arrow shard: {shard}")
        rows.extend(table.to_pylist())
    return rows


def normalize_answer(raw: Any) -> str:
    """AIME answers are integers 0-999; strip leading zeros ('025' -> '25')."""
    s = str(raw).strip()
    try:
        return str(int(s))
    except ValueError:
        return s  # leave anything non-integer untouched


def build_prompt(question: str, add_system: bool) -> list[dict[str, str]]:
    user = {"role": "user", "content": question}
    if add_system:
        return [{"role": "system", "content": SYSTEM_PROMPT}, user]
    return [user]


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--suite_dir", default="data/evaluation_suite_v2",
                   help="dir holding the {aime,aime25} HF-Arrow datasets")
    p.add_argument("--no_system", action="store_true",
                   help="keep the raw problem prompt (no boxed system instruction)")
    args = p.parse_args()

    add_system = not args.no_system
    for suite_name, out_dir, data_source in BENCHMARKS:
        src = os.path.join(args.suite_dir, suite_name)
        if not os.path.isdir(src):
            raise SystemExit(f"missing source dataset: {src}")
        raw = read_arrow_dir(src)

        records = []
        for i, row in enumerate(raw):
            question = row.get("problem") or row.get("question")
            answer = normalize_answer(row.get("answer", row.get("ground_truth")))
            extra = {"index": i, "split": "test", "benchmark": data_source}
            if "difficulty" in row:
                extra["difficulty"] = row["difficulty"]
            records.append({
                "data_source": data_source,
                "prompt": build_prompt(question, add_system),
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": extra,
            })
        df = pd.DataFrame(records)

        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "test.parquet")
        df.to_parquet(out, index=False)
        with open(os.path.join(out_dir, "test_example.json"), "w", encoding="utf-8") as f:
            json.dump(df.iloc[0].to_dict(), f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ {out}  ({len(df)} problems)  data_source={data_source!r}  "
              f"answers e.g. {[r['reward_model']['ground_truth'] for r in records[:5]]}")

    print("\nEval with the training script (Avg@8 is the default):")
    print("  bash examples/grpo_trainer/run_skywork_or1_r1distill_7b_16k_tmux.sh")
    print("  # val_files default to data/aime24 + data/aime25, val_kwargs.n=8")


if __name__ == "__main__":
    main()
