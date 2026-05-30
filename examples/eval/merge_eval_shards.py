"""Merge per-shard eval JSON files (raw_*shard{i}of{N}.json) into a single results dict.

Usage: python merge_eval_shards.py <save_dir>
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np


def main(save_dir: str) -> None:
    files = sorted(glob.glob(f"{save_dir}/raw_*_shard*.json"))
    if not files:
        raise SystemExit(f"no shard files under {save_dir}")
    print(f"[merge] {len(files)} shard files")

    merged: dict[str, list[dict]] = defaultdict(list)
    model_name = template = None
    for f in files:
        d = json.load(open(f))
        model_name = d["model_name"]
        template = d["template"]
        for task, rows in d["per_task_raw"].items():
            merged[task].extend(rows)

    results, avg_lens, max_lens = {}, {}, {}
    for task, rows in merged.items():
        rows.sort(key=lambda r: r["orig_idx"])
        rewards = [r["reward"] for r in rows]
        lens = [length for r in rows for length in r["lengths"]]
        results[task] = float(np.mean(rewards)) if rewards else 0.0
        avg_lens[task] = float(np.mean(lens)) if lens else 0.0
        max_lens[task] = int(np.max(lens)) if lens else 0
        print(f"  {task:20s}  n={len(rows):4d}  acc={results[task]:.4f}  "
              f"avg_len={avg_lens[task]:.0f}  max_len={max_lens[task]}")

    avg_score = float(np.mean(list(results.values()))) if results else 0.0
    final = {
        "model_name": model_name,
        "template": template,
        "results": results,
        "avg": avg_score,
        "avg_lens": avg_lens,
        "max_lens": max_lens,
        "n_shards": len(files),
    }
    out = f"{save_dir}/final.json"
    json.dump(final, open(out, "w"), indent=2)
    print(f"\n[merge] avg over {len(results)} tasks: {avg_score:.4f}")
    print(f"[merge] final -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python merge_eval_shards.py <save_dir>")
    main(sys.argv[1])
