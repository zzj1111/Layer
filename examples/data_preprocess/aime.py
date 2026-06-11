"""Build AIME 2024 + AIME 2025 eval sets EXACTLY as Skywork-OR1 uses them.

Source: Skywork-OR1's OWN eval parquets (or1_data/eval/aime{24,25}.parquet), already in verl
RLHF format. We copy the `prompt` and `reward_model.ground_truth` VERBATIM so the eval prompt is
byte-identical to the paper -- including Skywork's own quirks:
  * aime24 appends "Let's think step by step and output the final answer within \\boxed{}." to the
    user message;
  * aime25 does NOT append anything (raw problem);
  * neither uses a system prompt.
The only change is `data_source`, remapped so verl routes the reward to the boxed math grader and
reports each benchmark separately:
    test-math-aime24 -> aime24   (startswith "aime" -> verl math_dapo boxed grader)
    test-math-aime25 -> aime25

Output:
    data/aime24/test.parquet   data_source="aime24"   (30 problems)
    data/aime25/test.parquet   data_source="aime25"   (30 problems)

No huggingface_hub / datasets needed -- downloads two small parquet over plain HTTPS.

Usage:
    python examples/data_preprocess/aime.py
    python examples/data_preprocess/aime.py --local_dir or1_data/eval   # use pre-downloaded parquets
"""
import argparse
import json
import os
import urllib.request

import pandas as pd

SKYWORK_EVAL_URLS = {
    "aime24": "https://raw.githubusercontent.com/SkyworkAI/Skywork-OR1/main/or1_data/eval/aime24.parquet",
    "aime25": "https://raw.githubusercontent.com/SkyworkAI/Skywork-OR1/main/or1_data/eval/aime25.parquet",
}


def load_skywork_eval(name: str, local_dir: str | None, cache_dir: str) -> pd.DataFrame:
    """Load Skywork's eval parquet from a local dir or download it over HTTPS."""
    if local_dir:
        path = os.path.join(local_dir, f"{name}.parquet")
        if not os.path.isfile(path):
            raise SystemExit(f"--local_dir given but missing {path}")
        return pd.read_parquet(path)
    cache = os.path.join(cache_dir, f"{name}.parquet")
    if not os.path.isfile(cache):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"downloading {SKYWORK_EVAL_URLS[name]}")
        urllib.request.urlretrieve(SKYWORK_EVAL_URLS[name], cache)  # noqa: S310 (trusted host)
    return pd.read_parquet(cache)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--local_dir", default=None,
                   help="dir with pre-downloaded aime24.parquet / aime25.parquet (e.g. or1_data/eval)")
    p.add_argument("--cache_dir", default="/tmp/skywork_or1_eval_raw")
    args = p.parse_args()

    for name in ("aime24", "aime25"):
        df = load_skywork_eval(name, args.local_dir, args.cache_dir).copy()
        # remap data_source only (test-math-aimeXX -> aimeXX) for verl routing + per-benchmark metric;
        # prompt and reward_model.ground_truth are kept verbatim from Skywork.
        df["data_source"] = name
        out_dir = f"data/{name}"
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "test.parquet")
        df.to_parquet(out, index=False)
        with open(os.path.join(out_dir, "test_example.json"), "w", encoding="utf-8") as f:
            json.dump(df.iloc[0].to_dict(), f, indent=2, ensure_ascii=False, default=str)
        gts = [r["ground_truth"] for r in df["reward_model"][:5]]
        print(f"✓ {out}  ({len(df)} problems)  data_source={name!r}  "
              f"prompt+gt verbatim from Skywork  e.g. gt={gts}")

    print("\nThese are byte-identical to Skywork's or1_data/eval prompts (aime24 has the boxed")
    print("instruction, aime25 is raw -- Skywork's own setup). Used as the in-training val_files.")


if __name__ == "__main__":
    main()
