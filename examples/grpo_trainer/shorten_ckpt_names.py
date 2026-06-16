#!/usr/bin/env python3
"""Shorten the model tag in existing checkpoint / eval folder names (in place).

The Skywork-OR1 training script used to tag runs with the long model name
"DeepSeek-R1-Distill-Qwen-7B"; it now uses "R1Distill7B". Folders already on disk
keep the long name, which (a) is inconsistent with new runs and (b) overflows
wandb's 64-char tag limit (the full-RL exp name is 69 chars). This renames any
directory whose name CONTAINS --from so it uses --to instead -- covering the
checkpoint exp dirs, the eval_results "<exp>_stepN" dirs, etc.

SAFE BY DEFAULT: dry-run (just prints the planned renames). Pass --apply to do it.
Skips a rename when the target already exists; renames deepest-first so renaming
a child never invalidates a queued parent path. Only folder NAMES change; file
contents (final.json, weights) are untouched.

Usage:
  # preview everything under the skywork dir (ckpts + eval_results):
  python examples/grpo_trainer/shorten_ckpt_names.py /checkpoints/hongpaul-sandbox/rl-opt/skywork

  # actually rename:
  python examples/grpo_trainer/shorten_ckpt_names.py /checkpoints/hongpaul-sandbox/rl-opt/skywork --apply

  # custom mapping / multiple roots:
  python examples/grpo_trainer/shorten_ckpt_names.py <root1> <root2> --from LONG --to SHORT --apply
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def find_matches(root: Path, old: str, max_depth: int) -> list[Path]:
    """Dirs under `root` whose name contains `old` (descent is pruned at a match)."""
    matches: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        keep: list[str] = []
        for d in dirnames:
            if old in d:
                matches.append(Path(dirpath) / d)   # match -> record, don't descend into it
            elif depth < max_depth:
                keep.append(d)
        dirnames[:] = keep
    return matches


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("roots", nargs="+", help="directory(ies) to scan for folders to rename")
    p.add_argument("--from", dest="old", default="DeepSeek-R1-Distill-Qwen-7B",
                   help="old substring in folder names (default: DeepSeek-R1-Distill-Qwen-7B)")
    p.add_argument("--to", dest="new", default="R1Distill7B",
                   help="replacement substring (default: R1Distill7B)")
    p.add_argument("--apply", action="store_true",
                   help="actually rename (default: dry-run preview)")
    p.add_argument("--max-depth", type=int, default=4,
                   help="max directory depth to scan under each root (default 4)")
    args = p.parse_args()

    if args.old == args.new:
        raise SystemExit("--from and --to are identical; nothing to do")

    matches: list[Path] = []
    for r in args.roots:
        root = Path(r).expanduser()
        if not root.is_dir():
            print(f"[skip] not a directory: {root}")
            continue
        matches += find_matches(root, args.old, args.max_depth)

    # deepest-first: rename children before parents so a parent rename can't strand a child
    matches.sort(key=lambda x: len(x.parts), reverse=True)

    if not matches:
        print(f"No folders containing {args.old!r} found under: {', '.join(args.roots)}")
        return

    renamed = skipped = 0
    for src in matches:
        dst = src.with_name(src.name.replace(args.old, args.new))
        if dst.exists():
            print(f"[skip] target already exists: {dst}")
            skipped += 1
            continue
        warn = "" if len(dst.name) <= 64 else f"   (WARN: still {len(dst.name)} chars)"
        if args.apply:
            try:
                src.rename(dst)
            except OSError as e:
                print(f"[error] {src} -> {dst.name}: {e}")
                skipped += 1
                continue
            print(f"[renamed] {src.name}\n       -> {dst.name}{warn}")
        else:
            print(f"[dry-run] {src.name}\n       -> {dst.name}{warn}")
        renamed += 1

    verb = "Renamed" if args.apply else "Would rename"
    tail = "" if args.apply else "   (re-run with --apply to perform the renames)"
    print(f"\n{verb} {renamed}, skipped {skipped}.{tail}")


if __name__ == "__main__":
    main()
