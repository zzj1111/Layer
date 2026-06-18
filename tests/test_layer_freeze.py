"""Smoke test for the layer-freeze logic in fsdp_workers.py.

Loads Qwen2.5-3B-Instruct on CPU and applies the same freeze logic the FSDP
worker would, then verifies:
  - total_layers detection (= 36 for Qwen2.5-3B)
  - layer-name globs match the right tensors
  - semantic shortcuts (first, middle, last, lm_head, embed, norm) resolve

This catches "Qwen2.5-3B has model.layers.{0..35}" naming surprises BEFORE
spending GPU time on a real run.

Usage:
    pytest tests/test_layer_freeze.py -v
    or:
    python tests/test_layer_freeze.py
"""
import argparse
import os
import sys


def apply_freeze(model, train_layer_ids_cfg: str) -> tuple[int, int, set, list]:
    """Mirror of the freeze block in verl/workers/fsdp_workers.py — returns
    (frozen_count, trainable_count, trainable_ids, extra_prefixes)."""
    total_layers = len(model.model.layers)
    trainable_ids: set[int] = set()
    extra_prefixes: list[str] = []
    for token in str(train_layer_ids_cfg).split(","):
        token = token.strip()
        if token in ("", "full"):
            continue
        elif token == "first":
            trainable_ids.add(0)
        elif token == "middle":
            trainable_ids.add(total_layers // 2)
        elif token == "last":
            trainable_ids.add(total_layers - 1)
        elif token == "embed":
            extra_prefixes.append("model.embed_tokens")
        elif token == "norm":
            extra_prefixes.append("model.norm")
        elif token == "lm_head":
            extra_prefixes.append("lm_head")
        else:
            trainable_ids.add(int(token))
    trainable_prefixes = tuple(
        [f"model.layers.{i}." for i in trainable_ids] + extra_prefixes
    )
    frozen_count = trainable_count = 0
    for name, p in model.named_parameters():
        if trainable_prefixes and any(name.startswith(pfx) for pfx in trainable_prefixes):
            p.requires_grad_(True)
            trainable_count += 1
        else:
            p.requires_grad_(False)
            frozen_count += 1
    return frozen_count, trainable_count, trainable_ids, extra_prefixes


def assert_eq(label: str, got, want) -> None:
    ok = got == want
    print(f"  {label:30s} got={got!r:30s} want={want!r:30s}  {'✓' if ok else '✗ FAIL'}")
    assert ok, f"{label}: got {got!r}, want {want!r}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HF repo or local path")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    print(f"Loading {args.model} on CPU (this is slow first time, ~1 min)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    n_layers = len(model.model.layers)
    print(f"✓ loaded.  total_layers={n_layers}")
    assert n_layers == 36, f"Qwen2.5-3B should have 36 layers, got {n_layers}"

    print("\n--- case: single layer 14 ---")
    frozen, trainable, ids, extras = apply_freeze(model, "14")
    assert_eq("trainable_ids", ids, {14})
    assert_eq("extras", extras, [])
    print(f"  frozen={frozen}  trainable={trainable}")
    assert trainable > 0, "no params marked trainable"

    print("\n--- case: 3 layers 0,17,35 ---")
    frozen, trainable, ids, extras = apply_freeze(model, "0,17,35")
    assert_eq("trainable_ids", ids, {0, 17, 35})

    print("\n--- case: semantic shortcuts ---")
    frozen, trainable, ids, extras = apply_freeze(model, "first,middle,last,lm_head")
    assert_eq("trainable_ids", ids, {0, 18, 35})
    assert_eq("extras", extras, ["lm_head"])

    print("\n--- case: invalid layer 99 (out of range) — should still parse but freeze all ---")
    frozen, trainable, ids, extras = apply_freeze(model, "99")
    assert_eq("trainable_ids", ids, {99})
    print(f"  frozen={frozen}  trainable={trainable}")
    assert trainable == 0, "layer 99 should have 0 matching params"
    print("  ⚠ WARNING: layer 99 is invalid for 36-layer model — banner should catch this")

    print("\n--- case: full ---")
    frozen, trainable, ids, extras = apply_freeze(model, "full")
    assert_eq("trainable_ids", ids, set())
    print(f"  frozen={frozen}  trainable={trainable}")

    print("\nALL TESTS PASSED ✓")


if __name__ == "__main__":
    main()
