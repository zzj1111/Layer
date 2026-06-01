"""Auto-discover & evaluate all verl ckpts under a path. Logs per-task accuracy
+ avg + length stats to wandb (one run per exp_dir, step = global_step_N).
Persists final.json locally next to each ckpt for resumability.

Path auto-detection:
  - if <path>/global_step_*/actor/huggingface exists -> path is an exp_dir
  - else: recursively look for */global_step_*/actor/huggingface -> path is a root

Examples:
  # single exp, all steps:
  python examples/eval/eval_all_ckpts.py \\
      /scratch/.../checkpoints/DrGRPO-R1Distill7B/0529_xxx \\
      --wandb_project drgrpo_eval

  # whole ckpt root (all exps, all steps):
  python examples/eval/eval_all_ckpts.py \\
      /scratch/.../checkpoints/DrGRPO-R1Distill7B \\
      --wandb_project drgrpo_eval --gpu_list 0,1,2,3,4,5,6,7

  # dry-run to see what would be evaluated:
  python examples/eval/eval_all_ckpts.py <path> --dry_run

  # only step >= 200, force re-eval:
  python examples/eval/eval_all_ckpts.py <path> --step_filter ">=200" --force

Resume:
  By default, ckpts with an existing final.json are skipped.
  Use --force to re-evaluate.

WandB run identity:
  one run per exp_dir, deterministic id (md5(exp_name)[:8]) so re-runs
  resume the same run instead of creating duplicates. Each ckpt's metrics
  are logged at wandb_step = global_step_N.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ---- discovery ----

def is_exp_dir(p: Path) -> bool:
    """An exp dir contains direct global_step_N/actor/huggingface subdirs."""
    try:
        next(p.glob("global_step_*/actor/huggingface/config.json"))
        return True
    except StopIteration:
        return False


def discover(path: Path):
    """Return dict[exp_name -> list[(step, ckpt_huggingface_dir)]], sorted by step."""
    out = {}
    if is_exp_dir(path):
        exp_dirs = [path]
    else:
        seen = set()
        for cfg in path.rglob("global_step_*/actor/huggingface/config.json"):
            exp_dir = cfg.parent.parent.parent.parent
            seen.add(exp_dir)
        exp_dirs = sorted(seen)

    for exp in exp_dirs:
        items = []
        for cfg in exp.glob("global_step_*/actor/huggingface/config.json"):
            # cfg = <exp>/global_step_N/actor/huggingface/config.json
            # parents: huggingface -> actor -> global_step_N
            gs = cfg.parent.parent.parent
            m = re.match(r"global_step_(\d+)$", gs.name)
            if not m:
                continue
            items.append((int(m.group(1)), cfg.parent))
        if items:
            items.sort(key=lambda x: x[0])
            out[exp.name] = items
    return out


# ---- save location ----

def resolve_save_root(cli_arg):
    if cli_arg:
        return Path(cli_arg)
    ckpt_root = os.environ.get("CKPT_ROOT")
    if ckpt_root and Path(ckpt_root).is_dir():
        return Path(ckpt_root) / "eval_results"
    return Path.home() / "eval_results"


# ---- eval driver ----

def run_one_eval(ckpt: Path, save_dir: Path, args):
    """Invoke run_eval_dp8.sh for one ckpt; return parsed final.json (or None)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    final = save_dir / "final.json"
    if final.exists() and not args.force:
        try:
            print(f"[skip] {save_dir.name}: final.json exists")
            return json.load(open(final))
        except Exception as e:
            print(f"[redo] {save_dir.name}: final.json unreadable ({e})")

    env = os.environ.copy()
    env["GPU_LIST"] = args.gpu_list
    env["TEMPLATE"] = args.template
    env["MAX_TOKENS"] = str(args.max_tokens)
    env["MAX_MODEL_LEN"] = str(args.max_model_len)
    env["SAVE_DIR"] = str(save_dir)
    env["TASKS"] = "[" + ",".join(f'"{t.strip()}"' for t in args.tasks.split(",")) + "]"

    print(f"[eval] {save_dir.name}")
    print(f"       ckpt: {ckpt}")
    print(f"       save: {save_dir}")
    rc = subprocess.call(
        ["bash", args.eval_script, str(ckpt), save_dir.name],
        env=env, cwd=args.repo_dir,
    )
    if rc != 0:
        print(f"[ERR] eval failed for {save_dir.name} (rc={rc})")
        return None
    if not final.exists():
        print(f"[ERR] {save_dir.name}: eval exited 0 but no final.json")
        return None
    return json.load(open(final))


# ---- wandb ----

def log_to_wandb(exp_name, results, args):
    if args.no_wandb:
        return
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed, skipping upload")
        return

    run_id = hashlib.md5(f"eval_{exp_name}".encode()).hexdigest()[:8]
    run = wandb.init(
        project=args.wandb_project,
        id=run_id,
        name=f"eval_{exp_name}",
        group=exp_name,
        tags=["eval", args.template, exp_name],
        resume="allow",
        mode=os.environ.get("WANDB_MODE", "online"),
        config={
            "exp": exp_name,
            "template": args.template,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "tasks": [t.strip() for t in args.tasks.split(",")],
        },
    )
    n_logged = 0
    for step, final in results:
        if final is None:
            continue
        log = {f"acc/{k}": v for k, v in final["results"].items()}
        log["acc/avg"] = final["avg"]
        log.update({f"len_avg/{k}": v for k, v in final["avg_lens"].items()})
        log.update({f"len_max/{k}": v for k, v in final["max_lens"].items()})
        wandb.log(log, step=step)
        print(f"  [wandb] step={step}  avg={final['avg']:.4f}")
        n_logged += 1
    wandb.finish()
    print(f"[wandb] {exp_name}: {n_logged} steps logged (run id={run_id})")


# ---- summary ----

def write_summary(all_results, out_path: Path):
    """Markdown table of all evals (one section per exp)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Eval summary\n"]
    for exp, ckpts in all_results.items():
        lines.append(f"\n## {exp}\n")
        ok = [(s, f) for s, f in ckpts if f is not None]
        if not ok:
            lines.append("(no successful evals)\n")
            continue
        keys = sorted({k for _, f in ok for k in f["results"]})
        lines.append("| step | " + " | ".join(keys) + " | avg |")
        lines.append("|" + "---|" * (len(keys) + 2))
        for s, f in ok:
            r = f["results"]
            row = [str(s)] + [f"{r.get(k, float('nan')):.3f}" for k in keys] + [f"{f['avg']:.3f}"]
            lines.append("| " + " | ".join(row) + " |")
    out_path.write_text("\n".join(lines) + "\n")


# ---- step filter ----

def make_step_filter(spec):
    """Parse spec into a (step:int)->bool predicate."""
    if spec is None:
        return lambda s: True
    spec = spec.strip()
    if spec.startswith(">="): return lambda s, n=int(spec[2:]): s >= n
    if spec.startswith("<="): return lambda s, n=int(spec[2:]): s <= n
    if spec.startswith(">"):  return lambda s, n=int(spec[1:]): s > n
    if spec.startswith("<"):  return lambda s, n=int(spec[1:]): s < n
    # explicit list: "105|210" or "105,210"
    sep = "|" if "|" in spec else ","
    wanted = {int(x) for x in spec.split(sep)}
    return lambda s, ss=wanted: s in ss


# ---- main ----

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("path", help="exp_dir or ckpt root (auto-detected)")
    p.add_argument("--save_root", default=None,
                   help="Where to save final.json per ckpt "
                        "(default: $CKPT_ROOT/eval_results, else $HOME/eval_results)")
    p.add_argument("--gpu_list", default=os.environ.get("GPU_LIST", "0,1,2,3,4,5,6,7"),
                   help="Comma-separated GPU ids for DP eval (default 0..7)")
    p.add_argument("--template", default="r1d")
    p.add_argument("--max_tokens", type=int, default=32000)
    p.add_argument("--max_model_len", type=int, default=34816)
    p.add_argument("--tasks", default="aime,amc,math,minerva,olympiad_bench,aime25")
    p.add_argument("--wandb_project", default="drgrpo_eval")
    p.add_argument("--exp_filter", default=None, help="Regex to filter exp_dir names")
    p.add_argument("--step_filter", default=None,
                   help='Filter steps: ">=N", "<=N", or "N1|N2|N3" / "N1,N2"')
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate even when final.json exists")
    p.add_argument("--dry_run", action="store_true",
                   help="Print discovered ckpts and exit")
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--eval_script", default="examples/eval/run_eval_dp8.sh")
    p.add_argument("--repo_dir", default=None,
                   help="cwd for run_eval_dp8.sh (default: this repo root)")
    args = p.parse_args()

    if args.repo_dir is None:
        args.repo_dir = str(Path(__file__).resolve().parent.parent.parent)

    path = Path(args.path).resolve()
    if not path.is_dir():
        sys.exit(f"not a dir: {path}")

    save_root = resolve_save_root(args.save_root)
    print(f"[config] path        = {path}")
    print(f"[config] save_root   = {save_root}")
    print(f"[config] repo_dir    = {args.repo_dir}")
    print(f"[config] gpu_list    = {args.gpu_list}")
    print(f"[config] template    = {args.template}")
    print(f"[config] max_tokens  = {args.max_tokens}")
    print(f"[config] wandb_proj  = {args.wandb_project} ({'OFF' if args.no_wandb else os.environ.get('WANDB_MODE','online')})")

    discovered = discover(path)
    if args.exp_filter:
        rx = re.compile(args.exp_filter)
        discovered = {k: v for k, v in discovered.items() if rx.search(k)}
    step_pred = make_step_filter(args.step_filter)
    for k in list(discovered):
        discovered[k] = [(s, p) for s, p in discovered[k] if step_pred(s)]
        if not discovered[k]:
            del discovered[k]

    total = sum(len(v) for v in discovered.values())
    print(f"\n[discover] {len(discovered)} exps, {total} ckpts:")
    for exp, ckpts in discovered.items():
        print(f"  {exp}: steps={[s for s,_ in ckpts]}")
    if total == 0:
        print("nothing to do"); return
    if args.dry_run:
        print("\n[dry_run] not actually running anything"); return

    all_results = {}
    for exp_name, ckpts in discovered.items():
        results = []
        for step, ckpt_path in ckpts:
            tag = f"{exp_name}_step{step}"
            save_dir = save_root / tag
            results.append((step, run_one_eval(ckpt_path, save_dir, args)))
        all_results[exp_name] = results
        log_to_wandb(exp_name, results, args)

    summary = save_root / "summary.md"
    write_summary(all_results, summary)
    n_ok = sum(1 for ck in all_results.values() for _, f in ck if f is not None)
    print(f"\n[done] {n_ok}/{total} ckpts evaluated")
    print(f"[done] summary -> {summary}")


if __name__ == "__main__":
    main()
