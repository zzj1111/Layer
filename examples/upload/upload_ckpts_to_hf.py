"""Upload verl checkpoints (the actor/huggingface/ subdir) to HuggingFace Hub.

For each ckpt path you pass, this:
  1. Verifies actor/huggingface/config.json exists (a real HF model dir)
  2. Creates an HF repo named <exp_name>_step<N> in your namespace (or --org)
  3. Uploads the entire actor/huggingface/ folder (safetensors auto-LFS'd)
  4. Writes a README.md model card with the training context
  5. Prints final URL list

Auth:
  Pass --token, or export HF_TOKEN, or have ~/.cache/huggingface/token set.
  Token needs write permission on the target namespace.

Resumable:
  upload_folder() skips identical files, so re-running picks up where it
  left off after a network drop.

Examples:
  # Upload 3 ckpts to your personal namespace as private repos
  python upload_ckpts_to_hf.py \\
      /checkpoints/.../0522_1603_DrGRPO_..._L17_..._/global_step_465 \\
      /checkpoints/.../0521_1924_DrGRPO_..._L16_..._/global_step_744 \\
      /checkpoints/.../0521_1924_DrGRPO_..._L12_..._/global_step_465 \\
      --token hf_xxx --private

  # Upload all global_step_* under one exp dir, public
  python upload_ckpts_to_hf.py \\
      /checkpoints/.../0522_1603_DrGRPO_..._L17_..._/global_step_* \\
      --token hf_xxx
"""
import argparse
import os
import re
import sys
import textwrap
from pathlib import Path


def derive_repo_name(ckpt_path: Path) -> str:
    """ckpt_path = <root>/<exp_name>/global_step_<N>
    -> repo name: <exp_name>_step<N>
    """
    m = re.match(r"global_step_(\d+)$", ckpt_path.name)
    if not m:
        raise ValueError(f"{ckpt_path} doesn't match global_step_<N>")
    step = m.group(1)
    exp_name = ckpt_path.parent.name
    name = f"{exp_name}_step{step}"
    # HF repo names: alnum, dash, underscore, dot. Replace anything weird.
    name = re.sub(r"[^A-Za-z0-9._\-]", "-", name)
    if len(name) > 96:
        # HF cap ≈ 96 chars; trim from the front (keep step suffix)
        name = name[-96:]
    return name


def model_card(exp_name: str, step: int, ckpt_path: Path) -> str:
    """Minimal README.md with what we know from the exp name."""
    # parse exp like: 0521_1407_DrGRPO_Qwen2.5-Math-1.5B_full_n8_r3000_lr5e-6
    base_model = "unknown"
    if "Qwen2.5-Math-1.5B" in exp_name:
        base_model = "Qwen/Qwen2.5-Math-1.5B"
    elif "DeepSeek-R1-Distill-Qwen-7B" in exp_name:
        base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # parse layer setting
    layer = "full (all params)"
    m = re.search(r"_L(\d+(?:-\d+)*)_", exp_name)
    if m:
        layer = f"layer(s) {m.group(1)} only (rest frozen)"
    elif "_full_" in exp_name:
        layer = "full (all params)"

    # LR
    lr_m = re.search(r"_lr([\d.e+\-]+)$", exp_name)
    lr = lr_m.group(1) if lr_m else "?"

    # Rollout n
    n_m = re.search(r"_n(\d+)_", exp_name)
    rollout_n = n_m.group(1) if n_m else "?"

    # Max response
    r_m = re.search(r"_r(\d+)_", exp_name)
    max_resp = r_m.group(1) if r_m else "?"

    return textwrap.dedent(f"""\
        ---
        license: apache-2.0
        base_model: {base_model}
        tags:
        - dr-grpo
        - rl
        - math-reasoning
        - verl
        ---

        # {exp_name} — global_step_{step}

        Dr. GRPO fine-tune of `{base_model}`, checkpoint at training step **{step}**.

        ## Training config
        - **Algorithm**: Dr. GRPO (arXiv:2503.20783)
          - `norm_adv_by_std_in_grpo=False` (no /std on advantage)
          - `loss_agg_mode=drgrpo` (1/MAX_TOKENS, not 1/|o_i|)
          - No KL regularization (`kl_loss_coef=0`)
        - **Base model**: {base_model}
        - **Layer training**: {layer}
        - **Learning rate**: {lr} (constant)
        - **Rollout n**: {rollout_n}
        - **Max response length**: {max_resp} tokens
        - **Global step**: {step}
        - **Framework**: [verl](https://github.com/volcengine/verl) v0.7.0

        ## Source

        Original ckpt path: `{ckpt_path}`
        """)


def upload_one(ckpt_path: Path, args, api) -> dict:
    """Upload one ckpt; return dict with status + url."""
    hf_dir = ckpt_path / "actor" / "huggingface"
    cfg = hf_dir / "config.json"
    if not cfg.exists():
        return {"ckpt": str(ckpt_path), "status": "skipped",
                "reason": f"no config.json at {hf_dir}"}

    repo_name = derive_repo_name(ckpt_path)
    namespace = args.org or api.whoami()["name"]
    repo_id = f"{namespace}/{repo_name}"
    exp_name = ckpt_path.parent.name
    step = int(re.match(r"global_step_(\d+)$", ckpt_path.name).group(1))

    print(f"\n[upload] {repo_id}")
    print(f"         src: {hf_dir}")
    print(f"         private: {args.private}")
    if args.dry_run:
        return {"ckpt": str(ckpt_path), "status": "dry-run",
                "repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}"}

    # 1. create repo (idempotent)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=args.private, exist_ok=True)

    # 2. write model card to a tmp file alongside the model (or pass content)
    readme = hf_dir / "README.md"
    write_readme = not readme.exists() or args.overwrite_readme
    if write_readme:
        readme.write_text(model_card(exp_name, step, ckpt_path))
        print(f"         wrote README.md")

    # 3. upload entire folder (skips identical files = resumable)
    api.upload_folder(
        folder_path=str(hf_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload {exp_name} step {step}",
        ignore_patterns=[".cache*", "*.tmp", "tmp_*"],
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"         ✓ {url}")
    return {"ckpt": str(ckpt_path), "status": "ok",
            "repo_id": repo_id, "url": url}


def resolve_token(arg_token):
    if arg_token:
        return arg_token
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if tok:
        return tok
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.exists():
        return cached.read_text().strip()
    sys.exit("No HF token. Pass --token, set $HF_TOKEN, or `huggingface-cli login`")


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("ckpts", nargs="+",
                   help="One or more ckpt paths ending in global_step_<N>")
    p.add_argument("--token", default=None,
                   help="HF token (else $HF_TOKEN or ~/.cache/huggingface/token)")
    p.add_argument("--org", default=None,
                   help="HF namespace (default: token owner's username)")
    p.add_argument("--private", action="store_true",
                   help="Create private repos (default: public)")
    p.add_argument("--public", dest="private", action="store_false")
    p.set_defaults(private=False)
    p.add_argument("--overwrite-readme", action="store_true",
                   help="Always regenerate README.md (default: keep existing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be uploaded, don't actually upload")
    args = p.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub  # required")

    token = resolve_token(args.token)
    api = HfApi(token=token)
    who = api.whoami()
    print(f"[auth] logged in as: {who['name']}  (token type: {who.get('auth',{}).get('accessToken',{}).get('type','?')})")
    namespace = args.org or who["name"]
    print(f"[config] namespace : {namespace}")
    print(f"[config] private   : {args.private}")
    print(f"[config] ckpts     : {len(args.ckpts)}")

    # validate all paths first
    valid = []
    for s in args.ckpts:
        p_ = Path(s).resolve()
        if not p_.is_dir():
            print(f"  ✗ not a dir: {p_}"); continue
        if not re.match(r"global_step_\d+$", p_.name):
            print(f"  ✗ not global_step_<N>: {p_}"); continue
        valid.append(p_)
    print(f"[config] valid     : {len(valid)}/{len(args.ckpts)}")
    if not valid:
        sys.exit("nothing to upload")

    results = [upload_one(p_, args, api) for p_ in valid]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if r["status"] == "ok":
            print(f"  ✓ {r['url']}")
        elif r["status"] == "dry-run":
            print(f"  [dry] would upload -> {r['url']}")
        else:
            print(f"  ✗ {r['ckpt']}: {r.get('reason', r['status'])}")
    n_ok = sum(1 for r in results if r["status"] in ("ok", "dry-run"))
    print(f"\n{n_ok}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
