"""Aggregate mean±std across multiple eval runs.

Reads <base_dir>/run*/<exp>_step<N>/final.json, groups by (exp, step),
computes mean & stdev for each task accuracy across runs. Writes:
  - <base_dir>/aggregated.json    : per (exp,step) mean+std+per_run
  - <base_dir>/aggregated.md      : human-readable markdown table

Usage:
    python aggregate_multi_run.py <base_dir>
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def collect(base_dir: Path):
    """Return dict[(exp_name, step) -> list[{task: acc, ..., 'avg': X, 'run': i}]]."""
    groups = defaultdict(list)
    for run_dir in sorted(base_dir.glob("run*")):
        m = re.match(r"run(\d+)$", run_dir.name)
        if not m:
            continue
        run_idx = int(m.group(1))
        for sd in run_dir.glob("*_step*"):
            final = sd / "final.json"
            if not final.exists():
                continue
            tag = sd.name
            # tag = <exp_name>_step<N>
            m2 = re.match(r"(.+)_step(\d+)$", tag)
            if not m2:
                continue
            exp = m2.group(1)
            step = int(m2.group(2))
            d = json.load(open(final))
            row = {"run": run_idx, "avg": d["avg"], **d["results"]}
            groups[(exp, step)].append(row)
    return groups


def mean_std(xs):
    if not xs:
        return (float("nan"), float("nan"), 0)
    if len(xs) == 1:
        return (xs[0], 0.0, 1)
    return (statistics.mean(xs), statistics.stdev(xs), len(xs))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: aggregate_multi_run.py <base_dir>")
    base = Path(sys.argv[1]).resolve()
    if not base.is_dir():
        sys.exit(f"not a dir: {base}")

    groups = collect(base)
    if not groups:
        sys.exit(f"no run*/<exp>_step*/final.json under {base}")

    # collect all task names across runs
    task_set = set()
    for rows in groups.values():
        for r in rows:
            task_set.update(r.keys())
    task_set -= {"run"}
    tasks = sorted(task_set - {"avg"}) + ["avg"]

    out_json = {"base_dir": str(base), "per_exp_step": []}
    md_lines = [f"# Multi-run aggregated eval — {base.name}\n"]

    # group by exp
    by_exp = defaultdict(list)
    for (exp, step), rows in groups.items():
        by_exp[exp].append((step, rows))

    for exp, items in sorted(by_exp.items()):
        items.sort(key=lambda x: x[0])
        md_lines.append(f"\n## {exp}\n")
        n_runs_max = max(len(r) for _, r in items)
        md_lines.append(f"_(up to {n_runs_max} runs per step)_\n")
        header = "| step | " + " | ".join(tasks) + " | n |"
        sep = "|" + "---|" * (len(tasks) + 2)
        md_lines.append(header)
        md_lines.append(sep)
        for step, rows in items:
            row_md = [str(step)]
            entry = {"exp": exp, "step": step, "n_runs": len(rows), "tasks": {}}
            for t in tasks:
                vals = [r[t] for r in rows if t in r]
                mean, std, n = mean_std(vals)
                if n == 0:
                    row_md.append("—")
                elif n == 1:
                    row_md.append(f"{mean*100:.2f}")
                else:
                    row_md.append(f"{mean*100:.2f} ± {std*100:.2f}")
                entry["tasks"][t] = {"mean": mean, "std": std, "n": n,
                                     "per_run": [r[t] for r in rows if t in r]}
            row_md.append(str(len(rows)))
            md_lines.append("| " + " | ".join(row_md) + " |")
            out_json["per_exp_step"].append(entry)

        # also dump per-run rows table
        md_lines.append(f"\n<details><summary>Per-run breakdown ({exp})</summary>\n")
        md_lines.append("\n| step | run | " + " | ".join(tasks) + " |")
        md_lines.append("|" + "---|" * (len(tasks) + 2))
        for step, rows in items:
            for r in sorted(rows, key=lambda x: x["run"]):
                cells = [str(step), str(r["run"])]
                for t in tasks:
                    cells.append(f"{r.get(t,float('nan'))*100:.2f}" if t in r else "—")
                md_lines.append("| " + " | ".join(cells) + " |")
        md_lines.append("\n</details>\n")

    out_json_path = base / "aggregated.json"
    out_md_path = base / "aggregated.md"
    out_json_path.write_text(json.dumps(out_json, indent=2))
    out_md_path.write_text("\n".join(md_lines) + "\n")

    # also print summary to stdout
    print()
    print("=" * 70)
    print(f"AGGREGATED across runs (mean ± std, %)")
    print("=" * 70)
    for exp, items in sorted(by_exp.items()):
        print(f"\n{exp}")
        items.sort(key=lambda x: x[0])
        header = f"  {'step':>6} | " + " | ".join(f"{t:>14}" for t in tasks) + " | n"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for step, rows in items:
            cells = [f"{step:>6}"]
            for t in tasks:
                vals = [r[t] for r in rows if t in r]
                mean, std, n = mean_std(vals)
                if n == 0:
                    cells.append(f"{'—':>14}")
                elif n == 1:
                    cells.append(f"{mean*100:>13.2f} ")
                else:
                    cells.append(f"{mean*100:>5.2f}±{std*100:<5.2f}".rjust(14))
            cells.append(str(len(rows)))
            print(f"  {' | '.join(cells)}")
    print(f"\n→ {out_md_path}")
    print(f"→ {out_json_path}")


if __name__ == "__main__":
    main()
