"""
src/experiments/compare_results.py
────────────────────────────────────
Compare evaluation results across multiple experiments.

Usage:
    python -m src.experiments.compare_results \\
        --experiments baseline hybrid_retrieval reranking

    # Or compare all saved experiments:
    python -m src.experiments.compare_results --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

RESULTS_DIR = Path("experiments/results")


def load_metrics(experiment_name: str) -> dict | None:
    path = RESULTS_DIR / experiment_name / "metrics.json"
    if not path.exists():
        print(f"  [!] Not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(experiment_names: list[str]) -> None:
    all_results = {}
    for name in experiment_names:
        m = load_metrics(name)
        if m:
            all_results[name] = m

    if not all_results:
        print("No results found.")
        return

    # ── Retrieval Metrics Table ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RETRIEVAL METRICS")
    print("=" * 80)

    # Collect all metric keys
    ret_keys: list[str] = []
    for r in all_results.values():
        for k in r.get("retrieval_metrics", {}).keys():
            if k not in ret_keys:
                ret_keys.append(k)

    # Header
    col_w = 20
    header = f"{'Experiment':<30}" + "".join(f"{k:>{col_w}}" for k in ret_keys)
    print(header)
    print("-" * len(header))

    baseline_ret = {}
    for i, (name, result) in enumerate(all_results.items()):
        ret = result.get("retrieval_metrics", {})
        if i == 0:
            baseline_ret = ret
        row = f"{name:<30}"
        for k in ret_keys:
            val = ret.get(k, float("nan"))
            delta = ""
            if i > 0 and k in baseline_ret:
                d = val - baseline_ret[k]
                delta = f" ({'+' if d >= 0 else ''}{d:.3f})"
            row += f"{val:>{col_w - len(delta)}.4f}{delta}"
        print(row)

    # ── Generation Metrics Table ───────────────────────────────────────────────
    has_gen = any(r.get("generation_metrics") for r in all_results.values())
    if has_gen:
        print("\n" + "=" * 80)
        print("  GENERATION METRICS")
        print("=" * 80)

        gen_keys = [k for k in next(
            (r["generation_metrics"] for r in all_results.values() if r.get("generation_metrics")),
            {},
        ).keys() if not k.endswith("_coverage")]

        header2 = f"{'Experiment':<30}" + "".join(f"{k:>{col_w}}" for k in gen_keys)
        print(header2)
        print("-" * len(header2))

        baseline_gen = {}
        for i, (name, result) in enumerate(all_results.items()):
            gen = result.get("generation_metrics", {})
            if i == 0:
                baseline_gen = gen
            row = f"{name:<30}"
            for k in gen_keys:
                val = gen.get(k, float("nan"))
                delta = ""
                if i > 0 and k in baseline_gen and baseline_gen[k] >= 0:
                    d = val - baseline_gen[k]
                    delta = f" ({'+' if d >= 0 else ''}{d:.3f})"
                row += f"{val:>{col_w - len(delta)}.4f}{delta}"
            print(row)

    # ── System Metrics Table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SYSTEM METRICS (mean latency)")
    print("=" * 80)
    header3 = f"{'Experiment':<30}{'retrieval_s':>18}{'generation_s':>18}{'total_s':>18}"
    print(header3)
    print("-" * len(header3))
    for name, result in all_results.items():
        sys_m = result.get("system_metrics", {})
        ret_t  = sys_m.get("retrieval_time_s",  {}).get("mean", float("nan"))
        gen_t  = sys_m.get("generation_time_s", {}).get("mean", float("nan"))
        tot_t  = sys_m.get("total_latency_s",   {}).get("mean", float("nan"))
        print(f"{name:<30}{ret_t:>18.3f}{gen_t:>18.3f}{tot_t:>18.3f}")

    # ── Failure Analysis ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  FAILURE ANALYSIS")
    print("=" * 80)
    for name, result in all_results.items():
        fa = result.get("failure_analysis", {})
        total = result.get("num_samples", 1)
        print(f"\n  {name} (n={total}):")
        for mode, count in sorted(fa.items()):
            pct = count / total * 100
            print(f"    {mode:<30} {count:>4}  ({pct:.1f}%)")

    print("\n" + "=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare RAG experiment results")
    parser.add_argument("--experiments", nargs="+", help="Experiment names to compare")
    parser.add_argument("--all", action="store_true", help="Compare all saved experiments")
    args = parser.parse_args()

    if args.all:
        experiments = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
        if not experiments:
            print(f"No experiments found in {RESULTS_DIR}")
            sys.exit(1)
        # Put baseline first if it exists
        if "baseline" in experiments:
            experiments.remove("baseline")
            experiments.insert(0, "baseline")
    elif args.experiments:
        experiments = args.experiments
    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nComparing experiments: {', '.join(experiments)}")
    compare(experiments)


if __name__ == "__main__":
    main()
