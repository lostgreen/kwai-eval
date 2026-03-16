#!/usr/bin/env python3
"""
Compute evaluation metrics from inference result JSONL files.

Usage:
  # Single file
  python eval/metrics.py results/seeaot_AoTBench_QA_Qwen3-VL-4B_16f.jsonl

  # Multiple files at once
  python eval/metrics.py results/*.jsonl

  # Verbose: per-category breakdown (uses metadata.video_domain for FutureOmni)
  python eval/metrics.py results/*.jsonl --verbose
"""
import argparse
import json
import os
from collections import defaultdict


def load_results(result_file: str) -> list:
    results = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def accuracy(results: list) -> tuple:
    """Returns (acc_float, correct_int, total_int)."""
    if not results:
        return 0.0, 0, 0
    correct = sum(
        1 for r in results
        if r.get("pred", "").upper() == r.get("answer", "").upper()
    )
    return correct / len(results), correct, len(results)


def breakdown(results: list, key: str) -> dict:
    """Group results by metadata[key] and compute per-group accuracy."""
    groups: dict = defaultdict(list)
    for r in results:
        val = (r.get("metadata") or {}).get(key, "unknown")
        groups[val].append(r)
    return {k: accuracy(v) for k, v in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy from result JSONL files")
    parser.add_argument("result_files", nargs="+", help="One or more JSONL result files")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-category breakdown (video_domain, audio_type)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("  Evaluation Results")
    print("=" * 64)

    all_results = []
    for rf in args.result_files:
        if not os.path.exists(rf):
            print(f"\n[Skip] Not found: {rf}")
            continue

        results = load_results(rf)
        if not results:
            print(f"\n[Skip] Empty file: {rf}")
            continue

        all_results.extend(results)
        acc, correct, total = accuracy(results)
        name = os.path.basename(rf)
        print(f"\n  {name}")
        print(f"  Accuracy : {acc * 100:.2f}%  ({correct}/{total})")

        if args.verbose:
            # FutureOmni breakdowns
            for meta_key in ("video_domain", "audio_type", "forecasting_pattern"):
                bd = breakdown(results, meta_key)
                has_data = any(k != "unknown" for k in bd)
                if has_data:
                    print(f"\n  ── by {meta_key}:")
                    for cat, (a, c, t) in bd.items():
                        if cat == "unknown":
                            continue
                        print(f"     {cat:<28} {a * 100:.2f}%  ({c}/{t})")

    # Aggregate across all files if >1
    if len(args.result_files) > 1 and all_results:
        acc, correct, total = accuracy(all_results)
        print(f"\n{'─' * 64}")
        print(f"  OVERALL  Accuracy : {acc * 100:.2f}%  ({correct}/{total})")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
