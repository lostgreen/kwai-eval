#!/usr/bin/env python3
"""
Visualization script for comparing Base vs RL model evaluation results.

Typical workflow:
  1.  Run eval with Base model  → results land in OUTPUT_DIR
  2.  Run eval with RL model    → results land in same OUTPUT_DIR
  3.  Run this script to generate comparison figures

Usage:
  # Auto-detect all models in the results directory
  python eval/visualize.py results/

  # Specify model names explicitly (substring match on filenames)
  python eval/visualize.py results/ \
      --base Qwen3-VL-7B-Instruct \
      --rl   ArrowRL-Qwen3-VL-7B

  # Custom output directory for figures
  python eval/visualize.py results/ --save_dir ./figures

Generated figures:
  1. overview_bar.png          — Per-benchmark accuracy (grouped bars + deltas)
  2. overview_radar.png        — Radar chart across benchmarks
  3. mvbench_tasks.png         — 20 MVBench task breakdown
  4. longvideobench_duration.png — Accuracy vs video duration
  5. delta_breakdown.png       — RL improvement waterfall
"""
import argparse
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

COLOR_BASE   = "#4C72B0"
COLOR_RL     = "#DD8452"
COLOR_POS    = "#55A868"
COLOR_NEG    = "#C44E52"
COLOR_GRID   = "#E0E0E0"
COLOR_BG     = "#FAFAFA"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.facecolor": COLOR_BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
})

# ---------------------------------------------------------------------------
# Known benchmark / subset information
# ---------------------------------------------------------------------------

BENCHMARKS = ["futureomni", "seeaot", "mvbench", "longvideobench"]

SEEAOT_SUBSETS = ["ReverseFilm", "UCF101", "Rtime_t2v", "Rtime_v2t", "AoTBench_QA"]

# Short display names for plots
BENCH_DISPLAY = {
    "futureomni":     "FutureOmni",
    "seeaot":         "seeAoT",
    "mvbench":        "MVBench",
    "longvideobench": "LongVideoBench",
}

SEEAOT_DISPLAY = {
    "ReverseFilm": "ReverseFilm",
    "UCF101":      "UCF101",
    "Rtime_t2v":   "Rtime-T2V",
    "Rtime_v2t":   "Rtime-V2T",
    "AoTBench_QA": "AoTBench-QA",
}

DURATION_DISPLAY = {15: "8-15s", 60: "15-60s", 600: "3-10min", 3600: "10-60min"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> list:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def accuracy(results: list) -> float:
    if not results:
        return 0.0
    return sum(
        1 for r in results if r.get("pred", "").upper() == r.get("answer", "").upper()
    ) / len(results) * 100


def group_by_meta(results: list, key: str) -> Dict[str, list]:
    groups: Dict[str, list] = defaultdict(list)
    for r in results:
        val = (r.get("metadata") or {}).get(key, "unknown")
        if val is None:
            val = "unknown"
        groups[str(val)].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# File discovery and model identification
# ---------------------------------------------------------------------------

def detect_benchmark(filename: str) -> Optional[str]:
    """Return benchmark name from filename prefix."""
    for bm in BENCHMARKS:
        if filename.startswith(bm):
            return bm
    return None


def detect_seeaot_subset(filename: str) -> Optional[str]:
    """Return seeAoT subset name from filename."""
    for sub in SEEAOT_SUBSETS:
        if sub in filename:
            return sub
    return None


def find_model_files(result_dir: str, model_name: str) -> List[str]:
    """Find all JSONL files in result_dir whose name contains model_name."""
    files = []
    for f in sorted(os.listdir(result_dir)):
        if f.endswith(".jsonl") and model_name in f:
            files.append(os.path.join(result_dir, f))
    return files


def auto_detect_models(result_dir: str) -> List[str]:
    """
    Auto-detect model names from result filenames.

    Strategy: strip benchmark prefix, frame suffix, and backend suffix,
    then collect unique model name tokens.
    """
    models = set()
    for f in os.listdir(result_dir):
        if not f.endswith(".jsonl"):
            continue
        stem = f.replace(".jsonl", "")
        # strip _{Nf}_{backend} suffix
        m = re.match(r"^(.+)_(\d+)f_(vllm|tf)$", stem)
        if not m:
            # fallback: try without backend suffix (infer.py output)
            m = re.match(r"^(.+)_(\d+)f$", stem)
        if not m:
            continue

        prefix = m.group(1)
        # strip benchmark + subset prefix
        for bm in BENCHMARKS:
            if prefix.startswith(bm + "_"):
                remainder = prefix[len(bm) + 1:]
                # for seeaot, strip subset
                for sub in SEEAOT_SUBSETS:
                    if remainder.startswith(sub + "_"):
                        remainder = remainder[len(sub) + 1:]
                        break
                # for futureomni, strip data_name (first token)
                if bm == "futureomni" and "_" in remainder:
                    remainder = remainder.split("_", 1)[1]
                models.add(remainder)
                break
    return sorted(models)


# ---------------------------------------------------------------------------
# Build accuracy tables
# ---------------------------------------------------------------------------

def build_accuracy_table(
    result_dir: str, model_name: str
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Returns:
        overall: {benchmark_key: accuracy%} — seeaot sub-keys like "seeaot/ReverseFilm"
        breakdowns: {benchmark: {sub_category: accuracy%}}
    """
    files = find_model_files(result_dir, model_name)
    overall: Dict[str, float] = OrderedDict()
    breakdowns: Dict[str, Dict[str, float]] = {}

    all_seeaot_results = []

    for fpath in files:
        fname = os.path.basename(fpath)
        bm = detect_benchmark(fname)
        if bm is None:
            continue
        results = load_results(fpath)
        if not results:
            continue

        if bm == "seeaot":
            sub = detect_seeaot_subset(fname)
            key = f"seeaot/{sub}" if sub else "seeaot"
            overall[key] = accuracy(results)
            all_seeaot_results.extend(results)

        elif bm == "futureomni":
            overall["futureomni"] = accuracy(results)
            # breakdown by video_domain
            bd = {}
            for k, v in group_by_meta(results, "video_domain").items():
                if k != "unknown" and k is not None:
                    bd[k] = accuracy(v)
            if bd:
                breakdowns["futureomni_domain"] = bd
            # breakdown by audio_type
            bd2 = {}
            for k, v in group_by_meta(results, "audio_type").items():
                if k != "unknown" and k is not None:
                    bd2[k] = accuracy(v)
            if bd2:
                breakdowns["futureomni_audio"] = bd2

        elif bm == "mvbench":
            overall["mvbench"] = accuracy(results)
            bd = {}
            for k, v in group_by_meta(results, "task_type").items():
                if k != "unknown" and k is not None:
                    bd[k] = accuracy(v)
            if bd:
                breakdowns["mvbench_tasks"] = bd

        elif bm == "longvideobench":
            overall["longvideobench"] = accuracy(results)
            bd = {}
            for k, v in group_by_meta(results, "duration_group").items():
                if k != "unknown" and k is not None:
                    label = DURATION_DISPLAY.get(int(float(k)), f"{k}s")
                    bd[label] = accuracy(v)
            if bd:
                breakdowns["lvb_duration"] = bd
            bd2 = {}
            for k, v in group_by_meta(results, "question_category").items():
                if k != "unknown" and k is not None:
                    bd2[k] = accuracy(v)
            if bd2:
                breakdowns["lvb_category"] = bd2

    # seeaot aggregate
    if all_seeaot_results:
        overall["seeaot"] = accuracy(all_seeaot_results)

    return overall, breakdowns


# ---------------------------------------------------------------------------
# Figure 1: Overview grouped bar chart
# ---------------------------------------------------------------------------

def plot_overview_bar(
    base_overall: Dict[str, float],
    rl_overall: Dict[str, float],
    base_name: str,
    rl_name: str,
    save_path: str,
):
    """Horizontal grouped bar chart for all benchmarks + seeAoT subsets."""
    # Build ordered keys: top-level benchmarks first, then seeaot subsets
    keys = []
    for bm in ["futureomni", "mvbench", "longvideobench", "seeaot"]:
        if bm in base_overall or bm in rl_overall:
            keys.append(bm)
    for sub in SEEAOT_SUBSETS:
        k = f"seeaot/{sub}"
        if k in base_overall or k in rl_overall:
            keys.append(k)

    if not keys:
        return

    labels = []
    for k in keys:
        if "/" in k:
            labels.append("  " + SEEAOT_DISPLAY.get(k.split("/")[1], k.split("/")[1]))
        else:
            labels.append(BENCH_DISPLAY.get(k, k))

    base_vals = [base_overall.get(k, 0) for k in keys]
    rl_vals = [rl_overall.get(k, 0) for k in keys]

    n = len(keys)
    y = np.arange(n)
    h = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55 + 1.2)))

    bars_base = ax.barh(y + h / 2, base_vals, h, label=base_name, color=COLOR_BASE, edgecolor="white", linewidth=0.5)
    bars_rl   = ax.barh(y - h / 2, rl_vals,   h, label=rl_name,   color=COLOR_RL,   edgecolor="white", linewidth=0.5)

    # Value labels + delta
    for i in range(n):
        bv, rv = base_vals[i], rl_vals[i]
        ax.text(bv + 0.5, y[i] + h / 2, f"{bv:.1f}", va="center", fontsize=8, color=COLOR_BASE)
        ax.text(rv + 0.5, y[i] - h / 2, f"{rv:.1f}", va="center", fontsize=8, color=COLOR_RL)
        # delta annotation on the right
        delta = rv - bv
        if bv > 0 or rv > 0:
            color = COLOR_POS if delta >= 0 else COLOR_NEG
            sign = "+" if delta >= 0 else ""
            max_val = max(bv, rv)
            ax.text(
                max_val + 4, y[i], f"{sign}{delta:.1f}",
                va="center", fontsize=8, fontweight="bold", color=color,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, max(max(base_vals + [0]), max(rl_vals + [0])) + 10)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("Base vs RL — Benchmark Comparison", fontsize=14, fontweight="bold", pad=12)

    # Separator line between top-level and seeaot subsets
    n_toplevel = sum(1 for k in keys if "/" not in k)
    if n_toplevel < n:
        ax.axhline(y=n_toplevel - 0.5, color="#999999", linewidth=0.8, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: Radar chart
# ---------------------------------------------------------------------------

def plot_radar(
    base_overall: Dict[str, float],
    rl_overall: Dict[str, float],
    base_name: str,
    rl_name: str,
    save_path: str,
):
    """Radar chart with one axis per top-level benchmark."""
    top_keys = [bm for bm in ["futureomni", "seeaot", "mvbench", "longvideobench"]
                if bm in base_overall or bm in rl_overall]
    if len(top_keys) < 3:
        return  # radar needs at least 3 axes

    labels = [BENCH_DISPLAY[k] for k in top_keys]
    base_vals = [base_overall.get(k, 0) for k in top_keys]
    rl_vals = [rl_overall.get(k, 0) for k in top_keys]

    N = len(top_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]
    base_vals += base_vals[:1]
    rl_vals += rl_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_facecolor(COLOR_BG)

    ax.plot(angles, base_vals, "o-", color=COLOR_BASE, linewidth=2, label=base_name, markersize=6)
    ax.fill(angles, base_vals, alpha=0.15, color=COLOR_BASE)
    ax.plot(angles, rl_vals, "s-", color=COLOR_RL, linewidth=2, label=rl_name, markersize=6)
    ax.fill(angles, rl_vals, alpha=0.15, color=COLOR_RL)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    # Value annotations
    for angle, bv, rv in zip(angles[:-1], base_vals[:-1], rl_vals[:-1]):
        ax.text(angle, bv + 2, f"{bv:.1f}", ha="center", fontsize=8, color=COLOR_BASE, fontweight="bold")
        ax.text(angle, rv - 4, f"{rv:.1f}", ha="center", fontsize=8, color=COLOR_RL, fontweight="bold")

    # Gridlines
    all_vals = base_vals + rl_vals
    vmin = max(0, min(all_vals) - 10)
    vmax = min(100, max(all_vals) + 10)
    ax.set_ylim(vmin, vmax)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_rlabel_position(30)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), framealpha=0.9)
    ax.set_title("Model Capability Radar", fontsize=14, fontweight="bold", y=1.1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: MVBench task-level breakdown
# ---------------------------------------------------------------------------

def plot_mvbench_tasks(
    base_bd: Dict[str, float],
    rl_bd: Dict[str, float],
    base_name: str,
    rl_name: str,
    save_path: str,
):
    """Horizontal grouped bars for each of the 20 MVBench tasks."""
    all_tasks = sorted(set(list(base_bd.keys()) + list(rl_bd.keys())))
    if not all_tasks:
        return

    n = len(all_tasks)
    y = np.arange(n)
    h = 0.35

    base_vals = [base_bd.get(t, 0) for t in all_tasks]
    rl_vals = [rl_bd.get(t, 0) for t in all_tasks]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.38 + 1)))

    ax.barh(y + h / 2, base_vals, h, label=base_name, color=COLOR_BASE, edgecolor="white", linewidth=0.5)
    ax.barh(y - h / 2, rl_vals,   h, label=rl_name,   color=COLOR_RL,   edgecolor="white", linewidth=0.5)

    for i in range(n):
        bv, rv = base_vals[i], rl_vals[i]
        delta = rv - bv
        color = COLOR_POS if delta >= 0 else COLOR_NEG
        sign = "+" if delta >= 0 else ""
        max_val = max(bv, rv)
        ax.text(max_val + 1, y[i], f"{sign}{delta:.1f}", va="center", fontsize=7, fontweight="bold", color=color)

    ax.set_yticks(y)
    ax.set_yticklabels(all_tasks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, min(100, max(max(base_vals + [0]), max(rl_vals + [0])) + 12))
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("MVBench — Per-Task Comparison", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# ---------------------------------------------------------------------------
# Figure 4: LongVideoBench by duration group
# ---------------------------------------------------------------------------

def plot_lvb_duration(
    base_bd: Dict[str, float],
    rl_bd: Dict[str, float],
    base_name: str,
    rl_name: str,
    save_path: str,
):
    """Bar chart: accuracy vs video duration bucket."""
    order = ["8-15s", "15-60s", "3-10min", "10-60min"]
    durations = [d for d in order if d in base_bd or d in rl_bd]
    if not durations:
        return

    n = len(durations)
    x = np.arange(n)
    w = 0.35

    base_vals = [base_bd.get(d, 0) for d in durations]
    rl_vals = [rl_bd.get(d, 0) for d in durations]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_b = ax.bar(x - w / 2, base_vals, w, label=base_name, color=COLOR_BASE, edgecolor="white")
    bars_r = ax.bar(x + w / 2, rl_vals,   w, label=rl_name,   color=COLOR_RL,   edgecolor="white")

    for i in range(n):
        ax.text(x[i] - w / 2, base_vals[i] + 0.8, f"{base_vals[i]:.1f}", ha="center", fontsize=9, color=COLOR_BASE)
        ax.text(x[i] + w / 2, rl_vals[i] + 0.8,   f"{rl_vals[i]:.1f}",   ha="center", fontsize=9, color=COLOR_RL)

    ax.set_xticks(x)
    ax.set_xticklabels(durations, fontsize=11)
    ax.set_xlabel("Video Duration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, min(100, max(max(base_vals + [0]), max(rl_vals + [0])) + 10))
    ax.legend(framealpha=0.9)
    ax.set_title("LongVideoBench — Accuracy by Video Duration", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# ---------------------------------------------------------------------------
# Figure 5: Delta waterfall
# ---------------------------------------------------------------------------

def plot_delta(
    base_overall: Dict[str, float],
    rl_overall: Dict[str, float],
    base_name: str,
    rl_name: str,
    save_path: str,
):
    """Waterfall chart showing RL improvement (delta) for each benchmark/subset."""
    keys = []
    for bm in ["futureomni", "mvbench", "longvideobench"]:
        if bm in base_overall and bm in rl_overall:
            keys.append(bm)
    for sub in SEEAOT_SUBSETS:
        k = f"seeaot/{sub}"
        if k in base_overall and k in rl_overall:
            keys.append(k)

    if not keys:
        return

    labels = []
    for k in keys:
        if "/" in k:
            labels.append(SEEAOT_DISPLAY.get(k.split("/")[1], k.split("/")[1]))
        else:
            labels.append(BENCH_DISPLAY.get(k, k))

    deltas = [rl_overall[k] - base_overall[k] for k in keys]
    colors = [COLOR_POS if d >= 0 else COLOR_NEG for d in deltas]

    n = len(keys)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.0 + 1), 5))

    bars = ax.bar(x, deltas, 0.6, color=colors, edgecolor="white", linewidth=0.5)

    for i, (d, bar) in enumerate(zip(deltas, bars)):
        sign = "+" if d >= 0 else ""
        va = "bottom" if d >= 0 else "top"
        offset = 0.3 if d >= 0 else -0.3
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + offset,
            f"{sign}{d:.1f}",
            ha="center", va=va, fontsize=10, fontweight="bold", color=colors[i],
        )

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=25, ha="right")
    ax.set_ylabel("Accuracy Delta (%)")
    ax.set_title(
        f"RL Improvement  ({rl_name} - {base_name})",
        fontsize=14, fontweight="bold", pad=12,
    )

    # Overall average delta line
    if deltas:
        avg = np.mean(deltas)
        color = COLOR_POS if avg >= 0 else COLOR_NEG
        ax.axhline(avg, color=color, linewidth=1.2, linestyle="--", alpha=0.7)
        ax.text(
            n - 0.5, avg + 0.3, f"avg {avg:+.1f}",
            fontsize=9, color=color, fontweight="bold",
        )

    yabs = max(abs(min(deltas + [0])), abs(max(deltas + [0])))
    ax.set_ylim(-yabs - 5, yabs + 5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# ---------------------------------------------------------------------------
# Figure 6: FutureOmni breakdown (domain + audio type)
# ---------------------------------------------------------------------------

def plot_futureomni_breakdown(
    base_bd: Dict[str, Dict[str, float]],
    rl_bd: Dict[str, Dict[str, float]],
    base_name: str,
    rl_name: str,
    save_path: str,
):
    """Side-by-side charts for FutureOmni by video_domain and audio_type."""
    bd_domain_base = base_bd.get("futureomni_domain", {})
    bd_domain_rl   = rl_bd.get("futureomni_domain", {})
    bd_audio_base  = base_bd.get("futureomni_audio", {})
    bd_audio_rl    = rl_bd.get("futureomni_audio", {})

    panels = []
    if bd_domain_base or bd_domain_rl:
        cats = sorted(set(list(bd_domain_base.keys()) + list(bd_domain_rl.keys())))
        panels.append(("By Video Domain", cats, bd_domain_base, bd_domain_rl))
    if bd_audio_base or bd_audio_rl:
        cats = sorted(set(list(bd_audio_base.keys()) + list(bd_audio_rl.keys())))
        panels.append(("By Audio Type", cats, bd_audio_base, bd_audio_rl))

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, cats, bd_b, bd_r) in zip(axes, panels):
        n = len(cats)
        y = np.arange(n)
        h = 0.35
        bv = [bd_b.get(c, 0) for c in cats]
        rv = [bd_r.get(c, 0) for c in cats]

        ax.barh(y + h / 2, bv, h, label=base_name, color=COLOR_BASE, edgecolor="white")
        ax.barh(y - h / 2, rv, h, label=rl_name,   color=COLOR_RL,   edgecolor="white")

        for i in range(n):
            delta = rv[i] - bv[i]
            sign = "+" if delta >= 0 else ""
            color = COLOR_POS if delta >= 0 else COLOR_NEG
            mx = max(bv[i], rv[i])
            ax.text(mx + 1, y[i], f"{sign}{delta:.1f}", va="center", fontsize=8, fontweight="bold", color=color)

        ax.set_yticks(y)
        ax.set_yticklabels(cats, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("FutureOmni — Category Breakdown", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {save_path}")


# ---------------------------------------------------------------------------
# Summary table (printed to console)
# ---------------------------------------------------------------------------

def print_summary(
    base_overall: Dict[str, float],
    rl_overall: Dict[str, float],
    base_name: str,
    rl_name: str,
):
    all_keys = []
    for bm in ["futureomni", "mvbench", "longvideobench", "seeaot"]:
        if bm in base_overall or bm in rl_overall:
            all_keys.append(bm)
    for sub in SEEAOT_SUBSETS:
        k = f"seeaot/{sub}"
        if k in base_overall or k in rl_overall:
            all_keys.append(k)

    max_label = max(len(BENCH_DISPLAY.get(k, SEEAOT_DISPLAY.get(k.split("/")[-1], k))) for k in all_keys) if all_keys else 20
    bw = max(len(base_name), 8)
    rw = max(len(rl_name), 8)

    header = f"{'Benchmark':<{max_label+2}} | {base_name:>{bw}} | {rl_name:>{rw}} | {'Delta':>7}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for k in all_keys:
        if "/" in k:
            label = "  " + SEEAOT_DISPLAY.get(k.split("/")[1], k.split("/")[1])
        else:
            label = BENCH_DISPLAY.get(k, k)
        bv = base_overall.get(k, 0)
        rv = rl_overall.get(k, 0)
        delta = rv - bv
        sign = "+" if delta >= 0 else ""
        print(f"{label:<{max_label+2}} | {bv:>{bw}.1f} | {rv:>{rw}.1f} | {sign}{delta:>6.1f}")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize Base vs RL evaluation results",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "result_dir", help="Directory containing JSONL result files from both models"
    )
    parser.add_argument(
        "--base", default=None,
        help="Base model name (substring matched in filenames).\n"
             "If omitted, auto-detected from files.",
    )
    parser.add_argument(
        "--rl", default=None,
        help="RL model name (substring matched in filenames).\n"
             "If omitted, auto-detected from files.",
    )
    parser.add_argument(
        "--save_dir", default=None,
        help="Directory to save figures (default: <result_dir>/figures)",
    )
    args = parser.parse_args()

    result_dir = args.result_dir
    if not os.path.isdir(result_dir):
        print(f"[ERROR] Not a directory: {result_dir}")
        sys.exit(1)

    # Auto-detect models
    detected = auto_detect_models(result_dir)
    if not detected:
        print("[ERROR] No JSONL result files found in", result_dir)
        sys.exit(1)

    print(f"[Detected models] {detected}")

    base_name = args.base
    rl_name = args.rl

    if base_name is None or rl_name is None:
        if len(detected) == 1:
            base_name = base_name or detected[0]
            rl_name = rl_name or detected[0]
            print(f"[INFO] Only one model found: {detected[0]}. Showing single-model charts.")
        elif len(detected) == 2:
            base_name = base_name or detected[0]
            rl_name = rl_name or detected[1]
        else:
            print(f"[INFO] Found {len(detected)} models: {detected}")
            print("       Specify --base and --rl to select which two to compare.")
            base_name = base_name or detected[0]
            rl_name = rl_name or detected[1]

    print(f"[Base] {base_name}")
    print(f"[RL]   {rl_name}")

    save_dir = args.save_dir or os.path.join(result_dir, "figures")
    os.makedirs(save_dir, exist_ok=True)

    # Build accuracy tables
    base_overall, base_bd = build_accuracy_table(result_dir, base_name)
    rl_overall, rl_bd = build_accuracy_table(result_dir, rl_name)

    if not base_overall and not rl_overall:
        print("[ERROR] No results matched. Check model names and filenames.")
        sys.exit(1)

    # Console summary
    print_summary(base_overall, rl_overall, base_name, rl_name)

    # Generate figures
    print(f"\n[Figures] Saving to {save_dir}/")

    plot_overview_bar(base_overall, rl_overall, base_name, rl_name,
                      os.path.join(save_dir, "overview_bar.png"))

    plot_radar(base_overall, rl_overall, base_name, rl_name,
               os.path.join(save_dir, "overview_radar.png"))

    plot_delta(base_overall, rl_overall, base_name, rl_name,
               os.path.join(save_dir, "delta_breakdown.png"))

    # Conditional: FutureOmni breakdown
    if "futureomni_domain" in base_bd or "futureomni_domain" in rl_bd:
        plot_futureomni_breakdown(base_bd, rl_bd, base_name, rl_name,
                                  os.path.join(save_dir, "futureomni_breakdown.png"))

    # Conditional: MVBench tasks
    mvb_base = base_bd.get("mvbench_tasks", {})
    mvb_rl = rl_bd.get("mvbench_tasks", {})
    if mvb_base or mvb_rl:
        plot_mvbench_tasks(mvb_base, mvb_rl, base_name, rl_name,
                           os.path.join(save_dir, "mvbench_tasks.png"))

    # Conditional: LongVideoBench by duration
    lvb_base = base_bd.get("lvb_duration", {})
    lvb_rl = rl_bd.get("lvb_duration", {})
    if lvb_base or lvb_rl:
        plot_lvb_duration(lvb_base, lvb_rl, base_name, rl_name,
                          os.path.join(save_dir, "longvideobench_duration.png"))

    print(f"\nDone! {len(os.listdir(save_dir))} figures saved to {save_dir}/")


if __name__ == "__main__":
    main()
