#!/usr/bin/env python3
"""Generate the analysis figures from the merged dataset (and a row profile).

Reads the canonical ``merged.csv`` produced by ``merge_metrics.py`` and renders
the figures that carry the study's argument, in three groups:

Scalability (the load-bearing plots, all paradigms overlaid)
  * speedup     S(p) vs p, with the dashed ideal S = p as the reference the eye
                needs to judge whether a curve is good or poor;
  * efficiency  E(p) vs p, the same information normalised (starts at 1, falls);
  * Karp-Flatt  e(p) vs p, the thesis plot: a *rising* e(p) is the visual proof
                that the shortfall is overhead (imbalance/communication), not the
                serial fraction.

Per-paradigm (the mechanism)
  * OpenMP: static/dynamic/guided speedup on one axis (does dynamic recover the
            imbalance CoV predicted?);
  * MPI:    stacked compute-vs-communication time per rank count, per strategy,
            plus block/cyclic/master-worker speedup;
  * CUDA:   effective GFLOP/s vs theoretical peak, and throughput vs the warp-
            divergence proxy (the same irregularity, in SIMT clothing).

Baseline (level 1, deterministic, logically first)
  * the per-row work profile w_r vs r — the "mountain" that makes lambda and CoV
    visible; and serial cost T(1) vs problem size and vs max_iter.

The p axis is log2 (powers of two would otherwise crowd the high end). Colours
follow a fixed, colourblind-checked categorical palette assigned per entity (not
per rank), with distinct markers as a second channel for print/CVD.
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Palette (validated categorical set, light surface) ---------------------
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
VIOLET, RED, MAGENTA, ORANGE = "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# Colour + marker per entity (paradigm, schedule). Colour follows the entity, so
# a series keeps its identity across every figure it appears in.
STYLE = {
    ("openmp", "static"):        (BLUE, "o"),
    ("openmp", "dynamic"):       (GREEN, "s"),
    ("openmp", "guided"):        (VIOLET, "^"),
    ("mpi", "block"):            (RED, "D"),
    ("mpi", "cyclic"):           (ORANGE, "v"),
    ("mpi", "master_worker"):    (MAGENTA, "P"),
    ("cuda", "-"):               (AQUA, "X"),
    ("serial", "-"):             (MUTED, "."),
}
ENTITY_ORDER = list(STYLE.keys())
FALLBACK_COLORS = [YELLOW, MAGENTA, AQUA, ORANGE, VIOLET]
SEQUENCE_COLORS = [BLUE, ORANGE, GREEN, VIOLET, RED, AQUA]


def main(argv=None):
    args = parse_args(argv)
    setup_style()

    made, skipped = [], []

    def record(name, ok):
        (made if ok else skipped).append(name)

    rows = load_merged_rows(args.merged)
    if rows:
        groups = group_series(rows, "speedup")
        record("speedup", draw_speedup(groups, "Strong-scaling speedup",
                                       "speedup", args.outdir, args.formats))
        record("efficiency", plot_efficiency(rows, args.outdir, args.formats))
        record("karp_flatt", plot_karp_flatt(rows, args.outdir, args.formats))
        record("openmp_scheduling",
               draw_speedup(group_series(rows, "speedup", lambda k: k[0] == "openmp"),
                            "OpenMP scheduling comparison",
                            "openmp_scheduling", args.outdir, args.formats))
        record("mpi_time", plot_mpi_time_decomposition(rows, args.outdir, args.formats))
        record("mpi_strategies",
               draw_speedup(group_series(rows, "speedup", lambda k: k[0] == "mpi"),
                            "MPI decomposition strategies",
                            "mpi_strategies", args.outdir, args.formats))
        record("cuda_throughput",
               plot_cuda_throughput(rows, args.outdir, args.formats, args.gpu_peak_gflops))
        record("cuda_divergence", plot_cuda_divergence(rows, args.outdir, args.formats))
        record("baseline_growth", plot_baseline_growth(rows, args.outdir, args.formats))
    elif args.merged:
        print("warning: no rows in {}".format(args.merged), file=sys.stderr)

    if args.row_stats:
        for path in sorted(glob.glob(args.row_stats)):
            name = "row_profile:" + os.path.basename(path)
            record(name, plot_row_profile(path, args.outdir, args.formats))

    print("figures written to {}: {}".format(args.outdir, ", ".join(made) or "none"),
          file=sys.stderr)
    if skipped:
        print("skipped (no data): {}".format(", ".join(skipped)), file=sys.stderr)
    return 0 if made else 1


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, os.pardir, "data"))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--merged", default=os.path.join(data_dir, "merged.csv"),
                        help="merged dataset from merge_metrics.py (default: %(default)s)")
    parser.add_argument("--row-stats", default=None,
                        help="path or glob of a --row-stats CSV (row,iterations) for "
                             "the per-row load profile")
    parser.add_argument("--outdir", default=os.path.join(data_dir, "plots"),
                        help="output directory for figures (default: %(default)s)")
    parser.add_argument("--format", dest="formats", default="pdf,png",
                        help="comma-separated output formats (default: %(default)s)")
    parser.add_argument("--gpu-peak-gflops", type=float, default=None,
                        help="GPU theoretical peak, drawn as a reference on the "
                             "CUDA throughput plot")
    args = parser.parse_args(argv)
    args.formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    return args


def setup_style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "legend.fontsize": 10,
    })


def load_merged_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def group_series(rows, value_col, keep=None):
    """Return {(paradigm, schedule): [(p, value), ...] sorted} for present values."""
    groups = {}
    for row in rows:
        key = (row.get("paradigm", ""), row.get("schedule", ""))
        if keep is not None and not keep(key):
            continue
        p = to_int(row.get("p"))
        value = to_float(row.get(value_col))
        if p is None or value is None:
            continue
        groups.setdefault(key, []).append((p, value))
    for key in groups:
        groups[key].sort()
    return groups


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def draw_speedup(groups, title, name, outdir, formats):
    """Overlay a speedup curve per series with the dashed ideal S = p reference."""
    if not groups:
        return False
    fig, ax = new_axes()
    all_p = set()
    for key in ordered_keys(groups):
        xs = [p for p, _ in groups[key]]
        ys = [v for _, v in groups[key]]
        all_p.update(xs)
        color, marker = series_style(key)
        ax.plot(xs, ys, marker=marker, color=color, label=series_label(key),
                linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=0.8)

    p_ticks = sorted(all_p)
    ax.plot(p_ticks, p_ticks, linestyle="--", color=MUTED, linewidth=1.5,
            label=r"ideal  $S = p$", zorder=1)
    set_log2_p_axis(ax, p_ticks)
    ax.set_xlabel(r"processes / threads  $p$")
    ax.set_ylabel(r"speedup  $S(p)$")
    ax.set_title(title)
    ax.legend(frameon=False)
    save_figure(fig, outdir, name, formats)
    return True


def new_axes(width=6.4, height=4.2):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def ordered_keys(groups):
    def rank(key):
        return ENTITY_ORDER.index(key) if key in ENTITY_ORDER else len(ENTITY_ORDER)
    return sorted(groups, key=lambda key: (rank(key), key))


def series_style(key):
    if key in STYLE:
        return STYLE[key]
    idx = len(series_style.assigned)
    series_style.assigned.setdefault(key, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
    return (series_style.assigned[key], "o")


series_style.assigned = {}


def series_label(key):
    names = {"openmp": "OpenMP", "mpi": "MPI", "cuda": "CUDA",
             "serial": "Serial", "hybrid": "Hybrid"}
    paradigm, schedule = key
    base = names.get(paradigm, paradigm)
    if schedule and schedule != "-":
        return "{} ({})".format(base, schedule)
    return base


def set_log2_p_axis(ax, p_ticks):
    if len(p_ticks) > 1:
        ax.set_xscale("log", base=2)
    ax.set_xticks(p_ticks)
    ax.set_xticklabels([str(p) for p in p_ticks])
    ax.minorticks_off()


def save_figure(fig, outdir, name, formats):
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(outdir, "{}.{}".format(name, fmt)),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_efficiency(rows, outdir, formats):
    groups = group_series(rows, "efficiency")
    if not groups:
        return False
    fig, ax = new_axes()
    all_p = set()
    for key in ordered_keys(groups):
        xs = [p for p, _ in groups[key]]
        ys = [v for _, v in groups[key]]
        all_p.update(xs)
        color, marker = series_style(key)
        ax.plot(xs, ys, marker=marker, color=color, label=series_label(key),
                linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.axhline(1.0, linestyle="--", color=MUTED, linewidth=1.5, label=r"ideal  $E = 1$")
    set_log2_p_axis(ax, sorted(all_p))
    ax.set_xlabel(r"processes / threads  $p$")
    ax.set_ylabel(r"efficiency  $E(p)$")
    ax.set_title("Parallel efficiency")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "efficiency", formats)
    return True


def plot_karp_flatt(rows, outdir, formats):
    groups = group_series(rows, "karp_flatt_e")
    if not groups:
        return False
    fig, ax = new_axes()
    all_p = set()
    for key in ordered_keys(groups):
        xs = [p for p, _ in groups[key]]
        ys = [v for _, v in groups[key]]
        all_p.update(xs)
        color, marker = series_style(key)
        ax.plot(xs, ys, marker=marker, color=color, label=series_label(key),
                linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=0.8)
    set_log2_p_axis(ax, sorted(all_p))
    ax.set_xlabel(r"processes / threads  $p$")
    ax.set_ylabel(r"Karp-Flatt serial fraction  $e(p)$")
    ax.set_title("Karp-Flatt: rising $e(p)$ = overhead, not serial fraction")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "karp_flatt", formats)
    return True


def plot_mpi_time_decomposition(rows, outdir, formats):
    by_schedule = {}
    for row in rows:
        if row.get("paradigm") != "mpi":
            continue
        p = to_int(row.get("p"))
        total = to_float(row.get("T_min"))
        comm_fraction = to_float(row.get("comm_fraction"))
        if p is None or total is None or comm_fraction is None:
            continue
        by_schedule.setdefault(row.get("schedule", "-"), []).append((p, total, comm_fraction))

    made = False
    for schedule, points in by_schedule.items():
        points.sort()
        ranks = [p for p, _, _ in points]
        compute = [t * (1.0 - cf) for _, t, cf in points]
        comm = [t * cf for _, t, cf in points]
        positions = list(range(len(ranks)))

        fig, ax = new_axes()
        ax.bar(positions, compute, color=BLUE, label="compute",
               edgecolor=SURFACE, linewidth=2)
        ax.bar(positions, comm, bottom=compute, color=ORANGE, label="communication",
               edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(p) for p in ranks])
        ax.set_xlabel(r"MPI ranks  $p$")
        ax.set_ylabel(r"time per run  $T(p)$  [s]")
        ax.set_title("MPI time decomposition - {}".format(schedule))
        ax.legend(frameon=False)
        save_figure(fig, outdir, "mpi_time_{}".format(schedule), formats)
        made = True
    return made


def plot_cuda_throughput(rows, outdir, formats, gpu_peak):
    points = []
    for row in rows:
        if row.get("paradigm") != "cuda":
            continue
        gflops = to_float(row.get("gflops"))
        if gflops is None:
            continue
        points.append((row.get("resolution", "?"), gflops))
    if not points:
        return False

    points.sort()
    positions = list(range(len(points)))
    fig, ax = new_axes()
    ax.bar(positions, [g for _, g in points], color=AQUA, label="effective",
           edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(positions)
    ax.set_xticklabels([res for res, _ in points])
    if gpu_peak:
        ax.axhline(gpu_peak, linestyle="--", color=MUTED, linewidth=1.5,
                   label="theoretical peak ({:g})".format(gpu_peak))
    ax.set_xlabel("resolution")
    ax.set_ylabel("throughput  [GFLOP/s]")
    ax.set_title("CUDA effective throughput vs peak")
    ax.legend(frameon=False)
    save_figure(fig, outdir, "cuda_throughput", formats)
    return True


def plot_cuda_divergence(rows, outdir, formats):
    points = []
    for row in rows:
        if row.get("paradigm") != "cuda":
            continue
        divergence = to_float(row.get("warp_divergence"))
        gflops = to_float(row.get("gflops"))
        if divergence is None or gflops is None:
            continue
        points.append((divergence, gflops))
    if not points:
        return False

    fig, ax = new_axes()
    ax.scatter([d for d, _ in points], [g for _, g in points], color=VIOLET,
               s=64, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlabel("warp-divergence proxy  (mean per-warp variance)")
    ax.set_ylabel("throughput  [GFLOP/s]")
    ax.set_title("CUDA: throughput vs warp divergence")
    save_figure(fig, outdir, "cuda_divergence", formats)
    return True


def plot_baseline_growth(rows, outdir, formats):
    serial = [row for row in rows if row.get("paradigm") == "serial"]
    made = False

    by_max_iter = {}
    for row in serial:
        time = to_float(row.get("T_min"))
        resolution = parse_resolution(row.get("resolution"))
        max_iter = to_int(row.get("max_iter"))
        if time is None or resolution is None or max_iter is None:
            continue
        by_max_iter.setdefault(max_iter, []).append((resolution[0] * resolution[1], time))
    if any(len(v) >= 2 for v in by_max_iter.values()):
        fig, ax = new_axes()
        for i, max_iter in enumerate(sorted(by_max_iter)):
            points = sorted(by_max_iter[max_iter])
            ax.plot([px for px, _ in points], [t for _, t in points], marker="o",
                    color=SEQUENCE_COLORS[i % len(SEQUENCE_COLORS)], linewidth=2,
                    markersize=7, label="max_iter = {}".format(max_iter))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"pixels  ($W \cdot H$)")
        ax.set_ylabel(r"serial time  $T(1)$  [s]")
        ax.set_title("Serial cost vs problem size")
        ax.legend(frameon=False)
        save_figure(fig, outdir, "baseline_vs_pixels", formats)
        made = True

    by_resolution = {}
    for row in serial:
        time = to_float(row.get("T_min"))
        max_iter = to_int(row.get("max_iter"))
        if time is None or max_iter is None:
            continue
        by_resolution.setdefault(row.get("resolution", "?"), []).append((max_iter, time))
    if any(len(v) >= 2 for v in by_resolution.values()):
        fig, ax = new_axes()
        for i, resolution in enumerate(sorted(by_resolution)):
            points = sorted(by_resolution[resolution])
            ax.plot([m for m, _ in points], [t for _, t in points], marker="s",
                    color=SEQUENCE_COLORS[i % len(SEQUENCE_COLORS)], linewidth=2,
                    markersize=7, label=resolution)
        ax.set_xscale("log")
        ax.set_xlabel("max_iter")
        ax.set_ylabel(r"serial time  $T(1)$  [s]")
        ax.set_title("Serial cost vs max_iter")
        ax.legend(frameon=False)
        save_figure(fig, outdir, "baseline_vs_maxiter", formats)
        made = True

    return made


def parse_resolution(text):
    try:
        width, height = text.lower().split("x")
        return int(width), int(height)
    except (AttributeError, ValueError):
        return None


def plot_row_profile(path, outdir, formats):
    data = read_row_profile(path)
    if not data:
        return False

    indices = [r for r, _ in data]
    work = [w for _, w in data]
    n = len(work)
    mean = sum(work) / n
    std = (sum((w - mean) ** 2 for w in work) / n) ** 0.5
    peak = max(work)
    peak_index = max(range(n), key=lambda i: work[i])
    lam = peak / mean if mean else 0.0
    cov = std / mean if mean else 0.0

    fig, ax = new_axes(7.0, 4.2)
    ax.fill_between(indices, work, color=BLUE, alpha=0.22, linewidth=0)
    ax.plot(indices, work, color=BLUE, linewidth=1.2)
    ax.axhline(mean, linestyle="--", color=MUTED, linewidth=1.5,
               label=r"mean  $\bar{{w}}$ = {:.0f}".format(mean))
    ax.plot([indices[peak_index]], [peak], marker="v", color=RED, markersize=10,
            markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3,
            label=r"peak  $\lambda$ = {:.2f}".format(lam))
    ax.set_xlabel(r"row index  $r$")
    ax.set_ylabel(r"per-row work  $w_r$  [iterations]")
    ax.set_title(r"Per-row load profile   (CoV = {:.3f})".format(cov))
    ax.legend(frameon=False)
    save_figure(fig, outdir, "row_profile", formats)
    return True


def read_row_profile(path):
    if not os.path.exists(path):
        return []
    data = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            r = to_int(row.get("row"))
            w = to_float(row.get("iterations"))
            if r is not None and w is not None:
                data.append((r, w))
    data.sort()
    return data


if __name__ == "__main__":
    sys.exit(main())
