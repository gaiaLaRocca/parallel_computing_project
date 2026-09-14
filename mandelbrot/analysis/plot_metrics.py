#!/usr/bin/env python3
"""Render the report's figures from the per-job results.

Each figure answers one question the report asks, so the set is deliberately
small:

  row_profile                per-row work w_r of the serial baseline: the
                             "mountain" of contiguous expensive rows behind every
                             imbalance result.
  cpu_predicted_vs_measured  measured OpenMP and MPI speedup against the bound
                             p / lambda_p predicted from the serial profile, one
                             colour per assignment rule: the model is judged by
                             rule, not by paradigm.
  openmp_schedules           speedup of the five OpenMP policies: the fivefold
                             spread at 32 threads and the fixed ceiling of
                             dynamic,64.
  cuda_saturation            effective GPU throughput against image size
                             (Sweep B1): the card saturates from about four
                             million pixels.
  cuda_copy_bandwidth        effective device->host bandwidth against copy size,
                             with the least-squares fit (fixed cost per copy plus
                             asymptotic bandwidth) and the nominal PCIe rate.

The CPU figures read merged.csv at the reference problem (1024x1024, max_iter
1000). The CUDA figures read the per-job file of the problem-scaling sweep
directly: merged.csv carries no job id, and Sweep A also measured 256x1 at
1024x1024, so the scaling points cannot be told apart there.

Colours come from a validated categorical palette in fixed slot order and follow
the entity: an assignment rule keeps its hue in every figure it appears in.
Labels stay in English like the rest of the repository.

Needs matplotlib; everything else is the standard library.
"""

import argparse
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

REFERENCE_RESOLUTION = "1024x1024"
REFERENCE_MAX_ITER = "1000"
PCIE4_X16_GBPS = 31.5  # nominal per direction: 16 GT/s x 16 lanes, 128b/130b
SCALING_MAX_ITER = "1000"  # Sweep B1 varies the resolution at this N_max

# Validated categorical palette (dataviz reference), slots in fixed order.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

RULE_COLOR = {"block": SLOT[0], "cyclic": SLOT[1], "dynamic": SLOT[2]}
RULE_REALISATIONS = {
    "block": [("openmp", "static"), ("mpi", "block")],
    "cyclic": [("mpi", "cyclic")],
    "dynamic": [("openmp", "dynamic_1"), ("mpi", "dynamic")],
}
# (schedule in merged.csv, label, colour, marker). static and dynamic,1 share the
# colour of the rule they realise, so the two CPU figures read as one story.
OPENMP_POLICIES = [
    ("static", "static", RULE_COLOR["block"], "o"),
    ("dynamic_1", "dynamic,1", RULE_COLOR["dynamic"], "s"),
    ("dynamic_16", "dynamic,16", SLOT[3], "^"),
    ("dynamic_64", "dynamic,64", SLOT[4], "D"),
    ("guided", "guided", SLOT[5], "v"),
]
GPU_COLOR = SLOT[6]

RUN_FILE = re.compile(r"run_(\d+)\.csv$")


def main(argv=None):
    args = parse_args(argv)
    apply_style()

    reference_rows = [row for row in read_rows(args.merged) if is_reference_problem(row)]
    scaling_run = args.scaling_run or find_scaling_run(args.data_dir)
    scaling_rows = [row for row in read_rows(scaling_run) if row.get("paradigm") == "cuda"]

    figures = {
        "row_profile": lambda: plot_row_profile(read_row_profile(args.row_profile)),
        "cpu_predicted_vs_measured": lambda: plot_predicted_vs_measured(
            read_bounds(args.block_imbalance), reference_rows),
        "openmp_schedules": lambda: plot_openmp_schedules(reference_rows),
        "cuda_saturation": lambda: plot_cuda_saturation(scaling_rows),
        "cuda_copy_bandwidth": lambda: plot_copy_bandwidth(scaling_rows),
    }
    written = []
    for name, draw in figures.items():
        fig = draw()
        if fig is None:
            print("skipped {}: no data".format(name), file=sys.stderr)
            continue
        save_figure(fig, args.outdir, name, args.formats)
        written.append(name)

    print("scaling run: {}".format(scaling_run or "none found"), file=sys.stderr)
    print("wrote {} to {}".format(", ".join(written) or "nothing", args.outdir),
          file=sys.stderr)
    return 0 if written else 1


def plot_row_profile(profile):
    """Per-row work, with the mean and the peak row labelled directly."""
    if not profile:
        return None
    rows = [row for row, _ in profile]
    work = [iterations for _, iterations in profile]
    mean = sum(work) / len(work)
    peak_row, peak = max(profile, key=lambda item: item[1])

    fig, ax = new_axes()
    ax.fill_between(rows, work, color=SLOT[0], alpha=0.10, linewidth=0)
    ax.plot(rows, work, color=SLOT[0], linewidth=2)
    ax.axhline(mean, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(rows[-1], mean, "mean  {:,.0f}".format(mean), color=INK_SECONDARY,
            ha="right", va="bottom", fontsize=9)
    draw_point(ax, peak_row, peak, SLOT[0])
    ax.annotate("peak row {}: {:.2f}$\\times$ mean".format(peak_row, peak / mean),
                (peak_row, peak), xytext=(10, 0), textcoords="offset points",
                color=INK_SECONDARY, va="center", fontsize=9)

    ax.set_xlim(0, rows[-1])
    ax.set_ylim(0, peak * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "{:,.0f}".format(value)))
    ax.set_xlabel("row index  r")
    ax.set_ylabel("iterations per row  $w_r$")
    return fig


def plot_predicted_vs_measured(bounds, rows):
    """One panel per assignment rule: predicted bound as a line, measurements as markers.

    Small multiples rather than one plot: the cyclic and dynamic bounds coincide
    within 1% and would hide each other on a shared axis.
    """
    if not bounds:
        return None
    all_p = sorted({p for points in bounds.values() for p, _ in points})
    fig, axes = plt.subplots(1, len(RULE_COLOR), figsize=(6.8, 2.9), sharey=True)
    for ax, (rule, color) in zip(axes, RULE_COLOR.items()):
        style_axes(ax)
        draw_ideal(ax, all_p)
        predicted = sorted(bounds.get(rule, []))
        ax.plot([p for p, _ in predicted], [s for _, s in predicted],
                color=color, linewidth=2, zorder=2)
        for paradigm, schedule in RULE_REALISATIONS[rule]:
            draw_markers(ax, series(rows, paradigm, schedule, "speedup"), color, paradigm)
        set_log2_axes(ax, all_p)
        ax.set_title(rule, color=INK_SECONDARY, fontsize=10)
        ax.set_xlabel("processing units  p")
    axes[0].set_ylabel("speedup  S(p)")
    fig.legend(handles=encoding_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def plot_openmp_schedules(rows):
    """Speedup of every OpenMP policy against the ideal S = p."""
    fig, ax = new_axes()
    all_p = set()
    for schedule, label, color, marker in OPENMP_POLICIES:
        points = series(rows, "openmp", schedule, "speedup")
        if not points:
            continue
        all_p.update(p for p, _ in points)
        ax.plot([p for p, _ in points], [s for _, s in points], color=color,
                linewidth=2, marker=marker, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label, zorder=3)
    if not all_p:
        plt.close(fig)
        return None

    draw_ideal(ax, sorted(all_p))
    set_log2_axes(ax, sorted(all_p))
    ax.set_xlabel("threads  p")
    ax.set_ylabel("speedup  S(p)")
    ax.legend(title="OpenMP schedule", loc="upper left")
    return fig


def plot_cuda_saturation(rows):
    """Effective GPU throughput against image size, endpoints labelled."""
    points = sorted((pixels(row), float(row["gflops"])) for row in rows
                    if row.get("max_iter") == SCALING_MAX_ITER)
    if len({size for size, _ in points}) < 2:
        return None

    fig, ax = new_axes()
    sizes = [size for size, _ in points]
    throughput = [value for _, value in points]
    ax.plot(sizes, throughput, color=GPU_COLOR, linewidth=2, zorder=2)
    for size, value in points:
        draw_point(ax, size, value, GPU_COLOR)
    # The first point sits where the line rises, so its label goes below it.
    for (size, value), offset, va in ((points[0], -12, "top"), (points[-1], 10, "bottom")):
        ax.annotate("{:.0f} GFLOP/s".format(value), (size, value), xytext=(0, offset),
                    textcoords="offset points", ha="center", va=va,
                    color=INK_SECONDARY, fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(["{0}$\\times${0}".format(int(round(size ** 0.5))) for size in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] / 1.6, sizes[-1] * 1.6)
    ax.set_ylim(0, max(throughput) * 1.18)
    ax.set_xlabel("image size (pixels, log scale)")
    ax.set_ylabel("effective throughput  [GFLOP/s]")
    return fig


def plot_copy_bandwidth(rows):
    """Measured copy bandwidth, the fitted model and the nominal PCIe rate."""
    copies = sorted((pixels(row) * 4, float(row["transfer_time"])) for row in rows
                    if row.get("transfer_time"))
    if len({size for size, _ in copies}) < 2:
        return None
    fixed_seconds, seconds_per_byte = fit_line(copies)

    fig, ax = new_axes()
    mib = 1024 ** 2
    smallest, largest = copies[0][0], copies[-1][0]
    curve = [smallest / 1.5 * (largest * 2.25 / smallest) ** (i / 99) for i in range(100)]
    ax.plot([size / mib for size in curve],
            [size / (fixed_seconds + seconds_per_byte * size) / 1e9 for size in curve],
            color=GPU_COLOR, linewidth=2, zorder=2)
    for size, seconds in copies:
        draw_point(ax, size / mib, size / seconds / 1e9, GPU_COLOR)
    ax.axhline(PCIE4_X16_GBPS, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))

    ax.set_xscale("log", base=2)
    sizes_mib = sorted({size / mib for size, _ in copies})
    ax.set_xticks(sizes_mib)
    ax.set_xticklabels(["{:g}".format(size) for size in sizes_mib])
    ax.minorticks_off()
    ax.set_xlim(curve[0] / mib, curve[-1] / mib)
    ax.set_ylim(0, PCIE4_X16_GBPS * 1.15)
    ax.text(curve[0] / mib * 1.1, PCIE4_X16_GBPS, "nominal PCIe 4.0 x16",
            color=INK_SECONDARY, va="bottom", fontsize=9)
    ax.text(curve[-1] / mib / 1.1, 1 / seconds_per_byte / 1e9 * 0.62,
            "fit: {:.1f} GB/s asymptote,\n{:.2f} ms fixed cost per copy".format(
                1 / seconds_per_byte / 1e9, fixed_seconds * 1e3),
            color=INK_SECONDARY, ha="right", va="top", fontsize=9)
    ax.set_xlabel("copy size  [MiB, log scale]")
    ax.set_ylabel("effective bandwidth  [GB/s]")
    return fig


def draw_ideal(ax, all_p):
    ax.plot(all_p, all_p, color=AXIS, linewidth=1, zorder=1, label="ideal  S = p")


def draw_markers(ax, points, color, paradigm):
    """OpenMP as an open ring, MPI as a small filled square that stays visible inside it.

    The two paradigms agree within about 1% on most points, so the marks must be
    readable while sitting on top of each other.
    """
    if not points:
        return
    xs = [p for p, _ in points]
    ys = [s for _, s in points]
    if paradigm == "mpi":
        ax.plot(xs, ys, linestyle="none", marker="s", markersize=5, color=color, zorder=4)
    else:
        ax.plot(xs, ys, linestyle="none", marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=1.8, zorder=3)


def encoding_handles():
    return [
        Line2D([], [], color=AXIS, linewidth=1, label="ideal  S = p"),
        Line2D([], [], color=INK_SECONDARY, linewidth=2, label="predicted  p / $\\lambda_p$"),
        Line2D([], [], linestyle="none", marker="o", markersize=11, markerfacecolor="none",
               markeredgecolor=INK_SECONDARY, markeredgewidth=1.8, label="OpenMP measured"),
        Line2D([], [], linestyle="none", marker="s", markersize=5,
               color=INK_SECONDARY, label="MPI measured"),
    ]


def draw_point(ax, x, y, color):
    ax.plot([x], [y], marker="o", markersize=8, color=color, markeredgecolor=SURFACE,
            markeredgewidth=2, linestyle="none", zorder=3)


def set_log2_axes(ax, ticks):
    """Log2 scale on both axes, one plain tick per value: S = p is then a diagonal."""
    for axis, set_scale in ((ax.xaxis, ax.set_xscale), (ax.yaxis, ax.set_yscale)):
        set_scale("log", base=2)
        axis.set_ticks(ticks)
        axis.set_ticklabels([str(tick) for tick in ticks])
        axis.set_minor_locator(NullLocator())


def new_axes(width=6.4, height=3.6):
    fig, ax = plt.subplots(figsize=(width, height))
    style_axes(ax)
    return fig, ax


def style_axes(ax):
    """Recessive chrome: no top/right spines, hairline axes and horizontal grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(True, axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def apply_style():
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 10,
        "axes.labelcolor": INK_SECONDARY, "text.color": INK,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False, "legend.fontsize": 9, "legend.title_fontsize": 9,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def save_figure(fig, outdir, name, formats):
    os.makedirs(outdir, exist_ok=True)
    for fmt in formats:
        fig.savefig(os.path.join(outdir, "{}.{}".format(name, fmt)),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)


def series(rows, paradigm, schedule, column):
    """[(p, value)] sorted by p for one (paradigm, schedule) series."""
    points = []
    for row in rows:
        if row.get("paradigm") != paradigm or row.get("schedule") != schedule:
            continue
        try:
            points.append((int(row["p"]), float(row[column])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(points)


def fit_line(points):
    """Least-squares (intercept, slope) of y on x."""
    count = len(points)
    mean_x = sum(x for x, _ in points) / count
    mean_y = sum(y for _, y in points) / count
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in points)
             / sum((x - mean_x) ** 2 for x, _ in points))
    return mean_y - slope * mean_x, slope


def pixels(row):
    width, height = row["resolution"].lower().split("x")
    return int(width) * int(height)


def is_reference_problem(row):
    return (row.get("resolution") == REFERENCE_RESOLUTION
            and row.get("max_iter") == REFERENCE_MAX_ITER)


def find_scaling_run(data_dir):
    """Latest per-job file whose CUDA rows span more than one resolution."""
    candidates = []
    for path in glob.glob(os.path.join(data_dir, "run_*.csv")):
        match = RUN_FILE.search(os.path.basename(path))
        if not match:
            continue
        resolutions = {row.get("resolution") for row in read_rows(path)
                       if row.get("paradigm") == "cuda"}
        if len(resolutions) > 1:
            candidates.append((int(match.group(1)), path))
    return max(candidates)[1] if candidates else None


def read_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_row_profile(path):
    """[(row, iterations)] from a --row-stats dump, sorted by row."""
    profile = []
    for record in read_rows(path):
        try:
            profile.append((int(record["row"]), float(record["iterations"])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(profile)


def read_bounds(path):
    """{rule: [(p, speedup_bound)]} at the reference problem from block_imbalance.csv."""
    bounds = {}
    for record in read_rows(path):
        if not is_reference_problem(record) or record.get("scheme") not in RULE_COLOR:
            continue
        bounds.setdefault(record["scheme"], []).append(
            (int(record["p"]), float(record["speedup_bound"])))
    return bounds


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, os.pardir, "data"))
    profiles = sorted(glob.glob(os.path.join(
        data_dir, "rowprofile_*_res{}_nmax{}.csv".format(
            REFERENCE_RESOLUTION.split("x")[0], REFERENCE_MAX_ITER))))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=data_dir,
                        help="directory holding the per-job run files (default: %(default)s)")
    parser.add_argument("--merged", default=os.path.join(data_dir, "merged.csv"),
                        help="merged dataset from merge_metrics.py (default: %(default)s)")
    parser.add_argument("--block-imbalance", default=os.path.join(data_dir, "block_imbalance.csv"),
                        help="bounds from block_imbalance.py (default: %(default)s)")
    parser.add_argument("--row-profile", default=profiles[-1] if profiles else None,
                        help="--row-stats dump of the reference problem (default: %(default)s)")
    parser.add_argument("--scaling-run", default=None,
                        help="per-job file of the CUDA scaling sweep "
                             "(default: the latest run spanning several resolutions)")
    parser.add_argument("--outdir", default=os.path.join(data_dir, "plots"),
                        help="output directory for figures (default: %(default)s)")
    parser.add_argument("--format", dest="formats", default="pdf,png",
                        help="comma-separated output formats (default: %(default)s)")
    args = parser.parse_args(argv)
    args.formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    return args


if __name__ == "__main__":
    sys.exit(main())
