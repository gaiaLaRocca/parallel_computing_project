#!/usr/bin/env python3
"""Predict how each CUDA block geometry groups the escape-time matrix into warps.

The GPU never sees the 2D block the kernel declares: it linearises every block as
``id = threadIdx.x + threadIdx.y * blockDim.x`` and executes 32 consecutive ids
at a time as one warp, whose lanes advance in lockstep until the slowest one
escapes. Given the escape-time matrix ``n(r, c)`` of a ``--raw`` dump, that
grouping can be replayed exactly offline - the same traversal as
``warp_stats_blocked()`` in ``mandelbrot/src/metrics.cpp`` - so every number
below is a *prediction available before any kernel runs*: the GPU counterpart
of what ``block_imbalance.py`` does for the CPU row decompositions.

Per geometry ``BXxBY``:

  divergence         mean over warps of the within-warp variance of n. Must
                     match the ``warp_divergence`` column the CUDA binary writes.
  waste_mean         mean over warps of 1 - mean/max, every warp weighing the
                     same. Must match ``warp_wasted`` in the CUDA job log.
  waste_global       1 - W / lane_cycles: the masked share of *all* lane-cycles.
                     It weighs expensive warps more than waste_mean does, so it
                     is the one that converts into work.
  lane_cycles        sum over warps of lanes * max: what the geometry makes the
                     GPU execute, since a warp retires only with its slowest lane.
  block_cost_cov     coefficient of variation of the block costs, a block's cost
                     being the lane_cycles of its warps. A granularity descriptor
                     only: which SM runs which block is not observable without a
                     profiler, so no makespan is modelled. There is deliberately
                     no max/mean column: as soon as one block lies wholly inside
                     the set its cost is max_iter per lane, so max/mean collapses
                     to max_iter / (lane_cycles / N) and says nothing about shape.
  time_rel_pred      lane_cycles relative to the reference geometry (default
                     256x1, the Sweep B shape), under the model "time is
                     proportional to lane-cycles". Sweep A measured much smaller
                     differences than this model predicts; the gap between the
                     two is the analysis, not a bug of the prediction.

The ``scattered`` row is the GPU analogue of the MPI cyclic decomposition:
pixel ``p`` of the row-major matrix joins warp ``p mod (N / 32)``, as in
``warp_divergence_proxy_scattered()``. No block shape realises it, so its block
columns stay empty.

The dump must come from the *bare* kernel (no USE_PRUNING/USE_SYMMETRY): with
pruning, interior pixels store max_iter without iterating and every column above
overstates the work. The W printed on exit lets you check it against merged.csv.

Output: ``data/block_geometry.csv``, one row per (source, geometry).

Pure standard library: no numpy, so it runs anywhere Python 3 is available.
"""

import argparse
import array
import csv
import math
import os
import re
import sys

WARP_SIZE = 32
INT32_BYTES = 4

# The Sweep A shapes, in the order mandel_cuda_blocks.sh launches them.
DEFAULT_GEOMETRIES = ["256x1", "512x1", "128x1", "128x2", "64x4", "32x8",
                      "16x16", "8x32", "1x256"]
REFERENCE_GEOMETRY = "256x1"
SCATTERED = "scattered"

OUTPUT_COLUMNS = ["source", "resolution", "max_iter", "geometry", "block_x",
                  "block_y", "warps", "divergence", "waste_mean",
                  "waste_global", "lane_cycles", "blocks", "block_cost_cov",
                  "time_rel_pred"]

# figure_nmax<N>.raw - the name make_fractal_figures.sh gives the dumps.
NAME_PATTERN = re.compile(r"nmax(?P<nmax>\d+)")


def main(argv=None):
    args = parse_args(argv)
    if args.reference not in args.geometries:
        print("error: reference {} is not among --geometries".format(args.reference),
              file=sys.stderr)
        return 1
    try:
        matrix, width, height = read_raw(args.raw, args.resolution)
    except (OSError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1

    stats = {}
    for geometry in args.geometries:
        block_x, block_y = parse_geometry(geometry)
        stats[geometry] = blocked_stats(matrix, width, height, block_x, block_y)
    if not args.no_scattered:
        stats[SCATTERED] = scattered_stats(matrix)

    records = build_records(args, width, height, stats)
    write_csv(records, args.output)
    print("W = {} over {}x{} pixels".format(sum(matrix), width, height),
          file=sys.stderr)
    print("wrote {} rows to {}".format(len(records), args.output), file=sys.stderr)
    if not args.quiet:
        print_summary(records)
    return 0


def blocked_stats(matrix, width, height, block_x, block_y):
    """Warp statistics of one block shape, replaying the hardware's grouping."""
    return summarise(blocked_warps(matrix, width, height, block_x, block_y))


def scattered_stats(matrix):
    """Warp statistics when pixel p joins warp p mod (N / WARP_SIZE)."""
    return summarise(scattered_warps(matrix))


def blocked_warps(matrix, width, height, block_x, block_y):
    """Yield (block, lane values) for every warp of every block of the grid.

    Lanes of edge blocks that overhang the image are dropped, as the kernel's
    bounds check retires them, so the statistics measure the fractal and not
    the tiling.
    """
    threads_per_block = block_x * block_y
    for by in range((height + block_y - 1) // block_y):
        for bx in range((width + block_x - 1) // block_x):
            for start in range(0, threads_per_block, WARP_SIZE):
                lanes = []
                for thread_id in range(start, min(start + WARP_SIZE, threads_per_block)):
                    col = bx * block_x + thread_id % block_x
                    row = by * block_y + thread_id // block_x
                    if col < width and row < height:
                        lanes.append(matrix[row * width + col])
                if lanes:
                    yield (bx, by), lanes


def scattered_warps(matrix):
    """Yield (None, lane values) with warp w owning pixels w + k * (N / WARP_SIZE).

    Mirrors warp_divergence_proxy_scattered(): a tail shorter than one full warp
    is dropped, and there is no block to charge the cost to.
    """
    stride = len(matrix) // WARP_SIZE
    for warp in range(stride):
        yield None, [matrix[warp + k * stride] for k in range(WARP_SIZE)]


def summarise(warps):
    """Aggregate (block, lanes) pairs into one geometry's prediction."""
    variance_sum = waste_sum = 0.0
    work = lane_cycles = warp_count = 0
    block_costs = {}
    for block, lanes in warps:
        mean, peak, variance = warp_moments(lanes)
        cycles = len(lanes) * peak
        variance_sum += variance
        waste_sum += 1.0 - mean / peak if peak > 0 else 0.0
        work += sum(lanes)
        lane_cycles += cycles
        warp_count += 1
        if block is not None:
            block_costs[block] = block_costs.get(block, 0) + cycles

    return {
        "warps": warp_count,
        "divergence": variance_sum / warp_count,
        "waste_mean": waste_sum / warp_count,
        "waste_global": 1.0 - work / lane_cycles if lane_cycles else 0.0,
        "lane_cycles": lane_cycles,
        "blocks": len(block_costs) or None,
        "block_cost_cov": coefficient_of_variation(list(block_costs.values())),
    }


def warp_moments(lanes):
    """Mean, maximum and population variance of one warp's iteration counts."""
    count = len(lanes)
    mean = sum(lanes) / count
    variance = sum((n - mean) ** 2 for n in lanes) / count
    return mean, max(lanes), variance


def coefficient_of_variation(costs):
    """Population std / mean of the block costs; None when there are no blocks."""
    if not costs:
        return None
    mean = sum(costs) / len(costs)
    deviation = math.sqrt(sum((c - mean) ** 2 for c in costs) / len(costs))
    return deviation / mean


def build_records(args, width, height, stats):
    """One CSV record per geometry, with time normalised to the reference shape."""
    reference_cycles = stats[args.reference]["lane_cycles"]
    max_iter = args.max_iter or parse_max_iter(args.raw)
    records = []
    for geometry, geometry_stats in stats.items():
        block_x, block_y = ("", "") if geometry == SCATTERED else parse_geometry(geometry)
        records.append({
            "source": os.path.basename(args.raw),
            "resolution": "{}x{}".format(width, height),
            "max_iter": max_iter,
            "geometry": geometry,
            "block_x": block_x,
            "block_y": block_y,
            "warps": geometry_stats["warps"],
            "divergence": "{:.4f}".format(geometry_stats["divergence"]),
            "waste_mean": "{:.6f}".format(geometry_stats["waste_mean"]),
            "waste_global": "{:.6f}".format(geometry_stats["waste_global"]),
            "lane_cycles": geometry_stats["lane_cycles"],
            "blocks": format_optional(geometry_stats["blocks"], "{}"),
            "block_cost_cov": format_optional(geometry_stats["block_cost_cov"]),
            "time_rel_pred": "{:.6f}".format(geometry_stats["lane_cycles"] / reference_cycles),
        })
    return records


def format_optional(value, spec="{:.6f}"):
    return "" if value is None else spec.format(value)


def read_raw(path, resolution):
    """Load a --raw dump (int32 little-endian, row-major, no header) as (values, W, H).

    Without --resolution the image is assumed square, as in every sweep.
    """
    matrix = array.array("i")
    if matrix.itemsize != INT32_BYTES:
        raise ValueError("this platform's C int is not 32-bit")
    with open(path, "rb") as handle:
        matrix.frombytes(handle.read())
    if sys.byteorder != "little":
        matrix.byteswap()
    width, height = resolution or square_side(len(matrix))
    if width * height != len(matrix):
        raise ValueError("{} holds {} values, not {}x{}".format(
            path, len(matrix), width, height))
    return matrix, width, height


def square_side(count):
    side = int(round(math.sqrt(count)))
    if side * side != count:
        raise ValueError("{} values do not form a square image: pass --resolution"
                         .format(count))
    return side, side


def parse_max_iter(path):
    """Recover N_max from 'nmax<N>' in the file name, empty if absent."""
    match = NAME_PATTERN.search(os.path.basename(path))
    return match.group("nmax") if match else ""


def write_csv(records, path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records):
    """Compact table of the columns that go in the report."""
    row = "{:<10}{:>12}{:>10}{:>12}{:>12}{:>9}"
    print("\n" + row.format("geometry", "divergence", "waste", "waste_glob",
                            "blk_cov", "t_rel"))
    for record in records:
        print(row.format(record["geometry"], record["divergence"],
                         record["waste_mean"][:7], record["waste_global"][:7],
                         record["block_cost_cov"][:7] or "-",
                         record["time_rel_pred"][:6]))


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, os.pardir, "data"))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", default=os.path.join(data_dir, "figure_nmax1000.raw"),
                        help="bare-kernel --raw dump to analyse (default: %(default)s)")
    parser.add_argument("--resolution", type=parse_geometry, default=None,
                        help="WxH of the dump (default: inferred as a square)")
    parser.add_argument("--max-iter", default="",
                        help="N_max of the dump (default: read from 'nmax<N>' in the name)")
    parser.add_argument("--geometries", type=parse_geometry_list,
                        default=DEFAULT_GEOMETRIES,
                        help="comma-separated BXxBY shapes (default: {})".format(
                            ",".join(DEFAULT_GEOMETRIES)))
    parser.add_argument("--reference", default=REFERENCE_GEOMETRY,
                        help="geometry with time_rel_pred = 1 (default: %(default)s)")
    parser.add_argument("--no-scattered", action="store_true",
                        help="skip the scattered (cyclic-analogue) grouping")
    parser.add_argument("--output", default=os.path.join(data_dir, "block_geometry.csv"),
                        help="prediction table to write (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true",
                        help="write the CSV without printing the summary table")
    return parser.parse_args(argv)


def parse_geometry(text):
    """'BXxBY' -> (BX, BY), both positive integers."""
    match = re.fullmatch(r"(\d+)x(\d+)", text.strip())
    if not match or int(match.group(1)) < 1 or int(match.group(2)) < 1:
        raise argparse.ArgumentTypeError(
            "expected BXxBY with positive integers, got {!r}".format(text))
    return int(match.group(1)), int(match.group(2))


def parse_geometry_list(text):
    """Comma-separated geometries, validated and kept as their 'BXxBY' labels."""
    geometries = [geometry.strip() for geometry in text.split(",") if geometry.strip()]
    for geometry in geometries:
        parse_geometry(geometry)
    return geometries


if __name__ == "__main__":
    sys.exit(main())
