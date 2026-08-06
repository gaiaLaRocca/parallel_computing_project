#!/usr/bin/env python3
"""Predict the decomposition imbalance lambda(P) from the serial row profile.

The serial binary leaves ``lambda_block`` empty (``main.cpp``: it is only
meaningful for ``--p >= 2``, and the Level 1 baseline runs at ``--p 1``), so the
block-level bound has to be derived offline. It can be: the ``--row-stats`` dump
``rowprofile_*.csv`` carries the whole per-row work vector ``w_r``, which is all
the makespan of any *static* row decomposition depends on. Deriving it here
rather than re-running the binary with a fake ``--p`` also keeps the merged
dataset clean, since a serial run tagged ``p=8`` would be mistaken for a
parallel point and wrecked the ``T(1)`` matching.

Because ``w_r`` is deterministic in ``(resolution, max_iter)``, these numbers are
*predictions made before any parallel run exists* - which is what makes them
worth plotting against the measured MPI speedups. Agreement validates the model;
a measured curve *below* the prediction is the signature of a cost the model
omits (communication, scheduling overhead), and that gap is the analysis.

Three schemes, one definition. With work measured in iterations, a schedule's
makespan ``M(P)`` (in the same units) gives

    lambda(P) = P * M(P) / W        and        S(P) <= P / lambda(P) = W / M(P)

so lambda is the factor by which the heaviest worker exceeds the fair share
``W/P``, and ``lambda = 1`` is perfect balance.

  block    rows [b*H/P, (b+1)*H/P) - contiguous, the decomposition the C++
           ``lambda_block()`` assumes; ``M = max_b W_b``. Worst case for the
           Mandelbrot set, whose expensive interior rows are *contiguous*: one
           block inherits the whole mountain.
  cyclic   row r -> rank r mod P. Interleaving breaks up the mountain, so
           neighbouring ranks get statistically similar mixtures of cheap and
           expensive rows; ``M = max_b W_b`` over the strided groups.
  dynamic  perfect master-worker with rows as tasks. No static assignment, but
           two floors survive: no worker beats the fair share ``W/P``, and the
           makespan is at least the single longest row (whoever draws it works
           until the end). Hence ``M = max(W/P, max_r w_r)`` and
           ``S <= min(P, W / max_r w_r) = min(P, H / lambda_row)``. This is an
           idealisation - it charges nothing for the master's messages - so it
           is an upper bound the measured master-worker curve cannot exceed.

Output: ``data/block_imbalance.csv`` with one row per
``(source, scheme, p)``, consumed by ``plot_metrics.py --block-imbalance``.

Pure standard library: no pandas, so it runs anywhere Python 3 is available.
"""

import argparse
import csv
import glob
import os
import re
import sys

# Rank counts to evaluate. Powers of two up to the 64 cores of a single gnode01
# (no reachable partition offers a second node), plus the odd master-worker
# sizes np = W + 1 the MPI sweep uses, so the prediction lands on exactly the
# points the measurements will occupy.
DEFAULT_P_VALUES = [2, 3, 4, 5, 8, 9, 16, 17, 32, 64]

OUTPUT_COLUMNS = ["source", "resolution", "max_iter", "scheme", "p",
                  "lambda", "speedup_bound"]

# rowprofile_<jobid>_res<R>_nmax<N>.csv - the name the job scripts build.
NAME_PATTERN = re.compile(r"res(?P<res>\d+)(?:x\d+)?_nmax(?P<nmax>\d+)")


def main(argv=None):
    args = parse_args(argv)

    paths = sorted(glob.glob(args.row_stats))
    if not paths:
        print("error: no row profile matches {}".format(args.row_stats),
              file=sys.stderr)
        return 1

    records = []
    for path in paths:
        work = read_row_profile(path)
        if not work:
            print("warning: no usable rows in {}".format(path), file=sys.stderr)
            continue
        records.extend(analyse(path, work, args.p_values))

    if not records:
        print("error: no row profile yielded any prediction", file=sys.stderr)
        return 1

    write_csv(records, args.output)
    print("wrote {} rows to {}".format(len(records), args.output), file=sys.stderr)
    if not args.quiet:
        print_summary(records)
    return 0


def analyse(path, work, p_values):
    """Build the lambda(P) records for one row profile."""
    total = sum(work)
    height = len(work)
    heaviest_row = max(work)
    resolution, max_iter = parse_name(path)
    source = os.path.basename(path)

    records = []
    for p in p_values:
        # More ranks than rows leaves ranks idle: the decomposition is degenerate
        # and the bound meaningless, so skip rather than report a misleading 1.0.
        if p < 1 or p > height:
            continue
        for scheme, makespan in (
            ("block", block_makespan(work, p)),
            ("cyclic", cyclic_makespan(work, p)),
            ("dynamic", max(total / p, heaviest_row)),
        ):
            lam = p * makespan / total
            records.append({
                "source": source,
                "resolution": resolution,
                "max_iter": max_iter,
                "scheme": scheme,
                "p": p,
                "lambda": "{:.6f}".format(lam),
                "speedup_bound": "{:.6f}".format(p / lam),
            })
    return records


def block_makespan(work, p):
    """Heaviest contiguous block under the balanced split rows [b*H/P, (b+1)*H/P).

    Mirrors lambda_block() in mandelbrot/src/metrics.cpp exactly - same integer
    arithmetic, so a run made with --p >= 2 must reproduce these numbers.
    """
    height = len(work)
    heaviest = 0
    for b in range(p):
        begin = b * height // p
        end = (b + 1) * height // p
        heaviest = max(heaviest, sum(work[begin:end]))
    return heaviest


def cyclic_makespan(work, p):
    """Heaviest rank under round-robin assignment: row r goes to rank r mod P."""
    loads = [0] * p
    for r, w in enumerate(work):
        loads[r % p] += w
    return max(loads)


def read_row_profile(path):
    """Read a --row-stats dump (row,iterations) into w_r ordered by row index."""
    rows = []
    with open(path, newline="") as handle:
        for record in csv.DictReader(handle):
            try:
                rows.append((int(record["row"]), float(record["iterations"])))
            except (KeyError, TypeError, ValueError):
                continue
    rows.sort()
    return [w for _, w in rows]


def parse_name(path):
    """Recover (resolution, max_iter) from the file name, empty if unreadable."""
    match = NAME_PATTERN.search(os.path.basename(path))
    if not match:
        return "", ""
    side = match.group("res")
    return "{}x{}".format(side, side), match.group("nmax")


def write_csv(records, path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def print_summary(records):
    """Compact lambda table per configuration - the number that goes in the report."""
    configs = {}
    for record in records:
        key = (record["resolution"], record["max_iter"])
        configs.setdefault(key, {})[(record["scheme"], record["p"])] = record["lambda"]

    for (resolution, max_iter), cell in sorted(configs.items()):
        ps = sorted({p for _, p in cell})
        print("\n{}, max_iter={}".format(resolution or "?", max_iter or "?"))
        print("  {:<9}{}".format("P", "".join("{:>9}".format(p) for p in ps)))
        for scheme in ("block", "cyclic", "dynamic"):
            values = ["{:>9}".format(cell.get((scheme, p), "-")[:7]) for p in ps]
            print("  {:<9}{}".format(scheme, "".join(values)))


def parse_args(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, os.pardir, "data"))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row-stats",
                        default=os.path.join(data_dir, "rowprofile_*.csv"),
                        help="glob of --row-stats dumps to analyse (default: %(default)s)")
    parser.add_argument("--output", default=os.path.join(data_dir, "block_imbalance.csv"),
                        help="prediction table to write (default: %(default)s)")
    parser.add_argument("--p", dest="p_values", default=None,
                        help="comma-separated rank counts (default: {})".format(
                            ",".join(str(p) for p in DEFAULT_P_VALUES)))
    parser.add_argument("--quiet", action="store_true",
                        help="write the CSV without printing the summary table")
    args = parser.parse_args(argv)
    if args.p_values:
        args.p_values = sorted({int(v) for v in args.p_values.split(",") if v.strip()})
    else:
        args.p_values = DEFAULT_P_VALUES
    return args


if __name__ == "__main__":
    sys.exit(main())
