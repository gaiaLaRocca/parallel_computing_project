#!/usr/bin/env python3
"""Merge the per-job result files into the single canonical dataset.

Each benchmark job writes its own ``data/run_<SLURM_JOB_ID>.csv`` (header + one
row) instead of appending to a shared file: concurrent appends interleave or
corrupt rows on the cluster's network filesystem (NFS/Lustre), so per-job files
are lock-free by construction. This script is the *merge* stage that turns them
back into one dataset, in a single pass:

1. Concatenate every ``run_*.csv`` into one table, reconciling heterogeneous or
   partial per-job schemas into the canonical column order (missing columns are
   filled empty -> a sparse table).
2. Compute the *relational* metrics that a single binary cannot, because it only
   ever sees its own ``T(p)`` and never ``T(1)``:
       speedup      S(p) = T(1) / T(p)
       efficiency   E(p) = S(p) / p
       Karp-Flatt   e(p) = (1/S(p) - 1/p) / (1 - 1/p)
   Each parallel run is matched to its serial baseline ``T(1)`` by problem
   configuration (resolution, max_iter, pruning, symmetry) **and by job**. The
   baseline time is ``T_min`` (the least-noise estimator) of the serial run that
   the same job measured.

Why the baseline is per-job, not global. Every sweep script measures its own
``T(1)`` inside the same allocation as its parallel points, precisely so that
baseline and measurement share one node in one state. Picking instead the
fastest ``T(1)`` across all jobs -- the rule this script used originally --
makes previously published numbers change retroactively whenever a new job is
added: the OpenMP sweep's speedups shifted by 0.75% the day the MPI job landed,
without a single OpenMP run being repeated, because the MPI job happened to
measure a slightly faster serial baseline. Same-job matching keeps each sweep's
numbers stable for good, and is the more accurate comparison anyway. Pass
``--baseline-scope global`` to reproduce the old behaviour.

The job id is taken from the file name (``run_<jobid>.csv``, or a per-run shard
``run_<jobid>_<label>.csv``), since the canonical schema carries no job column.
A run whose job measured no matching baseline falls back to the global minimum
and is reported as such -- it is a degraded path, not a normal one.

Baseline / same-hardware caveat: the canonical schema carries no hardware
identity column, so this script matches only on the problem configuration and
the job, and *assumes* the serial baseline and the parallel points were run on
the same partition (guaranteed by the benchmark sweep discipline, and by
same-job matching in the usual case). For CUDA, ``S = T(1)/T_gpu`` is a
single-core-vs-single-GPU ratio by construction, since the baseline is the
serial run.

Pure standard library: no pandas, so it runs anywhere Python 3 is available.
"""

import argparse
import csv
import glob
import os
import re
import sys

# Canonical column order of the merged dataset. Kept in sync with the header
# written by write_run_csv() in mandelbrot/src/metrics.cpp.
CANONICAL_COLUMNS = [
    "paradigm", "schedule", "p", "nodes", "resolution", "max_iter",
    "pruning", "symmetry",
    "T_min", "T_median", "T_mean", "W", "lambda_row", "lambda_block", "cov",
    "speedup", "efficiency", "karp_flatt_e", "comm_fraction", "gflops",
    "occupancy", "warp_divergence", "transfer_time", "checksum",
]

# Columns that identify one problem configuration. Together with the job id this
# is the join key between a parallel run and its serial T(1) baseline.
CONFIG_KEY = ("resolution", "max_iter", "pruning", "symmetry")

# Per-row provenance, carried in memory only: the leading "_" keeps it out of the
# canonical schema, and write_merged() drops it.
JOB_COLUMN = "_job"

# "run_34137.csv" -> "34137"; a per-run shard "run_34137_cyclic_p8.csv" maps to
# the same job, which is what lets a shard-level merge behave like a job-level one.
JOB_FROM_FILENAME = re.compile(r"^run_([^_.]+)")


def main(argv=None):
    args = parse_args(argv)

    rows, file_count = read_rows(args.data_dir, args.pattern, args.output)
    if not rows:
        print("error: no rows found in {}/{}".format(args.data_dir, args.pattern),
              file=sys.stderr)
        return 1

    baselines, fallbacks = build_baselines(rows, args.baseline_paradigm)
    filled, borrowed = fill_relational(rows, baselines, fallbacks,
                                       args.baseline_scope)
    write_merged(rows, args.output)

    print(
        "merged {rows} row(s) from {files} file(s) -> {out}\n"
        "  {bases} baseline(s) ({bp}, p=1), matched per {scope}; "
        "{filled} row(s) with speedup/efficiency/Karp-Flatt".format(
            rows=len(rows), files=file_count, out=args.output,
            bases=len(baselines), bp=args.baseline_paradigm,
            scope=args.baseline_scope, filled=filled,
        ),
        file=sys.stderr,
    )
    if not baselines:
        print("warning: no {} p=1 baseline found; relational metrics left empty"
              .format(args.baseline_paradigm), file=sys.stderr)
    for job, config in sorted(borrowed):
        print("warning: job {job} measured no {bp} baseline for {config}; "
              "borrowed the global minimum, so its speedups are cross-job"
              .format(job=job or "<unknown>", bp=args.baseline_paradigm,
                      config="/".join(config)),
              file=sys.stderr)
    return 0


def parse_args(argv):
    default_data_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
    )
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=default_data_dir,
                        help="directory holding the per-job CSV files "
                             "(default: %(default)s)")
    parser.add_argument("--pattern", default="run_*.csv",
                        help="glob for the per-job files (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="merged dataset path (default: <data-dir>/merged.csv)")
    parser.add_argument("--baseline-paradigm", default="serial",
                        help="paradigm whose p=1 run provides T(1) "
                             "(default: %(default)s)")
    parser.add_argument("--baseline-scope", default="job", choices=("job", "global"),
                        help="'job' matches each run to the T(1) its own job "
                             "measured, so adding a new job never changes older "
                             "numbers; 'global' takes the fastest T(1) across all "
                             "jobs, the pre-2026-09 behaviour (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = os.path.join(args.data_dir, "merged.csv")
    return args


def read_rows(data_dir, pattern, output_path):
    """Read every matching per-job file into a list of canonical-shaped dicts."""
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    output_abs = os.path.abspath(output_path)

    rows = []
    unexpected = set()
    for path in paths:
        # Never fold a previous merge output back into the inputs.
        if os.path.abspath(path) == output_abs:
            continue
        job = job_from_path(path)
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for raw in reader:
                unexpected.update(k for k in raw if k not in CANONICAL_COLUMNS)
                # Reconcile to the canonical shape: keep known columns, default
                # the rest to empty so partial per-job schemas line up.
                row = {col: (raw.get(col) or "") for col in CANONICAL_COLUMNS}
                row[JOB_COLUMN] = job
                rows.append(row)

    if unexpected:
        print("warning: ignoring unexpected column(s): "
              + ", ".join(sorted(unexpected)), file=sys.stderr)
    return rows, len(paths)


def job_from_path(path):
    """Job id encoded in a per-job file name, or None if it does not carry one."""
    match = JOB_FROM_FILENAME.match(os.path.basename(path))
    return match.group(1) if match else None


def build_baselines(rows, baseline_paradigm):
    """Collect the serial T(1) times, per (job, config) and per config.

    Returns ``(baselines, fallbacks)``: ``baselines`` maps ``(job, config)`` to
    that job's own T(1), ``fallbacks`` maps ``config`` to the fastest T(1) seen
    in any job. A job that repeats the same config keeps the fastest of its own
    runs, T_min being the least-noise estimator.
    """
    baselines = {}
    fallbacks = {}
    for row in rows:
        if row["paradigm"] != baseline_paradigm or to_int(row["p"]) != 1:
            continue
        t1 = to_float(row["T_min"])
        if t1 is None or t1 <= 0.0:
            continue
        config = config_key(row)
        key = (row.get(JOB_COLUMN), config)
        if key not in baselines or t1 < baselines[key]:
            baselines[key] = t1
        if config not in fallbacks or t1 < fallbacks[config]:
            fallbacks[config] = t1
    return baselines, fallbacks


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


def config_key(row):
    return tuple(row[col] for col in CONFIG_KEY)


def fill_relational(rows, baselines, fallbacks, scope):
    """Fill speedup / efficiency / Karp-Flatt in place.

    Returns ``(filled, borrowed)``: the number of rows given relational metrics,
    and the set of ``(job, config)`` pairs that had to borrow another job's
    baseline because their own job measured none.
    """
    filled = 0
    borrowed = set()
    for row in rows:
        config = config_key(row)
        job = row.get(JOB_COLUMN)
        if scope == "job":
            t1 = baselines.get((job, config))
            if t1 is None:
                t1 = fallbacks.get(config)
                if t1 is not None:
                    borrowed.add((job, config))
        else:
            t1 = fallbacks.get(config)
        tp = to_float(row["T_min"])
        p = to_int(row["p"])
        if t1 is None or tp is None or tp <= 0.0 or p is None or p < 1:
            continue

        speedup = t1 / tp
        row["speedup"] = fmt(speedup)
        row["efficiency"] = fmt(speedup / p)
        # Karp-Flatt is undefined at p = 1 (division by 1 - 1/p = 0).
        if p > 1 and speedup > 0.0:
            row["karp_flatt_e"] = fmt((1.0 / speedup - 1.0 / p) / (1.0 - 1.0 / p))
        filled += 1
    return filled, borrowed


def fmt(value):
    return "{:.10g}".format(value)


def write_merged(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        # extrasaction="ignore" drops JOB_COLUMN: provenance is a merge-time
        # concern, and the canonical schema must stay exactly as documented.
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow(row)


def sort_key(row):
    max_iter = to_int(row["max_iter"])
    p = to_int(row["p"])
    return (
        row["paradigm"],
        row["resolution"],
        max_iter if max_iter is not None else -1,
        p if p is not None else -1,
        row["schedule"],
    )


if __name__ == "__main__":
    sys.exit(main())
