#!/bin/bash
# OpenMP strong-scaling sweep on a single node.
#
# Measures the serial baseline T(1) and the OpenMP T(p) for every combination of
# schedule policy and thread count, all on the SAME node so the speedups are
# comparable. Each run writes its own data/run_<jobid>_<...>.csv; the Python
# merge (analysis/merge_metrics.py) concatenates them and fills the relational
# columns (speedup/efficiency/karp_flatt_e) by matching each T(p) to this T(1).
#
# Submit from the experiment root:  sbatch mandelbrot/job_sbatch/mandel_omp.sh

#SBATCH --account=g.larocca-thesis             # billing account - REQUIRED, fill in
#SBATCH --job-name=mandel_omp
#SBATCH --partition=<cpu_partition>      # verify a CPU partition with `sinfo`
#SBATCH --nodes=1                        # OpenMP is shared-memory: one node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16               # max threads in the sweep
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=mandelbrot/job_logs/out_%x_%j.log  # relative to $SLURM_SUBMIT_DIR

set -euo pipefail

# --- toolchain -------------------------------------------------------------
# gcc 8.5.0 is the node's system compiler (RHEL 8 default) and survives
# `module purge`; both targets build with it (-fopenmp is a system-gcc flag), so
# no module is needed (there is no loadable amd/gcc-8.5.0 - it is a path prefix).
module purge
g++ --version | head -1        # log the compiler (expected: 8.5.0)

# --- build in-job ----------------------------------------------------------
# Compile where we measure: -march=native tunes for this CPU, so building on a
# different node than the one that runs would risk illegal instructions or,
# worse, silently incomparable timings.
cd "$SLURM_SUBMIT_DIR/mandelbrot/src"
make mandelbrot mandelbrot_omp

# --- experiment parameters -------------------------------------------------
# Bare kernel (no USE_PRUNING/USE_SYMMETRY): this is the scaling/imbalance
# reference, where W and w_r reflect the real work.
DATA="$SLURM_SUBMIT_DIR/mandelbrot/data"
RES=1024x1024   # square grid (square viewport -> square pixels); matches the
                # serial Level 1 characterisation. N_max fixed for the parallel study.
ITER=1000
REPEAT=5
JOB=$SLURM_JOB_ID

# --- thread pinning --------------------------------------------------------
# Bind threads to cores so they don't migrate and blur the timings.
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# --- T(1): true serial baseline on this node -------------------------------
./mandelbrot --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
    --p 1 --schedule - \
    --csv "$DATA/run_${JOB}_serial.csv"

# --- T(p): OpenMP sweep over schedule x thread count -----------------------
# static           : fixed contiguous blocks, no runtime rebalancing (worst
#                    case for imbalance, zero coordination overhead).
# dynamic,{1,16,64}: work queue handing out chunks of that many rows; the chunk
#                    sweep exposes the load-balancing vs coordination-overhead
#                    trade-off (small = better balance/more overhead).
# guided           : queue with shrinking chunks (large -> small), aiming for
#                    dynamic's balance at lower overhead.
# All at fixed N_max=1000: the parallel study isolates the parallel knobs; the
# N_max sensitivity of the imbalance is a separate serial (Level 1) job.
for SCHED in static "dynamic,1" "dynamic,16" "dynamic,64" guided; do
    LABEL=${SCHED//,/_}                  # "dynamic,16" -> "dynamic_16" for paths
    export OMP_SCHEDULE="$SCHED"
    for P in 2 4 8 16; do
        export OMP_NUM_THREADS=$P
        ./mandelbrot_omp --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
            --p "$P" --schedule "$SCHED" \
            --csv "$DATA/run_${JOB}_${LABEL}_p${P}.csv"
    done
done

# --- aggregate this job's per-run shards into one file ---------------------
# Within a single job the runs are sequential (a single writer), so merging the
# shards here is race-free - unlike a CSV shared across jobs. It leaves one file
# per job instead of one per run, which is friendlier to both the repo and the
# parallel filesystem (fewer small files = less metadata-server load). The shard
# glob has the "_" after the job id, so it never matches the aggregate itself;
# merge_metrics.py globs run_*.csv either way.
AGG="$DATA/run_${JOB}.csv"
shards=("$DATA"/run_${JOB}_*.csv)
head -n 1 "${shards[0]}" > "$AGG"          # canonical header, once
tail -q -n +2 "${shards[@]}" >> "$AGG"     # data rows from every run
rm -f "${shards[@]}"
