#!/bin/bash
# Serial baseline cross-sweep over N_max x resolution - the Level 1 analysis.
#
# The bare kernel (no USE_PRUNING/USE_SYMMETRY) is the imbalance reference: W,
# lambda_row, lambda_block and CoV reflect the real per-pixel work.
#
# Two axes. As measured on gnode01 (job 31391), both move the imbalance far less
# than expected - which is the finding, not a defect of the sweep:
#   N_max      -> WEAK, SATURATING effect. Raising the cap does make interior and
#                 boundary pixels iterate longer, but 10x (500 -> 5000) buys only
#                 +3.8% CoV and +3.1% lambda, in decreasing increments. The cheap
#                 rows are cap-independent (the exterior escapes in a few
#                 iterations regardless) while mean and max both grow with the
#                 cap, and their ratio therefore tends to a constant.
#   resolution -> NO measurable effect: lambda and CoV are ratios (max/mean,
#                 std/mean), hence scale-invariant - 512^2 and 1024^2 agree to
#                 the fourth decimal. What resolution drives is total work W
#                 (exactly 4x) and time.
# Conclusion: the imbalance is intrinsic to the geometry of the set, to be
# engineered around rather than dialled up. See README section "Revised by
# measurement".
#
# Each run also dumps its per-row work profile w_r for the "work per row" figure.
#
# Submit from the experiment root:  sbatch mandelbrot/job_sbatch/mandel_serial.sh

#SBATCH --account=g.larocca-thesis       # billing account
#SBATCH --job-name=mandel_serial
#SBATCH --partition=ulow                 # gnode01, QOS no-gpu: the CPU partition
                                         # this account may use. Same node as the
                                         # CUDA runs, so timings stay comparable.
                                         # PreemptMode=OFF, OverSubscribe=NO.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                # single core: this is the baseline
#SBATCH --gres=gpu:0
#SBATCH --time=00:45:00                  # 6 runs, up to 1024^2 x 5000 on 1 core
#SBATCH --output=mandelbrot/job_logs/out_%x_%j.log  # relative to $SLURM_SUBMIT_DIR

set -euo pipefail

# --- toolchain -------------------------------------------------------------
# gcc 8.5.0 is the node's system compiler (RHEL 8 default) and survives
# `module purge`; the bare serial target builds with it, so no module is needed
# (there is no loadable amd/gcc-8.5.0 - that string is a modulepath prefix).
module purge
g++ --version | head -1        # log the compiler (expected: 8.5.0)

# --- build in-job ----------------------------------------------------------
# Compile where we measure (-march=native tunes for this CPU). Default target is
# the bare serial kernel - exactly the imbalance reference we want here.
cd "$SLURM_SUBMIT_DIR/mandelbrot/src"
make mandelbrot

# --- experiment parameters -------------------------------------------------
DATA="$SLURM_SUBMIT_DIR/mandelbrot/data"
# Square grids only: the viewport is square (2.5 x 2.5), so W=H gives square
# pixels (dx=dy). Two sizes (4x pixel count) suffice to show lambda/CoV are
# nearly scale-invariant - the imbalance is driven by N_max, not by grid size.
RESOLUTIONS="512x512 1024x1024"
NMAX_VALUES="500 1000 5000"
REPEAT=3                                 # metrics are deterministic; 3 denoises time
JOB=$SLURM_JOB_ID

# --- N_max x resolution cross-sweep ----------------------------------------
# Imbalance metrics (W, lambda, CoV) are deterministic in the inputs; --repeat
# only denoises the timing/gflops. Each run writes its result shard plus a
# per-row work profile (rowprofile_*, a separate prefix so neither the
# aggregation nor the merge sweeps it up).
for RES in $RESOLUTIONS; do
    R=${RES%%x*}                         # "1024x1024" -> "1024" for the file tag
    for N in $NMAX_VALUES; do
        ./mandelbrot --resolution "$RES" --max-iter "$N" --repeat "$REPEAT" \
            --p 1 --schedule - \
            --csv "$DATA/run_${JOB}_res${R}_nmax${N}.csv" \
            --row-stats "$DATA/rowprofile_${JOB}_res${R}_nmax${N}.csv"
    done
done

# --- aggregate this job's per-run shards into one file ---------------------
# Sequential runs, single writer -> race-free concatenation (see mandel_omp.sh).
# The shard glob has the "_" after the job id, so it never matches the aggregate.
AGG="$DATA/run_${JOB}.csv"
shards=("$DATA"/run_${JOB}_*.csv)
head -n 1 "${shards[0]}" > "$AGG"          # canonical header, once
tail -q -n +2 "${shards[@]}" >> "$AGG"     # data rows from every N_max
rm -f "${shards[@]}"
