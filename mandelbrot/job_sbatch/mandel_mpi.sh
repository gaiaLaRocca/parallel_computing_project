#!/bin/bash
# MPI strong-scaling sweep across two nodes.
#
# Measures the serial baseline T(1) and the MPI T(p) for the three row
# decompositions (block, cyclic, dynamic master-worker) at fixed N_max and
# resolution, so the only knobs are the scheme and the process count. Each run
# writes its own data/run_<jobid>_<...>.csv; the Python merge
# (analysis/merge_metrics.py) concatenates them and fills the relational
# columns (speedup/efficiency/karp_flatt_e) by matching each T(p) to this T(1).
#
# Submit from the experiment root:  sbatch mandelbrot/job_sbatch/mandel_mpi.sh

#SBATCH --account=g.larocca-thesis             # billing account - REQUIRED, fill in
#SBATCH --job-name=mandel_mpi
#SBATCH --partition=<cpu_partition>      # verify a CPU partition with `sinfo`
#SBATCH --nodes=2                        # >1 node: exercise genuine inter-node messages
#SBATCH --ntasks-per-node=9              # 2x9 = 18 slots, enough for the largest run (np=17)
#SBATCH --cpus-per-task=1                # pure MPI: one core per rank
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=job_logs/out_%x_%j.log  # relative to $SLURM_SUBMIT_DIR

set -euo pipefail

# --- toolchain -------------------------------------------------------------
# Verify exact module names on the cluster with `module available` before use.
module purge
module load amd/gcc-8.5.0
module load amd/gcc-8.5.0/openmpi-4.1.6

# --- build in-job ----------------------------------------------------------
# Compile where we measure: -march=native tunes for this CPU, so building on a
# different node than the one that runs would risk illegal instructions or,
# worse, silently incomparable timings.
cd "$SLURM_SUBMIT_DIR/mandelbrot/src"
make mandelbrot mandelbrot_mpi

# --- experiment parameters -------------------------------------------------
# Bare kernel (no USE_PRUNING/USE_SYMMETRY): this is the scaling/imbalance
# reference, where W and w_r reflect the real work.
DATA="$SLURM_SUBMIT_DIR/mandelbrot/data"
RES=1024x1024   # same fixed grid and N_max as the OpenMP sweep, so the two
ITER=1000       # paradigms are compared on identical work.
REPEAT=5
JOB=$SLURM_JOB_ID
NODES=$SLURM_JOB_NUM_NODES

# Round-robin the ranks across the two nodes (-m cyclic): even the smallest
# run pays real network latency instead of packing onto one node, which is the
# communication regime this sweep is meant to measure. NB: this rank->node
# placement is unrelated to the row->rank decomposition also named "cyclic".
LAUNCH="srun --kill-on-bad-exit=1 -m cyclic"

# --- T(1): true serial baseline on this partition ---------------------------
# One rank of the same allocation: same hardware as the parallel points, as
# required for valid speedups. (The MPI sweep is multi-node while OpenMP is
# single-node; the nodes column records this for the analysis.)
srun -N 1 -n 1 ./mandelbrot \
    --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
    --p 1 --nodes 1 --schedule - \
    --csv "$DATA/run_${JOB}_serial.csv"

# --- T(p): static decompositions, block and cyclic ---------------------------
# block  : contiguous row slices - maximal imbalance (the heavy centre rows
#          land on few ranks), minimal communication (one gather).
# cyclic : row r -> rank r mod P - near-balanced by construction, still static
#          and still one gather; the hypothesised winner.
# One binary, scheme picked at runtime by MPI_DECOMP (sbatch exports it to the
# ranks); the --schedule label just tags the CSV row.
for DECOMP in block cyclic; do
    export MPI_DECOMP=$DECOMP
    for NP in 2 4 8 16; do
        $LAUNCH -n "$NP" ./mandelbrot_mpi \
            --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
            --p "$NP" --nodes "$NODES" --schedule "$DECOMP" \
            --csv "$DATA/run_${JOB}_${DECOMP}_p${NP}.csv"
    done
done

# --- T(p): dynamic master-worker ---------------------------------------------
# Rank 0 coordinates and computes nothing, so np = workers + 1: sweeping
# np in {3,5,9,17} keeps the COMPUTING units at {2,4,8,16}, aligned with the
# static schemes and the OpenMP thread sweep. For the same reason --p records
# the worker count, not np: speedup and efficiency are per computing unit
# (the idle master is a stated caveat in the report, recoverable as p+1).
export MPI_DECOMP=dynamic
for NP in 3 5 9 17; do
    W=$((NP - 1))
    $LAUNCH -n "$NP" ./mandelbrot_mpi \
        --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
        --p "$W" --nodes "$NODES" --schedule dynamic \
        --csv "$DATA/run_${JOB}_dynamic_p${W}.csv"
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
