#!/bin/bash
# MPI strong-scaling sweep on a single node.
#
# Measures the serial baseline T(1) and the MPI T(p) for the three row
# decompositions (block, cyclic, dynamic master-worker) at fixed N_max and
# resolution, so the only knobs are the scheme and the process count. Each run
# writes its own data/run_<jobid>_<...>.csv; the Python merge
# (analysis/merge_metrics.py) concatenates them and fills the relational
# columns (speedup/efficiency/karp_flatt_e) by matching each T(p) to this T(1).
#
# SINGLE NODE, AND THAT IS A CONSTRAINT, NOT A CHOICE. This account reaches only
# ulow, only-one-gpu and ice4hpc, and all three expose gnode01 alone, so a
# --nodes=2 request is rejected at submit time. Every message therefore travels
# through shared memory rather than the network, and comm_fraction is an
# optimistic lower bound on what a real multi-node run would pay - state this in
# the report rather than presenting the numbers as inter-node communication.
# The compensation is that T(1) and T(p) are same-hardware by construction, and
# that the MPI and OpenMP curves become directly comparable: both sweep the same
# process counts on the same socket of the same node, so their difference is the
# programming model and nothing else.
#
# Submit from the experiment root:  sbatch mandelbrot/job_sbatch/mandel_mpi.sh

#SBATCH --account=g.larocca-thesis             # billing account - REQUIRED, fill in
#SBATCH --job-name=mandel_mpi
#SBATCH --partition=ulow                 # gnode01, QOS no-gpu; same node as the
                                         # serial and OpenMP sweeps
#SBATCH --nodes=1                        # forced: no reachable partition has two
#SBATCH --ntasks=33                      # largest run is the master-worker at 32
                                         # workers, i.e. np = 33. The static
                                         # schemes use 32 of these, one socket.
#SBATCH --cpus-per-task=1                # pure MPI: one core per rank
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00                  # ~1 min of compute; a short limit is far
                                         # easier for backfill to place
#SBATCH --output=mandelbrot/job_logs/out_%x_%j.log  # relative to $SLURM_SUBMIT_DIR

set -euo pipefail

# --- toolchain -------------------------------------------------------------
# gcc 8.5.0 is the node's system compiler (RHEL 8 default) and survives
# `module purge`; only OpenMPI needs a module (it provides mpicxx, which wraps
# the system gcc). There is no loadable amd/gcc-8.5.0 - that string is a
# modulepath prefix, and amd/gcc-8.5.0/openmpi-4.1.6 is the real MPI module.
module purge
module load amd/gcc-8.5.0/openmpi-4.1.6
mpicxx --version | head -1     # log the compiler (expected: gcc 8.5.0)

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

# Launch with mpirun, NOT srun. This OpenMPI cannot be direct-launched by srun:
# its PMIx client (the ext3x shim of OMPI 4.1.6) cannot reach the PMIx v5 server
# this SLURM exposes, and every rank dies in MPI_Init with
#   OPAL ERROR: Unreachable in file ext3x_client.c at line 111
#   "the application appears to have been direct launched using srun, but OMPI
#    was not built with SLURM's PMI support"
# (verified interactively on gnode01, 2026-09-09; `srun --mpi=list` does show
# pmix_v5, so SLURM's side is not the problem - the two PMIx versions are).
# mpirun reads the allocation from the SLURM_* environment, starts its own
# daemons on the allocated node and bootstraps the ranks with OMPI's internal
# PMIx, so the handshake stays inside one MPI stack.
#
#   --map-by core --bind-to core : one rank per core, assigned in order, so the
#       ranks fill socket 0 before touching socket 1 (gnode01 is 2 sockets x 32
#       cores). This is the MPI analogue of the OpenMP sweep's OMP_PROC_BIND=
#       close: unpinned, the runtime may spread ranks over both sockets even for
#       np=2, and the small-np points would carry a cross-socket penalty the
#       large ones do not, bending the speedup curve for a reason that has
#       nothing to do with the decomposition. Only the np=33 master-worker point
#       spills one rank onto socket 1, unavoidably - 32 workers plus a master
#       exceed one socket. NB: "core" here is rank->core placement, unrelated to
#       the row->rank decomposition called "block" below.
#   -x MPI_DECOMP : mpirun forwards only OMPI_* variables by default (srun
#       exported the whole environment), and the kernel reads its decomposition
#       from MPI_DECOMP, so it must be listed explicitly. Without it every run
#       would silently fall back to the default scheme.
# A rank that aborts takes the whole mpirun down with a non-zero status, which
# `set -e` turns into a failed job - the same fail-fast the srun launcher got
# from --kill-on-bad-exit=1.
LAUNCH="mpirun --map-by core --bind-to core -x MPI_DECOMP"

# --- T(1): true serial baseline on this partition ---------------------------
# One rank of the same allocation: same hardware as the parallel points, as
# required for valid speedups - and, since the OpenMP sweep measures its own
# T(1) the same way on the same node, the two paradigms' speedups are on a
# common footing. This one keeps srun: the serial binary never calls MPI_Init,
# so it needs no PMI bootstrap and the incompatibility above does not apply.
srun -N 1 -n 1 ./mandelbrot \
    --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
    --p 1 --nodes 1 --schedule - \
    --csv "$DATA/run_${JOB}_serial.csv"

# --- T(p): static decompositions, block and cyclic ---------------------------
# block  : contiguous row slices - maximal imbalance (the heavy centre rows
#          land on few ranks), minimal communication (one gather).
# cyclic : row r -> rank r mod P - near-balanced by construction, still static
#          and still one gather; the hypothesised winner.
# One binary, scheme picked at runtime by MPI_DECOMP (exported here, forwarded
# to the ranks by mpirun's -x); the --schedule label just tags the CSV row.
for DECOMP in block cyclic; do
    export MPI_DECOMP=$DECOMP
    for NP in 2 4 8 16 32; do
        $LAUNCH -n "$NP" ./mandelbrot_mpi \
            --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
            --p "$NP" --nodes "$NODES" --schedule "$DECOMP" \
            --csv "$DATA/run_${JOB}_${DECOMP}_p${NP}.csv"
    done
done

# --- T(p): dynamic master-worker ---------------------------------------------
# Rank 0 coordinates and computes nothing, so np = workers + 1: sweeping
# np in {3,5,9,17,33} keeps the COMPUTING units at {2,4,8,16,32}, aligned with
# the static schemes, the OpenMP thread sweep and the P column of
# analysis/block_imbalance.py. For the same reason --p records the worker count,
# not np: speedup and efficiency are per computing unit, and comparing this
# curve against a prediction indexed by np would manufacture a shortfall that is
# an artefact of the abscissa (the idle master is a stated caveat in the report,
# recoverable as p+1).
export MPI_DECOMP=dynamic
for NP in 3 5 9 17 33; do
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
