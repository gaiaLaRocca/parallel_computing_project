#!/bin/bash
# CUDA block-shape sweep (Sweep A) on a single GPU.
#
# Fixes the problem (one resolution x N_max) and varies only the launch block
# shape via MANDEL_BLOCK, to find the geometry with the lowest kernel time and
# to explain WHY it wins from the recorded metrics:
#   * warp_divergence : the block-aware proxy (metrics.cpp) under each geometry.
#                       Horizontal warps (block_x a multiple of 32) minimise the
#                       within-warp spread of iteration counts; a warp that folds
#                       several rows or a scattered mapping diverges more.
#   * occupancy       : resident warps per SM the config sustains (register-bound;
#                       the -Xptxas -v line in the build log gives the reg count).
# The single serial T(1) below shares the fixed config, so the Python merge fills
# speedup for every block-shape run from it (S = T(1)/T_gpu, single core vs GPU).
#
# The block shape is NOT part of the merge's config key, so every shape lands as
# a distinct row tagged by --schedule; that is how the block-shape figure tells
# them apart while all match the one T(1).
#
# Submit from the experiment root:  sbatch mandelbrot/job_sbatch/mandel_cuda_blocks.sh

#SBATCH --account=g.larocca-thesis       # billing account
#SBATCH --job-name=mandel_cuda_blocks
#SBATCH --partition=only-one-gpu         # gnode01 (8x L40S), one-GPU partition.
                                         # ulow cannot: its QOS is no-gpu.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                 # host side is trivial; one core is plenty
#SBATCH --gres=gpu:1                      # one GPU (typed alt if rejected:
                                         # --gres=gpu:nvl40s_0:1)
#SBATCH --time=00:20:00
#SBATCH --output=mandelbrot/job_logs/out_%x_%j.log   # relative to $SLURM_SUBMIT_DIR

set -euo pipefail

# --- toolchain -------------------------------------------------------------
# gcc 8.5.0 is the node's system compiler (RHEL 8 default): it survives
# `module purge`, nvcc uses it as its host compiler, and it builds the serial
# baseline target too - so ONLY the CUDA toolkit needs a module. There is no
# loadable amd/gcc-8.5.0 (that string is a modulepath prefix, not a module);
# `module available` confirmed cuda-12.3.2 as the toolkit.
module purge
module load amd/nvidia/cuda-12.3.2
g++ --version | head -1        # log the host compiler (expected: 8.5.0)
nvcc --version | tail -1       # log the CUDA toolkit

# --- record the GPU used ---------------------------------------------------
# The effective-vs-peak GFLOP/s analysis needs the card's FP64 peak, which is
# not in the CSV: log the device identity so the peak can be looked up and
# passed to plot_metrics.py --gpu-peak-gflops later. On gnode01 the card is the
# L40S, whose FP64 is 1/64 of FP32 (~1.4 TFLOP/s = ~1400 GFLOP/s peak) - expect
# low absolute FP64 throughput and a strong case for a later FP32 experiment.
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true

# --- build in-job ----------------------------------------------------------
# Compile where we measure. NVARCH=-arch=native tunes the SASS for this GPU;
# --fmad=false (in the Makefile) keeps the escape counts bit-identical to the
# CPU baseline. The -Xptxas -v register report is emitted here into the log.
# If -arch=native is unavailable, pin it explicitly: sm_89 for gnode01's L40S,
# sm_86 for gnode02's A6000.
cd "$SLURM_SUBMIT_DIR/mandelbrot/src"
make mandelbrot mandelbrot_cuda NVARCH=-arch=native

# --- experiment parameters -------------------------------------------------
# Bare kernel (no USE_PRUNING/USE_SYMMETRY): the scaling/imbalance reference,
# where W and w_r reflect the real work. RES/ITER match the OpenMP sweep so the
# GPU point is comparable to the CPU paradigms at the same problem.
DATA="$SLURM_SUBMIT_DIR/mandelbrot/data"
RES=1024x1024
ITER=1000
REPEAT=30                                 # kernel is fast; more repeats denoise T.
                                          # 10 was not enough: in job 34334 the six
                                          # shapes that are *theoretically identical*
                                          # (blockDim.x a multiple of 32 => same warp
                                          # geometry, same divergence proxy, same
                                          # occupancy) still spread 7.6% in T_min,
                                          # swamping the 3.4% gap that separated the
                                          # nominal winner from 256x1. The ranking
                                          # cannot be read below that noise floor.
JOB=$SLURM_JOB_ID

# --- T(1): serial baseline on THIS node ------------------------------------
# Same config as every CUDA run below, so the merge matches it as their T(1).
# Runs on the GPU node's host CPU to keep the speedup same-hardware.
./mandelbrot --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
    --p 1 --schedule - \
    --csv "$DATA/run_${JOB}_serial.csv"

# --- T_gpu: block-shape sweep ----------------------------------------------
# 256x1/512x1/128x1 : pure horizontal warps (the EPFL-style config), coalesced
#                     writes and minimal divergence.
# 128x2/64x4/32x8   : 2D tiles whose warps still stay within one row group.
# 16x16/8x32        : block_x < 32, so a warp folds several image rows - tests
#                     whether a compact tile beats a strip, at the cost of split
#                     memory transactions.
# 1x256             : vertical warp - the coalescing worst case (32 writes,
#                     width*4 bytes apart) with divergence ~ horizontal; isolates
#                     the coalescing penalty from the divergence penalty.
for BLOCK in 256x1 512x1 128x1 128x2 64x4 32x8 16x16 8x32 1x256; do
    export MANDEL_BLOCK="$BLOCK"
    ./mandelbrot_cuda --resolution "$RES" --max-iter "$ITER" --repeat "$REPEAT" \
        --p 1 --schedule "$BLOCK" \
        --csv "$DATA/run_${JOB}_block${BLOCK}.csv"
done

# --- optional profiler confirmation (separate run, never the timed one) -----
# ncu perturbs timing, so it runs AFTER the measured sweep on the default shape
# only, and is guarded: on clusters that restrict performance counters to admins
# it fails harmlessly without failing the job. It confirms the occupancy and the
# divergence the deterministic proxy already reported.
export MANDEL_BLOCK=256x1
ncu --set basic --launch-count 1 --target-processes all \
    ./mandelbrot_cuda --resolution "$RES" --max-iter "$ITER" --repeat 1 \
    --p 1 --schedule 256x1 || echo "ncu unavailable (likely counter permissions); skipping"

# --- aggregate this job's per-run shards into one file ---------------------
# Sequential runs, single writer -> race-free concatenation (see mandel_omp.sh).
# The shard glob has the "_" after the job id, so it never matches the aggregate.
AGG="$DATA/run_${JOB}.csv"
shards=("$DATA"/run_${JOB}_*.csv)
head -n 1 "${shards[0]}" > "$AGG"          # canonical header, once
tail -q -n +2 "${shards[@]}" >> "$AGG"     # data rows from every block shape
rm -f "${shards[@]}"
