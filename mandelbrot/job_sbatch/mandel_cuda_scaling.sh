#!/bin/bash
# CUDA problem-scaling sweep (Sweep B) on a single GPU.
#
# Fixes the block shape (the Sweep A winner) and varies the PROBLEM, to trace how
# the GPU fills up and where it pays off:
#   B1  saturation : vary resolution at fixed N_max. Small grids launch too few
#                    warps to hide latency, so effective GFLOP/s (and speedup)
#                    are low; the curve rises and flattens towards the card's
#                    peak as the grid grows - the GPU's "scalability" story,
#                    since the core count is fixed and only utilisation varies.
#   B2  transfer   : vary N_max at fixed resolution. The device->host copy is a
#                    fixed byte count while compute grows ~ N_max, so
#                    transfer_time / T_min falls - the PCIe "Amdahl" term and its
#                    amortisation.
# Each problem point runs a serial T(1) on this same node right before the GPU
# run, so the merge fills speedup same-hardware (S = T(1)/T_gpu, core vs GPU).
#
# Submit from the experiment root:  sbatch mandelbrot/job_sbatch/mandel_cuda_scaling.sh

#SBATCH --account=g.larocca-thesis       # billing account
#SBATCH --job-name=mandel_cuda_scaling
#SBATCH --partition=only-one-gpu         # gnode01 (8x L40S), one-GPU partition.
                                         # Alt: ulow (default) / debug (short tests)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1                 # the serial T(1) baselines run here too
#SBATCH --gres=gpu:1                      # typed alt if rejected: --gres=gpu:nvl40s_0:1
#SBATCH --time=00:50:00                   # dominated by the large serial baselines
#SBATCH --output=job_logs/out_%x_%j.log   # relative to $SLURM_SUBMIT_DIR

set -euo pipefail

# --- toolchain -------------------------------------------------------------
# gcc 8.5.0 is the node's system compiler (RHEL 8 default): it survives
# `module purge`, nvcc uses it as its host compiler, and it builds the serial
# baseline target too - so ONLY the CUDA toolkit needs a module. There is no
# loadable amd/gcc-8.5.0 (that string is a modulepath prefix, not a module).
module purge
module load amd/nvidia/cuda-12.3.2
g++ --version | head -1        # log the host compiler (expected: 8.5.0)
nvcc --version | tail -1       # log the CUDA toolkit

# --- record the GPU used ---------------------------------------------------
# gnode01 = L40S: FP64 is 1/64 of FP32 (~1.4 TFLOP/s = ~1400 GFLOP/s peak), the
# number to pass to plot_metrics.py --gpu-peak-gflops for the throughput figure.
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true

# --- build in-job ----------------------------------------------------------
# --fmad=false keeps the escape counts bit-identical to the CPU baseline. If
# -arch=native is unavailable, pin it: sm_89 (gnode01 L40S) / sm_86 (gnode02 A6000).
cd "$SLURM_SUBMIT_DIR/mandelbrot/src"
make mandelbrot mandelbrot_cuda NVARCH=-arch=native

# --- experiment parameters -------------------------------------------------
DATA="$SLURM_SUBMIT_DIR/mandelbrot/data"
JOB=$SLURM_JOB_ID
REPEAT_CPU=3                              # serial metrics are deterministic
REPEAT_GPU=10                            # denoise the fast kernel timing

# Block shape held fixed across Sweep B. SET THIS TO THE SWEEP A WINNER before
# submitting; 256x1 is the default (horizontal warps) until Sweep A decides.
export MANDEL_BLOCK=256x1

# Runs the serial baseline and the CUDA point for one problem config. The two
# share resolution/N_max, so the merge pairs them; the serial one is written on
# the GPU node's CPU to keep the ratio same-hardware.
run_pair() {
    local res=$1 iter=$2 tag=$3
    ./mandelbrot --resolution "$res" --max-iter "$iter" --repeat "$REPEAT_CPU" \
        --p 1 --schedule - \
        --csv "$DATA/run_${JOB}_${tag}_serial.csv"
    ./mandelbrot_cuda --resolution "$res" --max-iter "$iter" --repeat "$REPEAT_GPU" \
        --p 1 --schedule "$MANDEL_BLOCK" \
        --csv "$DATA/run_${JOB}_${tag}_cuda.csv"
}

# --- B1: saturation vs problem size (fixed N_max=1000) ---------------------
# Span from an under-utilised GPU (512^2 = 0.26 Mpx) to a saturated one
# (4096^2 = 16.8 Mpx). Same N_max isolates size as the only variable.
for RES in 512x512 1024x1024 2048x2048 4096x4096; do
    R=${RES%%x*}
    run_pair "$RES" 1000 "b1_res${R}"
done

# --- B2: transfer fraction vs N_max (fixed resolution=2048^2) --------------
# Fixed pixel count => fixed device->host bytes, while compute grows ~ N_max:
# transfer_time/T_min should fall. 2048^2 is large enough to be GPU-saturated so
# the trend reflects compute vs copy, not launch under-utilisation. (2048/1000
# is already covered by B1, so only the other two N_max values are added.)
for N in 500 5000; do
    run_pair 2048x2048 "$N" "b2_nmax${N}"
done

# --- aggregate this job's per-run shards into one file ---------------------
# Sequential runs, single writer -> race-free concatenation (see mandel_omp.sh).
AGG="$DATA/run_${JOB}.csv"
shards=("$DATA"/run_${JOB}_*.csv)
head -n 1 "${shards[0]}" > "$AGG"          # canonical header, once
tail -q -n +2 "${shards[@]}" >> "$AGG"     # data rows from every problem point
rm -f "${shards[@]}"
