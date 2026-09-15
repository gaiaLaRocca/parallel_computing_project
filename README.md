# Parallel Computing Project — Mandelbrot Set

Serial, OpenMP, MPI and CUDA implementations of the Mandelbrot escape-time
computation, together with the SLURM sweeps, the analysis scripts and the
report that compare them.

The full analysis, in Italian, is in [`latex_report/`](latex_report/), which
contains the LaTeX sources and the compiled PDF (`report.pdf`). This README is
self-contained: it describes what the project studies, how the experiments are
designed and how to reproduce them. It does not report timings, since those
depend on the hardware the sweeps run on.

---

## Scope

Each pixel of the Mandelbrot set can be computed independently of all others,
so the problem is embarrassingly parallel. Its cost, however, is far from
uniform: the escape time `n(i, j)` of a pixel, the number of iterations of
`z ← z² + c` before `|z| > 2`, ranges from a few iterations for points far from
the set up to the cap `max_iter` for interior points. Expensive pixels
concentrate in the rows that cross the main cardioid and the period-2 bulb.

The project studies how this single irregularity affects each paradigm:

- **On CPUs (OpenMP, MPI)** it becomes *load imbalance*: the work is split into
  rows, and the parallel time is set by the most loaded thread or process. The
  question is how much each assignment rule — contiguous blocks, cyclic
  interleaving, dynamic queues — loses to it.
- **On the GPU (CUDA)** it becomes *warp divergence*: threads run in groups of 32
  (warps) that finish only when their slowest thread does. The question is
  whether the thread→pixel mapping, set by the block shape, changes the time.

A second goal is methodological: to check how far the parallel behaviour can be
**predicted from one serial run**. The serial baseline records the escape-time
matrix and the per-row work `w_r = Σ_j n(r, j)`. From these, before any parallel
code runs, the analysis scripts compute:

- for each CPU assignment rule, the effective imbalance
  `λ_p = max_k W_k / (W/p)` (where `W_k` is the work assigned to unit `k`) and
  the resulting speedup bound `S(p) ≤ p / λ_p`;
- for each CUDA block shape, the warp statistics: mean within-warp variance,
  masked (wasted) iterations and total lane-cycles.

The measured sweeps are then compared against these predictions.

## Methodology

- **One kernel per paradigm, everything else shared.** Driver, timing, I/O and
  metrics are the same code for all variants, so the only difference between
  two runs is the compute kernel.
- **Bit-identical results.** Every variant writes a 64-bit FNV-1a checksum of the
  escape-time matrix, which must match the serial one. Floating-point
  contraction is disabled (`-ffp-contract=off`, `--fmad=false` for `nvcc`) and
  `-ffast-math` is never used, since a fused multiply-add can change an escape
  count by ±1 near the boundary.
- **Bare kernel.** The benchmarks disable the two available optimisations
  (cardioid/bulb test and axial symmetry). Both would remove work unevenly and
  hide the imbalance under study.
- **Timing.** The timed region covers allocation and computation, excluding
  I/O. Each point is repeated and the minimum is reported, with median and mean
  as noise indicators.
- **Work and throughput.** Total work is `W = Σ n(i, j)`. Each iteration costs
  8 floating-point operations (FLOPs), so throughput is `8 W / T` FLOPs per
  second, an absolute metric comparable across CPU and GPU.
- **Speedup against the job's own baseline.** Every sweep measures its serial
  time `T(1)` inside the same job as its parallel points. The merge pairs each
  point with that baseline and computes speedup `S = T(1)/T(p)`, efficiency
  `S/p` and the Karp–Flatt serial fraction. For CUDA, `S` compares one GPU with
  one CPU core.
- **One result file per job.** Jobs never append to a shared CSV, since
  concurrent appends can corrupt it on a network filesystem. Each job writes
  `run_<jobid>.csv`, and a Python script merges them into one dataset.

## Experiments

Each sweep is a SLURM script in `mandelbrot/job_sbatch/`. All CPU sweeps use a
`1024×1024` grid of the region `[-2, 0.5] × [-1.25, 1.25]` with
`max_iter = 1000`, unless noted otherwise.

| Script | What varies | Fixed | Repeats |
|---|---|---|---|
| `mandel_serial.sh` | resolution `512², 1024²` × `max_iter` `500, 1000, 5000` | 1 core | 3 |
| `mandel_omp.sh` | schedule `static`, `dynamic,1`, `dynamic,16`, `dynamic,64`, `guided` × threads `2, 4, 8, 16, 32` | threads pinned to cores | 5 |
| `mandel_mpi.sh` | `block`, `cyclic` × processes `2, 4, 8, 16, 32`; master–worker × workers `2, 4, 8, 16, 32` (plus one master) | one core per rank | 5 |
| `mandel_cuda_blocks.sh` | block shape `256x1, 512x1, 128x1, 128x2, 64x4, 32x8, 16x16, 8x32, 1x256` | one GPU | 30 |
| `mandel_cuda_scaling.sh` | resolution `512²…4096²` at `max_iter = 1000`; `max_iter` `500, 1000, 5000` at `2048²` | one GPU, block `256x1` | 30 (GPU), 3 (serial) |

What each sweep is designed to show:

- **Serial.** Whether resolution and `max_iter` change the shape of the load, or
  only its amount: the imbalance indices `λ = max_r w_r / mean(w_r)` and
  `CoV = std(w_r) / mean(w_r)` are recorded, together with the full row profile
  of every run.
- **OpenMP.** How scheduling policy and chunk size trade load balance against
  coordination cost. `static` splits the rows into contiguous blocks,
  `dynamic,c` hands out chunks of `c` rows from a shared queue, and `guided`
  starts with large chunks and shrinks them.
- **MPI.** The same trade-off with explicit messages. `block` sends contiguous
  rows to each rank; `cyclic` interleaves them (row `r` to rank `r mod p`), with
  the same single gather; the master–worker scheme hands out one row at a time
  on request. The share of time spent in MPI calls is recorded as
  `comm_fraction`.
- **CUDA, block shapes.** Whether the block shape matters. A block width that is
  a multiple of 32 keeps every warp on a single row, so its writes to the
  row-major matrix are contiguous (coalesced). Narrower blocks fold rows into
  one warp, and compact tiles group pixels that are close in both directions.
  Each run records warp divergence (from the matrix), occupancy and
  device→host transfer time.
- **CUDA, problem size.** How GPU utilisation grows with the number of pixels,
  and how the fixed cost of copying the result back weighs as the work per pixel
  grows. The job also logs which CPU socket the process and the GPU sit on,
  because a copy that crosses sockets can be slower and the transfer time must
  be read against the placement the job actually received.

## Repository layout

```
parallel_computing_project/
├── README.md
├── latex_report/            report: LaTeX sources and report.pdf (Italian)
└── mandelbrot/
    ├── src/                 C++/CUDA sources and Makefile
    ├── job_sbatch/          SLURM scripts, one sweep each
    ├── analysis/            Python: merge, predictions, figures
    ├── data/                per-job results and derived tables (versioned)
    └── job_logs/            SLURM logs (git-ignored)
```

`CMakeLists.txt` at the root only serves IDE indexing of the serial sources;
the build uses the Makefile.

## Implementations

| Variant | Source | Make target | Parallelism |
|---|---|---|---|
| Serial baseline | `mandelbrot.cpp` | `mandelbrot` (default) | — |
| OpenMP | `mandelbrot_omp.cpp` | `mandelbrot_omp` | threads over rows, policy from `OMP_SCHEDULE` |
| MPI | `mandelbrot_mpi.cpp` | `mandelbrot_mpi` | rows split by `MPI_DECOMP` = `block` (default), `cyclic` or `dynamic` (master–worker) |
| CUDA | `mandelbrot_cuda.cu` | `mandelbrot_cuda` | one thread per pixel, block shape from `MANDEL_BLOCK=BXxBY` (default `256x1`) |

## Requirements

- g++ with C++17 and OpenMP
- an MPI implementation providing `mpicxx` (tested with OpenMPI 4.1)
- the CUDA toolkit (`nvcc`) for the GPU variant
- Python 3 for the analysis; only `plot_metrics.py` needs `matplotlib`
- SLURM for the job scripts

## Build

```bash
cd mandelbrot/src
make                    # serial baseline
make mandelbrot_omp     # OpenMP
make mandelbrot_mpi     # MPI
make mandelbrot_cuda    # CUDA
```

The build uses `-march=native`, so compile on the machine that runs the
benchmark. The two optimisations are off by default:

```bash
make OPT="-DUSE_PRUNING"                  # cardioid / period-2 bulb test
make OPT="-DUSE_PRUNING -DUSE_SYMMETRY"   # plus mirroring about the real axis
```

## Running

```
./mandelbrot [--resolution WxH] [--max-iter N] [--repeat N]
             [--center-re X --center-im Y --span S]
             [--csv FILE] [--ppm FILE] [--pgm FILE] [--raw FILE] [--row-stats FILE]
             [--p N] [--nodes N] [--schedule NAME]
```

- `--repeat N` runs the computation N times and reports the minimum, median and
  mean time.
- `--csv` writes one result row: times, total work, imbalance indices,
  throughput, checksum and the paradigm-specific fields.
- `--ppm`, `--pgm` and `--raw` write the image or the raw escape-time matrix;
  `--row-stats` writes the per-row work profile.
- `--center-re`, `--center-im` and `--span` select a zoom window instead of the
  default region.
- `--p`, `--nodes` and `--schedule` only label the CSV row.

Examples:

```bash
OMP_NUM_THREADS=8 OMP_SCHEDULE=dynamic,1 ./mandelbrot_omp --repeat 5 --p 8 --schedule dynamic_1
mpirun -x MPI_DECOMP=cyclic -n 8 ./mandelbrot_mpi --repeat 5 --p 8 --schedule cyclic
MANDEL_BLOCK=16x16 ./mandelbrot_cuda --repeat 30 --schedule 16x16
```

## Running the sweeps

Submit from the repository root, e.g. `sbatch mandelbrot/job_sbatch/mandel_omp.sh`.
Each script compiles inside the job, so that code is built for the CPU it runs
on, then writes `mandelbrot/data/run_<jobid>.csv` and its log to
`mandelbrot/job_logs/`. The `#SBATCH` account, partition and module lines refer
to the cluster used for the report and must be adapted elsewhere. The MPI script
requests a single node and launches ranks with `mpirun`; on one node messages
travel through shared memory, so `comm_fraction` is a lower bound for a
multi-node run.

## Analysis

Run from the repository root, after the sweeps:

```bash
python3 mandelbrot/analysis/merge_metrics.py     # run_*.csv -> merged.csv, with speedup, efficiency, Karp–Flatt
python3 mandelbrot/analysis/block_imbalance.py   # rowprofile_*.csv -> predicted λ_p and bounds for block/cyclic/dynamic
python3 mandelbrot/analysis/block_geometry.py    # raw matrix -> predicted warp statistics per CUDA block shape
python3 mandelbrot/analysis/plot_metrics.py      # report figures -> mandelbrot/data/plots/
```

- `block_geometry.py` reads by default the raw matrix written by
  `make_fractal_figures.sh`.
- `make_fractal_figures.sh` renders the fractal figures of the report through
  `render_fractal.py`; it needs no cluster.
- `plot_metrics.py` draws five figures:
  - the serial row profile;
  - OpenMP speedup per schedule;
  - measured CPU speedup against the predicted bound, one panel per assignment
    rule;
  - GPU throughput against image size;
  - device→host copy bandwidth against copy size.

`mandelbrot/data/` versions:
- the per-job results (`run_*.csv`) and the merged dataset;
- the serial row profiles;
- the two prediction tables (`block_imbalance.csv`, `block_geometry.csv`);
- short notes extracted from the CUDA job logs.

---

## Acknowledgements

Computational resources provided by hpc-ReGAInS@DISCo, a MUR Department of
Excellence Project within the Department of Informatics, Systems and
Communication at the University of Milan-Bicocca (https://www.disco.unimib.it).
