# Parallel Computing Project — Mandelbrot Set

Comparative study on the **parallelization of a fractal computation** (the
Mandelbrot set) across the main parallel programming paradigms, run and measured
on an HPC cluster.

The goal is not merely to "run Mandelbrot in parallel", but to understand **how**
the same computation behaves under different execution models (shared memory,
distributed memory, GPU), **how well** it scales, and **at what cost**
(communication, load imbalance, launch overhead), quantified with standard,
reproducible metrics.

> A detailed academic report (in Italian) accompanies this repository and is the
> primary reference for the analysis, derivations and results. See
> [`report.pdf`](report.pdf) *(added when available)*. This README summarises the
> idea and scope of the code; the report covers the theory and discussion in
> full. **The metric definitions below are the exact ones used in the report**
> (same symbols, same formulas) so that code and report never diverge.

> **Status (pre-cluster-run).** All four paradigms — serial, OpenMP, MPI and
> CUDA — are implemented, sharing one driver, metrics layer and per-job CSV. The
> compute kernel is the only file rewritten per paradigm, which is what makes
> the cross-variant checksum comparison meaningful. The CPU variants reproduce
> the serial checksum bit-for-bit locally; the CUDA kernel's checksum parity is
> the first thing validated on the cluster, since it needs a GPU. What remains
> is the measurement campaign: submit the sweeps, merge the per-job CSVs,
> generate the plots.

---

## Why the Mandelbrot set

Mandelbrot is a deceptively simple case study for parallel computing: trivial to
parallelize at the data level, yet non-trivial to parallelize *well*.

- **Embarrassingly parallel at the pixel level.** Each point of the complex
  plane is computed independently, with no data dependencies between pixels.
  Domain decomposition is therefore straightforward.
- **Strongly load-imbalanced.** Points *inside* the set iterate up to
  `max_iter`, while points *outside* escape after a few iterations. The per-pixel
  cost is a highly irregular (fractal) function of position, not a constant. This
  is what makes the problem interesting: it is an excellent testbed for
  *load-balancing* strategies, which are the central concern of the project.
- **Architecturally revealing.** The same imbalance manifests differently per
  paradigm: as idle workers under static CPU scheduling, and as **warp
  divergence** on the GPU, where threads in a warp that finish early must wait
  for the slowest.
- **Verifiable output.** The result is a matrix of iteration counts. Correctness
  across implementations is validated by comparing a checksum of that matrix
  (see *Correctness* below).

The single root cause — the per-pixel variability of the escape time — is the
thread that runs through the whole project: one algorithmic irregularity that
surfaces as **load imbalance** on CPUs and **warp divergence** on GPUs.

---

## Initial hypotheses (from parallelization theory)

These are the expectations the experiments are designed to test.

- **Serial baseline.** Because every iteration performs a fixed number of
  floating-point operations, the total work — and hence the serial time — is
  proportional to the sum of the iteration counts over all pixels. The per-pixel
  cost varies, but in the serial case the execution order is irrelevant.

- **OpenMP (shared memory).** A `parallel for` over the pixel grid should scale
  well *up to the cores of a single node*, bounded by **Amdahl's law** (the small
  serial fraction: setup, allocation). The key experiment is **scheduling**:
  with a load this irregular, `static` block scheduling is expected to leave
  workers idle, while `dynamic`/`guided` should recover most of the imbalance at
  the price of some scheduling overhead.

- **MPI (distributed memory).** Processes do not share memory and communicate by
  explicit messages, potentially *across nodes*. Two consequences are expected:
  (1) static decompositions (block, cyclic) pay **no** per-step communication but
  inherit the imbalance; (2) a **dynamic master–worker** scheme rebalances the
  load at the cost of request/response messages. Since computing a chunk vastly
  outweighs the cost of requesting one, the dynamic scheme is expected to win —
  this is the project's central comparison. Genuine inter-node communication cost
  only appears on a real cluster, not when all ranks share one machine's RAM.

- **CUDA (GPU).** Thousands of lightweight threads map one-to-one to pixels. The
  bottleneck is expected to shift from load balancing to **warp divergence** and
  to **host↔device transfer** of the result. Occupancy and block/grid geometry
  become the tuning levers.

---

## Paradigms under study

| Model | Memory | Scope | Role |
|---|---|---|---|
| **Serial** (baseline) | — | 1 core | Reference `T(1)` for speedup/efficiency |
| **OpenMP** | Shared | Intra-node threads | Parallelism over the cores of one node |
| **MPI** (OpenMPI) | Distributed | Inter-node processes | Scaling across multiple cluster nodes |
| **CUDA** | Device (GPU) | Thousands of GPU threads | Offloading the iteration kernel |
| **Hybrid** *(optional)* | Mixed | MPI+OpenMP / MPI+CUDA | Multiple nodes *and* cores/GPUs per node, time permitting |

### Domain decomposition strategies

Because the load is imbalanced, the partitioning strategy matters more than the
implementation details. The following are compared:

- **Block (static):** contiguous rows per worker. Simple, but prone to strong
  imbalance — whoever gets the interior region does far more work.
- **Cyclic / striped (static):** interleaved rows across workers. Balances the
  load much better at zero communication cost.
- **Dynamic / master–worker (MPI):** the master hands out work chunks on demand.
  Near-ideal balancing, but introduces communication overhead. This is the
  central comparison of the project.

On the **GPU** the analogue of the decomposition choice is the *thread↔pixel
mapping* — the block geometry — and the conclusion inverts. A warp runs in
lockstep and writes global memory fastest when its lanes touch contiguous
addresses, so mapping **adjacent** pixels to a warp (horizontal, warp-aligned
blocks) both minimizes divergence and coalesces the writes; the *cyclic*
interleaving that balances CPU workers is instead the worst case here. Same
irregularity, mirrored. The CUDA block-shape sweep (`MANDEL_BLOCK`) probes
exactly this trade-off.

---

## Metrics

Metrics are organized in **three levels**, mirroring the report. Keeping them
separate is what makes the three paradigms directly comparable.

- **Level 1 — Serial baseline metrics** (report §3.3): computed once, from the
  serial run. Deterministic (input-only, architecture-independent) except for
  the timing.
- **Level 2 — Common parallel metrics** (report §4.2): defined on times only,
  therefore identical across OpenMP, MPI and CUDA.
- **Level 3 — Paradigm-specific metrics** (report §4.3–4.5): measure the
  mechanism of each model; they have no meaning outside it.

### Level 1 — Serial baseline metrics

The reference artifact is the **escape-time matrix** `n(r, c)` — the iteration
count of each pixel — *not* the rendered image. All work metrics derive from it.

- **Wall-clock time** `T(1)`: compute part only (allocation of the matrix
  included, I/O excluded), monotonic clock, minimum over repetitions as the
  least-noise-contaminated estimator; median and mean also reported.
- **Total work** `W = Σ_r Σ_c n(r, c)` — the sum of iteration counts over all
  pixels. This is the exact computational-cost map.
- **FLOP model**: each iteration performs `φ = 8` FLOP (4 multiplications + 4
  additions/subtractions, counting the doubling `2·z_re` as a multiplication).
  Total FLOP `F = φ · W`; throughput `Π = F / (T(1) · 1e9)` in GFLOP/s.
  FLOP counting is **analytical** (derived from `n(r, c)` after the run), not
  instrumented at runtime.
- **Per-row work** `w_r = Σ_c n(r, c)` — the work metric is the **iteration
  count per row, not the per-row time** (per-row timing would inject systematic
  overhead noise). Mean `w̄ = W / H`, standard deviation `σ_w`.
- **Imbalance indices** (both dimensionless, computed from `{w_r}`):
  - `CoV = σ_w / w̄` — overall dispersion of the load; signals whether **dynamic
    scheduling** is worthwhile. `CoV = 0` is perfectly uniform.
  - `λ = max_r w_r / w̄` — peak-to-mean ratio at **row** granularity.

  **Granularity matters** (report §3.3.3). The row-level `λ` gives the
  *intrinsic* bound `S ≤ H / λ` (the heaviest row is indivisible). The bound
  `S ≤ P / λ_P` for a static **contiguous block** decomposition uses instead the
  *block-level* `λ_P = max_p W_p / W̄`, where `W_p` is the summed work of the
  block assigned to processor `p` and `W̄ = W / P`. Since aggregating rows
  averages out peaks, `λ_P ≤ λ`, with equality only when `P = H`. When computing
  a predicted bound, be explicit about which `λ` is used.

> **Baseline invariant.** The scaling reference is the *bare* kernel:
> `USE_PRUNING` and `USE_SYMMETRY` are **disabled** so that `n(r, c)` reflects
> the real iteration work. With pruning on, interior points store `max_iter`
> without iterating and `W` overstates the actual work — fine for a speed demo,
> but it masks the load balance and must never back the scaling numbers.

### Level 2 — Common parallel metrics (times only)

- **Speedup** `S(p) = T(1) / T(p)`; **efficiency** `E(p) = S(p) / p`. `T(1)` and
  `T(p)` must time the *same* code region (allocation included).
- **Amdahl (strong scaling):** fixed problem size; `S(p) ≤ 1 / ((1−P) + P/p)`.
  Here the serial fraction `1−P` is tiny (allocation and compute both scale
  `O(W·H)`), so Amdahl gives a near-ideal *reference ceiling* but does not
  explain the shortfalls.
- **Karp–Flatt** experimental serial fraction
  `e(p) = (1/S(p) − 1/p) / (1 − 1/p)`. It absorbs *all* non-ideal effects
  (imbalance, synchronization, communication) into one number. Its **trend** is
  the diagnostic: constant `e(p)` ⇒ genuine serial fraction; **rising** `e(p)`
  ⇒ overhead growing with `p` (load imbalance / communication). Given the
  negligible nominal serial fraction, a rising `e(p)` is the quantitative
  signature of the project's thesis.
- **Weak scaling (Gustafson):** problem size grown with `p`. **Caveat:** on
  Mandelbrot the work does *not* grow linearly with the pixel count — it depends
  on how many new pixels fall in the interior — so weak-scaling curves are read
  with this in mind, not as textbook Gustafson.
- **Predicted vs measured:** `λ`/`CoV` (Level 1, deterministic) predict the
  imbalance *before* any parallel code exists; `e(p)` measures its consequence
  *after*. Comparing the two is the analytical spine of the parallel sections.

### Level 3 — Paradigm-specific metrics (minimum required)

The following are the **minimum** each paradigm must report; add more where they
sharpen the analysis.

- **OpenMP** — for each `schedule` (`static`, `dynamic`, `guided`) and chunk
  size: `S(p)`, `E(p)`, `e(p)`; the scheduling that best recovers the imbalance
  predicted by `CoV`. *(Optional: per-thread work spread, chunk-size sweep.)*
- **MPI** — communication time vs compute time (fraction of `T(p)` spent in
  `scatter`/`gather`/messages); block vs cyclic vs dynamic master–worker;
  behaviour across ≥2 nodes. *(Optional: message counts, master contention.)*
- **CUDA** — effective throughput (GFLOP/s) vs theoretical peak; **warp
  divergence** as a *deterministic proxy from the escape-time matrix* (no
  profiler required, with block-aware and cyclic-worst-case variants); occupancy
  (runtime API); host↔device transfer time (CUDA events). The block/grid
  geometry is swept at runtime via `MANDEL_BLOCK`; the canonical kernel is
  double-precision for checksum parity with the CPU baseline. *(Optional: an
  FP32 throughput variant, shared-memory variants.)*

> **CUDA speedup denominator.** `S = T(1) / T_gpu` compares a single GPU against
> a single CPU core; state this explicitly and log the CPU baseline used. It is
> not "speedup over `p` cores" as in OpenMP/MPI — this is the classic way GPU
> Mandelbrot numbers get inflated.

> **Cross-hardware caveat.** If MPI is benchmarked on the multi-node cluster
> while OpenMP runs on a single node, their `T(1)` may come from different CPUs;
> absolute speedups are then not directly comparable. Per the report, trends —
> not absolute values — are what count.

### Result dataset — one file per job, merged offline

**Why not a single shared CSV.** Every paradigm is benchmarked on the *same
cluster* precisely so the comparison rests on one underlying architecture and the
numbers are as free of noise and confounds as possible. That same setting makes a
shared, append-to CSV unsafe: many `sbatch` jobs (OpenMP, MPI, CUDA) can finish
simultaneously, and concurrent appends interleave or corrupt rows — atomic append
is not guaranteed on the cluster's network filesystem (NFS/Lustre). A corrupted
dataset is itself a confound. So instead:

- **Each job writes its own file** `mandelbrot/data/run_${SLURM_JOB_ID}.csv`
  (header + its single result row, never appended to). This is lock-free by
  construction — no `flock`, no interleaving.
- **Binaries write only intra-run measurements** — everything computable from one
  execution: `T_*`, `W`, `lambda_row`, `lambda_block`, `cov`, `gflops`, the
  paradigm-specific fields, `checksum`. Per-job files may be partial: a serial run
  omits the CUDA/MPI columns, and that is fine.
- **The relational metrics are computed offline, in Python.** `speedup`,
  `efficiency` and `karp_flatt_e` need `T(1)` *and* `T(p)`; a single binary only
  ever sees its own `T(p)`. They are left empty by the binaries.
- **One Python merge stage** concatenates every `data/run_*.csv` (resolving the
  concurrency by construction), then computes the relational metrics on the union
  — matching each parallel run to its `T(1)` baseline measured on the *same
  hardware* (same resolution, `max_iter`), which is what keeps `S(p)` a clean
  comparison rather than a cross-machine artefact.

The **canonical schema below is the output of that merge** (missing columns filled
empty → a sparse table), so the plotting script stays paradigm-agnostic:

```
paradigm,schedule,p,nodes,resolution,max_iter,pruning,symmetry,
T_min,T_median,T_mean,W,lambda_row,lambda_block,cov,
speedup,efficiency,karp_flatt_e,comm_fraction,gflops,
occupancy,warp_divergence,transfer_time,checksum
```

- `paradigm` ∈ {serial, openmp, mpi, cuda, hybrid}; `schedule` ∈
  {"-", static, dynamic, guided, block, cyclic, master_worker}.
- `p` = threads / ranks / "1" for serial (CUDA: use "1" and record geometry
  separately or in `schedule`).
- `resolution` as `WxH` (e.g. `1024x1024`); `pruning`/`symmetry` ∈ {0, 1}.
- Fields not applicable to a paradigm are left empty (e.g. `comm_fraction` for
  serial/OpenMP; `occupancy`/`warp_divergence`/`transfer_time` for non-CUDA;
  `lambda_block` when `p < 2`; the relational columns until the merge fills them).
- `checksum` = FNV-1a of the escape-time matrix; identical inputs must yield an
  identical checksum.

---

## Test plan

**Experimental variables**

- **Resolution:** e.g. 512², 1024², 2048², 4096².
- **`max_iter`:** e.g. 100, 1000, 5000 — more iterations increase the imbalance.
- **Degree of parallelism:** number of OpenMP threads; number of MPI ranks and
  their distribution across nodes; CUDA block/grid geometry.

**Correctness**

For each configuration, the output is compared against the serial baseline via a
checksum of the iteration matrix. Two correct implementations produce an
identical result **provided floating-point contraction is disabled** (compiler
flags `-ffp-contract=off`, and `--fmad=false` for CUDA): without this, fused
multiply-add can flip the iteration count by ±1 on pixels near the escape
boundary. Also fixed for stability: `PALETTE_RANGE=256` (colour stability across
`max_iter`). This reproducibility requirement is itself part of the study.

**Methodology**

- Each configuration is run several times, reporting the minimum (least
  noise-contaminated estimator of compute time) alongside median and mean.
  The `--repeat N` flag drives this.
- Timing covers the compute part only; I/O is excluded.
- Official measurements are non-interactive scheduler jobs with explicit
  resource requests, for reproducibility and comparability.
- Result datasets (CSV, schema above) are saved for generating the scaling plots.

---

## Roadmap

- **Phase 0 — Setup and baseline.** *(implemented)* Sources organized into the
  cluster structure; correct, instrumented serial baseline (timing, image
  output, checksum, load-imbalance profiling: `W`, `w_r`, `λ`, `CoV`).
- **Phase 1 — Shared memory (OpenMP).** *(implemented)* Parallelize the loop over
  pixels; compare `static` / `dynamic` / `guided` scheduling against the imbalance.
- **Phase 2 — Distributed memory (MPI).** *(implemented)* Block and cyclic static
  decomposition; dynamic master–worker scheme; image gather and
  communication-overhead measurement.
- **Phase 3 — GPU (CUDA).** *(implemented; GPU correctness gate pending the first
  cluster run)* Iteration kernel, thread↔pixel mapping, host↔device transfer,
  block/grid tuning and occupancy.
- **Phase 4 — Hybrid** *(optional, not started)*. MPI+OpenMP / MPI+CUDA, time
  permitting.
- **Phase 5 — Analysis and report.** *(pending the cluster campaign; the merge
  and plotting scripts are in place)* Full benchmark campaign, scaling plots,
  discussion of results.

---

## Repository structure

```
parallel_computing_project/
│
├── .gitignore
├── README.md
│
└── mandelbrot/                  <-- First experiment (others may follow)
    │
    ├── src/                     <-- Sources (serial, OpenMP, MPI, CUDA) + Makefile
    ├── job_sbatch/              <-- Scheduler scripts (.sh): one sweep per paradigm
    ├── analysis/                <-- Python merge + plotting (paradigm-agnostic)
    ├── job_logs/                <-- Scheduler output logs (git-ignored)
    │   └── .gitkeep
    └── data/                    <-- Per-job CSVs (run_*.csv, versioned);
        └── .gitkeep                 images/figures git-ignored
```

The root is a container for multiple experiments: `mandelbrot/` is the first, and
other fractals or benchmarks can be added as sibling folders sharing the same
internal structure.

---

## Build

Sources are plain C++ (plus OpenMP / MPI / CUDA per variant); build with the
provided `Makefile` inside `mandelbrot/`. Toolchain and compilation must happen
**on the cluster**, since timings are only meaningful there and must all come
from the same machine. Compiled binaries, logs and generated data are excluded
from version control.

---

## Reproducibility

The report links this repository at an **immutable tag** (e.g. `v1.0-report`),
not the mutable `main` branch, so that the code backing the reported checksums
and timings is exactly recoverable. When citing the repo in the report, pin the
tag or commit SHA.

---

## Acknowledgements

Computational resources provided by hpc-ReGAInS@DISCo, a MUR Department of
Excellence Project within the Department of Informatics, Systems and
Communication at the University of Milan-Bicocca (https://www.disco.unimib.it).