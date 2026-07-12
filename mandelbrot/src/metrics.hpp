#pragma once

#include "mandelbrot.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// Floating-point operations per escape-time iteration:
// 4 mul + 3 add + 1 sub = 8.
constexpr int FLOPS_PER_ITERATION = 8;

// GPU warp width assumed by the warp-divergence proxy. Matches current NVIDIA
// hardware; the proxy is analytical, so this is the only place it is fixed.
constexpr int WARP_SIZE = 32;

// Sentinel for a metric that this run did not measure. It is serialised as an
// empty CSV field, so the Python merge sees a sparse column and fills it (for
// the relational metrics) or leaves it blank (for another paradigm's fields).
constexpr double METRIC_NA = std::numeric_limits<double>::quiet_NaN();

struct RowWorkProfile {
    std::vector<std::int64_t> iterations_per_row;
    std::int64_t total_iterations;
};

struct LoadBalanceStats {
    double mean;
    double stddev;
    std::int64_t min;
    std::int64_t max;
    double coefficient_of_variation;  // stddev / mean
    double imbalance_ratio;           // max / mean
};

// Sums the escape-time array. This equals the number of iterations actually
// executed ONLY for the bare kernel. With USE_PRUNING, interior points store
// max_iter without ever entering the loop; with USE_SYMMETRY, mirrored rows
// are a memcpy. In both cases the total is nominal, not measured.
RowWorkProfile build_row_work_profile(const MandelbrotImage& img);

LoadBalanceStats compute_load_balance(const RowWorkProfile& profile);

double throughput_gflops(std::int64_t total_iterations, double seconds);

// Block-level peak-to-mean imbalance for a static *contiguous* decomposition
// into `num_blocks` row ranges: lambda_block = max_p W_p / (W / num_blocks),
// where W_p is the summed row work of block p. This is the deterministic bound
// S <= P / lambda_block (README section 3.3.3), distinct from the row-level lambda.
// Since aggregating rows averages out peaks, lambda_block <= lambda_row.
// Returns METRIC_NA for num_blocks < 1.
double lambda_block(const RowWorkProfile& profile, int num_blocks);

// Deterministic proxy for GPU warp divergence, computed from the escape-time
// matrix without a profiler. Threads in a warp run in lockstep and the warp
// only retires when its slowest lane escapes, so the wasted work scales with
// the spread of iteration counts inside a warp. Flattening the matrix in
// row-major order (the natural 1D thread<->pixel mapping) and grouping every
// `warp_size` contiguous pixels into one warp, this returns the mean over all
// warps of the within-warp variance of the iteration counts. Zero means every
// warp is perfectly convergent; larger means more divergence.
double warp_divergence_proxy(const MandelbrotImage& img, int warp_size = WARP_SIZE);

// Human-readable tag written to the CSV `paradigm` column.
const char* paradigm_name(Paradigm paradigm);

// One row of the canonical result dataset. A binary fills the fields it can
// measure from its single execution and leaves the rest at METRIC_NA (doubles)
// or their defaults; the Python merge reconciles the per-job files into the
// full sparse schema and computes the relational columns afterwards.
//
// The relational fields (speedup/efficiency/karp_flatt_e) are intentionally
// absent here: a single binary knows only its own T(p), never T(1), so they
// are always empty in a per-job file and filled by the Python analysis.
struct RunRecord {
    Paradigm paradigm = Paradigm::Serial;
    std::string schedule = "-";
    int p = 1;
    int nodes = 1;
    int width = 0;
    int height = 0;
    int max_iter = 0;
    bool pruning = false;
    bool symmetry = false;

    // Level 1 / Level 2 timing, intra-run.
    double t_min = METRIC_NA;
    double t_median = METRIC_NA;
    double t_mean = METRIC_NA;

    // Level 1 work and imbalance, intra-run.
    std::int64_t total_work = 0;   // W = sum n(r, c)
    double lambda_row = METRIC_NA;
    double lambda_block = METRIC_NA;
    double cov = METRIC_NA;
    double gflops = METRIC_NA;

    // Level 3, paradigm-specific. Left METRIC_NA outside their paradigm.
    double comm_fraction = METRIC_NA;    // MPI: comm time / T(p)
    double occupancy = METRIC_NA;        // CUDA: cudaOccupancy... runtime API
    double warp_divergence = METRIC_NA;  // CUDA: warp_divergence_proxy(img)
    double transfer_time = METRIC_NA;    // CUDA: host<->device, CUDA events

    std::uint64_t checksum = 0;
};

// Write `record` as a standalone per-job CSV: the canonical header followed by
// this single row (the file is truncated, never appended to). One file per job
// is what keeps concurrent sbatch runs from interleaving rows on NFS/Lustre;
// the Python merge concatenates all data/run_*.csv into the single dataset.
bool write_run_csv(const RunRecord& record, const std::string& path);

// Dump per-row iteration counts to a CSV (row_index, iterations).
// Intended for gnuplot / matplotlib visualisation.
bool save_row_profile_csv(const RowWorkProfile& profile, const std::string& path);
