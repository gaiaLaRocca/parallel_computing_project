#pragma once

#include "mandelbrot.hpp"

#include <cstdint>
#include <string>
#include <vector>

// Floating-point operations per escape-time iteration:
// 4 mul + 3 add + 1 sub = 8.
constexpr int FLOPS_PER_ITERATION = 8;

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

// Dump per-row iteration counts to a CSV (row_index, iterations).
// Intended for gnuplot / matplotlib visualisation.
bool save_row_profile_csv(const RowWorkProfile& profile, const std::string& path);
