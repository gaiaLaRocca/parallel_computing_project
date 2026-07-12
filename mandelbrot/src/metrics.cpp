#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numeric>

RowWorkProfile build_row_work_profile(const MandelbrotImage& img) {
    RowWorkProfile profile;
    profile.iterations_per_row.resize(img.height, 0);

    for (int row = 0; row < img.height; ++row) {
        std::int64_t sum = 0;
        for (int col = 0; col < img.width; ++col) {
            sum += img.data[static_cast<std::size_t>(row) * img.width + col];
        }
        profile.iterations_per_row[row] = sum;
    }

    profile.total_iterations = std::accumulate(
        profile.iterations_per_row.begin(), profile.iterations_per_row.end(),
        std::int64_t{0});
    return profile;
}

LoadBalanceStats compute_load_balance(const RowWorkProfile& profile) {
    const auto& rows = profile.iterations_per_row;
    const int n = static_cast<int>(rows.size());

    const double mean =
        static_cast<double>(profile.total_iterations) / n;

    double variance_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = static_cast<double>(rows[i]) - mean;
        variance_sum += diff * diff;
    }
    const double stddev = std::sqrt(variance_sum / n);

    const auto [min_it, max_it] = std::minmax_element(rows.begin(), rows.end());

    return {mean, stddev, *min_it, *max_it,
            stddev / mean,                            // CoV
            static_cast<double>(*max_it) / mean};     // imbalance ratio
}

double throughput_gflops(std::int64_t total_iterations, double seconds) {
    return static_cast<double>(total_iterations) * FLOPS_PER_ITERATION /
           (seconds * 1e9);
}

bool save_row_profile_csv(const RowWorkProfile& profile, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << "row,iterations\n";
    for (std::size_t i = 0; i < profile.iterations_per_row.size(); ++i) {
        out << i << "," << profile.iterations_per_row[i] << "\n";
    }
    return out.good();
}
