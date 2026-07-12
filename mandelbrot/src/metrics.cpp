#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

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

double lambda_block(const RowWorkProfile& profile, int num_blocks) {
    const auto& rows = profile.iterations_per_row;
    const int height = static_cast<int>(rows.size());
    if (num_blocks < 1 || height == 0) {
        return METRIC_NA;
    }

    // Contiguous balanced split: block b owns rows [b*H/P, (b+1)*H/P), so block
    // sizes differ by at most one row. This is the decomposition the block-level
    // bound assumes; the heaviest block dictates the makespan.
    std::int64_t max_block = 0;
    for (int b = 0; b < num_blocks; ++b) {
        const int begin = static_cast<int>(static_cast<std::int64_t>(b) * height / num_blocks);
        const int end = static_cast<int>(static_cast<std::int64_t>(b + 1) * height / num_blocks);
        std::int64_t block_work = 0;
        for (int r = begin; r < end; ++r) {
            block_work += rows[r];
        }
        max_block = std::max(max_block, block_work);
    }

    const double mean_block =
        static_cast<double>(profile.total_iterations) / num_blocks;
    return static_cast<double>(max_block) / mean_block;
}

double warp_divergence_proxy(const MandelbrotImage& img, int warp_size) {
    if (warp_size < 1 || img.data.empty()) {
        return METRIC_NA;
    }

    const std::size_t total = img.data.size();
    double variance_sum = 0.0;
    std::size_t warp_count = 0;

    for (std::size_t base = 0; base < total; base += warp_size) {
        const std::size_t end = std::min(base + static_cast<std::size_t>(warp_size), total);
        const std::size_t n = end - base;

        double mean = 0.0;
        for (std::size_t i = base; i < end; ++i) {
            mean += img.data[i];
        }
        mean /= static_cast<double>(n);

        double var = 0.0;
        for (std::size_t i = base; i < end; ++i) {
            const double diff = static_cast<double>(img.data[i]) - mean;
            var += diff * diff;
        }
        variance_sum += var / static_cast<double>(n);
        ++warp_count;
    }

    return variance_sum / static_cast<double>(warp_count);
}

const char* paradigm_name(Paradigm paradigm) {
    switch (paradigm) {
        case Paradigm::Serial: return "serial";
        case Paradigm::OpenMP: return "openmp";
        case Paradigm::MPI:    return "mpi";
        case Paradigm::CUDA:   return "cuda";
        case Paradigm::Hybrid: return "hybrid";
    }
    return "unknown";
}

namespace {

// A METRIC_NA double serialises to an empty field; a real value keeps enough
// significant digits that two runs with identical inputs are byte-comparable.
std::string field(double value) {
    if (std::isnan(value)) {
        return std::string();
    }
    std::ostringstream os;
    os << std::setprecision(10) << value;
    return os.str();
}

}  // namespace

bool write_run_csv(const RunRecord& record, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }

    // Canonical schema. The relational columns are present but empty here; the
    // Python merge fills them from T(1). Order must match the merged dataset.
    out << "paradigm,schedule,p,nodes,resolution,max_iter,pruning,symmetry,"
           "T_min,T_median,T_mean,W,lambda_row,lambda_block,cov,"
           "speedup,efficiency,karp_flatt_e,comm_fraction,gflops,"
           "occupancy,warp_divergence,transfer_time,checksum\n";

    out << paradigm_name(record.paradigm) << ','
        << record.schedule << ','
        << record.p << ','
        << record.nodes << ','
        << record.width << 'x' << record.height << ','
        << record.max_iter << ','
        << (record.pruning ? 1 : 0) << ','
        << (record.symmetry ? 1 : 0) << ','
        << field(record.t_min) << ','
        << field(record.t_median) << ','
        << field(record.t_mean) << ','
        << record.total_work << ','
        << field(record.lambda_row) << ','
        << field(record.lambda_block) << ','
        << field(record.cov) << ','
        << /* speedup      */ ','
        << /* efficiency   */ ','
        << /* karp_flatt_e */ ','
        << field(record.comm_fraction) << ','
        << field(record.gflops) << ','
        << field(record.occupancy) << ','
        << field(record.warp_divergence) << ','
        << field(record.transfer_time) << ','
        << record.checksum << '\n';

    return out.good();
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
