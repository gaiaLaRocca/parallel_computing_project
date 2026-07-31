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

namespace {

// Shared traversal for the block-aware warp metrics. Walk every block_x*block_y
// block of the grid, cut each block into warps of `warp_size` consecutive linear
// thread ids (id = tx + ty*block_x, the hardware's own linearisation), map each
// lane back to its pixel, and accumulate both the within-warp variance and the
// wasted-lane fraction in one pass. Out-of-bounds lanes of edge blocks are
// skipped, so both figures measure fractal-induced divergence, not tiling edges.
struct WarpStats {
    double variance_mean;
    double wasted_mean;
};

WarpStats warp_stats_blocked(const MandelbrotImage& img, int block_x,
                             int block_y, int warp_size) {
    const int width = img.width;
    const int height = img.height;
    const long long threads_per_block =
        static_cast<long long>(block_x) * block_y;
    const int blocks_x = (width + block_x - 1) / block_x;
    const int blocks_y = (height + block_y - 1) / block_y;

    double variance_sum = 0.0;
    double wasted_sum = 0.0;
    long long warp_count = 0;

    std::vector<double> lane;
    lane.reserve(static_cast<std::size_t>(warp_size));

    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            for (long long start = 0; start < threads_per_block;
                 start += warp_size) {
                const long long end =
                    std::min(start + warp_size, threads_per_block);
                lane.clear();
                for (long long id = start; id < end; ++id) {
                    const int col =
                        bx * block_x + static_cast<int>(id % block_x);
                    const int row =
                        by * block_y + static_cast<int>(id / block_x);
                    if (col < width && row < height) {
                        lane.push_back(static_cast<double>(
                            img.data[static_cast<std::size_t>(row) * width +
                                     col]));
                    }
                }
                if (lane.empty()) {
                    continue;
                }

                double mean = 0.0;
                double maximum = lane.front();
                for (double v : lane) {
                    mean += v;
                    maximum = std::max(maximum, v);
                }
                mean /= static_cast<double>(lane.size());

                double var = 0.0;
                for (double v : lane) {
                    const double diff = v - mean;
                    var += diff * diff;
                }
                variance_sum += var / static_cast<double>(lane.size());
                wasted_sum += (maximum > 0.0) ? (1.0 - mean / maximum) : 0.0;
                ++warp_count;
            }
        }
    }

    if (warp_count == 0) {
        return {METRIC_NA, METRIC_NA};
    }
    return {variance_sum / static_cast<double>(warp_count),
            wasted_sum / static_cast<double>(warp_count)};
}

}  // namespace

double warp_divergence_proxy_blocked(const MandelbrotImage& img, int block_x,
                                     int block_y, int warp_size) {
    if (warp_size < 1 || img.data.empty()) {
        return METRIC_NA;
    }
    if (block_x < 1 || block_y < 1) {
        return warp_divergence_proxy(img, warp_size);
    }
    return warp_stats_blocked(img, block_x, block_y, warp_size).variance_mean;
}

double warp_wasted_fraction_blocked(const MandelbrotImage& img, int block_x,
                                    int block_y, int warp_size) {
    if (warp_size < 1 || block_x < 1 || block_y < 1 || img.data.empty()) {
        return METRIC_NA;
    }
    return warp_stats_blocked(img, block_x, block_y, warp_size).wasted_mean;
}

double warp_divergence_proxy_scattered(const MandelbrotImage& img,
                                       int warp_size) {
    if (warp_size < 1 || img.data.empty()) {
        return METRIC_NA;
    }
    const std::size_t total = img.data.size();
    const std::size_t num_warps = total / static_cast<std::size_t>(warp_size);
    if (num_warps == 0) {
        return warp_divergence_proxy(img, warp_size);
    }

    // Warp w owns pixels { w, w + num_warps, ..., w + (warp_size-1)*num_warps }:
    // a stride of num_warps ~ total/warp_size spreads its lanes across the whole
    // image, the maximally scattered assignment. The (w, k) -> w + k*num_warps
    // map is a bijection onto [0, num_warps*warp_size), so every full warp's
    // pixels are disjoint; any tail shorter than a full warp is dropped.
    double variance_sum = 0.0;
    for (std::size_t w = 0; w < num_warps; ++w) {
        double mean = 0.0;
        for (int k = 0; k < warp_size; ++k) {
            mean += img.data[w + static_cast<std::size_t>(k) * num_warps];
        }
        mean /= static_cast<double>(warp_size);

        double var = 0.0;
        for (int k = 0; k < warp_size; ++k) {
            const double diff =
                img.data[w + static_cast<std::size_t>(k) * num_warps] - mean;
            var += diff * diff;
        }
        variance_sum += var / static_cast<double>(warp_size);
    }
    return variance_sum / static_cast<double>(num_warps);
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
