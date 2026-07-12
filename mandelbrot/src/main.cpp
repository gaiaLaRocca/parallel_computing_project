#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "mandelbrot.hpp"
#include "metrics.hpp"

namespace {

struct Options {
    int width = 1024;
    int height = 1024;
    int max_iter = 1000;
    int repetitions = 1;
    Viewport view{-2.0, 0.5, -1.25, 1.25};
    std::string ppm_path;
    std::string pgm_path;
    std::string raw_path;
    std::string row_stats_path;
};

struct TimingSummary {
    double min;
    double median;
    double mean;
};

Options parse_arguments(int argc, char** argv);
void print_usage(const char* program);
MandelbrotImage run_timed_repetitions(const Options& opts, TimingSummary& timing);
TimingSummary summarise(std::vector<double> samples);
double elapsed_seconds(std::chrono::steady_clock::time_point start);
void print_metrics(const MandelbrotImage& img, const TimingSummary& timing);
void print_build_configuration();
bool write_requested_outputs(const MandelbrotImage& img, const Options& opts);
bool write_if_requested(bool (*writer)(const MandelbrotImage&, const std::string&),
                        const MandelbrotImage& img, const std::string& path);

}  // namespace

int main(int argc, char** argv) {
    const Options opts = parse_arguments(argc, argv);

    TimingSummary timing{};
    const MandelbrotImage img = run_timed_repetitions(opts, timing);

    print_build_configuration();
    print_metrics(img, timing);

    if (!opts.row_stats_path.empty()) {
        const RowWorkProfile profile = build_row_work_profile(img);
        if (!save_row_profile_csv(profile, opts.row_stats_path)) {
            std::cerr << "error: cannot write " << opts.row_stats_path << "\n";
        }
    }

    return write_requested_outputs(img, opts) ? EXIT_SUCCESS : EXIT_FAILURE;
}

namespace {

Options parse_arguments(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const bool has_value = (i + 1 < argc);
        if (std::strcmp(argv[i], "--width") == 0 && has_value) {
            opts.width = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--height") == 0 && has_value) {
            opts.height = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-iter") == 0 && has_value) {
            opts.max_iter = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--repeat") == 0 && has_value) {
            opts.repetitions = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--ppm") == 0 && has_value) {
            opts.ppm_path = argv[++i];
        } else if (std::strcmp(argv[i], "--pgm") == 0 && has_value) {
            opts.pgm_path = argv[++i];
        } else if (std::strcmp(argv[i], "--raw") == 0 && has_value) {
            opts.raw_path = argv[++i];
        } else if (std::strcmp(argv[i], "--row-stats") == 0 && has_value) {
            opts.row_stats_path = argv[++i];
        } else {
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return opts;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program
              << " [--width N] [--height N] [--max-iter N] [--repeat N]"
                 " [--ppm FILE] [--pgm FILE] [--raw FILE]"
                 " [--row-stats FILE]\n";
}

// Timing excludes all I/O. The image of the last repetition is returned; every
// repetition is deterministic, so which one we keep is irrelevant.
MandelbrotImage run_timed_repetitions(const Options& opts, TimingSummary& timing) {
    std::vector<double> samples;
    samples.reserve(opts.repetitions);

    MandelbrotImage img;
    for (int i = 0; i < opts.repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        img = compute_mandelbrot(opts.view, opts.width, opts.height, opts.max_iter);
        samples.push_back(elapsed_seconds(start));
    }

    timing = summarise(std::move(samples));
    return img;
}

// The minimum is the least noisy estimator of the true compute time: system
// noise can only add time, never remove it.
TimingSummary summarise(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());

    const std::size_t n = samples.size();
    const double median = (n % 2 == 1)
                              ? samples[n / 2]
                              : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);

    double sum = 0.0;
    for (double s : samples) {
        sum += s;
    }
    return {samples.front(), median, sum / n};
}

double elapsed_seconds(std::chrono::steady_clock::time_point start) {
    const std::chrono::duration<double> delta =
        std::chrono::steady_clock::now() - start;
    return delta.count();
}

void print_build_configuration() {
    std::cout << "pruning        " << (pruning_enabled() ? "on" : "off") << "\n";
    std::cout << "symmetry       " << (symmetry_enabled() ? "on" : "off") << "\n";
}

// Throughput is derived from the escape-time array. It is an exact measure of
// arithmetic work only for the bare kernel; with pruning or symmetry enabled
// the array counts iterations that were never executed, so the figure is
// nominal and is labelled as such.
void print_metrics(const MandelbrotImage& img, const TimingSummary& timing) {
    const RowWorkProfile profile = build_row_work_profile(img);
    const LoadBalanceStats lb = compute_load_balance(profile);
    const double gflops = throughput_gflops(profile.total_iterations, timing.min);
    const bool work_is_exact = !pruning_enabled() && !symmetry_enabled();

    std::cout << std::fixed;
    std::cout << std::setprecision(6);
    std::cout << "time_min_s     " << timing.min << "\n";
    std::cout << "time_median_s  " << timing.median << "\n";
    std::cout << "time_mean_s    " << timing.mean << "\n";
    std::cout << "checksum       " << std::hex << checksum(img) << std::dec << "\n";
    std::cout << "total_iter     " << profile.total_iterations
              << (work_is_exact ? "" : "  (nominal)") << "\n";
    std::cout << std::setprecision(3);
    std::cout << "gflops         " << gflops
              << (work_is_exact ? "" : "  (nominal)") << "\n";
    std::cout << std::setprecision(1);
    std::cout << "row_iter_mean  " << lb.mean << "\n";
    std::cout << "row_iter_std   " << lb.stddev << "\n";
    std::cout << "row_iter_min   " << lb.min << "\n";
    std::cout << "row_iter_max   " << lb.max << "\n";
    std::cout << std::setprecision(4);
    std::cout << "row_cov        " << lb.coefficient_of_variation << "\n";
    std::cout << "row_imbalance  " << lb.imbalance_ratio << "\n";
}

bool write_requested_outputs(const MandelbrotImage& img, const Options& opts) {
    return write_if_requested(save_raw, img, opts.raw_path) &&
           write_if_requested(save_pgm, img, opts.pgm_path) &&
           write_if_requested(save_ppm, img, opts.ppm_path);
}

bool write_if_requested(bool (*writer)(const MandelbrotImage&, const std::string&),
                        const MandelbrotImage& img, const std::string& path) {
    if (path.empty()) {
        return true;
    }
    if (writer(img, path)) {
        return true;
    }
    std::cerr << "error: cannot write " << path << "\n";
    return false;
}

}  // namespace
