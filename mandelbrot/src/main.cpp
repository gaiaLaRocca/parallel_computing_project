#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "mandelbrot.hpp"
#include "metrics.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace {

struct Options {
    int width = 1024;
    int height = 1024;
    int max_iter = 1000;
    int repetitions = 1;
    Viewport view{-2.0, 0.5, -1.25, 1.25};
    // Zoom window for the self-similarity figures: a centre in the complex
    // plane plus the width of the sampled real interval. A non-positive span
    // keeps the default viewport above, so every benchmark run samples exactly
    // the region it sampled before and its metrics stay comparable.
    double center_real = 0.0;
    double center_imag = 0.0;
    double span = 0.0;
    std::string ppm_path;
    std::string pgm_path;
    std::string raw_path;
    std::string row_stats_path;
    // Result-row metadata. The kernel reports the paradigm; the job script
    // supplies the degree of parallelism and topology it actually requested,
    // since the shared driver has no paradigm headers to query them itself.
    std::string csv_path;
    std::string schedule = "-";
    int parallelism = 1;  // threads / ranks; 1 for serial
    int nodes = 1;
};

struct TimingSummary {
    double min;
    double median;
    double mean;
};

Options parse_arguments(int argc, char** argv);
void print_usage(const char* program);
Viewport centred_viewport(double center_real, double center_imag, double span,
                          int width, int height);
MandelbrotImage run_timed_repetitions(const Options& opts, TimingSummary& timing);
TimingSummary summarise(std::vector<double> samples);
double elapsed_seconds(std::chrono::steady_clock::time_point start);
void print_metrics(const MandelbrotImage& img, const TimingSummary& timing);
RunRecord build_run_record(const MandelbrotImage& img, const TimingSummary& timing,
                           const Options& opts);
void print_build_configuration(const Viewport& view);
bool write_requested_outputs(const MandelbrotImage& img, const Options& opts);
bool write_if_requested(bool (*writer)(const MandelbrotImage&, const std::string&),
                        const MandelbrotImage& img, const std::string& path);

}  // namespace

int main(int argc, char** argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    const int rank = 0;
#endif

    const Options opts = parse_arguments(argc, argv);

    TimingSummary timing{};
    const MandelbrotImage img = run_timed_repetitions(opts, timing);

    // Only rank 0 holds the assembled matrix and performs all I/O; the other
    // ranks cooperated inside the kernel (block compute + gather) and now exit.
    int status = EXIT_SUCCESS;
    if (rank == 0) {
        print_build_configuration(opts.view);
        print_metrics(img, timing);

        if (!opts.row_stats_path.empty()) {
            const RowWorkProfile profile = build_row_work_profile(img);
            if (!save_row_profile_csv(profile, opts.row_stats_path)) {
                std::cerr << "error: cannot write " << opts.row_stats_path << "\n";
            }
        }

        if (!opts.csv_path.empty()) {
            const RunRecord record = build_run_record(img, timing, opts);
            if (!write_run_csv(record, opts.csv_path)) {
                std::cerr << "error: cannot write " << opts.csv_path << "\n";
            }
        }

        status = write_requested_outputs(img, opts) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return status;
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
        } else if ((std::strcmp(argv[i], "--max-iter") == 0 ||
                    std::strcmp(argv[i], "--max_iter") == 0) && has_value) {
            opts.max_iter = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--resolution") == 0 && has_value) {
            std::sscanf(argv[++i], "%dx%d", &opts.width, &opts.height);
        } else if (std::strcmp(argv[i], "--repeat") == 0 && has_value) {
            opts.repetitions = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--center-re") == 0 && has_value) {
            opts.center_real = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--center-im") == 0 && has_value) {
            opts.center_imag = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--span") == 0 && has_value) {
            opts.span = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--csv") == 0 && has_value) {
            opts.csv_path = argv[++i];
        } else if (std::strcmp(argv[i], "--schedule") == 0 && has_value) {
            opts.schedule = argv[++i];
        } else if (std::strcmp(argv[i], "--p") == 0 && has_value) {
            opts.parallelism = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--nodes") == 0 && has_value) {
            opts.nodes = std::atoi(argv[++i]);
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

    if (opts.span > 0.0) {
        opts.view = centred_viewport(opts.center_real, opts.center_imag, opts.span,
                                     opts.width, opts.height);
    }
    return opts;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program
              << " [--width N] [--height N] [--resolution WxH] [--max-iter N]"
                 " [--repeat N] [--p N] [--nodes N] [--schedule NAME]"
                 " [--center-re X] [--center-im Y] [--span S]"
                 " [--csv FILE] [--ppm FILE] [--pgm FILE] [--raw FILE]"
                 " [--row-stats FILE]\n";
}

// The imaginary half-extent follows the aspect ratio, so pixels stay square
// (dx = dy) at every zoom level and the shapes are not stretched.
Viewport centred_viewport(double center_real, double center_imag, double span,
                          int width, int height) {
    const double half_real = 0.5 * span;
    const double half_imag = half_real * height / width;
    return {center_real - half_real, center_real + half_real,
            center_imag - half_imag, center_imag + half_imag};
}

// Timing excludes all I/O. The image of the last repetition is returned; every
// repetition is deterministic, so which one we keep is irrelevant.
MandelbrotImage run_timed_repetitions(const Options& opts, TimingSummary& timing) {
    std::vector<double> samples;
    samples.reserve(opts.repetitions);

    MandelbrotImage img;
    for (int i = 0; i < opts.repetitions; ++i) {
#ifdef USE_MPI
        // Align every rank's start so the measured region is the parallel
        // compute + gather, not a staggered launch.
        MPI_Barrier(MPI_COMM_WORLD);
#endif
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

// The viewport is echoed because it is now a runtime choice: a zoomed run is
// otherwise indistinguishable from a default one in the log.
void print_build_configuration(const Viewport& view) {
    std::cout << "pruning        " << (pruning_enabled() ? "on" : "off") << "\n";
    std::cout << "symmetry       " << (symmetry_enabled() ? "on" : "off") << "\n";
    // Deep zooms need far more digits than the default 6: at span 1e-6 the
    // bounds would print identical.
    const std::streamsize digits = std::cout.precision();
    std::cout << std::setprecision(15);
    std::cout << "viewport       [" << view.x_min << ", " << view.x_max << "] x ["
              << view.y_min << ", " << view.y_max << "]\n";
    std::cout.precision(digits);
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

    // CUDA-only: the divergence proxy under the launched block geometry (the
    // value that reaches the CSV), the scattered/cyclic worst case for the
    // GPU-vs-MPI contrast, and the interpretable wasted-lane fraction. These
    // land in the sweep log; occupancy/transfer are in the CSV, not reprinted.
    if (kernel_paradigm() == Paradigm::CUDA) {
        const int bx = kernel_block_x();
        const int by = kernel_block_y();
        std::cout << "block_shape    " << bx << "x" << by << "\n";
        std::cout << std::setprecision(4);
        std::cout << "warp_div_block " << warp_divergence_proxy_blocked(img, bx, by) << "\n";
        std::cout << "warp_div_scat  " << warp_divergence_proxy_scattered(img) << "\n";
        std::cout << "warp_wasted    " << warp_wasted_fraction_blocked(img, bx, by) << "\n";
    }
}

// Assemble the canonical result row from a single run. Only intra-run metrics
// are filled; the relational columns stay empty (the binary never sees T(1)).
// Paradigm-specific fields are gated on the kernel's own paradigm, so a serial
// or OpenMP build leaves the CUDA/MPI columns blank.
RunRecord build_run_record(const MandelbrotImage& img, const TimingSummary& timing,
                           const Options& opts) {
    const RowWorkProfile profile = build_row_work_profile(img);
    const LoadBalanceStats lb = compute_load_balance(profile);
    const Paradigm paradigm = kernel_paradigm();

    RunRecord r;
    r.paradigm = paradigm;
    r.schedule = opts.schedule;
    r.p = opts.parallelism;
    r.nodes = opts.nodes;
    r.width = img.width;
    r.height = img.height;
    r.max_iter = img.max_iter;
    r.pruning = pruning_enabled();
    r.symmetry = symmetry_enabled();

    r.t_min = timing.min;
    r.t_median = timing.median;
    r.t_mean = timing.mean;

    r.total_work = profile.total_iterations;
    r.lambda_row = lb.imbalance_ratio;
    r.cov = lb.coefficient_of_variation;
    r.gflops = throughput_gflops(profile.total_iterations, timing.min);

    // lambda_block predicts the imbalance of a contiguous P-way block split;
    // it is only meaningful for a genuine decomposition (P >= 2).
    if (opts.parallelism >= 2) {
        r.lambda_block = lambda_block(profile, opts.parallelism);
    }

    // Communication fraction is an MPI concept: time in gather/scatter/messages
    // over T(p). kernel_comm_seconds() is the last call's comm time (0 for the
    // other paradigms); timing.min is the reported T(p).
    if (paradigm == Paradigm::MPI && timing.min > 0.0) {
        r.comm_fraction = kernel_comm_seconds() / timing.min;
    }

    // The GPU-only columns: the divergence proxy comes from the matrix (shared
    // code) under the block shape the kernel actually launched, so a block-shape
    // sweep records the divergence of each geometry; occupancy and transfer time
    // come from the CUDA kernel's own accessors.
    if (paradigm == Paradigm::CUDA) {
        r.warp_divergence =
            warp_divergence_proxy_blocked(img, kernel_block_x(), kernel_block_y());
        r.occupancy = kernel_occupancy();
        r.transfer_time = kernel_transfer_seconds();
    }

    r.checksum = checksum(img);
    return r;
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
