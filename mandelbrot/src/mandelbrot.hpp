#pragma once

#include <vector>

// Rectangular region of the complex parameter plane to be sampled.
struct Viewport {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
};

// Escape-time counts, row-major. Row 0 corresponds to y_max.
// A pixel that never escapes stores max_iter.
struct MandelbrotImage {
    int width;
    int height;
    int max_iter;
    std::vector<int> data;
};

// Pure function: the result depends only on the arguments.
// Sampling happens at pixel centres, which makes the grid exactly symmetric
// about the real axis whenever the viewport is.
MandelbrotImage compute_mandelbrot(const Viewport& view, int width, int height,
                                   int max_iter);

// Report which compile-time optimisations this translation unit was built with.
// Both are off by default: the bare kernel is the scaling baseline.
bool pruning_enabled();
bool symmetry_enabled();

// The parallel programming model this kernel translation unit implements.
// Reported by the kernel (not chosen at runtime) so a run can never mislabel
// itself: the driver reads it to tag the result row and to gate the
// paradigm-specific metrics.
enum class Paradigm { Serial, OpenMP, MPI, CUDA, Hybrid };

// Set at compile time by the kernel source. The serial kernel returns Serial;
// each parallel kernel overrides it with its own model.
Paradigm kernel_paradigm();

// Seconds spent in interprocess communication during the last
// compute_mandelbrot call (MPI gather/scatter/messages). Only the MPI kernel
// measures a non-zero value; the serial and OpenMP kernels return 0. The shared
// driver divides it by T(p) to fill the MPI comm_fraction metric.
double kernel_comm_seconds();

// Occupancy of the CUDA launch configuration: resident threads per SM over the
// hardware maximum, from cudaOccupancyMaxActiveBlocksPerMultiprocessor. Only
// the CUDA kernel reports a non-zero value; the driver copies it into the
// record's occupancy field.
double kernel_occupancy();

// Seconds of host<->device transfer during the last compute_mandelbrot call,
// measured with CUDA events around the device->host copy of the escape-time
// matrix. Only the CUDA kernel reports a non-zero value; the driver copies it
// into the record's transfer_time field.
double kernel_transfer_seconds();

// The CUDA launch block shape (blockDim.x, blockDim.y) actually used by the last
// run. Only the CUDA kernel reports non-zero values; the other kernels return 0,
// which the driver reads as "no block shape" and falls back to the flat
// row-major warp-divergence proxy. Reporting the real shape here (rather than
// re-parsing MANDEL_BLOCK in the driver) keeps the CSV's warp_divergence aligned
// with the geometry the kernel was actually launched with.
int kernel_block_x();
int kernel_block_y();
