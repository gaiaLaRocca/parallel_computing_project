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
