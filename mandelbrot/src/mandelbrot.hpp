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
