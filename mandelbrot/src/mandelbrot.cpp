#include "mandelbrot.hpp"

#include <cmath>
#include <cstddef>

namespace {

constexpr double ESCAPE_RADIUS_SQ = 4.0;
constexpr double SYMMETRY_TOLERANCE = 1e-9;

#ifdef USE_PRUNING
constexpr double PERIOD_TWO_BULB_RADIUS_SQ = 0.0625;  // (1/4)^2
#endif

bool should_exploit_symmetry(const Viewport& view);
int rows_to_compute(int height, bool symmetric);
int escape_time(double c_real, double c_imag, int max_iter);
void mirror_computed_rows(MandelbrotImage& img, int computed_rows);

#ifdef USE_PRUNING
bool lies_in_main_cardioid(double c_real, double c_imag_sq);
bool lies_in_period_two_bulb(double c_real, double c_imag_sq);
#endif

}  // namespace

bool pruning_enabled() {
#ifdef USE_PRUNING
    return true;
#else
    return false;
#endif
}

bool symmetry_enabled() {
#ifdef USE_SYMMETRY
    return true;
#else
    return false;
#endif
}

Paradigm kernel_paradigm() { return Paradigm::Serial; }

double kernel_comm_seconds() { return 0.0; }

MandelbrotImage compute_mandelbrot(const Viewport& view, int width, int height,
                                   int max_iter) {
    MandelbrotImage img;
    img.width = width;
    img.height = height;
    img.max_iter = max_iter;
    img.data.resize(static_cast<std::size_t>(width) * height);

    const double dx = (view.x_max - view.x_min) / width;
    const double dy = (view.y_max - view.y_min) / height;

    const bool symmetric = should_exploit_symmetry(view);
    const int computed_rows = rows_to_compute(height, symmetric);

    for (int row = 0; row < computed_rows; ++row) {
        const double c_imag = view.y_max - (row + 0.5) * dy;
        for (int col = 0; col < width; ++col) {
            const double c_real = view.x_min + (col + 0.5) * dx;
            img.data[static_cast<std::size_t>(row) * width + col] =
                escape_time(c_real, c_imag, max_iter);
        }
    }

    if (symmetric) {
        mirror_computed_rows(img, computed_rows);
    }
    return img;
}

namespace {

// Symmetry is an opt-in optimisation: it halves the work but makes the
// per-row work profile meaningless for the mirrored half, and complicates
// domain decomposition under MPI.
bool should_exploit_symmetry(const Viewport& view) {
#ifdef USE_SYMMETRY
    return std::abs(view.y_min + view.y_max) < SYMMETRY_TOLERANCE;
#else
    (void)view;
    return false;
#endif
}

int rows_to_compute(int height, bool symmetric) {
    if (!symmetric) {
        return height;
    }
    return height / 2 + height % 2;  // odd heights keep the middle row
}

// Iterating on the squares avoids both a square root and two multiplications
// per step. Order matters: z_imag must be updated before z_real, and the
// squares only after both, otherwise the recurrence mixes generations.
//
// WARNING: with USE_PRUNING the returned value is max_iter for interior points
// that were never iterated. The escape-time array then overstates the actual
// arithmetic work, so FLOP counts derived from it become nominal, not measured.
int escape_time(double c_real, double c_imag, int max_iter) {
#ifdef USE_PRUNING
    const double c_imag_sq = c_imag * c_imag;
    if (lies_in_main_cardioid(c_real, c_imag_sq) ||
        lies_in_period_two_bulb(c_real, c_imag_sq)) {
        return max_iter;
    }
#endif

    double z_real = 0.0;
    double z_imag = 0.0;
    double z_real_sq = 0.0;
    double z_imag_sq = 0.0;

    int iter = 0;
    while (iter < max_iter && z_real_sq + z_imag_sq <= ESCAPE_RADIUS_SQ) {
        z_imag = 2.0 * z_real * z_imag + c_imag;
        z_real = z_real_sq - z_imag_sq + c_real;
        z_real_sq = z_real * z_real;
        z_imag_sq = z_imag * z_imag;
        ++iter;
    }
    return iter;
}

// Valid only when the sampling grid is symmetric about the real axis, which
// pixel-centre sampling guarantees for a symmetric viewport.
void mirror_computed_rows(MandelbrotImage& img, int computed_rows) {
    for (int row = computed_rows; row < img.height; ++row) {
        const int source_row = img.height - 1 - row;
        for (int col = 0; col < img.width; ++col) {
            img.data[static_cast<std::size_t>(row) * img.width + col] =
                img.data[static_cast<std::size_t>(source_row) * img.width + col];
        }
    }
}

#ifdef USE_PRUNING

bool lies_in_main_cardioid(double c_real, double c_imag_sq) {
    const double shifted = c_real - 0.25;
    const double q = shifted * shifted + c_imag_sq;
    return q * (q + shifted) <= 0.25 * c_imag_sq;
}

bool lies_in_period_two_bulb(double c_real, double c_imag_sq) {
    const double shifted = c_real + 1.0;
    return shifted * shifted + c_imag_sq <= PERIOD_TWO_BULB_RADIUS_SQ;
}

#endif

}  // namespace
