#include "mandelbrot.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

// MPI kernel - static decomposition, BLOCK or CYCLIC, selected at runtime by the
// MPI_DECOMP environment variable (default "block"). This mirrors how the
// OpenMP kernel reads OMP_SCHEDULE: one binary sweeps both schemes without a
// recompile, and the job passes a matching --schedule label for the CSV.
//
//   block  : rank r owns a contiguous range of rows (rows/size, remainder over
//            the first ranks). The heavy middle rows land on a few ranks, so the
//            block-level imbalance is high.
//   cyclic : rank r owns rows r, r+size, r+2*size, ... Every rank gets an even
//            mix of light (edge) and heavy (centre) rows, so the load is nearly
//            balanced - at essentially no extra communication.
//
// Each rank computes only its own rows; MPI_Gatherv assembles the full
// escape-time matrix on rank 0. The per-pixel arithmetic is identical to the
// serial kernel, so the checksum matches (validated with mpirun).
//
// Symmetry is deliberately NOT exploited (it complicates decomposition; for a
// symmetric viewport computing every row yields the same matrix anyway).
// Pruning is fine - it is a per-pixel test, independent across ranks.

namespace {

constexpr double ESCAPE_RADIUS_SQ = 4.0;

#ifdef USE_PRUNING
constexpr double PERIOD_TWO_BULB_RADIUS_SQ = 0.0625;  // (1/4)^2
#endif

enum class Decomp { Block, Cyclic };

int escape_time(double c_real, double c_imag, int max_iter);

#ifdef USE_PRUNING
bool lies_in_main_cardioid(double c_real, double c_imag_sq);
bool lies_in_period_two_bulb(double c_real, double c_imag_sq);
#endif

// Communication time (seconds) of the last compute_mandelbrot call: here the
// single MPI_Gatherv. Read by the shared driver to fill comm_fraction.
double g_last_comm_seconds = 0.0;

Decomp read_decomp() {
    const char* env = std::getenv("MPI_DECOMP");
    if (env != nullptr && std::strcmp(env, "cyclic") == 0) {
        return Decomp::Cyclic;
    }
    return Decomp::Block;  // default, and what the master-worker scheme will override
}

}  // namespace

bool pruning_enabled() {
#ifdef USE_PRUNING
    return true;
#else
    return false;
#endif
}

// MPI does not exploit symmetry (see file header); kept for interface parity.
bool symmetry_enabled() {
#ifdef USE_SYMMETRY
    return true;
#else
    return false;
#endif
}

Paradigm kernel_paradigm() { return Paradigm::MPI; }

double kernel_comm_seconds() { return g_last_comm_seconds; }

MandelbrotImage compute_mandelbrot(const Viewport& view, int width, int height,
                                   int max_iter) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const Decomp decomp = read_decomp();

    // Row counts are the same for both schemes (base rows each, the first `rem`
    // ranks take one extra); only which rows a rank owns differs. row_displs is
    // the contiguous send/gather layout: for block it is also the destination
    // offset, for cyclic it just packs each rank's rows before the reorder.
    std::vector<int> row_counts(size);
    std::vector<int> row_displs(size);
    const int base = height / size;
    const int rem = height % size;
    int acc = 0;
    for (int r = 0; r < size; ++r) {
        row_counts[r] = base + (r < rem ? 1 : 0);
        row_displs[r] = acc;
        acc += row_counts[r];
    }

    const int my_rows = row_counts[rank];

    const double dx = (view.x_max - view.x_min) / width;
    const double dy = (view.y_max - view.y_min) / height;

    // Compute this rank's owned rows, packed contiguously into a local buffer.
    // block:  global row = row_displs[rank] + i    (contiguous range)
    // cyclic: global row = rank + i*size           (strided)
    std::vector<int> local(static_cast<std::size_t>(my_rows) * width);
    for (int i = 0; i < my_rows; ++i) {
        const int row = (decomp == Decomp::Block) ? (row_displs[rank] + i)
                                                  : (rank + i * size);
        const double c_imag = view.y_max - (row + 0.5) * dy;
        for (int col = 0; col < width; ++col) {
            const double c_real = view.x_min + (col + 0.5) * dx;
            local[static_cast<std::size_t>(i) * width + col] =
                escape_time(c_real, c_imag, max_iter);
        }
    }

    MandelbrotImage img;
    img.width = width;
    img.height = height;
    img.max_iter = max_iter;

    // Gather counts/displacements in units of pixels (int elements).
    std::vector<int> px_counts(size);
    std::vector<int> px_displs(size);
    for (int r = 0; r < size; ++r) {
        px_counts[r] = row_counts[r] * width;
        px_displs[r] = row_displs[r] * width;
    }

    // For block, gather straight into the final matrix (contiguous per rank).
    // For cyclic, gather into a temporary packed buffer, then rank 0 scatters
    // the strided rows into their real positions.
    std::vector<int> gather_buf;
    int* recvbuf = nullptr;
    if (rank == 0) {
        img.data.resize(static_cast<std::size_t>(width) * height);
        if (decomp == Decomp::Block) {
            recvbuf = img.data.data();
        } else {
            gather_buf.resize(static_cast<std::size_t>(width) * height);
            recvbuf = gather_buf.data();
        }
    }

    // Barrier first so the timed gather is genuine data movement, not the
    // load-imbalance wait (see block vs comm_fraction reasoning); the idle time
    // is parked in the barrier, still inside the timed region, so T(p) is fair.
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    MPI_Gatherv(local.data(), my_rows * width, MPI_INT, recvbuf,
                px_counts.data(), px_displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
    g_last_comm_seconds = MPI_Wtime() - t0;

    // Un-interleave: rank r's i-th packed row belongs at global row r + i*size.
    if (rank == 0 && decomp == Decomp::Cyclic) {
        for (int r = 0; r < size; ++r) {
            for (int i = 0; i < row_counts[r]; ++i) {
                const int row = r + i * size;
                const int* src =
                    &gather_buf[static_cast<std::size_t>(px_displs[r]) +
                                static_cast<std::size_t>(i) * width];
                std::copy(src, src + width,
                          &img.data[static_cast<std::size_t>(row) * width]);
            }
        }
    }

    // Non-root ranks never have their image read by the driver (I/O guarded to
    // rank 0); keep their local block so the returned object is well-formed.
    if (rank != 0) {
        img.data = std::move(local);
    }
    return img;
}

namespace {

// Identical to the serial kernel: this is what keeps the checksum bit-for-bit
// comparable across paradigms (with -ffp-contract=off).
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
