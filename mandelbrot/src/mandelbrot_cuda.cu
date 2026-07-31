#include "mandelbrot.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// CUDA kernel. Identical arithmetic to the serial baseline (mandelbrot.cpp):
// each device thread runs the same escape-time recurrence on one pixel, so with
// --fmad=false (the nvcc analogue of -ffp-contract=off) the escape-time
// matrix - and thus the checksum - is bit-for-bit the same as the CPU kernels'.
//
// Thread->pixel mapping: threadIdx.x walks columns, so the 32 lanes of a warp
// own horizontally adjacent pixels of one row (as long as blockDim.x is a
// multiple of the warp width). That single choice buys both coalesced
// global-memory writes (one transaction per warp) and low warp divergence: the
// warp only retires when its slowest lane escapes, and horizontally adjacent
// pixels have similar escape times.

namespace {

constexpr double ESCAPE_RADIUS_SQ = 4.0;
constexpr double SYMMETRY_TOLERANCE = 1e-9;

#ifdef USE_PRUNING
constexpr double PERIOD_TWO_BULB_RADIUS_SQ = 0.0625;  // (1/4)^2
#endif

// Default launch geometry: warp-aligned horizontal strips ("256x1"). The shape
// is overridable at runtime with MANDEL_BLOCK=BXxBY so a single binary sweeps
// block shapes without a recompile - the job script sets MANDEL_BLOCK and the
// matching --schedule label, mirroring how the OpenMP sweep uses OMP_SCHEDULE.
constexpr int DEFAULT_BLOCK_X = 256;
constexpr int DEFAULT_BLOCK_Y = 1;
constexpr int MAX_THREADS_PER_BLOCK = 1024;  // hardware limit on every arch

// Measured on the last compute_mandelbrot call, reported via the accessors.
double g_last_transfer_seconds = 0.0;
double g_occupancy = 0.0;
int g_block_x = 0;
int g_block_y = 0;

void check(cudaError_t result, const char* what);
dim3 block_shape_from_env();
bool should_exploit_symmetry(const Viewport& view);
int rows_to_compute(int height, bool symmetric);
void mirror_computed_rows(MandelbrotImage& img, int computed_rows);

// The first CUDA API call creates the device context, which costs hundreds of
// milliseconds. Paying it in a static initialiser keeps that one-off setup out
// of the first timed repetition (the shared driver times every repetition).
struct ContextInit {
    ContextInit() { check(cudaFree(nullptr), "context initialisation"); }
};
const ContextInit g_context_init;

#ifdef USE_PRUNING

__device__ bool lies_in_main_cardioid(double c_real, double c_imag_sq) {
    const double shifted = c_real - 0.25;
    const double q = shifted * shifted + c_imag_sq;
    return q * (q + shifted) <= 0.25 * c_imag_sq;
}

__device__ bool lies_in_period_two_bulb(double c_real, double c_imag_sq) {
    const double shifted = c_real + 1.0;
    return shifted * shifted + c_imag_sq <= PERIOD_TWO_BULB_RADIUS_SQ;
}

#endif

// Iterating on the squares avoids both a square root and two multiplications
// per step. Order matters: z_imag must be updated before z_real, and the
// squares only after both, otherwise the recurrence mixes generations.
//
// WARNING: with USE_PRUNING the returned value is max_iter for interior points
// that were never iterated. The escape-time array then overstates the actual
// arithmetic work, so FLOP counts derived from it become nominal, not measured.
__device__ int escape_time(double c_real, double c_imag, int max_iter) {
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

// One thread per pixel. The grid is sized with ceiling division, so edge
// blocks overhang the image; the bounds check retires those lanes idle.
__global__ void mandelbrot_kernel(int* escape_counts, int width, int rows,
                                  Viewport view, double dx, double dy,
                                  int max_iter) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= rows) {
        return;
    }

    const double c_real = view.x_min + (col + 0.5) * dx;
    const double c_imag = view.y_max - (row + 0.5) * dy;
    escape_counts[static_cast<std::size_t>(row) * width + col] =
        escape_time(c_real, c_imag, max_iter);
}

// Theoretical occupancy of this kernel at the given block size: resident
// threads per SM over the hardware maximum, as bounded by the register and
// block budget the compiler actually spent (-Xptxas -v in the build log shows
// the per-thread register count behind this number).
double launch_occupancy(dim3 block) {
    const int threads_per_block = static_cast<int>(block.x * block.y);

    int active_blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_blocks, mandelbrot_kernel, threads_per_block, 0),
          "occupancy query");

    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "device properties query");
    return static_cast<double>(active_blocks) * threads_per_block /
           props.maxThreadsPerMultiProcessor;
}

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

Paradigm kernel_paradigm() { return Paradigm::CUDA; }

double kernel_comm_seconds() { return 0.0; }

double kernel_occupancy() { return g_occupancy; }

double kernel_transfer_seconds() { return g_last_transfer_seconds; }

int kernel_block_x() { return g_block_x; }

int kernel_block_y() { return g_block_y; }

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

    // The block shape is fixed for the whole process, so the occupancy query
    // runs once; the grid is re-derived per call because it depends on the
    // image size (transparent scalability: fix the tile, the grid adapts).
    static const dim3 block = block_shape_from_env();
    static const double occupancy = launch_occupancy(block);
    g_occupancy = occupancy;
    g_block_x = static_cast<int>(block.x);
    g_block_y = static_cast<int>(block.y);

    const dim3 grid((width + block.x - 1) / block.x,
                    (computed_rows + block.y - 1) / block.y);
    const std::size_t bytes =
        static_cast<std::size_t>(computed_rows) * width * sizeof(int);

    int* d_escape_counts = nullptr;
    check(cudaMalloc(&d_escape_counts, bytes), "device allocation");

    mandelbrot_kernel<<<grid, block>>>(d_escape_counts, width, computed_rows,
                                       view, dx, dy, max_iter);
    check(cudaGetLastError(), "kernel launch");

    // The device->host copy of the escape-time matrix is the only host<->device
    // traffic (the inputs travel as kernel arguments). The events bracket just
    // the copy: before_copy fires only once the kernel has drained the stream,
    // so the elapsed time is pure transfer, no compute.
    cudaEvent_t before_copy = nullptr;
    cudaEvent_t after_copy = nullptr;
    check(cudaEventCreate(&before_copy), "event creation");
    check(cudaEventCreate(&after_copy), "event creation");

    check(cudaEventRecord(before_copy), "event record");
    check(cudaMemcpy(img.data.data(), d_escape_counts, bytes,
                     cudaMemcpyDeviceToHost),
          "device->host copy");
    check(cudaEventRecord(after_copy), "event record");
    check(cudaEventSynchronize(after_copy), "event synchronisation");

    float transfer_ms = 0.0f;
    check(cudaEventElapsedTime(&transfer_ms, before_copy, after_copy),
          "event elapsed time");
    g_last_transfer_seconds = transfer_ms / 1000.0;

    check(cudaEventDestroy(before_copy), "event destruction");
    check(cudaEventDestroy(after_copy), "event destruction");
    check(cudaFree(d_escape_counts), "device deallocation");

    if (symmetric) {
        mirror_computed_rows(img, computed_rows);
    }
    return img;
}

namespace {

void check(cudaError_t result, const char* what) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", what,
                     cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

dim3 block_shape_from_env() {
    int block_x = DEFAULT_BLOCK_X;
    int block_y = DEFAULT_BLOCK_Y;
    const char* spec = std::getenv("MANDEL_BLOCK");
    if (spec != nullptr &&
        (std::sscanf(spec, "%dx%d", &block_x, &block_y) != 2 || block_x < 1 ||
         block_y < 1 || block_x * block_y > MAX_THREADS_PER_BLOCK)) {
        std::fprintf(stderr,
                     "error: MANDEL_BLOCK must be BXxBY with 1 <= BX*BY <= %d, "
                     "got \"%s\"\n",
                     MAX_THREADS_PER_BLOCK, spec);
        std::exit(EXIT_FAILURE);
    }
    return dim3(static_cast<unsigned>(block_x), static_cast<unsigned>(block_y));
}

// Symmetry is an opt-in optimisation: it halves the work but makes the
// per-row work profile meaningless for the mirrored half. The mirroring runs
// on the host after the copy, exactly as in the serial kernel.
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

}  // namespace
