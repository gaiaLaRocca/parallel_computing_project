#include "image_io.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <vector>

namespace {

struct Rgb {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// Fixed normalisation range: colours stay stable across different max_iter
// values, so the only visual change between panels is structural detail.
constexpr int PALETTE_RANGE = 256;

const char* raw_bytes(const std::vector<int>& data);
std::size_t raw_size(const std::vector<int>& data);
double normalise_iter(int iter, int max_iter);
Rgb bernstein_palette(double t);
unsigned char to_gray(int iter, int max_iter);

}  // namespace

// Assumes a little-endian host. Cross-machine comparison of dumps produced on
// mixed-endian hardware is meaningless; compare the checksum instead.
bool save_raw(const MandelbrotImage& img, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out.write(raw_bytes(img.data), static_cast<std::streamsize>(raw_size(img.data)));
    return out.good();
}

std::uint64_t checksum(const MandelbrotImage& img) {
    constexpr std::uint64_t FNV_OFFSET = 1469598103934665603ULL;
    constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

    const auto* bytes = reinterpret_cast<const unsigned char*>(raw_bytes(img.data));
    std::uint64_t hash = FNV_OFFSET;
    for (std::size_t i = 0; i < raw_size(img.data); ++i) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

bool save_pgm(const MandelbrotImage& img, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P5\n" << img.width << " " << img.height << "\n255\n";

    std::vector<unsigned char> row(img.width);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            row[x] = to_gray(img.data[static_cast<std::size_t>(y) * img.width + x],
                             img.max_iter);
        }
        out.write(reinterpret_cast<const char*>(row.data()), img.width);
    }
    return out.good();
}

bool save_ppm(const MandelbrotImage& img, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P6\n" << img.width << " " << img.height << "\n255\n";

    std::vector<unsigned char> row(static_cast<std::size_t>(img.width) * 3);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            const int iter = img.data[static_cast<std::size_t>(y) * img.width + x];
            const Rgb color = (iter >= img.max_iter)
                                  ? Rgb{0, 0, 0}
                                  : bernstein_palette(normalise_iter(iter, img.max_iter));
            row[3 * x + 0] = color.r;
            row[3 * x + 1] = color.g;
            row[3 * x + 2] = color.b;
        }
        out.write(reinterpret_cast<const char*>(row.data()),
                  static_cast<std::streamsize>(row.size()));
    }
    return out.good();
}

namespace {

const char* raw_bytes(const std::vector<int>& data) {
    return reinterpret_cast<const char*>(data.data());
}

std::size_t raw_size(const std::vector<int>& data) {
    return data.size() * sizeof(int);
}

// Classic Bernstein polynomial ramp: cheap, smooth, and monotone in luminance.
Rgb bernstein_palette(double t) {
    const double u = 1.0 - t;
    const double r = 9.0 * u * t * t * t;
    const double g = 15.0 * u * u * t * t;
    const double b = 8.5 * u * u * u * t;
    return {static_cast<unsigned char>(255.0 * r),
            static_cast<unsigned char>(255.0 * g),
            static_cast<unsigned char>(255.0 * b)};
}

// Clamp to [0, 1] using PALETTE_RANGE as denominator. Interior points
// (iter >= max_iter) are handled by the callers before reaching this.
double normalise_iter(int iter, int max_iter) {
    (void)max_iter;
    const double t = static_cast<double>(iter) / PALETTE_RANGE;
    return t < 1.0 ? t : 1.0;
}

unsigned char to_gray(int iter, int max_iter) {
    if (iter >= max_iter) {
        return 0;
    }
    return static_cast<unsigned char>(255.0 * normalise_iter(iter, max_iter));
}

}  // namespace
