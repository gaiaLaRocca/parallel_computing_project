#pragma once

#include <cstdint>
#include <string>

#include "mandelbrot.hpp"

// Canonical validation artifact: little-endian int32 escape counts, no header.
// Two implementations agree iff the two dumps are byte-identical.
bool save_raw(const MandelbrotImage& img, const std::string& path);

// FNV-1a over the same bytes save_raw would emit. Cheap regression check.
std::uint64_t checksum(const MandelbrotImage& img);

// Binary PGM (P5), 8-bit grayscale. Escape counts are linearly rescaled.
bool save_pgm(const MandelbrotImage& img, const std::string& path);

// Binary PPM (P6), 24-bit RGB, Bernstein palette. Interior is black.
bool save_ppm(const MandelbrotImage& img, const std::string& path);
