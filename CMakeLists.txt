cmake_minimum_required(VERSION 4.2)
project(parallel_computing_project)

set(CMAKE_CXX_STANDARD 14)

include_directories(mandelbrot/src)

add_executable(parallel_computing_project
        mandelbrot/src/image_io.cpp
        mandelbrot/src/image_io.hpp
        mandelbrot/src/main.cpp
        mandelbrot/src/mandelbrot.cpp
        mandelbrot/src/mandelbrot.hpp
        mandelbrot/src/metrics.cpp
        mandelbrot/src/metrics.hpp)
