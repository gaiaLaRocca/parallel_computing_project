CXX      := g++

# Algorithmic optimisations are opt-in. The bare kernel is the scaling
# baseline: pruning removes the most expensive pixels and masks the load
# imbalance under study, symmetry makes half the rows a memcpy.
#   make                                          -> bare kernel
#   make OPT="-DUSE_PRUNING"                      -> cardioid + bulb test
#   make OPT="-DUSE_PRUNING -DUSE_SYMMETRY"       -> both
OPT      ?=

# -ffp-contract=off keeps results bit-identical across compilers and against
# the CUDA port (nvcc -fmad=false). Never enable -ffast-math: it can flip
# escape counts on pixels near the boundary and break validation.
CXXFLAGS := -std=c++17 -O3 -march=native -ffp-contract=off -Wall -Wextra $(OPT)

# OpenMP is enabled only on the parallel kernel's object and on its link line;
# the shared objects contain no pragmas, so they build unchanged.
OMPFLAGS := -fopenmp

# I/O, metrics and driver are shared by every kernel; only the compute kernel
# object differs between the serial and the OpenMP executables.
SHARED_OBJECTS := image_io.o metrics.o main.o

TARGET     := mandelbrot
OMP_TARGET := mandelbrot_omp

# Default build is the serial baseline; `make mandelbrot_omp` builds the OpenMP
# variant (the job script does this before submitting the schedule sweep).
$(TARGET): mandelbrot.o $(SHARED_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OMP_TARGET): mandelbrot_omp.o $(SHARED_OBJECTS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^

# The OpenMP kernel object needs -fopenmp so the #pragma omp directives are
# honoured (without it they are silently ignored and the code stays serial).
mandelbrot_omp.o: mandelbrot_omp.cpp mandelbrot.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c -o $@ $<

mandelbrot.o: mandelbrot.cpp mandelbrot.hpp
image_io.o:   image_io.cpp image_io.hpp mandelbrot.hpp
metrics.o:    metrics.cpp metrics.hpp mandelbrot.hpp
main.o:       main.cpp image_io.hpp mandelbrot.hpp metrics.hpp

clean:
	rm -f mandelbrot.o mandelbrot_omp.o $(SHARED_OBJECTS) $(TARGET) $(OMP_TARGET)

.PHONY: clean
