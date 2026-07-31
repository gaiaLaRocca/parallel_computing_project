CXX      := g++
MPICXX   := mpicxx
NVCC     := nvcc

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

# --fmad=false is the nvcc analogue of -ffp-contract=off: without it the
# compiler fuses multiply-adds and escape counts flip near the boundary,
# breaking checksum validation against the CPU kernels. Never --use_fast_math.
# -Xptxas -v prints the per-thread register count at build time (it explains
# the occupancy figure); -lineinfo lets an optional ncu run map counters back
# to source lines. NVARCH is empty by default so the build also works where no
# GPU is visible (PTX is JIT-compiled at load time); when compiling inside the
# job on the GPU node, pass NVARCH=-arch=native for code tuned to its GPU.
NVARCH    ?=
NVCCFLAGS := -std=c++17 -O3 --fmad=false -lineinfo -Xptxas -v $(NVARCH) $(OPT)

# I/O, metrics and driver are shared by every kernel; only the compute kernel
# object differs between the serial and the OpenMP executables.
SHARED_OBJECTS := image_io.o metrics.o main.o

TARGET      := mandelbrot
OMP_TARGET  := mandelbrot_omp
MPI_TARGET  := mandelbrot_mpi
CUDA_TARGET := mandelbrot_cuda

# Default build is the serial baseline; `make mandelbrot_omp` / `mandelbrot_mpi`
# build the parallel variants (the job scripts do this before submitting).
$(TARGET): mandelbrot.o $(SHARED_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OMP_TARGET): mandelbrot_omp.o $(SHARED_OBJECTS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^

# MPI needs the mpicxx wrapper and -DUSE_MPI on both the kernel and the driver
# (the driver is compiled into a distinct main_mpi.o because the plain main.o is
# built without USE_MPI). The MPI-agnostic I/O and metrics objects are reused.
$(MPI_TARGET): mandelbrot_mpi.o main_mpi.o image_io.o metrics.o
	$(MPICXX) $(CXXFLAGS) -o $@ $^

# The OpenMP kernel object needs -fopenmp so the #pragma omp directives are
# honoured (without it they are silently ignored and the code stays serial).
mandelbrot_omp.o: mandelbrot_omp.cpp mandelbrot.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c -o $@ $<

mandelbrot_mpi.o: mandelbrot_mpi.cpp mandelbrot.hpp
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI -c -o $@ $<

# The CUDA kernel object needs nvcc; the shared objects are plain host code
# compiled by g++ and are reused unchanged, nvcc drives the host linker.
$(CUDA_TARGET): mandelbrot_cuda.o $(SHARED_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

mandelbrot_cuda.o: mandelbrot_cuda.cu mandelbrot.hpp
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

main_mpi.o: main.cpp image_io.hpp mandelbrot.hpp metrics.hpp
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI -c -o $@ $<

mandelbrot.o: mandelbrot.cpp mandelbrot.hpp
image_io.o:   image_io.cpp image_io.hpp mandelbrot.hpp
metrics.o:    metrics.cpp metrics.hpp mandelbrot.hpp
main.o:       main.cpp image_io.hpp mandelbrot.hpp metrics.hpp

clean:
	rm -f mandelbrot.o mandelbrot_omp.o mandelbrot_mpi.o mandelbrot_cuda.o \
	      main_mpi.o $(SHARED_OBJECTS) \
	      $(TARGET) $(OMP_TARGET) $(MPI_TARGET) $(CUDA_TARGET)

.PHONY: clean
