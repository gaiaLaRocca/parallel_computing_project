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

OBJECTS  := mandelbrot.o image_io.o metrics.o main.o
TARGET   := mandelbrot

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

mandelbrot.o: mandelbrot.cpp mandelbrot.hpp
image_io.o:   image_io.cpp image_io.hpp mandelbrot.hpp
metrics.o:    metrics.cpp metrics.hpp mandelbrot.hpp
main.o:       main.cpp image_io.hpp mandelbrot.hpp metrics.hpp

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: clean
