FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cpp planets.hpp config.hpp vector.hpp $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< -o $@
compute.o: cuda_compute.cu vector.hpp $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< -o $@

clean:
	rm -f *.o nbody