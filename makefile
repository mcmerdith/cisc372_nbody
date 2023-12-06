FLAGS= -g --keep -lineinfo -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.hpp config.hpp vector.hpp $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< -o $@
compute.o: cuda_compute.cu vector.hpp $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< -o $@

test: test.cu out
	nvcc $(FLAGS) -c $< -o out/$@

clean:
	rm -f nbody test *.o *.ii *.cudafe1.* *.fatbin* *.module_id *.ptx *.cubin *.reg.*