NVCC          := /usr/local/cuda/bin/nvcc
NVCCFLAGS     := -ccbin g++ -m64
LIBRARIES     := -lcurand

es_cuda: main.o es_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $+ $(LIBRARIES)

main.o:main.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

es_cuda.o:es_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

run: es_cuda
	./es_cuda

clean:
	rm -f es_cuda main.o es_cuda.o

