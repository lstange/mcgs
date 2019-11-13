//
// Average extreme spread of five-shot group assuming impact coordinates follow standard normal distribution
//
// This is host code. It only calls the device code, shows the result and time. All heavy lifting is in es_cuda.cu
//
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>

double es_cuda(unsigned int numSims, unsigned int threadBlockSize, unsigned int seed);

int main(int argc, char **argv)
{
  try {
    unsigned seed            = 1234;
    unsigned numSims         = 10000000;
    unsigned threadBlockSize = 128;

    // Check requested size is valid
    cudaDeviceProp deviceProperties;
    cudaError_t cudaResult = cudaGetDeviceProperties(&deviceProperties, 0);
    if (cudaResult != cudaSuccess) {
      fprintf(stderr, "cound not get device properties.\n");
      return(EXIT_FAILURE);
    }
    if ( threadBlockSize < 32
      || threadBlockSize > static_cast<unsigned int>(deviceProperties.maxThreadsPerBlock)
       ) {
      fprintf(stderr, "specified block size (%d) is invalid, must be between 32 and %d.\n",
              threadBlockSize, deviceProperties.maxThreadsPerBlock);
      return(EXIT_FAILURE);
    }

    // Evaluate on GPU
    auto start_time = std::chrono::system_clock::now();
    double result = es_cuda(numSims, threadBlockSize, seed);
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = end_time - start_time;

    // Print results
    std::cout.precision(14);
    std::cout << "code,threads,power_of_4,min,avg,max,time\n";
    std::cout << "CUDA," << deviceProperties.multiProcessorCount << "," << numSims << ",," << result << ",," << seconds.count() << "\n";
  } catch (std::runtime_error &e) { // es_cuda() can throw runtime exceptions
    fprintf(stderr, "runtime error (%s)\n", e.what());
    return(EXIT_FAILURE);
  }
  return(EXIT_SUCCESS);
}
