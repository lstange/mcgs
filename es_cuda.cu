//
// Average extreme spread of five-shot group assuming impact coordinates follow standard normal distribution
//
// Building:
//   nvcc -std=c++11 es_cuda.cu -o es_cuda -lcurand
//
// Running:
//   for run in {1..10}; do ./es_cuda 15 | tee -a es_cuda.csv; done
//
#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <typeinfo>
#include <cooperative_groups.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <curand.h>

namespace cg = cooperative_groups;

using std::string;
using std::vector;

// First level of reduction
__device__ double reduce_sum(double in, cg::thread_block cta)
{
  extern __shared__ double sdata[];

  // Write to shared memory
  unsigned ltid = threadIdx.x;
  sdata[ltid] = in;
  cg::sync(cta);

  // Do reduction in shared memory
  for (unsigned s = blockDim.x / 2 ; s > 0 ; s >>= 1) {
    if (ltid < s) {
      sdata[ltid] += sdata[ltid + s];
    }
    cg::sync(cta);
  }

  return sdata[0];
}

// Estimator kernel
__global__ void computeValue(double* const results,
                             const double* const points,
                             const unsigned numGroups)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  // Determine thread ID
  unsigned bid = blockIdx.x;
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned step = gridDim.x * blockDim.x;

  // Shift the input/output pointers
  const double* pointx = points + tid;
  const double* pointy = pointx + 5 * numGroups;

  // Sum up extreme spread of all groups
  double sum = 0;

  for (unsigned i = tid ; i < numGroups; i += step, pointx += step * 5, pointy += step * 5) {

    // Pairwise distances
    double dx[10], dy[10];

    // Unroll nested comparison loops
    dx[0] = pointx[0] - pointx[1]; dy[0] = pointy[0] - pointy[1];
    dx[1] = pointx[0] - pointx[2]; dy[1] = pointy[0] - pointy[2];
    dx[2] = pointx[0] - pointx[3]; dy[2] = pointy[0] - pointy[3];
    dx[3] = pointx[0] - pointx[4]; dy[3] = pointy[0] - pointy[4];
    dx[4] = pointx[1] - pointx[2]; dy[4] = pointy[1] - pointy[2];
    dx[5] = pointx[1] - pointx[3]; dy[5] = pointy[1] - pointy[3];
    dx[6] = pointx[1] - pointx[4]; dy[6] = pointy[1] - pointy[4];
    dx[7] = pointx[2] - pointx[3]; dy[7] = pointy[2] - pointy[3];
    dx[8] = pointx[2] - pointx[4]; dy[8] = pointy[2] - pointy[4];
    dx[9] = pointx[3] - pointx[4]; dy[9] = pointy[3] - pointy[4];

    double max_d2 = 0;
    for (unsigned j = 0; j < 10; j++) {
      auto candidate_d2 = dx[j] * dx[j] + dy[j] * dy[j];
      max_d2 = max(max_d2, candidate_d2);
    }
    double es = sqrt(max_d2);
    sum += es;
  }

  // Reduce within the block
  sum = reduce_sum(sum, cta);

  // Store the result
  if (threadIdx.x == 0) {
    results[bid] = sum;
  }
}

double es_cuda(unsigned power_of_4, unsigned seed)
{
  // Get device properties
  struct cudaDeviceProp deviceProperties;
  cudaError_t cudaResult = cudaGetDeviceProperties(&deviceProperties, 0);
  if (cudaResult != cudaSuccess) {
    string msg("Could not get device properties: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Check precision is valid
  if (deviceProperties.major < 1 || (deviceProperties.major == 1 && deviceProperties.minor < 3)) {
    throw std::runtime_error("Device does not have double precision support");
  }

  // Check requested size is valid
  const unsigned threadBlockSize = 128;
  if (threadBlockSize > (deviceProperties.maxThreadsPerBlock)) {
    throw std::runtime_error("Thread block size is greater than maxThreadsPerBlock");
  }
  dim3 block;
  block.x = threadBlockSize;

  // Attach to GPU
  cudaResult = cudaSetDevice(0);
  if (cudaResult != cudaSuccess) {
    string msg("Could not set CUDA device: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Aim to launch around ten or more times as many blocks as there
  // are multiprocessors on the target device.
  dim3 grid;
  const unsigned numGroups = 1 << (2 * power_of_4);
  grid.x  = numGroups / threadBlockSize;
  while (grid.x > 20 * deviceProperties.multiProcessorCount) {
    grid.x >>= 1;
  }

  // Get computeValue function properties and check the maximum block size
  struct cudaFuncAttributes funcAttributes;
  cudaResult = cudaFuncGetAttributes(&funcAttributes, computeValue);
  if (cudaResult != cudaSuccess) {
    string msg("Could not get function attributes: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  if (block.x > (unsigned)funcAttributes.maxThreadsPerBlock) {
    throw std::runtime_error("Block X dimension is too large for computeValue kernel");
  }

  // Check the dimensions are valid
  if (block.x > (unsigned)deviceProperties.maxThreadsDim[0]) {
    throw std::runtime_error("Block X dimension is too large for device");
  }
  if (grid.x > (unsigned)deviceProperties.maxGridSize[0]) {
    throw std::runtime_error("Grid X dimension is too large for device");
  }

  // Allocate memory for points
  // Each simulation has ten random numbers to give five pairs of X and Y coordinates
  double* d_points = 0;
  cudaResult = cudaMalloc((void **)&d_points, 10 * numGroups * sizeof(double));
  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for random numbers: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Allocate memory for result
  // Each thread block will produce one result
  double* d_results = 0;
  cudaResult = cudaMalloc((void**)&d_results, grid.x * sizeof(double));
  if (cudaResult != cudaSuccess) {
    string msg("Could not allocate memory on device for partial results: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Generate random points
  curandStatus_t curandResult;
  curandGenerator_t prng;
  curandResult = curandCreateGenerator(&prng,  CURAND_RNG_PSEUDO_DEFAULT);
  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not create pseudo-random number generator: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  curandResult = curandSetPseudoRandomGeneratorSeed(prng, seed);
  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not set seed for pseudo-random number generator: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  curandResult = curandGenerateNormalDouble(prng, (double*)d_points, 10 * numGroups, 0, 1);
  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not generate pseudo-random numbers: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  curandResult = curandDestroyGenerator(prng);
  if (curandResult != CURAND_STATUS_SUCCESS) {
    string msg("Could not destroy pseudo-random number generator: ");
    msg += curandResult;
    throw std::runtime_error(msg);
  }

  // Calculate and average group size
  computeValue<<<grid, block, block.x * sizeof(double)>>>(d_results, d_points, numGroups);

  // Copy the results back to host
  vector<double> results(grid.x);
  cudaResult = cudaMemcpy(&results[0], d_results, grid.x * sizeof(double), cudaMemcpyDeviceToHost);
  if (cudaResult != cudaSuccess) {
    string msg("Could not copy results to host: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Complete sum-reduction
  double sum = std::accumulate(results.begin(), results.end(), double(0));

  // Cleanup
  if (d_points) {
    cudaFree(d_points);
  }
  if (d_results) {
    cudaFree(d_results);
  }

  // Divide sum by count to get the average
  return sum / numGroups;
}

int main(int argc, char **argv)
{
  unsigned power_of_4 = 12;
  if (argc == 2) {
    power_of_4 = atoi(argv[1]);
  }
  unsigned nt = 12;
  if (power_of_4 > 12) {
    nt <<= 2 * (power_of_4 - 12);
    power_of_4 = 12;
  }
  try {
    auto start_time = std::chrono::system_clock::now();
    double avg = 0, min = 100, max = 0;
    __uint128_t mcg128_state = time(NULL) | 1; // can be seeded to any odd number
    for (unsigned j = 0; j < nt; j++) {
      double r = es_cuda(power_of_4, (unsigned)(mcg128_state >> 64));
      avg += r;
      min = fmin(r, min);
      max = fmax(r, max);
      mcg128_state *= 0xda942042e4dd58b5ULL;
    }
    avg /= nt;
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> seconds = end_time - start_time;
    std::cout.precision(14);
    std::cout << "code,threads,power_of_4,min,avg,max,time\n";
    std::cout << "CUDA," << nt << "," << power_of_4 << "," << min << "," << avg << "," << max << "," << seconds.count() << "\n";
  } catch (std::runtime_error &e) { // es_cuda() can throw runtime exceptions
    fprintf(stderr, "runtime error (%s)\n", e.what());
    return(EXIT_FAILURE);
  }
  return(EXIT_SUCCESS);
}
