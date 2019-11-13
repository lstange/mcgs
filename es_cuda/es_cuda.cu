//
// Average extreme spread of five-shot group assuming impact coordinates follow standard normal distribution
//
#include <string>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <typeinfo>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <curand.h>

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
                             const unsigned int numSims)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  // Determine thread ID
  unsigned bid = blockIdx.x;
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned step = gridDim.x * blockDim.x;

  // Shift the input/output pointers
  const double* pointx = points + tid;
  const double* pointy = pointx + numSims;

  // Count the number of points which lie inside the unit quarter-circle
  double pointsInside = 0;

  for (unsigned int i = tid ; i < numSims ; i += step, pointx += step, pointy += step) {
    double x = *pointx;
    double y = *pointy;
    double l2norm2 = x * x + y * y;

    if (l2norm2 < 1) {
      pointsInside += 1;
    }
  }

  // Reduce within the block
  pointsInside = reduce_sum(pointsInside, cta);

  // Store the result
  if (threadIdx.x == 0) {
    results[bid] = pointsInside;
  }
}

double es_cuda(unsigned int numSims, unsigned int threadBlockSize, unsigned int seed)
{
  // Get device properties
  struct cudaDeviceProp  deviceProperties;
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

  // Attach to GPU
  cudaResult = cudaSetDevice(0);
  if (cudaResult != cudaSuccess) {
    string msg("Could not set CUDA device: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Determine how to divide the work between cores
  dim3 block;
  dim3 grid;
  block.x = threadBlockSize;
  grid.x  = (numSims + threadBlockSize - 1) / threadBlockSize;

  // Aim to launch around ten or more times as many blocks as there
  // are multiprocessors on the target device.
  unsigned blocksPerSM = 10;
  unsigned numSMs      = deviceProperties.multiProcessorCount;
  while (grid.x > 2 * blocksPerSM * numSMs) {
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
  // Each simulation has two random numbers to give X and Y coordinate
  double *d_points = 0;
  cudaResult = cudaMalloc((void **)&d_points, 2 * numSims * sizeof(double));
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

  // Generate random points in unit square
  curandStatus_t curandResult;
  curandGenerator_t prng;
  curandResult = curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

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

  curandResult = curandGenerateUniformDouble(prng, (double*)d_points, 2 * numSims);
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

  // Count the points inside unit quarter-circle
  computeValue<<<grid, block, block.x * sizeof(double)>>>(d_results, d_points, numSims);

  // Copy partial results back
  vector<double> results(grid.x);
  cudaResult = cudaMemcpy(&results[0], d_results, grid.x * sizeof(double), cudaMemcpyDeviceToHost);

  if (cudaResult != cudaSuccess) {
    string msg("Could not copy partial results to host: ");
    msg += cudaGetErrorString(cudaResult);
    throw std::runtime_error(msg);
  }

  // Complete sum-reduction on host
  double value = std::accumulate(results.begin(), results.end(), 0);

  // Determine the proportion of points inside the quarter-circle,
  // i.e. the area of the unit quarter-circle
  value /= numSims;

  // Value is currently an estimate of the area of a unit quarter-circle, so we can
  // scale to a full circle by multiplying by four. Now since the area of a circle
  // is pi * r^2, and r is one, the value will be an estimate for the value of pi.
  value *= 4;

  // Cleanup
  if (d_points) {
    cudaFree(d_points);
    d_points = 0;
  }

  if (d_results) {
    cudaFree(d_results);
    d_results = 0;
  }

  return value;
}
