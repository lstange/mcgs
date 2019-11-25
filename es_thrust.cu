// sudo apt install nvidia-cuda-toolkit
// nvcc -O3 -std=c++11 es_thrust.cu -o es_thrust
//
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

const unsigned power_of_4 = 8;
const unsigned numGroups = 1 << 2 * power_of_4;

__constant__ uint64_t g_seed[1];

__device__
unsigned hash(unsigned a)
{
  a = (a+0x7ed55d16) + (a<<12);
  a = (a^0xc761c23c) ^ (a>>19);
  a = (a+0x165667b1) + (a<<5);
  a = (a+0xd3a2646c) ^ (a<<9);
  a = (a+0xfd7046c5) + (a<<3);
  a = (a^0xb55a4f09) ^ (a>>16);
  return a;
}

// Adopted from http://prng.di.unimi.it/xoshiro256plus.c
__device__
double xoshiro256plus(uint64_t s[4]) {
  const uint64_t result = s[0] + s[3];
  const uint64_t t = s[1] << 17;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = (s[3] << 45) | (s[3] >> (64 - 45));
  // random_real53() from http://mumble.net/~campbell/tmp/random_real.c
  return (double)(result >> 11) / (1ULL << 53);
}

struct estimate_es : public thrust::unary_function<unsigned, double>
{
  __device__
  double operator()(unsigned thread_id)
  {
    __uint64_t prng_state[4];
    __uint64_t mask = hash(thread_id);
    for (unsigned i = 0; i < 4; i++) {
      mask = (mask << 32) | hash(mask);
      mask = (mask << 32) | hash(mask);
      prng_state[i] = g_seed[0] ^ mask;
    }
    double sum = 0;
    for (unsigned group = 0; group < numGroups; group++) {
      double x[5]; double y[5];
      for (unsigned shot = 0; shot < 5; shot++) {
        auto u = xoshiro256plus(prng_state);
        auto v = xoshiro256plus(prng_state);
        
        // Box-Muller transform
        auto r = sqrt(-2 * log(u));
        auto theta = 2 * M_PI * v;
        x[shot] = r * cos(theta);
        y[shot] = r * sin(theta);
      }
      double es2 = 0;
      for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = i + 1; j < 5; j++) {
          double dx = x[i] - x[j];
          double dy = y[i] - y[j];
          double d2 = dx * dx + dy * dy;
          if (es2 < d2) {
            es2 = d2;
          }
        }
      }
      sum += sqrt(es2);
    }
    return sum / numGroups;
  }
};

int main(void)
{
  auto start_time = std::chrono::system_clock::now();
  __uint128_t mcg128_state = ((__uint128_t)time(NULL) << 1) | 1;
  double avg = 0, min = 100, max = 0;
  const unsigned nt = 12;
  for (unsigned j = 0; j < nt; j++) {
    // MCG 128 PRNG from http://www.pcg-random.org/posts/on-vignas-pcg-critique.html
    mcg128_state *= 0xda942042e4dd58b5ULL;
    uint64_t seed[1] = {(uint64_t)(mcg128_state >> 64)};
    cudaMemcpyToSymbol(g_seed, seed, sizeof(uint64_t));
    double r = thrust::transform_reduce(thrust::counting_iterator<unsigned>(0),
                                        thrust::counting_iterator<unsigned>(numGroups),
                                        estimate_es(),
                                        double(0),
                                        thrust::plus<double>());
    avg += r / numGroups;
    min = fmin(r / numGroups, min);
    max = fmax(r / numGroups, max);
  }
  avg /= nt;
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end_time - start_time;
  std::cout.precision(14);
  std::cout << "code,threads,power_of_4,min,avg,max,time\n";
  std::cout << "CUDA thrust," << nt << "," << power_of_4 << "," << min << "," << avg << "," << max << "," << seconds.count() << "\n";
  return 0;
}
