//
// Extreme spread of five-shot group assuming impact coordinates follow standard normal distribution
//
// Monte-Carlo using AVX2 instructions, parallelized with OpenMP 
//
// Building:
//
// g++-8 -Wall -O3 -std=c++11 -march=native es_omp_avx2.cpp -lm -g -o es_omp_avx2 -fopenmp
//
// Running:
//
// for run in {1..10}; do ./es_omp_avx2 14 | tee -a es_omp_avx2.csv; done
//
#include <omp.h>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <x86intrin.h>

//#define USE_MCG

extern "C" {
  __m256d _ZGVdN4v_log(__m256d x);                    
  void    _ZGVdN4vvv_sincos(__m256d x, __m256i ptrs, __m256i ptrc);
}

// Vectorized version of http://prng.di.unimi.it/xoshiro256plus.c
void inline xoshiro256plus(__m256i state[4], double output[4])
{
  uint64_t result[4];
  _mm256_storeu_si256((__m256i*)&result[0], _mm256_add_epi64(state[0], state[3]));
  const __m256i t = _mm256_slli_epi64(state[1], 17);
  state[2] = _mm256_xor_si256(state[2], state[0]);
  state[3] = _mm256_xor_si256(state[3], state[1]);
  state[1] = _mm256_xor_si256(state[1], state[2]);
  state[0] = _mm256_xor_si256(state[0], state[3]);
  state[2] = _mm256_xor_si256(state[2], t);
  state[3] = _mm256_or_si256( _mm256_slli_epi64(state[3], 45)
                            , _mm256_srli_epi64(state[3], 64 - 45)
                            );
  for (unsigned i = 0; i < 4; i++) {
    output[i] = ldexp(result[i], -64);
  }
}

double recurse(
  #ifdef USE_MCG
    __uint128_t& prng_state,
  #else
    __m256i prng_state[40],
  #endif 
     unsigned zeroes)
{
  double sum = 0;
  if (zeroes) {
    for (unsigned group = 0; group < 4; group++) {
      sum += recurse(prng_state, zeroes - 1);
    }
  } else {
    double prng_output[40];
    #ifdef USE_MCG
      // Advance all PRNGs, one by one
      for (unsigned k = 0; k < 40; k++) {
        // MCG 128 PRNG from http://www.pcg-random.org/posts/on-vignas-pcg-critique.html
        prng_output[k] = ldexp((prng_state *= 0xda942042e4dd58b5ULL) >> 64, -64);
      }
    #else
      // Advance all PRNGs, four at a time
      for (unsigned k = 0; k < 40; k += 4) {
        xoshiro256plus(&prng_state[k], &prng_output[k]);
      }
    #endif

    // Box-Muller transform, four at a time
    const auto minus2 = _mm256_set_pd(-2, -2, -2, -2);
    const auto twopi = _mm256_set_pd(2 * M_PI, 2 * M_PI, 2 * M_PI, 2 * M_PI); 
    double z_s[4], z_c[4];
    auto ptrs = _mm256_set_epi64x((uint64_t)&z_s[3],(uint64_t)&z_s[2],(uint64_t)&z_s[1],(uint64_t)&z_s[0]);
    auto ptrc = _mm256_set_epi64x((uint64_t)&z_c[3],(uint64_t)&z_c[2],(uint64_t)&z_c[1],(uint64_t)&z_c[0]);
    __m256d x[20], y[20];
    for (unsigned i = 0, j = 0; i < 20; i += 4, j++) {
      auto u = _mm256_set_pd(prng_output[i+3], prng_output[i+2], prng_output[i+1], prng_output[i]);
      auto v = _mm256_set_pd(prng_output[i+23], prng_output[i+22], prng_output[i+21], prng_output[i+20]);
      auto r = _mm256_sqrt_pd(_mm256_mul_pd(minus2, _ZGVdN4v_log(u))); 
      auto theta = _mm256_mul_pd(twopi, v);
      _ZGVdN4vvv_sincos(theta, ptrs, ptrc);
      x[j] = _mm256_mul_pd(r, _mm256_set_pd(z_s[3],z_s[2],z_s[1],z_s[0]));
      y[j] = _mm256_mul_pd(r, _mm256_set_pd(z_c[3],z_c[2],z_c[1],z_c[0]));
    }

    // Pairwise distances
    __m256d dx[10], dy[10];

    // Unroll nested comparison loops
    dx[0] = x[0] - x[1]; dy[0] = y[0] - y[1];
    dx[1] = x[0] - x[2]; dy[1] = y[0] - y[2];
    dx[2] = x[0] - x[3]; dy[2] = y[0] - y[3];
    dx[3] = x[0] - x[4]; dy[3] = y[0] - y[4];
    dx[4] = x[1] - x[2]; dy[4] = y[1] - y[2];
    dx[5] = x[1] - x[3]; dy[5] = y[1] - y[3];
    dx[6] = x[1] - x[4]; dy[6] = y[1] - y[4];
    dx[7] = x[2] - x[3]; dy[7] = y[2] - y[3];
    dx[8] = x[2] - x[4]; dy[8] = y[2] - y[4];
    dx[9] = x[3] - x[4]; dy[9] = y[3] - y[4];

    auto max_d2 = _mm256_set_pd(0, 0, 0, 0);
    for (unsigned i = 0; i < 10; i++) {
      auto dx2 = _mm256_mul_pd(dx[i], dx[i]);
      auto dy2 = _mm256_mul_pd(dy[i], dy[i]);
      auto d2 = _mm256_add_pd(dx2, dy2);
      max_d2 = _mm256_max_pd(max_d2, d2);
    }
    auto max_d = _mm256_sqrt_pd(max_d2);
    double es[4];
    _mm256_storeu_pd(&es[0], max_d);

    sum += (es[0] + es[1] + es[2] + es[3]);
  }
  return sum / 4;
}

int main(int argc, char* argv[])
{
  unsigned power = 12;
  if (argc == 2) {
  	power = atoi(argv[1]);
  }
  unsigned nt = omp_get_max_threads();
  double avg = 0, min = 100, max = 0;
  auto start_time = std::chrono::system_clock::now();
  #pragma omp parallel for reduction (+:avg) reduction (min:min) reduction (max:max)
  for (unsigned j = 0; j < nt; j++) {
    #ifdef USE_MCG
      __uint128_t mcg128_state; // can be seeded to any odd number
      _rdrand64_step((unsigned long long *)(&mcg128_state));
      mcg128_state = (mcg128_state << 1) | 1;
      auto r = recurse(mcg128_state, power);
    #else
      // Need 40 PRNGs to get impact coordinates for 4 groups of 5 shots at once
      __m256i xoshiro_state[40];
      for (unsigned i = 0; i < 40; i++) {
        uint64_t init[4];
        for (unsigned k = 0; k < 4; k++) {
          _rdrand64_step((unsigned long long *)&init[k]);
        }
        xoshiro_state[i] = _mm256_set_epi64x(init[3], init[2], init[1], init[0]);
      }
      auto r = recurse(xoshiro_state, power);
    #endif

    avg += r;
    min = fmin(r, min);
    max = fmax(r, max);
  }
  avg /= nt;
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end_time - start_time;
  std::cout.precision(14);
  std::cout << "code,threads,power_of_4,min,avg,max,time\n";
  std::cout << "AVX2," << nt << "," << power << "," << min << "," << avg << "," << max << "," << seconds.count() << "\n";
  return 0;
}
