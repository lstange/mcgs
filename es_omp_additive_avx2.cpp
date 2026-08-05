//
// Extreme spread of five-shot group assuming impact coordinates follow standard normal distribution
//
// Monte-Carlo using AVX2 instructions, parallelized with OpenMP 
//
// Building:
//
// g++-9 -Wall -O3 -std=c++11 -march=native es_omp_additive_avx2.cpp -lm -g -o es_omp_additive_avx2 -fopenmp
//
// Running:
//
// for run in {1..10}; do ./es_omp_additive_avx2 14 | tee -a es_omp_additive_avx2.csv; done
//
#include <cassert>
#include <omp.h>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <x86intrin.h>

#define USE_ROOTS_OF_PRIMES

extern "C" {
  __m256d _ZGVdN4v_log(__m256d x);                    
  void    _ZGVdN4vvv_sincos(__m256d x, __m256i ptrs, __m256i ptrc);
}

void inline advance_qrng(__m256i& state, __m256i step, double output[4])
{
  state = _mm256_add_epi64(state, step);
  uint64_t result[4];
  _mm256_storeu_si256((__m256i*)&result, state);
  for (int i = 0; i < 4; i++) {
    output[i] = ldexp(result[i], -64);
  }
}

double recurse(
  __m256i qrng_state[10],
  const __m256i qrng_step[10],
  unsigned zeroes)
{
  double sum = 0;
  if (zeroes) {
    for (unsigned group = 0; group < 4; group++) {
      sum += recurse(qrng_state, qrng_step, zeroes - 1);
    }
  } else {
    double qrng_output[40];

    // Advance all QRNGs
    for (unsigned k = 0; k < 10; k++) {
      advance_qrng(qrng_state[k], qrng_step[k], &qrng_output[4 * k]);
    }

    // Box-Muller transform, four at a time
    const auto minus2 = _mm256_set_pd(-2, -2, -2, -2);
    const auto twopi = _mm256_set_pd(2 * M_PI, 2 * M_PI, 2 * M_PI, 2 * M_PI); 
    double z_s[4], z_c[4];
    auto ptrs = _mm256_set_epi64x((uint64_t)&z_s[3],(uint64_t)&z_s[2],(uint64_t)&z_s[1],(uint64_t)&z_s[0]);
    auto ptrc = _mm256_set_epi64x((uint64_t)&z_c[3],(uint64_t)&z_c[2],(uint64_t)&z_c[1],(uint64_t)&z_c[0]);
    __m256d x[20], y[20];
    for (unsigned i = 0, j = 0; i < 20; i += 4, j++) {
      auto u = _mm256_set_pd(qrng_output[i+3], qrng_output[i+2], qrng_output[i+1], qrng_output[i]);
      auto v = _mm256_set_pd(qrng_output[i+23], qrng_output[i+22], qrng_output[i+21], qrng_output[i+20]);
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

static uint64_t fractional_part(long double d)
{
  long double i; // should be zero
  union {
    long double d;
    uint64_t bits;
    uint16_t exp_sign;
  } u;
  assert(sizeof(u) == sizeof(d));
  u.d = modfl(d, &i);
  assert(i == 0);
  int e = ilogbl(d);
  uint64_t f = u.bits;
  if (e < 0) {
    f >>= -e - 1;
  } 
  return f;
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
  auto duration_in_seconds = std::chrono::duration<double>(start_time.time_since_epoch());
  double num_seconds = duration_in_seconds.count();
  #pragma omp parallel for reduction (+:avg) reduction (min:min) reduction (max:max)
  for (unsigned j = 0; j < nt; j++) {
    // Need 40 QRNGs to get impact coordinates for 4 groups of 5 shots at once.
    // Each __m256i has 4 QRNGs, hence 10 of those.
    // Steps are the same for each of the four groups, initial states are different
    __m256i qrng_state[10];
    __m256i qrng_step[10];
#ifndef USE_ROOTS_OF_PRIMES
    long double x = 1.5; // initial approximation
    for (unsigned i = 0; i < 30; i++) { // Newton-Raphson
      long double xd = 1;
      for (unsigned j = 0; j < 10; j++) {
        xd *= x;
      }
      long double new_x = x - (xd * x - x - 1) / ((10 + 1) * xd - 1);
      if (new_x == x) {
        break;
      }
      x = new_x;
    }
#endif
    long double a = 1.;
    for (unsigned i = 0; i < 10; i++) {
#ifdef USE_ROOTS_OF_PRIMES
      const int prime[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47};
      a = sqrt((long double)prime[i]);
      a -= floor(a);
#else
      a /= x;
#endif
      uint64_t step = fractional_part(a);
      qrng_step[i] = _mm256_set_epi64x(step, step, step, step);
      uint64_t states[4];
      for (int j = 0; j < 4; j++) {
        long long unsigned int random_number = omp_get_thread_num() * (uint64_t)num_seconds; // fallback
        _rdrand64_step(&random_number);
        states[j] = (uint64_t)random_number;
      }
      qrng_state[i] = _mm256_set_epi64x(states[0], states[1], states[2], states[3]);
    }
    auto r = recurse(qrng_state, qrng_step, power);
    avg += r;
    min = fmin(r, min);
    max = fmax(r, max);
  }
  avg /= nt;
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end_time - start_time;
  std::cout.precision(14);
  ///std::cout << "code,threads,power_of_4,min,avg,max,time\n";
#ifdef USE_ROOTS_OF_PRIMES
  std::cout << "Additive AVX2 (prime roots),";
#else
  std::cout << "Additive AVX2,";
#endif
  std::cout<< nt << "," << power << "," << min << "," << avg << "," << max << "," << seconds.count() << "\n";
  return 0;
}
