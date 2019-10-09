//
// Extreme spread of N shot group assuming impact coordinates follow standard normal distribution
//
// Monte-Carlo using AVX2 instructions, parallelized with OpenMP 
//
// Building:
//
// g++ -Wall -O3 -march=native omp_es_avx2.cpp -lm -g -o omp_es_avx2 -fopenmp
//
#include <omp.h>
#include <iostream>
#include <numeric>
#include <complex>
#include <x86intrin.h>

extern "C" {
  __m256d _ZGVdN4v_log(__m256d x);                    
  void    _ZGVdN4vvv_sincos(__m256d x, __m256i ptrs, __m256i ptrc);
}

void xorshiro256plus(__m256i state[4], double output[4])
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

double recurse(__m256i prng_state[104], unsigned zeroes)
{
  double x[52], y[52];
  if (!zeroes) {
    // Advance all PRNGs, four at a time
    double prng_output[104];
    for (unsigned k = 0; k < 104; k += 4) {
      xorshiro256plus(&prng_state[k], &prng_output[k]);
    }

    // Box-Muller transform, four at a time
    const __m256d minus2 = _mm256_set_pd(-2, -2, -2, -2);
    const __m256d twopi = _mm256_set_pd(2 * M_PI, 2 * M_PI, 2 * M_PI, 2 * M_PI); 
    double z_s[4], z_c[4];
    __m256i ptrs = _mm256_set_epi64x((uint64_t)&z_s[3],(uint64_t)&z_s[2],(uint64_t)&z_s[1],(uint64_t)&z_s[0]);
    __m256i ptrc = _mm256_set_epi64x((uint64_t)&z_c[3],(uint64_t)&z_c[2],(uint64_t)&z_c[1],(uint64_t)&z_c[0]);             /* Pointers to the elements of z_s and z_c    */ 
    for (unsigned i = 0; i < 52; i += 4) {
      __m256d u = _mm256_set_pd(prng_output[i+3], prng_output[i+2], prng_output[i+1], prng_output[i]);
      __m256d v = _mm256_set_pd(prng_output[i+55], prng_output[i+54], prng_output[i+53], prng_output[i+52]);
      __m256d r = _mm256_sqrt_pd(_mm256_mul_pd(minus2, _ZGVdN4v_log(u))); 
      __m256d theta = _mm256_mul_pd(twopi, v);
      _ZGVdN4vvv_sincos(theta, ptrs, ptrc);
      _mm256_storeu_pd(&x[i], _mm256_mul_pd(r, _mm256_set_pd(z_s[3],z_s[2],z_s[1],z_s[0])));
      _mm256_storeu_pd(&y[i], _mm256_mul_pd(r, _mm256_set_pd(z_c[3],z_c[2],z_c[1],z_c[0])));
    }
  }
  double sum = 0;
  unsigned pos = 0;
  for (unsigned group = 0; group < 10; group++) {
    if (zeroes) {
  	  sum += recurse(prng_state, zeroes - 1);
    } else {
      std::complex<double> impact[5];
      for (unsigned shot = 0; shot < 5; shot++) {
        for (;;) {
          double xx = x[pos];
          double yy = y[pos];
          pos++;
          if (pos >= 104) {
            pos = 0;
          }
          if (std::isfinite(xx) && std::isfinite(yy)) {
            impact[shot] = std::complex<double>(xx, yy);
            break;
          }
        }
      }
      double extreme_spread = 0;
      for (unsigned i = 0; i < 5 - 1; i++) {
        for (unsigned j = i + 1; j < 5; j++) {
          double candidate = std::abs(impact[i] - impact[j]);
          if (extreme_spread < candidate) {
            extreme_spread = candidate;
          }
        }
      }
      sum += extreme_spread;
    }
  }
  return sum / 10;
}

int main(int argc, char* argv[])
{
  unsigned power = 6;
  if (argc == 2) {
  	power = atoi(argv[1]);
  }
  unsigned nt = omp_get_max_threads();
  double avg = 0, min = 100, max = 0;
  #pragma omp parallel for reduction (+:avg) reduction (min:min) reduction (max:max)
  for (unsigned j = 0; j < nt; j++) {

    // Need at least 100 PRNGs to get impact coordinates of 50 shots at a time
    // Actually 104 is easier because 50 is not divisible by 4, but 52 is
    __m256i prng_state[104];

    // Initialize xorshiro256+ state using MCG 128 PRNG 
    // from http://www.pcg-random.org/posts/on-vignas-pcg-critique.html
    __uint128_t mcg128_state = 2 * omp_get_thread_num() + 1; // can be seeded to any odd number
    for (unsigned i = 0; i < 104; i++) {
      uint64_t init[4];
      for (unsigned k = 0; k < 4; k++) {
        init[k] = (mcg128_state *= 0xda942042e4dd58b5ULL);
      }
      prng_state[i] = _mm256_set_epi64x(init[3], init[2], init[1], init[0]);
    }

    double r = recurse(prng_state, power);
    avg += r;
    min = fmin(r, min);
    max = fmax(r, max);
  }
  avg /= nt;
  std::cout.precision(8);
  std::cout << "threads=" << nt << "\n" 
            << "power=" << power << "\n" 
            << "min="<< min << "\n" 
            << "avg=" << avg << "\n" 
            << "max=" << max << "\n";
  return 0;
}
