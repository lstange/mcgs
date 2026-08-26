//
// Extreme spread of TEN-shot group assuming impact coordinates follow standard normal distribution
//
// Monte-Carlo using AVX2 instructions, parallelized with OpenMP
//
// Ported from es_omp_additive_avx2.cpp (five-shot). Same additive (Richtmyer)
// quasi-random generator, same Box-Muller, same balanced-tree averaging; only
// the shot count changes, from 5 shots / 10 dimensions / 10 pairwise distances
// to 10 shots / 20 dimensions / 45 pairwise distances.
//
// Building:
//
// g++-9 -Wall -O3 -std=c++11 -march=native es10_omp_additive_avx2.cpp -lm -lmvec -g -o es10_omp_additive_avx2 -fopenmp
//
// Running:
//
// for run in {1..10}; do ./es10_omp_additive_avx2 14 | tee -a es10_omp_additive_avx2.csv; done
//
// Expect roughly 3-4x the runtime of the five-shot version at the same power:
// 45 pairwise distances instead of 10, 20 QRNGs instead of 10, 10 Box-Muller
// pairs instead of 5. Reference value to converge on: E[ES10] = 3.8115826 sigma.
//
#include <cassert>
#include <omp.h>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <x86intrin.h>

#define USE_ROOTS_OF_PRIMES

// Ten shots, two coordinates each: 20 quasi-random dimensions. Radius of shot j
// is drawn from dimension j, its angle from dimension j + SHOTS -- the same
// "split" layout the five-shot version uses.
static const unsigned SHOTS = 10;
static const unsigned DIMS = 2 * SHOTS;   // 20
static const unsigned PAIRS = SHOTS * (SHOTS - 1) / 2;  // 45, for reference

#ifndef USE_ROOTS_OF_PRIMES
#error "The generalized-golden-ratio (R_d) generator must not be used at 20 dimensions with this dimension layout."
// Why this is an #error rather than a comment:
//
// R_d sets alpha_j = g^-j, with g the positive root of x^(d+1) = x + 1. Those
// alphas form a geometric progression, so under the split layout above EVERY
// shot's (radius, angle) pair shares the identical ratio alpha_{j+SHOTS} /
// alpha_j = g^-SHOTS. All ten shots then inherit the same aliasing error in the
// same 2-D projection and it accumulates coherently instead of ten unrelated
// errors partly cancelling.
//
// Measured on exactly this integrand, 32 randomized shifts, standard error of
// one run (lower is better):
//
//        N      sqrt(primes)   R_d
//        2^14     2.1e-03      5.1e-03
//        2^16     1.3e-03      5.9e-03   <- R_d not converging at all
//        2^18     5.0e-04      4.6e-03
//        2^20     1.6e-04      9.2e-04
//
// R_d simply stops improving between 2^14 and 2^18 while sqrt(primes) converges
// normally. It is NOT broken in general -- at 10 dimensions (the five-shot
// case) it is fully competitive, and on a smooth product test function it is
// fine even at 20 dimensions. The failure is specific to the interaction
// between its algebraic structure and this pairing of dimensions, and it is
// silent: the answer is simply wrong-ish and stops getting better.
//
// If you want R_d here anyway, either pair each shot's radius and angle from
// ADJACENT dimensions (2j, 2j+1) or shuffle the assignment of alphas to
// dimensions -- both break the shared ratio and restore normal convergence.
#endif

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
  __m256i qrng_state[DIMS],
  const __m256i qrng_step[DIMS],
  unsigned zeroes)
{
  double sum = 0;
  if (zeroes) {
    for (unsigned group = 0; group < 4; group++) {
      sum += recurse(qrng_state, qrng_step, zeroes - 1);
    }
  } else {
    double qrng_output[4 * DIMS];

    // Advance all QRNGs
    for (unsigned k = 0; k < DIMS; k++) {
      advance_qrng(qrng_state[k], qrng_step[k], &qrng_output[4 * k]);
    }

    // Box-Muller transform, four at a time.
    // qrng_output[4*k + l] is dimension k, lane l -- and each lane is its own
    // independent group, so all four groups advance in lockstep.
    const auto minus2 = _mm256_set_pd(-2, -2, -2, -2);
    const auto twopi = _mm256_set_pd(2 * M_PI, 2 * M_PI, 2 * M_PI, 2 * M_PI);
    double z_s[4], z_c[4];
    auto ptrs = _mm256_set_epi64x((uint64_t)&z_s[3],(uint64_t)&z_s[2],(uint64_t)&z_s[1],(uint64_t)&z_s[0]);
    auto ptrc = _mm256_set_epi64x((uint64_t)&z_c[3],(uint64_t)&z_c[2],(uint64_t)&z_c[1],(uint64_t)&z_c[0]);
    __m256d x[SHOTS], y[SHOTS];
    for (unsigned j = 0; j < SHOTS; j++) {
      const unsigned ur = 4 * j;                 // radius: dimension j
      const unsigned ua = 4 * (j + SHOTS);       // angle:  dimension j + SHOTS
      auto u = _mm256_set_pd(qrng_output[ur+3], qrng_output[ur+2], qrng_output[ur+1], qrng_output[ur]);
      auto v = _mm256_set_pd(qrng_output[ua+3], qrng_output[ua+2], qrng_output[ua+1], qrng_output[ua]);
      auto r = _mm256_sqrt_pd(_mm256_mul_pd(minus2, _ZGVdN4v_log(u)));
      auto theta = _mm256_mul_pd(twopi, v);
      _ZGVdN4vvv_sincos(theta, ptrs, ptrc);
      // Sine into x and cosine into y, as in the five-shot version. The
      // swap versus the usual convention is a rotation of the whole group and
      // cannot change a distance between two of its shots.
      x[j] = _mm256_mul_pd(r, _mm256_set_pd(z_s[3],z_s[2],z_s[1],z_s[0]));
      y[j] = _mm256_mul_pd(r, _mm256_set_pd(z_c[3],z_c[2],z_c[1],z_c[0]));
    }

    // Extreme spread: largest of the 45 pairwise distances, four groups at once.
    // The five-shot version spelled its 10 pairs out longhand; at 45 pairs that
    // would need 90 live vectors against 16 architectural registers, so the max
    // is accumulated in place instead and nothing is kept alive across pairs.
    auto max_d2 = _mm256_setzero_pd();
    for (unsigned a = 0; a + 1 < SHOTS; a++) {
      for (unsigned b = a + 1; b < SHOTS; b++) {
        auto dx = _mm256_sub_pd(x[a], x[b]);
        auto dy = _mm256_sub_pd(y[a], y[b]);
        auto d2 = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
        max_d2 = _mm256_max_pd(max_d2, d2);
      }
    }
    auto max_d = _mm256_sqrt_pd(max_d2);
    double es[4];
    _mm256_storeu_pd(&es[0], max_d);

    sum += (es[0] + es[1] + es[2] + es[3]);
  }
  // Every level averages exactly four equal-weight children, so the whole
  // recursion is a balanced summation tree -- far better conditioned than
  // accumulating 4^(power+1) values into one running total.
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

// splitmix64 -- only used to build a fallback seed if RDRAND is unavailable or
// keeps failing, so that every (dimension, lane) still starts somewhere
// different. See seed_state() below for why that matters.
static uint64_t splitmix64(uint64_t& z)
{
  z += 0x9e3779b97f4a7c15ULL;
  uint64_t r = z;
  r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ULL;
  r = (r ^ (r >> 27)) * 0x94d049bb133111ebULL;
  return r ^ (r >> 31);
}

// One random starting offset for a single (dimension, lane).
//
// The five-shot version assigned a fallback value before calling
// _rdrand64_step and ignored the return code -- but on failure the instruction
// writes 0 to its destination, overwriting that fallback. Worse, the fallback
// was thread_num * seconds, identical across all dimensions and all four lanes,
// so a machine without working RDRAND would have started every lane of every
// dimension at the same offset. The four lanes are the independent randomized
// shifts whose spread becomes the min/max column, so that would silently
// collapse them into one and make the reported spread far too narrow --
// exactly the wrong failure when the point of the run is counting significant
// digits. Retry, then fall back to something genuinely distinct per slot.
static uint64_t seed_state(uint64_t& fallback_z)
{
  unsigned long long r = 0;
  for (int attempt = 0; attempt < 100; attempt++) {
    if (_rdrand64_step(&r) && r != 0) {
      return (uint64_t)r;
    }
  }
  return splitmix64(fallback_z);
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
    // Need 4 * DIMS QRNGs to get impact coordinates for 4 groups of SHOTS shots
    // at once. Each __m256i holds 4 of them (one per group), hence DIMS of those.
    // Steps are the same for each of the four groups, initial states differ.
    __m256i qrng_state[DIMS];
    __m256i qrng_step[DIMS];
#ifndef USE_ROOTS_OF_PRIMES
    long double x = 1.5; // initial approximation
    for (unsigned i = 0; i < 30; i++) { // Newton-Raphson for x^(DIMS+1) = x + 1
      long double xd = 1;
      for (unsigned k = 0; k < DIMS; k++) {
        xd *= x;
      }
      long double new_x = x - (xd * x - x - 1) / ((DIMS + 1) * xd - 1);
      if (new_x == x) {
        break;
      }
      x = new_x;
    }
#endif
    // Distinct per thread, so the fallback path (if it is ever reached) still
    // gives every thread its own stream rather than a shared one.
    uint64_t fallback_z =
      (uint64_t)(omp_get_thread_num() + 1) * 0x9e3779b97f4a7c15ULL
      ^ (uint64_t)(num_seconds * 1e6);

    long double a = 1.;
    for (unsigned i = 0; i < DIMS; i++) {
#ifdef USE_ROOTS_OF_PRIMES
      // DIMS entries required -- the five-shot version's table had 15 and used
      // only the first 10, which would read past the end at 20 dimensions.
      const int prime[DIMS] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71};
      a = sqrt((long double)prime[i]);
      a -= floor(a);
#else
      a /= x;
#endif
      uint64_t step = fractional_part(a);
      qrng_step[i] = _mm256_set_epi64x(step, step, step, step);
      uint64_t states[4];
      for (int k = 0; k < 4; k++) {
        states[k] = seed_state(fallback_z);
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
  std::cout << "Additive AVX2 10-shot (prime roots),";
#else
  std::cout << "Additive AVX2 10-shot,";
#endif
  std::cout<< nt << "," << power << "," << min << "," << avg << "," << max << "," << seconds.count() << "\n";
  return 0;
}
