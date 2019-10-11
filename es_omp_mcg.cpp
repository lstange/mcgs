//
// Extreme spread of five shot group assuming impact coordinates follow standard normal distribution
//
// Monte-Carlo parallelized with OpenMP
//
// Building:
//
// g++-8 -Wall -O3 -std=c++11 -march=native es_omp_mcg.cpp -lm -g -o es_omp_mcg -fopenmp
//
// Running:
//
// for run in {1..10}; do ./es_omp_mcg | tee -a es_omp_mcg.csv; done
//
#include <omp.h>
#include <iostream>
#include <numeric>
#include <complex>
#include <chrono>

// MCG 128 PRNG from http://www.pcg-random.org/posts/on-vignas-pcg-critique.html
inline double mcg128(__uint128_t& mcg128_state)
{
  return ldexp((mcg128_state *= 0xda942042e4dd58b5ULL) >> 64, -64);
}

double recurse(__uint128_t& mcg_state, unsigned zeroes)
{
  double sum = 0;
  for (unsigned group = 0; group < 4; group++) {
    if (zeroes) {
  	  sum += recurse(mcg_state, zeroes - 1);
    } else {
      std::complex<double> impact[5];
      for (unsigned shot = 0; shot < 5; shot++) {
        for (;;) {
          // Box-Muller transform
          auto r = sqrt(-2 * log(mcg128(mcg_state)));
          auto theta = 2 * M_PI * mcg128(mcg_state);
          auto x = r * cos(theta);
          auto y = r * sin(theta);
          if (std::isfinite(x) && std::isfinite(y)) {
            impact[shot] = std::complex<double>(x, y);
            break;
          }
        }
      }
      double extreme_spread = 0;
      for (unsigned i = 0; i < 5 - 1; i++) {
        for (unsigned j = i + 1; j < 5; j++) {
          auto candidate = std::abs(impact[i] - impact[j]);
          if (extreme_spread < candidate) {
            extreme_spread = candidate;
          }
        }
      }
      sum += extreme_spread;
    }
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
  auto duration_in_seconds = std::chrono::duration<double>(start_time.time_since_epoch());
  double num_seconds = duration_in_seconds.count();
  #pragma omp parallel for reduction (+:avg) reduction (min:min) reduction (max:max)
  for (unsigned j = 0; j < nt; j++) {
    __uint128_t prng_state = ((__uint128_t)num_seconds << 8)
                           + (omp_get_thread_num() << 1)
                           + 1; // can be seeded to any odd number
    auto r = recurse(prng_state, power);
    avg += r;
    min = fmin(r, min);
    max = fmax(r, max);
  }
  avg /= nt;
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end_time - start_time;
  std::cout.precision(14);
  std::cout << "code,threads,power_of_4,min,avg,max,time\n";
  std::cout << "Regular," << nt << "," << power << "," << min << "," << avg << "," << max << "," << seconds.count() << "\n";
  return 0;
}
