
// Building:
//
// g++ -Wall -O3 omp_es_mcg.cpp -lm -o omp_es_mcg -fopenmp -pg
//
#include <omp.h>
#include <iostream>
#include <numeric>
#include <complex>

// MCG 128 PRNG from http://www.pcg-random.org/posts/on-vignas-pcg-critique.html
inline double mcg128(__uint128_t& mcg128_state)
{
  return ldexp((mcg128_state *= 0xda942042e4dd58b5ULL) >> 64, -64);
}

double recurse(__uint128_t& mcg_state, unsigned zeroes)
{
  double sum = 0;
  for (unsigned group = 0; group < 10; group++) {
    if (zeroes) {
  	  sum += recurse(mcg_state, zeroes - 1);
    } else {
      std::complex<double> impact[5];
      for (unsigned shot = 0; shot < 5; shot++) {
        for (;;) {
          // Box-Muller transform
          double r = sqrt(-2 * log(mcg128(mcg_state)));
          double theta = 2 * M_PI * mcg128(mcg_state);
          double x = r * cos(theta);
          double y = r * sin(theta);
          if (std::isfinite(x) && std::isfinite(y)) {
            impact[shot] = std::complex<double>(x, y);
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
    __uint128_t prng_state = 2 * omp_get_thread_num() + 1; // can be seeded to any odd number
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
