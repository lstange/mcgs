// ladder.cpp
//
// Rifle A produces 5 shot groups with 1 MOA average size.
// Rifle B produces 5 shot groups with k MOA average size, where k >= 1.
// Plot probability that a group from rifle A is larger than a group from rifle B.
//
// Probability that a 5 shot group from 1 MOA rifle is larger than a 5 shot group from 1.5 MOA rifle is approximately 15%
// 
#include <iostream>
#include <random>
#include <complex>

int main(void)
{
  const unsigned SHOTS_IN_GROUP = 5;
  const unsigned K_POINTS = 41;
  const double K_STEP = .05;
  unsigned groups = 1000000000, seed = 0;
  std::mt19937_64 gen(seed);
  std::normal_distribution<double> dist;
  unsigned a_gt_b_count[K_POINTS] = {};
  double prev_es = 0;
  std::complex<double> first_group[SHOTS_IN_GROUP];
  for (unsigned group = 0; group <= groups; group++) { // one more pass than groups because we reuse previous group
    std::complex<double> impact[SHOTS_IN_GROUP];
    double extreme_spread = 0;
    for (unsigned i = 0; i < SHOTS_IN_GROUP; i++) {
      std::complex<double> point(dist(gen), dist(gen));
      impact[i] = point;
      for (unsigned j = 0; j < i; j++) {
        double candidate = std::abs(point - impact[j]);
        if (extreme_spread < candidate) {
          extreme_spread = candidate;
        }
      }
    }
    if (group > 0) {
      // check if PRNG wrapped around and returned the same impacts
      bool wraparound = true;
      for (unsigned i = 0; i < SHOTS_IN_GROUP; i++) {
        if (first_group[i] != impact[i]) { // exact comparison, zero tolerance
          wraparound = false;
          break;
        }
      }
      if (wraparound) {
        std::cerr << "WARNING: PRNG wraparaound detected at group " << group << "\n";
      }

      for (unsigned k = 0; k < K_POINTS; k++) {
        if (extreme_spread > prev_es * (1 + k * K_STEP)) {
          a_gt_b_count[k]++;
        }
      } 
    } else { // save the very first group
      for (unsigned i = 0; i < SHOTS_IN_GROUP; i++) {
        first_group[i] = impact[i];
      }
    }
    prev_es = extreme_spread;
  }
  std::cout << "k\tP(ESa>ESb)\n";
  for (unsigned k = 0; k < K_POINTS; k++) {
    std::cout << (1 + k * K_STEP) << "\t" << 100. * a_gt_b_count[k] / groups << "%\n";
  }
  return 0;
}

// 5 shot groups
//
// k P(ESa>ESb)
// 1 49.998%
// 1.05  44.9864%
// 1.1 40.284%
// 1.15  35.9227%
// 1.2 31.9194%
// 1.25  28.2772%
// 1.3 24.9857%
// 1.35  22.032%
// 1.4 19.3935%
// 1.45  17.0488%
// 1.5 14.9708%
// 1.55  13.1363%
// 1.6 11.5209%
// 1.65  10.1013%
// 1.7 8.8562%
// 1.75  7.76493%
// 1.8 6.80971%
// 1.85  5.97425%
// 1.9 5.24483%
// 1.95  4.60698%
// 2 4.04886%
// 2.05  3.56101%
// 2.1 3.13466%
// 2.15  2.76154%
// 2.2 2.43522%
// 2.25  2.14967%
// 2.3 1.89955%
// 2.35  1.68017%
// 2.4 1.48789%
// 2.45  1.31881%
// 2.5 1.1704%
// 2.55  1.0399%
// 2.6 0.924795%
// 2.65  0.823675%
// 2.7 0.734218%
// 2.75  0.655321%
// 2.8 0.585333%
// 2.85  0.523511%
// 2.9 0.468741%
// 2.95  0.420244%
// 3 0.377059%
