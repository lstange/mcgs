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
  std::default_random_engine gen(seed);
  std::normal_distribution<double> dist;
  unsigned a_gt_b_count[K_POINTS] = {};
  double prev_es = 0;
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
      for (unsigned k = 0; k < K_POINTS; k++) {
        if (extreme_spread > prev_es * (1 + k * K_STEP)) {
          a_gt_b_count[k]++;
        }
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
// 1 49.9979%
// 1.05  44.9833%
// 1.1 40.2822%
// 1.15  35.9235%
// 1.2 31.9215%
// 1.25  28.2775%
// 1.3 24.986%
// 1.35  22.0332%
// 1.4 19.393%
// 1.45  17.0474%
// 1.5 14.9729%
// 1.55  13.1383%
// 1.6 11.5222%
// 1.65  10.1043%
// 1.7 8.86029%
// 1.75  7.76888%
// 1.8 6.81299%
// 1.85  5.97747%
// 1.9 5.24739%
// 1.95  4.60836%
// 2 4.05072%
//
// 3 shot groups
// k P(ESa>ESb)
// 1 50.0011%
// 1.05  46.4175%
// 1.1 43.0309%
// 1.15  39.8464%
// 1.2 36.8669%
// 1.25  34.0895%
// 1.3 31.5094%
// 1.35  29.1178%
// 1.4 26.906%
// 1.45  24.8649%
// 1.5 22.9834%
// 1.55  21.2504%
// 1.6 19.6576%
// 1.65  18.1919%
// 1.7 16.8453%
// 1.75  15.6066%
// 1.8 14.4686%
// 1.85  13.424%
// 1.9 12.4624%
// 1.95  11.5775%
// 2 10.7647%
// 2.05  10.0154%
// 2.1 9.32671%
// 2.15  8.69314%
// 2.2 8.10861%
// 2.25  7.56911%
// 2.3 7.07117%
// 2.35  6.61095%
// 2.4 6.18543%
// 2.45  5.79244%
// 2.5 5.42827%
// 2.55  5.09172%
// 2.6 4.77927%
// 2.65  4.48947%
// 2.7 4.22031%
// 2.75  3.97044%
// 2.8 3.73753%
// 2.85  3.52107%
// 2.9 3.31962%
// 2.95  3.13179%
// 3 2.95684%
