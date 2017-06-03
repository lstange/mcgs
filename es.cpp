// es.cpp
//
// Monte-Carlo simulation to estimate expected values of metrics for 2 to N shot groups
// with impact coordinates pulled from bivariate normal distribution
//
// es      Extreme spread
// pdwr    Average pairwise distance weighted by rank
// pdwrt   Average pairwise distance weighted by rank, trimmed to exclude N - 1 greatest distances
// rwmrwpd Rank weighted mean of right Winsorized pairwise distances
//
// Use dynamic programming: start with one shot group, add shots one by one. Keep track 
// of extreme spread (group size) so far, update it as new shots are being added.
//
// Keep ordered list of pairwise distances between shots in a group, use it to calculate pdwr and pdwrt
//
#include <iostream>
#include <random>
#include <set>
#include <complex>

int main(int argc, char* argv[])
{
  const unsigned N = 10;
  unsigned groups = 1000000, seed = 0;
  switch (argc) {
    case 3: seed = atoi(argv[2]);
    case 2: groups = atoi(argv[1]);
  }
  std::default_random_engine gen(seed);
  std::normal_distribution<double> dist;
  double es_mean     [N] = {}; unsigned es_count     [N] = {};
  double pdwr_mean   [N] = {}; unsigned pdwr_count   [N] = {};
  double pdwrt_mean  [N] = {}; unsigned pdwrt_count  [N] = {};
  double rwmrwpd_mean[N] = {}; unsigned rwmrwpd_count[N] = {};
  for (unsigned group = 0; group < groups; group++) {
    std::complex<double> impact[N];
    double extreme_spread = 0;
    std::multiset<double> pairwise_distance; // multiset keeps distances ordered
    for (unsigned i = 0; i < N; i++) {
      std::complex<double> point(dist(gen), dist(gen));
      impact[i] = point;
      for (unsigned j = 0; j < i; j++) {
        double candidate = std::abs(point - impact[j]);
        pairwise_distance.insert(candidate);
        if (extreme_spread < candidate) {
          extreme_spread = candidate;
        }
      }
      es_mean[i] += (extreme_spread - es_mean[i]) / (++es_count[i]);

      double pdwr = 0, pdwrt = 0, rwmrwpd = 0, winsor = 0;
      unsigned rank = 0;
      for (auto it = pairwise_distance.begin(); it != pairwise_distance.end(); ++it) {
        rank++;
        pdwr += *it * rank;
        if (rank == (i + 1) * i / 2 - i) {
          winsor = *it;
        }
        if (rank <= (i + 1) * i / 2 - i) {
          pdwrt += *it * rank;
          rwmrwpd += *it * rank;
        } else {
          rwmrwpd += winsor * rank;
        }
      }
      // Divide by sum of weights (ranks) to get weighted average
      pdwr /= rank * (rank + 1) / 2;
      pdwrt /= (rank - i) * (rank - i + 1) / 2;
      rwmrwpd /= rank * (rank + 1) / 2;

      pdwr_mean   [i] += (pdwr    - pdwr_mean   [i]) / (++pdwr_count   [i]);
      pdwrt_mean  [i] += (pdwrt   - pdwrt_mean  [i]) / (++pdwrt_count  [i]);
      rwmrwpd_mean[i] += (rwmrwpd - rwmrwpd_mean[i]) / (++rwmrwpd_count[i]);
    }
  }
  std::cout << "shots\tes\tpdwr\tpdwrt\trwmrwpd\n";
  for (unsigned i = 1; i < N; i++) {
    std::cout << (i + 1) << "\t" << es_mean[i] << "\t" << pdwr_mean[i] 
              << "\t" << pdwrt_mean[i] << "\t" << rwmrwpd_mean[i] << "\n";
  }
  return 0;
}
