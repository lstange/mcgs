// Extreme spread of 1 to N shot groups, with impact coordinates pulled from bivariate 
// normal distribution. Use dynamic programming: start with one shot group, add shots 
// one by one. Keep track of group size so far, update it as new shots are being added.
// Update mean extreme spread for each group size from 1 to N. 
#include <iostream>
#include <random>
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
  double mean[N] = {};
  unsigned count[N] = {};
  for (unsigned group = 0; group < groups; group++) {
    std::complex<double> impact[N];
    double d = 0;
    for (unsigned i = 0; i < N; i++) {
      std::complex<double> p(dist(gen), dist(gen));
      impact[i] = p;
      for (unsigned j = 0; j < i; j++) {
        double candidate = std::abs(p - impact[j]);
        if (d < candidate) {
          d = candidate;
        }
      }
      mean[i] += (d - mean[i]) / (++count[i]);
    }
  }
  for (unsigned i = 0; i < N; i++) {
    std::cout << (i + 1) << "\t" << mean[i] << "\n";
  }
  return 0;
}
