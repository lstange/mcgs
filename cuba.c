/*

  Extreme spread of N shot group assuming impact coordinates follow standard normal distribution

  VEGAS from http://www.feynarts.de/cuba/

  Building:

	gcc -o cuba -O3 cuba.c -lcuba -lm

   3.0658794642
+- 0.0000015887
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuba.h"

#define N 5

static int Integrand(const int *ndim, const cubareal xx[],
  const int *ncomp, cubareal ff[], void *userdata) 
{
  double x[N], y[N], xxx, yyy, r, theta, candidate, extreme_spread2 = 0;
  unsigned i, j;
  for (i = 0; i < N; i++) {
    r = sqrt(-2 * log(xx[i]));
    theta = 2 * M_PI * xx[i + N];
    xxx = r * cos(theta);
    yyy = r * sin(theta);
    if (isfinite(xxx) && isfinite(yyy)) {
      x[i] = xxx;
      y[i] = yyy;
    } else {
      x[i] = 0;
      y[i] = 0;
    }
  }
  for (i = 0; i < N - 1; i++) {
    for (j = i + 1; j < N; j++) {
       candidate = (x[i] - x[j]) * (x[i] - x[j])+(y[i] - y[j]) * (y[i] - y[j]);
       if (extreme_spread2 < candidate) {
         extreme_spread2 = candidate;
       }
    }
  }
  ff[0] = sqrt(extreme_spread2);
  return 0;
}

int main(int argc, char* argv[])
{
  unsigned zeroes = 8;
  long long calls = 10, comp, nregions, neval;
  int fail;
  cubareal integral[1], error[1], prob[1];

  if (argc == 2) {
    zeroes = atoi(argv[1]);
  }
  while (--zeroes) {
    calls *= 10;
  }
  llVegas(2*N, 1, Integrand, NULL, 1, 1e-16, 1e-6, 
    0, 0, // 0,0 is Sobol, 3<<8,42 is RanLux, 0,42 is Mersenne Twister 
    0, calls, 1000, 500, 1000, 0, NULL, NULL,
    &neval, &fail, integral, error, prob);
  printf("VEGAS result:\tneval %lld\tfail %d\n", neval, fail);
  printf("   %.10f\n"
         "+- %.10f\n", (double)integral[0], (double)error[0]);
  return 0;
}
