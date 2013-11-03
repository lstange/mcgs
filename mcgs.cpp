/* mcgs.cpp

  Monte-Carlo simulation of group sizes under bivariate normal distribution

*/
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <stdlib.h>

class ShotGroup {
  private:
    std::vector<double> x_;
    std::vector<double> y_;
  
  public:
    void add(double x, double y) {
      x_.push_back(x);
      y_.push_back(y);
    }
    
    double distance(int a, int b) {
      double x = x_.at(a) - x_.at(b);
      double y = y_.at(a) - y_.at(b);
      return sqrt(x * x + y * y);
    }
    
    double group_size(void) {
      if (x_.size() < 2) return 0;
      double best_so_far = distance(0, 1);
      for (unsigned i = 0; i < x_.size() - 1; i++) {
        for (unsigned j = i + 1; j < x_.size(); j++) {
          double candidate = distance(i, j);
          if (best_so_far < candidate) {
            best_so_far = candidate;
          }
        }
      }
      return best_so_far;
    }
};

static std::pair<double, double> pull_from_bivariate_normal(void)
{
  int iu; do { iu = rand(); } while (!iu);
  int iv; do { iv = rand(); } while (!iv);
  double u = (double)iu / RAND_MAX;
  double v = (double)iv / RAND_MAX;

  // Box-Muller transform
  double r = sqrt(-2 * log(u));
  double x = r * cos(2 * M_PI * v);
  double y = r * sin(2 * M_PI * v);

  return std::make_pair<double, double>(x, y);
}

static void descriptive_stats(const std::vector<double>& x, const char* metric)
{
  double average = accumulate(x.begin(), x.end(), 0.) / x.size();
  double variance = 0;
  for (unsigned i = 0; i < x.size(); i++) {
    double deviation = x.at(i) - average;
    variance += deviation * deviation;
  }
  variance /= (x.size() - 1);
  std::cout << metric << " mean=" << average << ", CV=" << sqrt(variance) / average << "\n";
}

static double median(std::vector<double>& x)
{
  std::vector<double>::iterator median_it = x.begin() + x.size() / 2; 
  nth_element(x.begin(), median_it, x.end());
  if (x.size() % 2) { // Odd number of elements
    return *median_it;
  }
  // Even number of elements, return average of two elements in the middle
  double b = *median_it--;
  nth_element(x.begin(), median_it, x.end());
  double a = *median_it;
  return (a + b) / 2;  
}

int main(int argc, char* argv[])
{
  int experiments = 0;
  if (argc > 1) {
    experiments = atoi(argv[1]);
    srand(experiments);
  } else {
    std::cerr << "Usage: mcgs experiments [[shots_in_group] groups_in_experiment]\n";
    return -1;
  }
  int shots_in_group = 5;
  if (argc > 2) {
    shots_in_group = atoi(argv[2]);
  }
  int groups_in_experiment = 1;
  if (argc > 3) {
    groups_in_experiment = atoi(argv[3]);
  }
  std::cout << experiments << " experiments, " << shots_in_group << " shots in group";
  if (groups_in_experiment > 1) {
    std::cout << ", " << groups_in_experiment << " group(s) per experiment"; 
  }
  std::cout << "\n";
  const double rayleigh_cep_factor = sqrt(4 * log(2) / M_PI); // 0.9394
  std::vector<double> gs_v;
  std::vector<double> bgs_v;
  std::vector<double> ags_v;
  std::vector<double> rayleigh_v;
  std::vector<double> median_r_v;
  std::vector<double> all_r;
  while (experiments --> 0) {
    double best_gs = 0, avg_gs = 0;
    for (int j = 0; j < groups_in_experiment; j++) { 
      ShotGroup g; 
      std::vector<double> r;
      for (int i = 0; i < shots_in_group; i++) {
        std::pair<double, double> p = pull_from_bivariate_normal();
        double x = p.first;
        double y = p.second;
        double ri = sqrt(x * x + y * y);
        g.add(x, y);
        r.push_back(ri);
        all_r.push_back(ri);
      }
      double this_gs = g.group_size();
      gs_v.push_back(this_gs);
      if (j) {
        if (best_gs > this_gs) {
          best_gs = this_gs;
        }
      } else {
        best_gs = this_gs;
      }
      avg_gs += this_gs;
      double rayleigh = rayleigh_cep_factor * accumulate(r.begin(), r.end(), 0.) / r.size();
      rayleigh_v.push_back(rayleigh);
      median_r_v.push_back(median(r));
    }
    avg_gs /= groups_in_experiment;
    bgs_v.push_back(best_gs);
    ags_v.push_back(avg_gs);
  }
  descriptive_stats(gs_v, "Group size");
  if (groups_in_experiment > 1) {
    descriptive_stats(ags_v, "Average group size");
    descriptive_stats(bgs_v, "Best group size");
  }
  descriptive_stats(rayleigh_v, "Rayleigh CEP estimator");
  descriptive_stats(median_r_v, "Median CEP estimator");

  std::vector<double>::iterator all_r50_it = all_r.begin() + all_r.size() / 2; 
  nth_element(all_r.begin(), all_r50_it, all_r.end());
  std::vector<double>::iterator all_r90_it = all_r.begin() + all_r.size() - all_r.size() / 10; 
  nth_element(all_r.begin(), all_r90_it, all_r.end());
  std::vector<double>::iterator all_r95_it = all_r.begin() + all_r.size() - all_r.size() / 20; 
  nth_element(all_r.begin(), all_r95_it, all_r.end());
  std::vector<double>::iterator all_r99_it = all_r.begin() + all_r.size() - all_r.size() / 100; 
  nth_element(all_r.begin(), all_r99_it, all_r.end());
  std::cout << "Observed R50=" << *all_r50_it << ", R90=" << *all_r90_it << ", R95=" << *all_r95_it << ", R99=" << *all_r99_it << "\n";
  std::cout << "Expected R50=" << sqrt(-2*log(0.5)) << ", R90=" << sqrt(-2*log(0.1)) << ", R95=" << sqrt(-2*log(0.05)) << ", R99=" << sqrt(-2*log(0.01)) << "\n";

  return 0;
}
