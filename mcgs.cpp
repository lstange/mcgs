/* mcgs.cpp

  Monte-Carlo simulation of group sizes under bivariate (possibly contaminated) normal distribution

*/
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <iostream>
#include <stdlib.h>

class ShotGroup {
  private:
    std::vector<double> x_;
    std::vector<double> y_;
  
  public:
    void add(double x, double y)
    {
      x_.push_back(x);
      y_.push_back(y);
    }
    
    double distance(int a, int b)
    {
      double x = x_.at(a) - x_.at(b);
      double y = y_.at(a) - y_.at(b);
      return sqrt(x * x + y * y);
    }
    
    double group_size(void)
    {
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

    double avg_miss_radius(void)
    {
      if (x_.size() < 2) return 0;
      double center_x = 0;
      double center_y = 0;
      for (int i = 0; i < x_.size(); i++) {
        center_x += x_.at(i);
        center_y += y_.at(i);
      }
      center_x /= x_.size();
      center_y /= y_.size();
      double amr = 0;
      for (int i = 0; i < x_.size(); i++) {
        amr += sqrt( (x_.at(i) - center_x) * (x_.at(i) - center_x) 
                   + (y_.at(i) - center_y) * (y_.at(i) - center_y)
                   );
      }
      amr /= x_.size();
      return amr;
    }
};

class DescriptiveStat
{
  public:
    DescriptiveStat() : n_(0) {}
    
    void push(double x)
    {
      if (!n_++) {
        m_ = x;
        s_ = 0;
      } else {
        double new_m = m_ + (x - m_) / n_;
        double new_s = s_ + (x - m_) * (x - new_m);
        m_ = new_m;
        s_ = new_s;
      }
    }

    double mean(void) const
    { 
      return (n_ > 0) ? m_ : 0;
    }

    double variance(void) const
    {
      return ( (n_ > 1) ? s_ / (n_ - 1) : 0 );
    }

    double stdev(void) const
    {
      return sqrt(variance());
    }

    int count(void) const
    {
      return n_;
    }

    void show(const char* metric) {
      std::cout << metric << " mean=" << mean() << ", CV=" << stdev() / mean() << "\n";
    }

  private:
    int n_;
    double m_;
    double s_;
};

static std::pair<double, double> pull_from_bivariate_normal(std::mt19937& gen, double deviation = 1.)
{
  std::uniform_real_distribution<> dis(0., 1.);
  double u; do { u = dis(gen); } while (!u);
  double v; do { v = dis(gen); } while (!v);

  // Box-Muller transform
  double r = deviation * sqrt(-2 * log(u));
  double x = r * cos(2 * M_PI * v);
  double y = r * sin(2 * M_PI * v);

  return std::make_pair<double, double>(x, y);
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
  } else {
    std::cerr << "Usage: mcgs experiments [[[[shots_in_group] groups_in_experiment] proportion_of_outliers] outlier_severity]\n";
    return -1;
  }
  std::mt19937 prng(experiments);
  std::uniform_real_distribution<> unit_distr(0., 1.);
  int shots_in_group = 5;
  if (argc > 2) {
    shots_in_group = atoi(argv[2]);
  }
  int groups_in_experiment = 1;
  if (argc > 3) {
    groups_in_experiment = atoi(argv[3]);
  }
  double proportion_of_outliers = 0.;
  if (argc > 4) {
    proportion_of_outliers = atof(argv[4]);
  }
  double outlier_severity = 10.;
  if (argc > 5) {
    outlier_severity = atof(argv[5]);
  }
  std::cout << experiments << " experiments, " << shots_in_group << " shots in group";
  if (groups_in_experiment > 1) {
    std::cout << ", " << groups_in_experiment << " group(s) per experiment"; 
  }
  std::cout << "\n";
  if (proportion_of_outliers > 0) {
    std::cerr << "Using contaminated normal distribution: " << 
    (proportion_of_outliers * 100) << "% of observations are pulled from distribution with " <<
    outlier_severity << " times higher deviation\n";
  }
  const double rayleigh_cep_factor = sqrt(4 * log(2) / M_PI); // 0.9394
  DescriptiveStat gs_s, bgs_s, ags_s, mgs_s, wgs_s, amr_s, aamr_s, rayleigh_s, median_r_s;
  std::vector<double> all_r;
  while (experiments --> 0) {
    double best_gs = 0, worst_gs = 0;
    DescriptiveStat gs, amr;
    std::vector<double> gsg_v;
    for (int j = 0; j < groups_in_experiment; j++) { 
      ShotGroup g; 
      std::vector<double> r;
      for (int i = 0; i < shots_in_group; i++) {
        std::pair<double, double> p;
        if (proportion_of_outliers > 0 && unit_distr(prng) < proportion_of_outliers) {
          p = pull_from_bivariate_normal(prng, outlier_severity);
        } else {
          p = pull_from_bivariate_normal(prng);
        }
        double x = p.first;
        double y = p.second;
        double ri = sqrt(x * x + y * y);
        g.add(x, y);
        r.push_back(ri);
        all_r.push_back(ri);
      }

      double this_gs = g.group_size();
      gs_s.push(this_gs);
      gsg_v.push_back(this_gs);
      if (j) {
        if (best_gs > this_gs) {
          best_gs = this_gs;
        }
        if (worst_gs < this_gs) {
          worst_gs = this_gs;
        }
      } else {
        best_gs = this_gs;
        worst_gs = this_gs;
      }
      gs.push(this_gs);

      double this_amr = g.avg_miss_radius();
      amr_s.push(this_amr);
      amr.push(this_amr);

      double rayleigh = rayleigh_cep_factor * accumulate(r.begin(), r.end(), 0.) / r.size();
      rayleigh_s.push(rayleigh);
      median_r_s.push(median(r));
    }
    bgs_s.push(best_gs);
    ags_s.push(gs.mean());
    if (groups_in_experiment > 1) {
      mgs_s.push(median(gsg_v));
    }
    wgs_s.push(worst_gs);
    aamr_s.push(amr.mean());
  }
  gs_s.show("Group size:");
  if (groups_in_experiment > 1) {
    bgs_s.show("Best group size:");
    ags_s.show("Average group size:");
    mgs_s.show("Median group size:");
    wgs_s.show("Worst group size:");
  }
  amr_s.show("Average Miss Radius:");
  if (groups_in_experiment > 1) {
    aamr_s.show("Average of AMR of groups:");
  }  
  rayleigh_s.show("Rayleigh CEP estimator:");
  median_r_s.show("Median CEP estimator:");
  std::vector<double>::iterator all_r50_it = all_r.begin() + all_r.size() / 2; 
  nth_element(all_r.begin(), all_r50_it, all_r.end());
  std::vector<double>::iterator all_r90_it = all_r.begin() + all_r.size() - all_r.size() / 10; 
  nth_element(all_r.begin(), all_r90_it, all_r.end());
  std::vector<double>::iterator all_r95_it = all_r.begin() + all_r.size() - all_r.size() / 20; 
  nth_element(all_r.begin(), all_r95_it, all_r.end());
  std::vector<double>::iterator all_r99_it = all_r.begin() + all_r.size() - all_r.size() / 100; 
  nth_element(all_r.begin(), all_r99_it, all_r.end());
  std::cout << "Observed R50=" << *all_r50_it << ", R90=" << *all_r90_it << ", R95=" << *all_r95_it << ", R99=" << *all_r99_it << "\n";
  if (proportion_of_outliers == 0) {
    std::cout << "Expected R50=" << sqrt(-2*log(0.5)) << ", R90=" << sqrt(-2*log(0.1)) << ", R95=" << sqrt(-2*log(0.05)) << ", R99=" << sqrt(-2*log(0.01)) << "\n";
  }
  return 0;
}
