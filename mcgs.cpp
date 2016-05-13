/* mcgs.cpp

  Monte-Carlo simulation of group sizes under bivariate (possibly contaminated) normal distribution

*/
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>
#include <stdlib.h>

// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct { uint64_t state;  uint64_t inc; } pcg32_random_t;

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

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
    
    double distance(int a, int b) const
    {
      double x = x_.at(a) - x_.at(b);
      double y = y_.at(a) - y_.at(b);
      return hypot(x, y);
    }
    
    double group_size(void) const
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
    
    // As per NSD (page 181) should be within 15 cm at 100 m
    // Group size 2.79295 corresponds to kuchnost~=3.15863,
    // so this is equivalent to 4.56 MOA.
    double nsd_kuchnost(void) const
    {
      // Only defined for 4 shot groups
      if (x_.size() != 4) return  std::numeric_limits<double>::quiet_NaN();
      
      // Find STP using all 4 shots
      double stp_x = (x_.at(0) + x_.at(1) + x_.at(2) + x_.at(3)) / 4;
      double stp_y = (y_.at(0) + y_.at(1) + y_.at(2) + y_.at(3)) / 4;
      
      // Find minimum radius of circle with center at STP that encloses all shots 
      double r = 0;
      for (unsigned i = 0; i < 4; i++) {
        double candidate = hypot(stp_x - x_.at(i), stp_y - y_.at(i));
        if (r < candidate) {
          r = candidate;
        }
      }
      
      // Exclude outlier, if any
      for (unsigned i = 0; i < 4; i++) {
      
        // STP excluding this shot
        double stp3_x = (stp_x * 4 - x_.at(i)) / 3;
        double stp3_y = (stp_y * 4 - y_.at(i)) / 3;
        
        // Minimum radius of circle with center at STP of three shots excluding this shot
        double r2 = 0;
        for (unsigned j = 0; j < 4; j++) {
          if (i == j) {
            continue;
          }
          double candidate = hypot(stp_x - x_.at(j), stp_y - y_.at(j));
          if (r2 < candidate) {
            r2 = candidate;
          }
        }

        // Outlier is a shot 2.5x or more distant from STP of other three shots
        // than radius of the circle centered as STP of other three shots
        // that covers these three shots
        if (hypot(stp3_x - x_.at(i), stp3_y - y_.at(i)) > 2.5 * r2 && r > r2) {
          r = r2;
        }
      }
      
      // Diameter
      return 2 * r;
    }

    double group_size_excluding_worst(void) const
    {
      // With two shots, excluding one results in a zero size group
      if (x_.size() < 3) return 0;
      
      // Find the two impacts defining extreme spread
      double best_so_far = distance(0, 1);
      int index_a = 0;
      int index_b = 1;
      for (unsigned i = 0; i < x_.size() - 1; i++) {
        for (unsigned j = i + 1; j < x_.size(); j++) {
          double candidate = distance(i, j);
          if (best_so_far < candidate) {
            best_so_far = candidate;
            index_a = i;
            index_b = j;
          }
        }
      }

      // Worst shot must be one of the impacts defining extreme spread.
      // Calculate group size without either one, return the smaller number.
      double best_so_far_excluding_a = 0;
      double best_so_far_excluding_b = 0;
      for (unsigned i = 0; i < x_.size() - 1; i++) {
        for (unsigned j = i + 1; j < x_.size(); j++) {
          double candidate = distance(i, j);
          if (i != index_a && j != index_a && best_so_far_excluding_a < candidate) {
            best_so_far_excluding_a = candidate;
          }
          if (i != index_b && j != index_b && best_so_far_excluding_b < candidate) {
            best_so_far_excluding_b = candidate;
          }
        }
      }
      return std::min(best_so_far_excluding_a, best_so_far_excluding_b);
    }

    double avg_miss_radius(void) const
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
        amr += hypot(x_.at(i) - center_x, y_.at(i) - center_y);
      }
      amr /= x_.size();
      return amr;
    }
    
    double worst_miss_radius(void) const
    {
      double max_r = 0;
      for (int i = 0; i < x_.size(); i++) {
        double r = hypot(x_.at(i), y_.at(i));
        if (max_r < r) {
          max_r = r;
        }
      }
      return max_r;
    }

    double second_worst_miss_radius(void) const
    {
      double max_r = 0;
      int max_i = 0;
      for (int i = 0; i < x_.size(); i++) {
        double r = hypot(x_.at(i), y_.at(i));
        if (max_r < r) {
          max_r = r;
          max_i = i;
        }
      }
      double second_r = 0;
      for (int i = 0; i < x_.size(); i++) {
        if (i == max_i) {
          continue;
        }
        double r = hypot(x_.at(i), y_.at(i));
        if (second_r < r) {
          second_r = r;
        }
      }
      return second_r;
    }

    double third_worst_miss_radius(void) const
    {
      double max_r = 0;
      int max_i = 0;
      for (int i = 0; i < x_.size(); i++) {
        double r = hypot(x_.at(i), y_.at(i));
        if (max_r < r) {
          max_r = r;
          max_i = i;
        }
      }
      double second_r = 0;
      int second_i = 0;
      for (int i = 0; i < x_.size(); i++) {
        if (i == max_i) {
          continue;
        }
        double r = hypot(x_.at(i), y_.at(i));
        if (second_r < r) {
          second_r = r;
          second_i = i;
        }
      }
      double third_r = 0;
      for (int i = 0; i < x_.size(); i++) {
        if (i == max_i || i == second_i) {
          continue;
        }
        double r = hypot(x_.at(i), y_.at(i));
        if (third_r < r) {
          third_r = r;
        }
      }
      return third_r;
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

    void show(const char* metric, double theoretical = 0) {
      std::cout << metric << " mean=" << mean(); 
      if (theoretical > 0) {
        std::cout << " (cf. " << theoretical << ")";
      }      
      std::cout << ", CV=" << stdev() / mean() << "\n";
    }

  private:
    int n_;
    double m_;
    double s_;
};

static std::pair<double, double> pull_from_bivariate_normal(pcg32_random_t* prng, double deviation = 1.)
{
  double u; do { u = ldexp(pcg32_random_r(prng), -32); } while (!u);
  double v; do { v = ldexp(pcg32_random_r(prng), -32); } while (!v);

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
  pcg32_random_t prng = {0};
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
    std::cout << ", " << groups_in_experiment << " groups per experiment"; 
  }
  std::cout << "\n";
  if (proportion_of_outliers > 0) {
    std::cerr << "Using contaminated normal distribution: " << 
    (proportion_of_outliers * 100) << "% of observations are pulled from distribution with " <<
    outlier_severity << " times higher deviation\n";
  }
  const double rayleigh_cep_factor = sqrt(4 * log(2) / M_PI) / shots_in_group; // 0.9394/shots
  
  double rayleigh_mle_factor = sqrt(log(2) / M_PI);
  for (int i = 0; i < shots_in_group; i++) {
    rayleigh_mle_factor *= 4;
    if (i != shots_in_group - 1) {
      rayleigh_mle_factor *= i + 1;
    }
    rayleigh_mle_factor /= i + 1 + shots_in_group;
  }

  DescriptiveStat gs_s, gs_s2, bgs_s, ags_s, ags_s2, mgs_s, wgs_s, amr_s, aamr_s, rayleigh_s, median_r_s;
  DescriptiveStat rayleigh_mle, worst_r_s, second_worst_r_s, third_worst_r_s;
  DescriptiveStat nsd_s, wr_s, swr_s, twr_s;
  int hits_within_double_median = 0;
  int hits_within_r50hat = 0;
  int hits_within_r90hat = 0;
  int hits_within_r90hat2 = 0;
  int hits_within_r90hat3 = 0;
  int hits_within_r95hat = 0;
  double double_med = 0;
  double r50hat = 0;
  double r90hat = 0;
  double r90hat2 = 0;
  double r90hat3 = 0;
  double r95hat = 0;
  for (int experiment = 0; experiment < experiments; experiment++) {
    double best_gs = 0, worst_gs = 0;
    DescriptiveStat gs, gs2, amr, wr, swr, twr;
    std::vector<double> gsg_v;
    for (int j = 0; j < groups_in_experiment; j++) { 
      ShotGroup g; 
      std::vector<double> r;
      std::vector<double> r2;
      for (int i = 0; i < shots_in_group; i++) {
        std::pair<double, double> p;
        if (proportion_of_outliers > 0 && ldexp(pcg32_random_r(&prng), -32) < proportion_of_outliers) {
          p = pull_from_bivariate_normal(&prng, outlier_severity);
        } else {
          p = pull_from_bivariate_normal(&prng);
        }
        double x = p.first;
        double y = p.second;
        double ri = hypot(x, y);
        g.add(x, y);
        r.push_back(ri);
        r2.push_back(ri*ri);
      }

      double this_gs = g.group_size();
      gs_s.push(this_gs);
      if (shots_in_group == 4) {
        nsd_s.push(g.nsd_kuchnost());
      }
      double this_minus1 = g.group_size_excluding_worst();
      gs_s2.push(this_minus1);
      gs2.push(this_minus1);
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

      double rayleigh = rayleigh_cep_factor * accumulate(r.begin(), r.end(), 0.);
      rayleigh_s.push(rayleigh);
      rayleigh_mle.push(rayleigh_mle_factor * sqrt(accumulate(r2.begin(), r2.end(), 0.)));
      
      double this_wr = g.worst_miss_radius();
      wr.push(this_wr);
      worst_r_s.push(this_wr);
      
      double this_swr = g.second_worst_miss_radius();
      swr.push(this_swr);
      second_worst_r_s.push(this_swr);
      
      double this_twr = g.third_worst_miss_radius();
      twr.push(this_twr);
      third_worst_r_s.push(this_twr);
      
      double med = median(r);
      median_r_s.push(med);
      
      // Use R90 estimates based on previous group to avoid correlation
      if (!(experiment == 0 && j == 0)) { // If there is a prior group
        for (int i = 0; i < shots_in_group; i++) {
          if (r.at(i) < double_med) {
            hits_within_double_median++;
          }
          if (r.at(i) < r50hat) {
            hits_within_r50hat++;
          }
          if (r.at(i) < r90hat) {
            hits_within_r90hat++;
          }
          if (r.at(i) < r90hat2) {
            hits_within_r90hat2++;
          }
          if (r.at(i) < r90hat3) {
            hits_within_r90hat3++;
          }
          if (r.at(i) < r95hat) {
            hits_within_r95hat++;
          }
        }
      }
      switch (shots_in_group) {
        case  1:
          // wxMaxima: find_root(integrate(x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.5, k, 0.1, 10);
          r50hat = this_wr * 1;
          // wxMaxima: find_root(integrate(x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.9, k, 1, 10);
          r90hat = this_wr * 3;
          // wxMaxima: find_root(integrate(x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.95, k, 1, 10);
          r95hat = this_wr * 4.358898943540671;
          break;
        case  2: 
          // wxMaxima: find_root(integrate(2*(1-exp(-x^2/2))*x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.5, k, 0.1, 10); 
          r50hat = this_wr * 0.7493682758222623;
          // wxMaxima: find_root(integrate(2*(1-exp(-x^2/2))*x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.9, k, 1, 10); 
          r90hat = this_wr * 1.732050807568877;
          // wxMaxima: find_root(integrate(2*(1-exp(-x^2/2))*x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.95, k, 1, 10); 
          r95hat = this_wr * 2.200974504673954;
          break;
        case  3: 
          // wxMaxima: find_root(integrate(3*x*(1-exp(-x^2/2))^2*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.5, k, 0.1, 10);
          r50hat = this_wr * 0.6594250285035448;
          // wxMaxima: find_root(integrate(3*x*(1-exp(-x^2/2))^2*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
          r90hat = this_wr * 1.414213562373095;
          // wxMaxima: find_root(integrate(3*x*(1-exp(-x^2/2))^2*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.95, k, 1, 10);
          r95hat = this_wr * 1.732050807568877;
          break;
        case  5: 
          // wxMaxima: find_root(integrate(5*x*(1-exp(-x^2/2))^4*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.5, k, 0.1, 10);
          r50hat = this_wr * 0.5779767551239063;
          // wxMaxima: find_root(integrate(5*x*(1-exp(-x^2/2))^4*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
          r90hat = this_wr  * 1.17215228421396;
          r90hat2 = this_wr  * 1.2;
          r90hat3 = this_gs  * 0.8;
          // wxMaxima: find_root(integrate(5*x*(1-exp(-x^2/2))^4*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.95, k, 1, 10);
          r95hat = this_wr  * 1.398425341757205;
          break;
        case  8: 
          // wxMaxima: find_root(integrate(56*x*(1-exp(-x^2/2))^6*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.5, k, 0.1, 10);
          r50hat = this_swr * 0.6550734805178998;
          // wxMaxima: find_root(integrate(56*x*(1-exp(-x^2/2))^6*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
          r90hat = this_swr * 1.281217078686304;
          // wxMaxima: find_root(integrate(56*x*(1-exp(-x^2/2))^6*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.95, k, 1, 10);
          r95hat = this_swr * 1.506180776008639;
          break;
        case 10: 
          // wxMaxima: find_root(integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.5, k, 0.1, 10);
          r50hat = this_swr * 0.614857338705793;
          // wxMaxima: find_root(integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
          r90hat = this_swr * 1.187140545277712;
          r90hat2 = this_swr  * 1.2;
          r90hat3 = this_minus1  * 0.75;
          // wxMaxima: find_root(integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.95, k, 1, 10);
          r95hat = this_swr * 1.387586460106418;
          break;
      }
      double_med = med * 2;
    }
    if (groups_in_experiment > 1) {
      bgs_s.push(best_gs);
      ags_s.push(gs.mean());
      ags_s2.push(gs2.mean());
      mgs_s.push(median(gsg_v));
      wgs_s.push(worst_gs);
      aamr_s.push(amr.mean());
      wr_s.push(wr.mean());
      swr_s.push(swr.mean());
      twr_s.push(twr.mean());
    }
  }
  if (groups_in_experiment == 1) {
    std::cout << "--- One " << shots_in_group << "-shot group ---\n";
    gs_s.show("Group size:");
    if (shots_in_group == 4) {
      nsd_s.show("Kuchnost:");
    }
    gs_s2.show("Group size (excluding worst shot in group):");
    amr_s.show("Average Miss Radius:");
    double theoretical_cep = 0;
    if (proportion_of_outliers == 0) {
      theoretical_cep =  sqrt(-2*log(0.5));
    }
    rayleigh_s.show("Rayleigh CEP estimator:", theoretical_cep);
    rayleigh_mle.show("Maximum likelihood CEP estimator:", theoretical_cep);
    median_r_s.show("Median CEP estimator:"); // Not showing theoretical_cep because median has known bias
    double theoretical_worst = 0;
    if (proportion_of_outliers == 0) {
      switch (shots_in_group) {
        case 2:
          theoretical_worst = (2 * sqrt(2) - 1) * sqrt(M_PI) / 2;
          break;
        case 3:
          theoretical_worst = (-9 + 9 * sqrt(2) + sqrt(6)) * sqrt(M_PI) / 6;
          break;
        case 5:
          theoretical_worst =(-300 + 75 * sqrt(2) + 100 * sqrt(6) + 6 * sqrt(10)) * sqrt(M_PI) / 60;
          break;
      }
    }
    worst_r_s.show("Worst miss radius:", theoretical_worst);
    double theoretical_second_worst = 0;
    if (proportion_of_outliers == 0) {
      switch (shots_in_group) {
        case 5:
          theoretical_second_worst =
            ( 300 
            + 225 * sqrt(2) 
            - 200 * sqrt(6) 
            -  24 * sqrt(10) 
            ) * sqrt(M_PI) / 60;
          break;
        case 8:
          theoretical_second_worst =
            ( 6615 
            + 22050 * sqrt(2) 
            +  9800 * sqrt(3) 
            -  7840 * sqrt(6) 
            -  9408 * sqrt(10) 
            -  1440 * sqrt(14)
            ) * sqrt(M_PI) / 420;
          break;
        case 10:
          theoretical_second_worst =
            ( 42525 
            + 60550 * sqrt(2) 
            + 73500 * sqrt(3) 
            +   378 * sqrt(5) 
            - 16800 * sqrt(6) 
            - 42336 * sqrt(10) 
            - 21600 * sqrt(14)
            ) * sqrt(M_PI) / 420;
          break;
      }
    }
    second_worst_r_s.show("Second worst miss radius:", theoretical_second_worst);
    double theoretical_third_worst = 0;
    if (proportion_of_outliers == 0) {
      switch (shots_in_group) {
        case 5:
          theoretical_third_worst = (-225 + 100 * sqrt(3) +  36 * sqrt(5)) * sqrt(M_PI / 2) / 30;
          break;
      }
    }
    third_worst_r_s.show("Third worst miss radius:", theoretical_third_worst);
    std::cout << "Percent of hits within double median radius: " << 100. * hits_within_double_median / shots_in_group / ((double)groups_in_experiment * experiments - 1) << "%\n";
    double denominator = shots_in_group * ((double)groups_in_experiment * experiments - 1);
    switch (shots_in_group) {
      case 1: case 2: case 3: case 5: 
        std::cout << "Percent of hits within R50hat based on worst miss radius: " << 100. * hits_within_r50hat / denominator << "%\n";
        std::cout << "Percent of hits within R90hat based on worst miss radius: " << 100. * hits_within_r90hat / denominator << "%\n";
        std::cout << "Percent of hits within R95hat based on worst miss radius: " << 100. * hits_within_r95hat / denominator << "%\n";
        break;
      case 8: case 10:
        std::cout << "Percent of hits within R50hat based on second worst miss radius: " << 100. * hits_within_r50hat / denominator << "%\n";
        std::cout << "Percent of hits within R90hat based on second worst miss radius: " << 100. * hits_within_r90hat / denominator << "%\n";
        std::cout << "Percent of hits within R95hat based on second worst miss radius: " << 100. * hits_within_r95hat / denominator << "%\n";
        break;
    }
    switch (shots_in_group) {
      case 5: 
        std::cout << "Percent of hits within 1.2 * worst miss radius: " 
          << 100. * hits_within_r90hat2 / denominator << "%"
          // wxMaxima: float(1-integrate(5*(1-exp(-x^2/2))^4*x*exp(-x^2/2-1.2^2*x^2/2), x, 0, inf));
          << " (cf. " << 100 * 0.9080894764538276 << "%)\n";
        std::cout << "Percent of hits within 0.8 * group size: " 
          << 100. * hits_within_r90hat3 / denominator << "%\n";
        break;
      case 10:
        std::cout << "Percent of hits within 1.2 * second worst miss radius: " 
          << 100. * hits_within_r90hat2 / denominator << "%"
          // wxMaxima: float(1-integrate(90*(1-exp(-x^2/2))^8*x*exp(-x^2-1.2^2*x^2/2), x, 0, inf));
          << " (cf. " << 100 * 0.9042093867664657 << "%)\n";
        std::cout << "Percent of hits within 0.75 * group size excluding worst shot: " 
          << 100. * hits_within_r90hat3 / denominator << "%\n";
        break;
    }
  } else {
    std::cout << "--- Aggregate of " << groups_in_experiment << " " << shots_in_group << "-shot groups ---\n";
    bgs_s.show("Best group size:");
    ags_s.show("Average group size:");
    ags_s2.show("Average group size (excluding worst shot in group):");
    mgs_s.show("Median group size:");
    wgs_s.show("Worst group size:");
    aamr_s.show("Average of AMR of groups:");
    wr_s.show("Average worst miss radius:");
    swr_s.show("Average second worst miss radius:");
    twr_s.show("Average third worst miss radius:");
  }  
  return 0;
}
