/* mcgs.cpp

  Monte-Carlo simulation of group sizes under bivariate (possibly contaminated) normal distribution

  Building:
  
  g++ -O3 -Wall -std=c++11 -o mcgs mcgs.cpp

*/
#include <vector>
#include <map>
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
      if (x_.size() != 4) return std::numeric_limits<double>::quiet_NaN();
      
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
      if (x_.size() < 2) return std::numeric_limits<double>::quiet_NaN();;
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
      return (n_ > 0) ? m_ : std::numeric_limits<double>::quiet_NaN();
    }

    double variance(void) const
    {
      return ( (n_ > 1) ? s_ / (n_ - 1) : 0 );
    }

    double stdev(void) const
    {
      return sqrt(variance());
    }

    double cv(void) const
    {
      return stdev() / mean();
    }

    int count(void) const
    {
      return n_;
    }

    void show(const char* metric, double theoretical = 0) {
      std::cout << metric << " mean=" << mean(); 
      if (theoretical > 0) {
        std::cout << " (expected " << theoretical << ")";
      }      
      std::cout << ", CV=" << cv() << "\n";
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

  return std::make_pair(x, y);
}

static double median(std::vector<double>& x)
{
  if (x.size() == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
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

// Rank is zero-based
static double kth_miss_radius(std::vector<double>& x, int k)
{
  if (k < 0 || k >= x.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::vector<double>::iterator it = x.begin() + k; 
  nth_element(x.begin(), it, x.end());
  return *it;  
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
  
  double mle_factor = sqrt(log(2) / M_PI);
  for (int i = 0; i < shots_in_group; i++) {
    mle_factor *= 4;
    if (i != shots_in_group - 1) {
      mle_factor *= i + 1;
    }
    mle_factor /= i + 1 + shots_in_group;
  }
  double miss_radius_to_r90hat_factor  = 0;
  double miss_radius_to_r90hat_factor_rounded = 0;
  double group_size_to_r90hat_factor = 0;
  double sixtynine_to_r90hat_factor = 0;
  double rayleigh_to_r90hat_factor = 0;
  double mle_to_r90hat_factor = 0;
  std::pair<int, int> sixtynine_rank( (shots_in_group * 6 - 5) / 10 // 60th percentile
                                    , (shots_in_group * 9 - 5) / 10 // 90th percentile
                                    );
  if (groups_in_experiment == 1) {
    switch (shots_in_group) {
      case  1:
        // wxMaxima: find_root(integrate(x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        miss_radius_to_r90hat_factor = 3;
        break;
      case  3: 
        // wxMaxima: find_root(integrate(3*x*(1-exp(-x^2/2))^2*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        miss_radius_to_r90hat_factor = 1.414213562373095;
        break;
      case  5: 
        // wxMaxima: find_root(integrate(5*x*(1-exp(-x^2/2))^4*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        miss_radius_to_r90hat_factor = 1.17215228421396;
        miss_radius_to_r90hat_factor_rounded = 1.2;
        group_size_to_r90hat_factor = 0.8;
        break;
      case 10: 
        // wxMaxima: find_root(integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        miss_radius_to_r90hat_factor = 1.187140545277712;
        miss_radius_to_r90hat_factor_rounded = 1.2;
        group_size_to_r90hat_factor = 0.75;
        break;
      case 20: 
        rayleigh_to_r90hat_factor = 1.76;
        mle_to_r90hat_factor = 0.35;
        break;
    }
  } else if (groups_in_experiment == 2) {
    switch (shots_in_group) {
      case 1:
        // wxMaxima:
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=x*exp(-x^2/2);
        // p2(t):=''(integrate(2*p(u)*p(2*t-u),u,0,2*t));
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1, 3);
        miss_radius_to_r90hat_factor = 2.22197649;
        miss_radius_to_r90hat_factor_rounded = 2.2;
        break;
      case 3:
        // wxMaxima: 
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=3*x*(1-exp(-x^2/2))^2*exp(-x^2/2);
        // p2(t):=''(integrate(2*p(u)*p(2*t-u),u,0,2*t));
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1, 3);
        miss_radius_to_r90hat_factor = 1.28875916;
        miss_radius_to_r90hat_factor_rounded = 1.3;
        break;
      case 5:
        // wxMaxima:
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=5*x*(1-exp(-x^2/2))^4*exp(-x^2/2);
        // p2(t):=romberg(2*p(u)*p(2*t-u),u,0,2*t);
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1, 1.3);
        miss_radius_to_r90hat_factor = 1.10343798;
        miss_radius_to_r90hat_factor_rounded = 1;
        break;
      case 10:
        // wxMaxima:
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=90*x*(1-exp(-x^2/2))^8*exp(-x^2);
        // p2(t):=romberg(2*p(u)*p(2*t-u),u,0,2*t);
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1.1, 1.15);
        miss_radius_to_r90hat_factor = 1.149216;
        miss_radius_to_r90hat_factor_rounded = 1.15;
        break;
    }
  } else if (groups_in_experiment == 4) {
    switch (shots_in_group) {
      case  5:
        group_size_to_r90hat_factor = 0.723;
        break;
      default:
        break;
    }
  } else if (groups_in_experiment == 5) {
    switch (shots_in_group) {
      case  5:
        group_size_to_r90hat_factor = 0.72;
        break;
      default:
        break;
    }
  } else if (groups_in_experiment > 10) { // Asymptotic approximation for large number of groups in experiment
    switch (shots_in_group) {
      case  1:
        // wxMaxima: float(sqrt(2*log(10))/integrate(x*exp(-x^2/2)*x, x, 0, inf));
        miss_radius_to_r90hat_factor  = 1.712233160383746;
        miss_radius_to_r90hat_factor_rounded = 1.7;
        break;
      case  3:
        // wxMaxima: float(sqrt(2*log(10))/integrate(3*(1-exp(-x^2/2))^2*x*exp(-x^2/2)*x, x, 0, inf));
        miss_radius_to_r90hat_factor  = 1.175960143568417;
        miss_radius_to_r90hat_factor_rounded = 1.2;
        break;
      case  5:
        // wxMaxima: float(sqrt(2*log(10))/integrate(5*(1-exp(-x^2/2))^4*x*exp(-x^2/2)*x, x, 0, inf));
        miss_radius_to_r90hat_factor  = 1.037938194579831;
        miss_radius_to_r90hat_factor_rounded = 1;
        group_size_to_r90hat_factor = 0.7;
        break;
      case 10:
        // wxMaxima: float(sqrt(2*log(10))/integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2)*x, x, 0, inf));
        miss_radius_to_r90hat_factor  = 1.112257194707586;
        miss_radius_to_r90hat_factor_rounded = 1.1;
        group_size_to_r90hat_factor = 0.7;
        break;
      case 20:
        mle_to_r90hat_factor = 0.3414341089001782;
        break;
    }
  }
  if (shots_in_group == 10) {
    switch (groups_in_experiment) {
      case  1:
        // wxMaxima:
        // assume(x>0,y>0,x<=y,k>0);
        // pdf(x,y):=3*7*8*9*10*(1-exp(-x^2/2))^5*(exp(-x^2/2)-exp(-y^2/2))^2*exp(-y^2/2)*x*exp(-x^2/2)*y*exp(-y^2/2);
        // find_root(romberg(integrate(pdf(x,y)*(1-exp(-(k*(x+y))^2/2)),y,x,inf),x,0,50)=9/10,k,0.6,0.8);
        sixtynine_to_r90hat_factor = 0.7076687;
        break;
      case 2:
        // wxMaxima:
        // assume(x>0,y>0,x<=y);
        // pdf(x,y):=15120*(1-exp(-x^2/2))^5*(exp(-x^2/2)-exp(-y^2/2))^2*exp(-y^2/2)*x*exp(-x^2/2)*y*exp(-y^2/2);
        // assume(u>0,v>0,v<=u);
        // p2(u):=romberg(pdf((u-v)/2,(u+v)/2)/2,v,0,u);
        // p2s(t):=romberg(2*p2(u)*p2(2*t-u),u,0,2*t);
        // c2(k):=romberg(p2s(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1/10, k, 0.6, 0.8);
        sixtynine_to_r90hat_factor = 0.68860849;
        break;
      default:
        // wxMaxima:
        // assume(x>0,y>0,x<=y);
        // pdf(x,y):=3*7*8*9*10*(1-exp(-x^2/2))^5*(exp(-x^2/2)-exp(-y^2/2))^2*exp(-y^2/2)*x*exp(-x^2/2)*y*exp(-y^2/2);
        // float(sqrt(2*log(10)))/float(integrate(integrate(pdf(x,y)*(x+y),y,x,inf),x,0,inf));
        sixtynine_to_r90hat_factor = 0.67024464399177286;
        break;
    }
  } else if (shots_in_group > 7) {
    sixtynine_to_r90hat_factor = 0.69;
  }
  DescriptiveStat gs_s, gs_s2, bgs_s, ags_s, ags_s2, mgs_s, amr_s, aamr_s, rayleigh_s, mle_s, median_r_s;
  DescriptiveStat worst_r_s, second_worst_r_s;
  std::map< std::pair<int, int>, DescriptiveStat> sixtynine_r_s;
  DescriptiveStat nsd_s, wr_s, swr_s, sixtynine_s;
  int hits_within_r50hat = 0;
  int hits_within_r90hat = 0;
  int hits_within_r90hat2 = 0;
  int hits_within_r90hat3 = 0;
  int hits_within_r90hat4 = 0;
  int hits_within_r90hat5 = 0;
  int hits_within_r90hat6 = 0;
  double r50hat = 0;
  double r90hat = 0;
  double r90hat2 = 0;
  double r90hat3 = 0;
  double r90hat4 = 0;
  double r90hat5 = 0;
  double r90hat6 = 0;
  for (int experiment = 0; experiment < experiments; experiment++) {
    double best_gs = 0;
    DescriptiveStat gs, gs2, amr, wr, swr, sixtynine, rayleigh, mle;
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

        // Use R90 estimates based on previous experiment to avoid correlation
        if (experiment != 0) { // If there is a prior group
          if (ri < r50hat)  hits_within_r50hat++;
          if (ri < r90hat)  hits_within_r90hat++;
          if (ri < r90hat2) hits_within_r90hat2++;
          if (ri < r90hat3) hits_within_r90hat3++;
          if (ri < r90hat4) hits_within_r90hat4++;
          if (ri < r90hat5) hits_within_r90hat5++;
          if (ri < r90hat6) hits_within_r90hat6++;
        }
      } // Next shot

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
      } else {
        best_gs = this_gs;
      }
      gs.push(this_gs);

      double this_amr = g.avg_miss_radius();
      amr_s.push(this_amr);
      amr.push(this_amr);

      double this_rayleigh = rayleigh_cep_factor * accumulate(r.begin(), r.end(), 0.);
      rayleigh.push(this_rayleigh);
      rayleigh_s.push(this_rayleigh);
      
      double this_mle = sqrt(accumulate(r2.begin(), r2.end(), 0.));
      mle.push(this_mle);
      mle_s.push(mle_factor * this_mle);
      
      double this_wr = kth_miss_radius(r, shots_in_group - 1);
      wr.push(this_wr);
      worst_r_s.push(this_wr);
      
      double this_swr = kth_miss_radius(r, shots_in_group - 2);
      swr.push(this_swr);
      second_worst_r_s.push(this_swr);
            
      double med = median(r);
      median_r_s.push(med);
      
      if (sixtynine_rank.first != sixtynine_rank.second) {
        double this_sixtynine = kth_miss_radius(r, sixtynine_rank.first) + kth_miss_radius(r, sixtynine_rank.second);
        sixtynine.push(this_sixtynine);
        if (shots_in_group <= 100) {
          // Remember all rank pairs, will choose the best one later
          for (int rank_a = 0; rank_a < shots_in_group - 1; rank_a++) {
            for (int rank_b = rank_a + 1; rank_b < shots_in_group; rank_b++) {
              double e = kth_miss_radius(r, rank_a) + kth_miss_radius(r, rank_b);
              sixtynine_r_s[std::make_pair(rank_a, rank_b)].push(e);
            }
          }
        }
      }
    } // Next group
    if (groups_in_experiment > 1) {
      bgs_s.push(best_gs);
      ags_s.push(gs.mean());
      ags_s2.push(gs2.mean());
      mgs_s.push(median(gsg_v));
      aamr_s.push(amr.mean());
      wr_s.push(wr.mean());
      swr_s.push(swr.mean());
      sixtynine_s.push(sixtynine.mean());
    }
    if (shots_in_group <= 5) {
      r90hat  = wr.mean()  * miss_radius_to_r90hat_factor;
      r90hat2 = wr.mean()  * miss_radius_to_r90hat_factor_rounded;
      r90hat3 = gs.mean()  * group_size_to_r90hat_factor;
    } else {
      r90hat  = swr.mean() * miss_radius_to_r90hat_factor;
      r90hat2 = swr.mean() * miss_radius_to_r90hat_factor_rounded;
      r90hat3 = gs2.mean() * group_size_to_r90hat_factor;
    }
    r90hat4 = sixtynine.mean() * sixtynine_to_r90hat_factor;
    r90hat5 = rayleigh.mean() * rayleigh_to_r90hat_factor / sqrt(4 * log(2) / M_PI);
    if (shots_in_group == 20) {
      r90hat6 = mle.mean() * mle_to_r90hat_factor;
    }
  } // Next experiment
  if (groups_in_experiment == 1) {
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
    mle_s.show("Maximum likelihood CEP estimator:", theoretical_cep);
    median_r_s.show("Median CEP estimator:"); // Not showing theoretical_cep because median has known bias
    double theoretical_worst = 0;
    if (proportion_of_outliers == 0) {
      switch (shots_in_group) {
        case 3:
          // wxMaxima: float(integrate(3*(1-exp(-x^2/2))^2*x*exp(-x^2/2)*x, x, 0, inf));
          theoretical_worst = 1.824862890146495;
          break;
        case 5:
          // wxMaxima: float(integrate(5*(1-exp(-x^2/2))^4*x*exp(-x^2/2)*x, x, 0, inf));
          theoretical_worst = 2.067527755983637;
          break;
      }
    }
    worst_r_s.show("Worst miss radius:", theoretical_worst);
    double theoretical_second_worst = 0;
    double theoretical_sixtynine = 0;
    if (proportion_of_outliers == 0) {
      switch (shots_in_group) {
        case 10:
          // wxMaxima: float(integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2)*x, x, 0, inf));
          theoretical_second_worst = 1.929379316672818;
          
          // wxMaxima:
          // assume(x>0,y>0,x<=y);
          // pdf(x,y):=3*7*8*9*10*(1-exp(-x^2/2))^5*(exp(-x^2/2)-exp(-y^2/2))^2*exp(-y^2/2)*x*exp(-x^2/2)*y*exp(-y^2/2);
          // integrate(integrate(pdf(x,y)*(x+y),y,x,inf),x,0,inf);
          theoretical_sixtynine = 3.201765293569168;
          break;
      }
    }
    second_worst_r_s.show("Second worst miss radius:", theoretical_second_worst);
    // Find combination of two rank order statistics with lowest CV
    {
      int best_pos = 0;
      double best_cv = 0;
      int pos = 0;
      for (auto it = sixtynine_r_s.begin(); it != sixtynine_r_s.end(); it++, pos++) {
        if (pos == 0 || best_cv > it->second.cv()) {
          best_cv = it->second.cv();
          best_pos = pos; 
        }
      }
      pos = 0;
      for (auto it = sixtynine_r_s.begin(); it != sixtynine_r_s.end(); it++, pos++) {
        if (pos == best_pos) {
          std::cout << "Lowest CV pair: R" << (it->first.first + 1) << ":" << shots_in_group 
                    << "+R" << (it->first.second + 1) << ":" << shots_in_group;
          it->second.show(",");
          it = sixtynine_r_s.find(sixtynine_rank);
          if (it != sixtynine_r_s.end()) {
            std::cout << "60/90 pair: R" << (sixtynine_rank.first  + 1) << ":" << shots_in_group 
                      <<            "+R" << (sixtynine_rank.second + 1) << ":" << shots_in_group;
            it->second.show(",", theoretical_sixtynine);
          }
        }
      }
    }
  } else {
    bgs_s.show("Best group size:");
    ags_s.show("Average group size:");
    ags_s2.show("Average group size (excluding worst shot in group):");
    mgs_s.show("Median group size:");
    aamr_s.show("Average of AMR of groups:");
    wr_s.show("Average worst miss radius:");
    swr_s.show("Average second worst miss radius:");
    if (sixtynine_rank.first != sixtynine_rank.second) {
      std::cout << "Average R" << (sixtynine_rank.first  + 1) << ":" << shots_in_group 
                << "+R"        << (sixtynine_rank.second + 1) << ":" << shots_in_group;
      sixtynine_s.show(":");
    }
  }  
  double denominator = shots_in_group * ((double)groups_in_experiment * (experiments - 1));
  if (miss_radius_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << miss_radius_to_r90hat_factor 
              << " * " << (shots_in_group > 5 ? "second " : "") << "worst miss radius: " 
              << 100. * hits_within_r90hat / denominator << "%\n";
  }
  if (miss_radius_to_r90hat_factor_rounded > 0) {
    std::cout << "Percent of hits within " << miss_radius_to_r90hat_factor_rounded 
              << " * " << (shots_in_group > 5 ? "second " : "") << "worst miss radius: " 
              << 100. * hits_within_r90hat2 / denominator << "%\n";
  }
  if (group_size_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << group_size_to_r90hat_factor 
              << " * group size" << (shots_in_group > 5 ? " (excluding worst shot in group)" : "") << ": " 
              << 100. * hits_within_r90hat3 / denominator << "%\n";
  }
  if (sixtynine_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << sixtynine_to_r90hat_factor 
              << " * (R" << (sixtynine_rank.first  + 1) << ":" << shots_in_group 
              << " + R"    << (sixtynine_rank.second + 1) << ":" << shots_in_group << "): "
              << 100. * hits_within_r90hat4 / denominator << "%\n";
  }
  if (rayleigh_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << rayleigh_to_r90hat_factor 
              << " * average miss radius: " 
              << 100. * hits_within_r90hat5 / denominator << "%\n";
  }
  if (mle_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << mle_to_r90hat_factor 
              << " * square root of sum of squares: " 
              << 100. * hits_within_r90hat6 / denominator << "%\n";
  }
  return 0;
}
