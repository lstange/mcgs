/* mcgs.cpp

  Monte-Carlo simulation of group sizes under bivariate (possibly contaminated) normal distribution

  Building:
  
  g++ -O3 -Wall -Wextra -std=c++11 -march=native -g -I ~/boost_1_64_0 -o mcgs mcgs.cpp

*/
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <complex>
#include <random>
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <algorithm>
#include <boost/math/distributions/chi_squared.hpp>

// Abstract base class (interface) for random shot group generator, implementations follow
class RandomNumberGenerator {
  public:
    virtual void advance() = 0;
    virtual void reset() = 0;
    virtual std::complex<double> point(unsigned dimension) = 0;
};

//
// Baseline pseudorandom number generator, wraps the default provided by standard library
//
class DefaultRandomNumberGenerator : public RandomNumberGenerator {
    std::default_random_engine gen_;
    std::normal_distribution<double> dist_;
  
  public:
    DefaultRandomNumberGenerator() 
    {
      reset();
    }
  
    void advance() {}

    void reset() 
    {
      gen_.seed(42);
    }

    std::complex<double> point(unsigned dimension __attribute__((unused)))
    {
      return std::complex<double>(dist_(gen_), dist_(gen_));
    }
};

// 
// Additive quasi-random generator
//
// To get the next point, add a fixed step modulo 1 to the previous point. 
// Use square roots of prime numbers as steps because they are irrational.
//
class AdditiveRandomNumberGenerator: public RandomNumberGenerator {
    std::vector<double> x_;
    std::vector<double> step_;

  public:
    explicit AdditiveRandomNumberGenerator(unsigned dimensions)
    {
      unsigned nprimes = dimensions * 2; // Two coordinates for each point
      x_.resize(nprimes);
      step_.resize(nprimes);
      //
      // To initialize the steps, we need to find some prime numbers. Do that with 
      // sieve of Eratosthenes.
      //
      // To estimate how big the sieve should be to get nprimes primes, start with 
      // approximation by Gauss and Legendre:
      //
      //   nprimes = PrimePi(x) ~ x/log(x)
      //
      // Here PrimePi(x) is prime-counting function: number of prime numbers that are 
      // less than or equal to x. This is a lower bound for all but the lowest values
      // of x (which we'll handle separately), so x will be slightly larger than strictly 
      // necessary.
      //
      // To solve for x without resorting to Lambert-W function, rewrite as follows:
      //
      //   x ~ nprimes * log(x)
      //
      // Now use fixed point iteration to find x. Stop iterating when the whole part
      // of x stops changing.
      // 
      unsigned sieve_size = 0;
      if (nprimes < 5) {  // Approximation breaks below 5, floor sieve_size
        sieve_size = 7;
      } else {
        double x = nprimes;
        for (unsigned i = 0; i < 30; i++) {
          unsigned before = (unsigned)ceil(x);
          x = nprimes * log(x); // Fixed point iteration
          unsigned after = (unsigned)ceil(x);
          if (before == after) {
            sieve_size = before;
            break;
          }      
        }
        if (sieve_size == 0) {
          std::cerr << "Fixed point iteration failed to converge\n";
          exit(-1);
        }
      }
    
      // The sieve itself
      std::vector<bool> prime(sieve_size + 1, true);
      unsigned top = sqrt(sieve_size);
      for (unsigned i = 2; i < top; i++) {
        if (prime.at(i)) {
          for(unsigned j = i * i; j <= sieve_size; j += i) {
            prime.at(j) = false;
          }
        }
      }
    
      // Collect prime numbers left in the sieve, take square roots, assign to steps.
      unsigned index = 0;
      for (unsigned j = 2; index < nprimes; j++) {
        if (j >= sieve_size) {
          std::cerr << "Ran out of primes\n"; // Shouldn't happen if we did the math right
          exit(-1);
        }
        if (prime[j]) {
          step_.at(index++) = sqrt(j);
        }
      }
    }
  
    void advance()
    {
      for (unsigned j = 0; j < step_.size(); j++) {
        double int_part;
        x_.at(j) = modf(x_.at(j) + step_.at(j), &int_part);
      }
    }

    void reset()
    {
      for (unsigned j = 0; j < step_.size(); j++) {
        x_.at(j) = 0;
      }
    }

    std::complex<double> point(unsigned dimension)
    {
      double x, y;
      if (dimension >= step_.size() / 2) {
        x = y = std::numeric_limits<double>::quiet_NaN();
      } else {
        double u = x_.at(dimension);
        double v = x_.at(dimension + step_.size() / 2);
        // Box-Muller transform
        double r = sqrt(-2 * log(u));
        x = r * cos(2 * M_PI * v);
        y = r * sin(2 * M_PI * v);
      }
      return std::complex<double>(x, y);
    }
};

//
// Sobol quasi-random number generator
//
class SobolRandomNumberGenerator : public RandomNumberGenerator {
    unsigned x_[20];
    unsigned v_[20][32 + 1];
    unsigned i_;
  
public:
    explicit SobolRandomNumberGenerator(unsigned dimensions): i_(0)
    {
      // Primitive polynomials and initial direction numbers for the first 20 dimensions
      // recommended by Stephen Joe and Frances Kuo in new-joe-kuo-6.21201
      // http://web.maths.unsw.edu.au/~fkuo/sobol/index.html
      const unsigned s[20] = {1, 2, 3, 3, 4, 4, 5, 5, 5,  5,  5, 5,  6,  6,  6,  6,  6,  6, 7, 7};
      const unsigned a[20] = {0, 1, 1, 2, 1, 4, 2, 4, 7, 11, 13, 14, 1, 13, 16, 19, 22, 25, 1, 4};
      const unsigned m[20][7] = {
        {1, 0, 0,  0,  0,  0,   0},
        {1, 3, 0,  0,  0,  0,   0},
        {1, 3, 1,  0,  0,  0,   0},
        {1, 1, 1,  0,  0,  0,   0},
        {1, 1, 3,  3,  0,  0,   0},
        {1, 3, 5, 13,  0,  0,   0},
        {1, 1, 5,  5, 17,  0,   0},
        {1, 1, 5,  5,  5,  0,   0},
        {1, 1, 7, 11, 19,  0,   0}, 
        {1, 1, 5,  1,  1,  0,   0},
        {1, 1, 1,  3, 11,  0,   0}, 
        {1, 3, 5,  5, 31,  0,   0}, 
        {1, 3, 3,  9,  7, 49,   0}, 
        {1, 1, 1, 15, 21, 21,   0}, 
        {1, 3, 1, 13, 27, 49,   0}, 
        {1, 1, 1, 15,  7,  5,   0}, 
        {1, 3, 1, 15, 13, 25,   0}, 
        {1, 1, 5,  5, 19, 61,   0}, 
        {1, 3, 7, 11, 23, 15, 103}, 
        {1, 3, 7, 13, 13, 15,  69} 
      };
      if (dimensions > 10) {
        std::cerr << "Sobol direction numbers available for at most 10 shot group\n";
        exit(-1);
      }
      for (unsigned j = 0; j < 20; j++) {
        if (j == 0) {
          for (unsigned i = 0; i < 32; i++) {
            v_[j][i] = 1 << (32 - i - 1);
          }
        } else {
          unsigned ss = s[j - 1];
          for (unsigned i = 0; i < ss; i++) {
            v_[j][i] = m[j - 1][i] << (32 - i - 1);
          }
          for (unsigned i = ss; i < 32; i++) {
	        v_[j][i] = v_[j][i - ss] ^ (v_[j][i - ss] >> ss); 
	        for (unsigned k = 1; k < ss; k++) { 
	          v_[j][i] ^= (((a[j - 1] >> (ss - 1 - k)) & 1) * v_[j][i - k]);
	        } 
          }
        }
        x_[j] = 0;
      }
      
      // Burn-in: discard initial values
      for (unsigned i = 0; i < 32; i++) {
        advance();
      }
    }
    
    void reset()
    {
      for (unsigned j = 0; j < 20; j++) {
        x_[j] = 0;
      }
    }

    void advance()
    {
      unsigned b = i_++;
      unsigned z = 0; // Position of the first zero bit in binary representation of i_
                      // counting from the right
      if (b) {
        while (b & 1) {
          b >>= 1;
          z++;
        }
      }
      for (unsigned j = 0; j < 20; j++) {
        x_[j] ^= v_[j][z];
      }
    }
    
    std::complex<double> point(unsigned dimension)
    {
      double x, y;
      if (dimension >= 10) {
        x = y = std::numeric_limits<double>::quiet_NaN();
      } else {
        double u = ldexp(x_[dimension], -32);
        double v = ldexp(x_[dimension + 10], -32);

        // Box-Muller transform
        double r = sqrt(-2 * log(u));
        x = r * cos(2 * M_PI * v);
        y = r * sin(2 * M_PI * v);
      }
      return std::complex<double>(x, y);
    }
};

struct ConvexHullPoint {
  public:  
    explicit ConvexHullPoint(const std::complex<double>& p) : point_(p) {}
  
    // For lexicographic sort: first by x, then by y
    bool operator<(const ConvexHullPoint& other) const
    {
      if (point_.real() < other.point_.real()) {
        return true;
      } else if (point_.real() > other.point_.real()) {
        return false;
      } else {
        return point_.imag() < other.point_.imag();
      }
    }

    bool operator!=(const std::complex<double>& other) const
    {
      return point_ != other;
    }
  
    // Distance to another point
    double distanceTo(const ConvexHullPoint& other) const
    {
      return std::abs(point_ - other.point_);
    }

    double inline x(void) const { return point_.real(); }
    double inline y(void) const { return point_.imag(); }

  private:
    std::complex<double> point_;
};

class ShotGroup {
  private:
    // Use complex type to store impact coordinates
    std::vector<std::complex<double> > impact_;
  
    // Cross product of vectors OA and OB, positive if OAB makes a CCW turn
    double cross_product( const ConvexHullPoint& O
                        , const ConvexHullPoint& A
                        , const ConvexHullPoint& B
                        ) const
    {
      return (A.x() - O.x()) * (B.y() - O.y())
           - (A.y() - O.y()) * (B.x() - O.x());
    }

    double distance(unsigned a, unsigned b) const
    {
      return std::abs(impact_.at(a) - impact_.at(b));
    }
    
  public:
    void add(const std::complex<double>& p)
    {
      impact_.push_back(p);
    }
    
    // Brute force implementation, asymptotic complexity O(N^2)
    double group_size_brute_force(double* excluding_worst = NULL) const
    {
      unsigned n = impact_.size();
      if (n < 2) {
        if (excluding_worst) {
          *excluding_worst = 0;
        }
        return 0;
      }
      
      // Find the two impacts defining extreme spread
      double extreme_spread = 0;
      unsigned index_a = 0;
      unsigned index_b = 0;
      for (unsigned i = 0; i < n - 1; i++) {
        for (unsigned j = i + 1; j < n; j++) {
          double candidate = distance(i, j);
          if (extreme_spread < candidate) {
            extreme_spread = candidate;
            index_a = i;
            index_b = j;
          }
        }
      }
      
      if (excluding_worst) {
        // Worst shot must be one of the impacts defining extreme spread.
        // Calculate group size without either one, return the smaller number.
        double extreme_spread_excluding_a = 0;
        double extreme_spread_excluding_b = 0;
        for (unsigned i = 0; i < n - 1; i++) {
          for (unsigned j = i + 1; j < n; j++) {
            double candidate = distance(i, j);
            if (i != index_a && j != index_a && extreme_spread_excluding_a < candidate) {
              extreme_spread_excluding_a = candidate;
            }
            if (i != index_b && j != index_b && extreme_spread_excluding_b < candidate) {
              extreme_spread_excluding_b = candidate;
            }
          }
        }
        *excluding_worst = std::min(extreme_spread_excluding_a, extreme_spread_excluding_b);
      }
      return extreme_spread;
    }

    // Same as group_size_brute_force(), but using convex hull.
    // Asymptotic complexity O(N log N).
    //
    // Pass 0: find impacts a and b defining extreme spread. 
    //         Return extreme spread if excluding_worst == false.
    // Pass 1: find extreme spread excluding a
    // Pass 2: find extreme spread excluding b
    //
    // Return the smaller of extreme spread excluding a and extreme spread excluding b
    //
    double group_size_convex_hull(double* excluding_worst = NULL) const
    {
      unsigned n = impact_.size();
      if (n < 2) {
        if (excluding_worst) {
          *excluding_worst = 0;
        }
        return 0;
      }
      double extreme_spread = 0;
      ConvexHullPoint a(impact_.at(0)); double extreme_spread_excluding_a = 0;
      ConvexHullPoint b(impact_.at(1)); double extreme_spread_excluding_b = 0;
      for (unsigned pass = 0; pass < 3; pass++) {

        // Use Andrew's monotone chain 2D algorithm to construct convex hull 
        std::vector<ConvexHullPoint> lower_hull;
        std::vector<ConvexHullPoint> upper_hull;
        {
          std::vector<ConvexHullPoint> p;
          for (unsigned i = 0; i < n; i++) {
            if ( pass == 0
              || ((pass == 1) && (a != impact_.at(i)))
              || ((pass == 2) && (b != impact_.at(i)))
               )
            p.push_back(ConvexHullPoint(impact_.at(i)));
          }
          std::sort(p.begin(), p.end());
      
          unsigned k = 0;
          for (unsigned i = 0; i < p.size(); i++) { // lower hull
            while (k >= 2 && cross_product(lower_hull.at(k - 2), lower_hull.at(k - 1), p.at(i)) <= 0) {
              lower_hull.pop_back();
              k--;
            }
            lower_hull.push_back(p.at(i));
            k++;
          }
          k = 0;
          for (unsigned i = 0; i < p.size(); i++) { // upper hull
            while (k >= 2 && cross_product(upper_hull.at(k - 2), upper_hull.at(k - 1), p.at(i)) >= 0) {
              upper_hull.pop_back();
              k--;
            }
            upper_hull.push_back(p.at(i));
            k++;
          }
        }

        // Use rotating calipers algoritm to find most distant antipodal pair of hull points
        double diameter = 0;
        {
          unsigned i = 0;
          unsigned j = lower_hull.size() - 1;
          while (i < upper_hull.size() - 1 || j > 0) {
            double d = upper_hull.at(i).distanceTo(lower_hull.at(j));
            if (diameter < d) {
              diameter = d;
              if (pass == 0) {
                a = upper_hull.at(i);
                b = lower_hull.at(j);
              }
            }
            if (i == upper_hull.size() - 1) {
              j--;
            } else if (j == 0) {
              i++;
            } else if ( (upper_hull.at(i + 1).y() - upper_hull.at(i).y()) 
                      * (lower_hull.at(j).x() - lower_hull.at(j - 1).x())
                      > (upper_hull.at(i + 1).x() - upper_hull.at(i).x()) 
                      * (lower_hull.at(j).y() - lower_hull.at(j - 1).y())
                      ) {
              i++;
            } else {
              j--;
            }
          }
        }
        switch (pass) {
          case 0:
            if (excluding_worst) {
              extreme_spread = diameter;
              break;
            }
            return diameter;
          case 1:
            extreme_spread_excluding_a = diameter;
            break;
          case 2:
            extreme_spread_excluding_b = diameter;
            break;
        }
      }
      *excluding_worst = std::min(extreme_spread_excluding_a, extreme_spread_excluding_b);
      return extreme_spread;
    }
    
    // As per NSD (page 181) should be within 15 cm at 100 m
    // Group size 2.79295 corresponds to kuchnost~=3.15863,
    // so this is equivalent to 4.56 MOA.
    double nsd_kuchnost(void) const
    {
      // Only defined for 4 shot groups
      if (impact_.size() != 4) return std::numeric_limits<double>::quiet_NaN();
      
      // STP using all 4 shots
      auto stp = (impact_.at(0) + impact_.at(1) + impact_.at(2) + impact_.at(3)) / 4.;
      
      // Minimum radius of circle with center at STP that encloses all shots 
      double r = 0;
      for (unsigned i = 0; i < 4; i++) {
        double candidate = std::abs(stp - impact_.at(i));
        if (r < candidate) {
          r = candidate;
        }
      }
      
      // Exclude outlier, if any
      for (unsigned i = 0; i < 4; i++) {
      
        // STP excluding this shot
        auto stp3 = (stp * 4. - impact_.at(i)) / 3.;
        
        // Minimum radius of circle with center at STP of three shots excluding this shot
        double r2 = 0;
        for (unsigned j = 0; j < 4; j++) {
          if (i == j) {
            continue;
          }
          double candidate = std::abs(stp - impact_.at(i));
          if (r2 < candidate) {
            r2 = candidate;
          }
        }

        // Outlier is a shot 2.5x or more distant from STP of other three shots
        // than radius of the circle centered as STP of other three shots
        // that covers these three shots
        if (std::abs(stp3 - impact_.at(i)) > 2.5 * r2 && r > r2) {
          r = r2;
        }
      }
      
      // Diameter
      return 2 * r;
    }

    double avg_miss_radius(void) const
    {
      unsigned n = impact_.size();
      if (n < 2) return std::numeric_limits<double>::quiet_NaN();
      std::complex<double> center = 0;
      for (unsigned i = 0; i < n; i++) {
        center += impact_.at(i);
      }
      center /= n;
      double amr = 0;
      for (unsigned i = 0; i < n; i++) {
        amr += std::abs(impact_.at(i) - center);
      }
      amr /= n;
      return amr;
    }

    //
    // Ballistic Accuracy Class (before rounding)
    //
    // For each group g, calculate group center (mean)
    // For each shot i, find its radius squared (r2) relative to group center
    // Gaussian correction factor cG=1/EXP(LN(SQRT(2/(2n-2)))+GAMMALN((2n-1)/2)-GAMMALN((2n-2)/2))
    // Upper 90% confidence value sigma_U=cG*SQRT(SUM(r2)/CHIINV(0.9,2n-2))
    // Ballistic Accuracy Class is ROUND(sigma_U,0)
    //
    // http://ballistipedia.com/index.php?title=Ballistic_Accuracy_Classification
    //
    double bac(void) const
    {
      unsigned n = impact_.size();
      if (n < 2) return std::numeric_limits<double>::quiet_NaN();
      std::complex<double> center = 0;
      for (unsigned i = 0; i < n; i++) {
        center += impact_.at(i);
      }
      center /= n;
      double sum_r2 = 0;
      for (unsigned i = 0; i < n; i++) {
        sum_r2 += std::norm(impact_.at(i) - center);
      }

      // Memoize BAC factor to avoid recalculating it for each group
      static unsigned memoized_n = 0;
      static double memoized_factor = 0;
      if (n != memoized_n) {
        double cg = 1 / (exp(log(sqrt(2. / (2 * n - 2))) + lgamma((2. * n - 1) / 2) - lgamma((2. * n - 2) / 2)));
std::cout << "n=" << n << " cg=" << cg << " (should be 1.0139785698 for n = 10) \n";
        boost::math::chi_squared ch2(2 * n - 2);
        double chisq_inv_rt = boost::math::quantile(ch2, 1 - 0.9);
std::cout << "n=" << n << " chisq_inv_rt=" << chisq_inv_rt << " (should be 10.86494 for n = 10) \n";
        memoized_factor = cg / sqrt(chisq_inv_rt);
        memoized_n = n;
      }
      return memoized_factor * sqrt(sum_r2);
    }

    void show(void) const
    {
      for (unsigned i = 0; i < impact_.size(); i++) {
        std::cout << "g.add(std::complex<double>(" << impact_.at(i).real() << ", " << impact_.at(i).imag() << "));\n";
      }
    }
};

class DescriptiveStat
{
  public:
    DescriptiveStat() : n_(0), m_(0), s_(0) {}
    
    void push(double x)
    {
      double new_m = m_ + (x - m_) / (++n_);
      double new_s = s_ + (x - m_) * (x - new_m);
      m_ = new_m;
      s_ = new_s;
    }

    double mean(void) const
    { 
      return (n_ > 0) ? m_ : std::numeric_limits<double>::quiet_NaN();
    }

    double variance(void) const
    {
      return (n_ > 1) ? s_ / (n_ - 1) : std::numeric_limits<double>::quiet_NaN();
    }

    double stdev(void) const
    {
      return sqrt(variance());
    }

    double cv(void) const
    {
      return stdev() / mean();
    }

    void show(const char* metric, double theoretical = 0)
    {
      std::cout << metric << " mean=" << mean(); 
      if (theoretical > 0) {
        std::cout << " (expected " << theoretical << ")";
      }      
      std::cout << ", CV=" << cv() << "\n";
    }

  private:
    unsigned n_;
    double m_;
    double s_;
};

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

int main(int argc, char* argv[])
{
  DefaultRandomNumberGenerator pseudo_rng;
  long long experiments = 0;
  if (argc > 1) {
    experiments = atoi(argv[1]);
    
    // Self-test
    if (experiments == 0) {
      std::cout << "Comparing group_size_brute_force() and group_size_convex_hull()\n";
      {
        double max_diff = 0;
        for (unsigned i = 0; i < 1e6; i++) {
          unsigned shots = (i & 0xf) + 2;
          ShotGroup g;
          pseudo_rng.advance();
          for (unsigned j = 0; j < shots; j++) {
            g.add(pseudo_rng.point(j));
          }
          double bf2 = 0, ch2 = 0;
          double bf = g.group_size_brute_force((i & 0x10) ? &bf2 : NULL);
          double ch = g.group_size_convex_hull((i & 0x10) ? &ch2 : NULL);
          double diff_1 = fabs(bf - ch);
          if (max_diff < diff_1) {
            max_diff = diff_1;
          }
          if (diff_1 > 1e-8) {
            std::cout << "Expected group size " << bf << ", got " << ch << "\n";
            g.show();
            return 0;
          }
          if (i & 0x10) {
            double diff_2 = fabs(bf2 - ch2);
            if (diff_2 > 1e-8) {
              std::cout << "Expected group size excluding worst " << bf2 << ", got " << ch2 << "\n";
              g.show();
              return 0;
            }
            if (max_diff < diff_2) {
              max_diff = diff_2;
            }
          }
        }
        std::cout << "Max difference " << max_diff << "\n";
      }
      std::cout << "\tgroup_size_brute_force()\tgroup_size_convex_hull()\n";
      for (unsigned shots = 4; shots <= 256; shots *= 2) {
        std::cout << shots << " shots:\t";
        pseudo_rng.reset();
        time_t start_time;
        time(&start_time);
        double a = 0;
        for (unsigned i = 0; i < 1e6; i++) {
          ShotGroup g;
          pseudo_rng.advance();
          for (unsigned j = 0; j < shots; j++) {
            g.add(pseudo_rng.point(j));
          }
          a += g.group_size_brute_force();
        }
        time_t end_time;
        time(&end_time);
        std::cout << (end_time - start_time) << " µs, sum=" << a;

        pseudo_rng.reset();
        std::cout << "\t";
        time(&start_time);
        double b = 0;
        for (unsigned i = 0; i < 1e6; i++) {
          ShotGroup g;
          pseudo_rng.advance();
          for (unsigned j = 0; j < shots; j++) {
            g.add(pseudo_rng.point(j));
          }
          b += g.group_size_convex_hull();
        }
        time(&end_time);
        std::cout << (end_time - start_time) << " µs, sum=" << b << "\n";
      }
      return 0;
    }
  } else {
    std::cerr << "Usage: mcgs experiments [[[[shots_in_group] groups_in_experiment] proportion_of_outliers] outlier_severity]\n"
                 "0 experiments to run a self-test,\n"
                 "negative experiments to use additive quasi-random number generator,\n"
                 "negative shots_in_group to use Sobol quasi-random number generator\n";
    return -1;
  }
  std::unique_ptr<RandomNumberGenerator> rng(std::unique_ptr<RandomNumberGenerator>(new DefaultRandomNumberGenerator()));
  unsigned shots_in_group;
  {
    int isig = 5;
    if (argc > 2) {
      isig = atoi(argv[2]);
    }
    if (isig < 0) {
      shots_in_group = (unsigned)(-isig);
      std::cout << "Using Sobol quasi-random number generator for impact coordinates\n";
      rng = std::unique_ptr<RandomNumberGenerator>(new SobolRandomNumberGenerator(shots_in_group));
    } else {
      shots_in_group = (unsigned)(isig);
    }
  }
  if (experiments < 0) {
    experiments = -experiments;
    std::cout << "Using additive quasi-random number generator for impact coordinates\n";
    rng = std::unique_ptr<RandomNumberGenerator>(new AdditiveRandomNumberGenerator(shots_in_group));
  }
  unsigned groups_in_experiment = 1;
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
  for (unsigned i = 0; i < shots_in_group; i++) {
    mle_factor *= 4;
    if (i != shots_in_group - 1) {
      mle_factor *= i + 1;
    }
    mle_factor /= i + 1 + shots_in_group;
  }
  double wmr_to_r90hat_factor = 0;
  double swmr_to_r90hat_factor = 0;
  double swmrr_to_r90hat_factor = 0;
  double gs_to_r90hat_factor = 0;
  double sixtynine_to_r90hat_factor = 0;
  double rayleigh_to_r90hat_factor = 0;
  double mle_to_r90hat_factor = 0;
  
  // http://ballistipedia.com/images/7/7a/Statistical_Inference_for_Rayleigh_Distributions_-_Siddiqui%2C_1964.pdf
  std::pair<unsigned, unsigned> sixtynine_rank( (unsigned)(0.639 * (shots_in_group + 1)) - 1
                                              , (unsigned)(0.927 * (shots_in_group + 1)) - 1
                                              );
  if (shots_in_group == 10) { // Suboptimal, but more robust
    sixtynine_rank.first = 6 - 1;
    sixtynine_rank.second = 9 - 1;
  }
  if (groups_in_experiment == 1) {
    switch (shots_in_group) {
      case  1:
        // wxMaxima: find_root(integrate(x*exp(-x^2/2)*exp(-x^2*k^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        wmr_to_r90hat_factor = 3;
        break;
      case  3: 
        // wxMaxima: find_root(integrate(3*x*(1-exp(-x^2/2))^2*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        wmr_to_r90hat_factor = 1.414213562373095;
        break;
      case  5: 
        // wxMaxima: find_root(integrate(5*x*(1-exp(-x^2/2))^4*exp(-x^2/2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        wmr_to_r90hat_factor = 1.17215228421396;
        // wxMaxima:
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=20*x*(1-exp(-x^2/2))^3*exp(-x^2);
        // c(k):=integrate(p(q)*exp(-q^2*k^2/2),q,0,inf);
        // find_root(c(k)=1-0.9, k, 1, 2);
        swmr_to_r90hat_factor = 1.578643529936508;
        gs_to_r90hat_factor = 0.8;
        break;
      case 10: 
        // wxMaxima: find_root(integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2-k^2*x^2/2), x, 0, inf)=1-0.9, k, 1, 10);
        swmr_to_r90hat_factor = 1.187140545277712;
        swmrr_to_r90hat_factor = 1.2;
        gs_to_r90hat_factor = 0.6;
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
        wmr_to_r90hat_factor = 2.22197649;
        break;
      case 3:
        // wxMaxima: 
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=3*x*(1-exp(-x^2/2))^2*exp(-x^2/2);
        // p2(t):=''(integrate(2*p(u)*p(2*t-u),u,0,2*t));
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1, 3);
        wmr_to_r90hat_factor = 1.28875916;
        break;
      case 5:
        // wxMaxima:
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=5*x*(1-exp(-x^2/2))^4*exp(-x^2/2);
        // p2(t):=romberg(2*p(u)*p(2*t-u),u,0,2*t);
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1, 1.3);
        wmr_to_r90hat_factor = 1.10343798;
        // wxMaxima:
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=20*x*(1-exp(-x^2/2))^3*exp(-x^2);
        // p2(t):=romberg(2*p(u)*p(2*t-u),u,0,2*t);
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1, 2);
        swmr_to_r90hat_factor = 1.478456;
        break;
      case 10:
        // wxMaxima:
        // assume(x>0,t>0,u>0,q>0,k>1);
        // p(x):=90*x*(1-exp(-x^2/2))^8*exp(-x^2);
        // p2(t):=romberg(2*p(u)*p(2*t-u),u,0,2*t);
        // c2(k):=romberg(p2(q)*exp(-q^2*k^2/2),q,0,10);
        // find_root(c2(k)=1-0.9, k, 1.1, 1.15);
        swmr_to_r90hat_factor = 1.149216;
        swmrr_to_r90hat_factor = 1.15;
        break;
    }
  } else if (groups_in_experiment == 4) {
    switch (shots_in_group) {
      case  5:
        swmr_to_r90hat_factor = 1.43;
        gs_to_r90hat_factor = 0.723;
        break;
      default:
        break;
    }
  } else if (groups_in_experiment == 5) {
    switch (shots_in_group) {
      case  5:
        swmr_to_r90hat_factor = 1.42;
        gs_to_r90hat_factor = 0.72;
        break;
      default:
        break;
    }
  } else if (groups_in_experiment > 10) { // Asymptotic approximation for large number of groups in experiment
    switch (shots_in_group) {
      case  1:
        // wxMaxima: float(sqrt(2*log(10))/integrate(x*exp(-x^2/2)*x, x, 0, inf));
        wmr_to_r90hat_factor  = 1.712233160383746;
        break;
      case  3:
        // wxMaxima: float(sqrt(2*log(10))/integrate(3*(1-exp(-x^2/2))^2*x*exp(-x^2/2)*x, x, 0, inf));
        wmr_to_r90hat_factor  = 1.175960143568417;
        break;
      case  5:
        // wxMaxima: float(sqrt(2*log(10))/integrate(5*(1-exp(-x^2/2))^4*x*exp(-x^2/2)*x, x, 0, inf));
        wmr_to_r90hat_factor  = 1.037938194579831;

        // wxMaxima: float(sqrt(2*log(10))/integrate(20*x*(1-exp(-x^2/2))^3*exp(-x^2)*x, x, 0, inf));
        swmr_to_r90hat_factor = 1.38619009633813;

        gs_to_r90hat_factor = 0.7;
        break;
      case 10:
        // wxMaxima: float(sqrt(2*log(10))/integrate(90*x*(1-exp(-x^2/2))^8*exp(-x^2)*x, x, 0, inf));
        swmr_to_r90hat_factor  = 1.112257194707586;
        swmrr_to_r90hat_factor = 1.1;
        gs_to_r90hat_factor = 0.7;
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
  DescriptiveStat gs_s, gs_s2, bgs_s, ags_s, ags_s2, mgs_s, amr_s, bac_s, aamr_s, rayleigh_s, mle_s, median_r_s;
  DescriptiveStat worst_r_s, second_worst_r_s;
  std::map< std::pair<unsigned, unsigned>, DescriptiveStat> sixtynine_r_s;
  DescriptiveStat nsd_s, wr_s, swr_s, sixtynine_s;
  unsigned hits_wmr = 0;
  unsigned hits_swmr = 0;
  unsigned hits_swmrr = 0;
  unsigned hits_gs = 0;
  unsigned hits_sixtynine = 0;
  unsigned hits_rayleigh = 0;
  unsigned hits_mle = 0;
  unsigned bac_gt_1_ct = 0;
  double r90hat_wmr = 0;
  double r90hat_swmr = 0;
  double r90hat_swmrr = 0;
  double r90hat_gs = 0;
  double r90hat_sixtynine = 0;
  double r90hat_rayleigh = 0;
  double r90hat_mle = 0;
  std::default_random_engine outlier_generator;
  std::uniform_real_distribution<double> outlier_distribution;
  for (unsigned experiment = 0; experiment < experiments; experiment++) {
    double best_gs = 0;
    DescriptiveStat gs, gs2, amr, bac, wr, swr, sixtynine, rayleigh, mle;
    std::vector<double> gsg_v;
    for (unsigned j = 0; j < groups_in_experiment; j++) { 
      ShotGroup g; 
      rng->advance();
      std::vector<double> r;
      std::vector<double> r2;
      for (unsigned i = 0; i < shots_in_group; i++) {
        std::complex<double> p = rng->point(i);
        if ( proportion_of_outliers > 0 
          && outlier_distribution(outlier_generator) < proportion_of_outliers
           ) {
          p *= outlier_severity;
        }
        double ri = std::abs(p);
        g.add(p);
        r.push_back(ri);
        r2.push_back(ri*ri);

        // Use R90 estimates based on previous experiment to avoid correlation
        if (experiment != 0) { // If there is a prior group
          if (ri < r90hat_wmr) hits_wmr++;
          if (ri < r90hat_swmr) hits_swmr++;
          if (ri < r90hat_swmrr) hits_swmrr++;
          if (ri < r90hat_gs) hits_gs++;
          if (ri < r90hat_sixtynine) hits_sixtynine++;
          if (ri < r90hat_rayleigh) hits_rayleigh++;
          if (ri < r90hat_mle) hits_mle++;
        }
      } // Next shot

      std::sort(r.begin(), r.end()); // We'll need many ranks, faster to sort once

      double this_minus1 = 0;
      double this_gs = (shots_in_group < 32) 
                     ? g.group_size_brute_force(&this_minus1) 
                     : g.group_size_convex_hull(&this_minus1);
      gs_s.push(this_gs);
      gsg_v.push_back(this_gs);
      gs_s2.push(this_minus1);
      gs2.push(this_minus1);
      if (shots_in_group == 4) {
        nsd_s.push(g.nsd_kuchnost());
      }
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

      double this_bac = g.bac();
      bac_s.push(this_bac);
      bac.push(this_bac);
      if (this_bac > 1) {
        bac_gt_1_ct++;
      }

      double this_rayleigh = rayleigh_cep_factor * accumulate(r.begin(), r.end(), 0.);
      rayleigh.push(this_rayleigh);
      rayleigh_s.push(this_rayleigh);
      
      double this_mle = sqrt(accumulate(r2.begin(), r2.end(), 0.));
      mle.push(this_mle);
      mle_s.push(mle_factor * this_mle);
      
      double this_wr = r.at(shots_in_group - 1);
      wr.push(this_wr);
      worst_r_s.push(this_wr);
      
      double this_swr = r.at(shots_in_group - 2);
      swr.push(this_swr);
      second_worst_r_s.push(this_swr);
            
      double med = median(r);
      median_r_s.push(med);
      
      if (sixtynine_rank.first != sixtynine_rank.second) {
        double this_sixtynine = r.at(sixtynine_rank.first) + r.at(sixtynine_rank.second);
        sixtynine.push(this_sixtynine);
        if (shots_in_group <= 100) {
          // Remember all rank pairs, will choose the best one later
          for (unsigned rank_a = 0; rank_a < shots_in_group - 1; rank_a++) {
            for (unsigned rank_b = rank_a + 1; rank_b < shots_in_group; rank_b++) {
              double e = r.at(rank_a) + r.at(rank_b);
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
    r90hat_wmr = wr.mean()  * wmr_to_r90hat_factor;
    r90hat_swmr = swr.mean() * swmr_to_r90hat_factor;
    r90hat_swmrr = swr.mean() * swmrr_to_r90hat_factor;
    r90hat_gs = gs.mean()  * gs_to_r90hat_factor;
    r90hat_sixtynine = sixtynine.mean() * sixtynine_to_r90hat_factor;
    r90hat_rayleigh = rayleigh.mean() * rayleigh_to_r90hat_factor / sqrt(4 * log(2) / M_PI);
    if (shots_in_group == 20) {
      r90hat_mle = mle.mean() * mle_to_r90hat_factor;
    }
  } // Next experiment
  if (groups_in_experiment == 1) {
    std::cout << "--- Precision estimators ---\n"; 
    gs_s.show("Group size:");
    if (shots_in_group == 4) {
      nsd_s.show("Kuchnost:");
    }
    amr_s.show("Average Miss Radius:");
    bac_s.show("Ballistic Accuracy Class:");
    std::cout << "Percent of groups with BAC>1: " 
              << 100. * bac_gt_1_ct / groups_in_experiment / experiments << "%, expected 90%\n";
    std::cout << "--- Robust precision estimators ---\n"; 
    gs_s2.show("Group size (excluding worst shot in group):");
    std::cout << "--- Hit probability estimators ---\n"; 
    double theoretical_cep = 0;
    if (proportion_of_outliers == 0) {
      theoretical_cep =  sqrt(-2*log(0.5));
    }
    rayleigh_s.show("Rayleigh CEP estimator:", theoretical_cep);
    mle_s.show("Maximum likelihood CEP estimator:", theoretical_cep);
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
    std::cout << "--- Robust hit probability estimators ---\n"; 
    median_r_s.show("Median CEP estimator:"); // Not showing theoretical_cep because median has known bias
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
    std::cout << "--- Combinations of two order statistics ---\n"; 
    // Find combination of two order statistics with lowest CV
    {
      unsigned best_pos = 0;
      double best_cv = 0;
      unsigned pos = 0;
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
            std::cout << "R" << (sixtynine_rank.first  + 1) << ":" << shots_in_group 
                      << "+R" << (sixtynine_rank.second + 1) << ":" << shots_in_group;
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
  std::cout << "--- R90 estimators ---\n"; 
  double denominator = shots_in_group * ((double)groups_in_experiment * (experiments - 1));
  if (wmr_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << wmr_to_r90hat_factor 
              << " * worst miss radius: " 
              << 100. * hits_wmr / denominator << "%\n";
  }
  if (swmr_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << swmr_to_r90hat_factor
              << " * second worst miss radius: " 
              << 100. * hits_swmr / denominator << "%\n";
  }
  if (swmrr_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << swmrr_to_r90hat_factor 
              << " * second worst miss radius: " 
              << 100. * hits_swmrr / denominator << "%\n";
  }
  if (gs_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << gs_to_r90hat_factor 
              << " * group size: " 
              << 100. * hits_gs / denominator << "%\n";
  }
  if (sixtynine_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << sixtynine_to_r90hat_factor 
              << " * (R" << (sixtynine_rank.first  + 1) << ":" << shots_in_group 
              << " + R"  << (sixtynine_rank.second + 1) << ":" << shots_in_group << "): "
              << 100. * hits_sixtynine / denominator << "%\n";
  }
  if (rayleigh_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << rayleigh_to_r90hat_factor 
              << " * average miss radius: " 
              << 100. * hits_rayleigh / denominator << "%\n";
  }
  if (mle_to_r90hat_factor > 0) {
    std::cout << "Percent of hits within " << mle_to_r90hat_factor 
              << " * square root of sum of squares: " 
              << 100. * hits_mle / denominator << "%\n";
  }
  return 0;
}
