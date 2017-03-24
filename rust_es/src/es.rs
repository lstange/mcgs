// Extreme spread of 1 to N shot groups, with impact coordinates pulled from bivariate 
// normal distribution. Use dynamic programming: start with one shot group, add shots 
// one by one. Keep track of group size so far, update it as new shots are being added.
// Update mean extreme spread for each group size from 1 to N. 
extern crate rand;
use rand::distributions::{Normal, IndependentSample};
use rand::Rng;
use std::env;

fn main()
{
  let normal = Normal::new(0.0, 1.0);
  let mut rng = rand::thread_rng();
  let args: Vec<String> = env::args().collect();
  const N: usize = 10;
  let mut groups = 1_000_000u64;
  if args.len() == 2 {
    groups = args[1].parse().unwrap_or(0);
  }

  let mean: [f64; N] = [0f64; N];
  let count: [u64; N] = [0u64; N];
  for group in 0..groups {
    let impactx: [f64; N];
    let impacty: [f64; N];
    let mut d = 0f64;
    for i in 0..N {
      let x = normal.ind_sample(&mut rng);
      let y = normal.ind_sample(&mut rng);
      impactx[i] = x;
      impacty[i] = y;
      for j in 0..i {
        let candidate = (x - impactx[j]).hypot(y - impacty[j]);
        if d < candidate {
          d = candidate;
      }
        }
      count[i] += 1;
      mean[i] += (d - mean[i]) / (count[i] as f64);
    }
  }
  for i in 0..N {
    println!("{} {}", i + 1, mean[i]);
  }
}
