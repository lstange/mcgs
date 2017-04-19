//
// Extreme spread of 2 to N shot groups, with impact coordinates pulled from bivariate 
// standard normal distribution. Use dynamic programming: start with one shot group, add 
// shots one by one. Keep track of group size so far, update it as new shots are being added.
// Update mean extreme spread for each group size from 2 to N. 
//
// Multithreaded vestion. To run, do "cargo run". To build release, "cargo build --release"
//
// Output: number of shots in group, average extreme spread
//
extern crate rand;
use rand::distributions::{Normal, IndependentSample};
use std::env;
use std::thread;

fn main()
{
  const N: usize = 10; // Maximum number of shots in group
  const THREADS: usize = 4;
  let mut children = vec![];

  // Number of groups, can be changed from the command line
  let mut groups = 1_000_000u64;
  let args: Vec<String> = env::args().collect();
  if args.len() == 2 {
    groups = args[1].parse().unwrap_or(0);
  }
  groups /= THREADS as u64; // divide groups equally among threads

  for _ in 0..THREADS {
    children.push(thread::spawn(move || {
      // Thread begins here
      let normal = Normal::new(0.0, 1.0);
      let mut rng = rand::thread_rng();
      let mut mean: [f64; N] = [0f64; N];
      let mut count: [u64; N] = [0u64; N];
      for _ in 0..groups {
        let mut impactx: [f64; N] = [0f64; N];
        let mut impacty: [f64; N] = [0f64; N];
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
      mean // Each thread returns its own averages
    }));
  }

  // Wait for the threads to finish, average averages from all threads, show results
  let mut sum: [f64; N] = [0f64; N];
  for child in children {
    let m = child.join().unwrap();
    for i in 1..N {
      sum[i] += m[i];
    }
  }
  for i in 1..N {
    println!("{} {}", i + 1, sum[i] / (THREADS as f64));
  }
}
