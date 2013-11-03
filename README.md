# Group Size, CEP, and Hit Probability

People often measure firearm [precision](http://en.wikipedia.org/wiki/Accuracy_and_precision) in terms of group size. This program lets you run Monte Carlo simulation to determine relationships between group size and other metrics. Impact coordinates are pulled from the same [bivariate normal distribution](http://en.wikipedia.org/wiki/Bivariate_normal_distribution) with mean 0 and variance 1 to make results comparable. If you don't want to run the simulations yourself, this page covers some common cases.

### Group Size

Group size is maximum distance between the centers of two shots in a group.

![Measuring group size](https://github.com/lstange/mcgs/raw/gs_400.jpg)

Here are some group sizes pulled from our reference distribution:

|                    | Mean |  CV  |
| ------------------ | ----:|-----:|
| 3 shot group size  | 2.41 | 0.37 |
| 5 shot group size  | 3.07 | 0.27 |
| 10 shot group size | 3.81 | 0.19 |

CV is [coefficient of variation](http://en.wikipedia.org/wiki/Coefficient_of_variation): the ratio of standard deviation to mean. It can be thought of as *noise to signal ratio*. As you can see there's quite a bit of noise, meaning that one group does not let us measure precision well.

### Target Radius

Radial miss distances for bivariate normal distribution follow [Rayleigh distribution](http://en.wikipedia.org/wiki/Rayleigh_distribution). The radius of a circle containing centers of given proportion of shots can be calculated analytically:

|             |      Exact          | Approximate |
|-------------|---------------------|------------:|
| R50 aka CEP | `sqrt(-2*ln(0.5))`  |        1.18 |
| R90         | `sqrt(-2*ln(0.1))`  |        2.15 |
| R95         | `sqrt(-2*ln(0.05))` |        2.45 |
| R99         | `sqrt(-2*ln(0.01))` |        3.03 |

Here R50 is radius of a circle containing centers of half the impacts, R90 contains 90% and so on.

Using the tables below, one can convert between group size and radius of the circle containing given proportion of impacts. This conversion assumes ideal accuracy (perfect zero). More on that later.

|                  |3 shot group size|5 shot group size|10 shot group size|R50 |R90 |R95 |R99 |
|------------------|----------------:|----------------:|-----------------:|---:|---:|---:|---:|
|3 shot group size |             1.00|             1.27|              1.58|0.49|0.89|1.02|1.26|
|5 shot group size |             0.79|             1.00|              1.24|0.38|0.70|0.80|0.99|
|10 shot group size|             0.63|             0.80|              1.00|0.31|0.56|0.64|0.80|
|R50               |             2.05|             2.60|              3.24|1.00|1.82|2.08|2.58|
|R90               |             1.12|             1.43|              1.78|0.55|1.00|1.14|1.41|
|R95               |             0.98|             1.25|              1.56|0.48|0.88|1.00|1.24|
|R99               |             0.79|             1.01|              1.26|0.39|0.71|0.81|1.00|

> **Example 1** 4" 5 shot group corresponds to R95 = 4" * 2.45 / 3.07 = 4" * 0.8 = 3.2"

### Best Group

Sometimes people report best group size rather than average group size. Let's do the comparison.

|                  |Mean| CV |                     |Mean| CV |
|------------------|---:|---:|---------------------|---:|---:|
|One 5 shot group  |3.07|0.27|One 5 shot group     |3.07|0.27|
|Best of 2 groups  |2.60|0.24|Average of 2 groups  |3.07|0.19|
|Best of 5 groups  |2.15|0.21|Average of 5 groups  |3.07|0.12|
|Best of 10 groups |1.89|0.20|Average of 10 groups |3.07|0.08|
|Best of 100 groups|1.31|0.17|Average of 100 groups|3.07|0.03|    

Note how noisy best group size is compared to average group size. Average of two groups has less noise (CV=0.19) than best of 10 groups (CV=0.20), and it takes 10 rounds rather than 50.

> **Example 2** If the best of 10 five-shot groups measures 4", that corresponds to R95 = 4" * 2.45 / 1.89 = 5.2". Compare this number to 3.2" from Example 1.

> **Example 3** Averaging group sizes of two 5 shot groups works about as well as one 10 shot group size (in both cases CV is approximately 0.19).

### CEP

If accuracy is less than ideal, then group size alone does not mean much. 2" group 2' above the target is not particularly useful. But there is a way to estimate  hit probability that does not have this problem. It works by estimating CEP rather than group size.

CEP stands for [Circular Error Probable](http://en.wikipedia.org/wiki/Circular_error_probable) <A href=""></A>: minimum radius of a circle centered on the target that contains half the impacts. CEP is sometimes called R50. If we only care about precision we can center the circle about the mean, but then it won't help with hit probability.

There are several ways to estimate CEP. The easiest two are median and Rayleigh estimators. Both look at *radial miss distances* - distances from the center of the target to the center of the impact.

*Median CEP estimator* is the simplest one possible: rank order shots by radial miss distance, then take the median. For example, in a 5 shot group discard two impacts 
closest to the center of the target and two impacts furthest from the center of the target, then measure the distance between the center of the target and the center of remaining impact. This gives you estimated CEP.

![Estimating CEP as median radial miss](https://github.com/lstange/mcgs/raw/r50_400.jpg)

*Rayleigh CEP estimator* is a bit more work: measure all radial miss distances, take the average, then multiply it by &radic;<span style="text-decoration:overline;">&nbsp;(2 ln 4)&nbsp;/&nbsp;&pi;&nbsp;</span>&nbsp;&asymp;&nbsp;0.9394. This magic number comes from the observation that mean of Rayleigh distribution (that we just estimated by averaging radial miss distances) is &sigma;&nbsp;&radic;<span style="text-decoration:overline;">&nbsp;&pi;&nbsp;/&nbsp;2&nbsp;</span> and CEP is median of this distribution, or &sigma;&nbsp;&radic;<span style="text-decoration:overline;">&nbsp;ln&nbsp;4&nbsp;</span>.

|             |Median Estimator Mean|Median Estimator CV|Rayleigh Estimator Mean|Rayleigh Estimator CV|
|-------------|--------------------:|------------------:|----------------------:|--------------------:|
|3 shot group |                 1.21|               0.37|                   1.18|                 0.30|
|5 shot group |                 1.20|               0.30|                   1.18|                 0.23|
|10 shot group|                 1.19|               0.21|                   1.18|                 0.16|

In this simulation CV of Rayleigh estimator is consistently lower, but that's to be expected. Rayleigh estimator is parametric - it assumes the data follows a certain distribution, and in case of our Monte Carlo simulation that's certainly true. If shots follow a different distribution, especially one with heavy tails, the picture can be different.

Median estimator is non-parametric (it does not rely on assumptions about underlying distributions) and is more robust (less sensitive to outliers). It works well with large number of shots, but even for small groups it's good enough.

### Rules of Thumb

  + 3 shot group size &asymp; R95 (assuming perfect zero)
  + 5 shot group size  &asymp; R99 (assuming perfect zero)
  + R95 &asymp; 2 * CEP


