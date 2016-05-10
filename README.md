## Group Size, AMR, CEP, and Hit Probability

People often measure firearm [precision](http://en.wikipedia.org/wiki/Accuracy_and_precision) in terms of group size. This program lets you run Monte Carlo simulation to determine relationships between group size and other metrics. Except where noted, impact coordinates are pulled from the same [bivariate normal distribution](http://en.wikipedia.org/wiki/Bivariate_normal_distribution) with mean 0 and variance 1 to make results comparable. If you don't want to run the simulations yourself, this page covers some common cases.

### Group Size

Group size is maximum distance between the centers of two shots in a group.

![Measuring group size](group_size.jpg?raw=true)

Here are some group sizes pulled from our reference distribution:

|                    | Mean |  CV  |
| ------------------ | ----:|-----:|
| 3 shot group size  | 2.41 | 0.37 |
| 5 shot group size  | 3.06 | 0.27 |
| 10 shot group size | 3.81 | 0.19 |

CV is [coefficient of variation](http://en.wikipedia.org/wiki/Coefficient_of_variation): the ratio of standard deviation to mean. It can be thought of as *noise to signal ratio*. As you can see there's quite a bit of noise, meaning that one group does not let us measure precision well.

### AMR

AMR is average miss radius, measured from the mean point of impact of the group.

|           |  AMR |  CV  |
|-----------|-----:|-----:|
| 3 shots   | 1.02 | 0.37 |
| 5 shots   | 1.12 | 0.26 |
| 10 shots  | 1.19 | 0.17 |
| 100 shots | 1.25 | 0.05 |

### Kuchnost

Accuracy metric used by the Soviets and described in their NSD (*Nastavlenie po Strelkovomu delu*). It is based on four shots and calculated as follows:

* Find mean point of impact of the four shots
* Using this point as the center, find minimum radius of circle that encloses all shots
* Unless there is an outlier, in which case discard the outlier an repeat the procedure
with the three remaining shots. 

Outlier is a shot 2.5 times or more distant from mean point of impact of the other three shots than any of these three shots.

Page 181 of NSD states that AKM should be within 15 cm at 100 m, which corresponds to 4.56 MOA average 5-shot group size. 

### Miss Radius

Radial miss distances for bivariate normal distribution follow [Rayleigh distribution](http://en.wikipedia.org/wiki/Rayleigh_distribution). The radius of a circle containing centers of given proportion of shots can be calculated analytically:

|                        |      Exact          | Approximate |
|------------------------|---------------------|------------:|
| R<sub>50</sub> aka CEP | `sqrt(-2*ln(0.5))`  |        1.18 |
| R<sub>90</sub>         | `sqrt(-2*ln(0.1))`  |        2.15 |
| R<sub>95</sub>         | `sqrt(-2*ln(0.05))` |        2.45 |
| R<sub>99</sub>         | `sqrt(-2*ln(0.01))` |        3.03 |

Here R<sub>50</sub> is radius of a circle containing centers of half the impacts, R<sub>90</sub> contains 90% and so on.

Using the tables below, one can convert between group size and radius of the circle containing given proportion of impacts. This conversion assumes ideal accuracy (perfect zero). More on that later.

|                  |3 shot group size|5 shot group size|10 shot group size|R<sub>50</sub> |R<sub>90</sub> |R<sub>95</sub> |R<sub>99</sub> |
|------------------|----------------:|----------------:|-----------------:|---:|---:|---:|---:|
|3 shot group size |             1.00|             1.27|              1.58|0.49|0.89|1.02|1.26|
|5 shot group size |             0.79|             1.00|              1.24|0.38|0.70|0.80|0.99|
|10 shot group size|             0.63|             0.80|              1.00|0.31|0.56|0.64|0.80|
|R<sub>50</sub>    |             2.05|             2.60|              3.24|1.00|1.82|2.08|2.58|
|R<sub>90</sub>               |             1.12|             1.43|              1.78|0.55|1.00|1.14|1.41|
|R<sub>95</sub>    |             0.98|             1.25|              1.56|0.48|0.88|1.00|1.24|
|R<sub>99</sub>    |             0.79|             1.01|              1.26|0.39|0.71|0.81|1.00|

> **Example 1**
> 4" 5 shot group corresponds to R<sub>95</sub> = 4" * 2.45 / 3.07 = 4" * 0.8 = 3.2"

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

> **Example 2** 
> If the best of 10 five-shot groups measures 4", that corresponds to R<sub>95</sub> = 4" * 2.45 / 1.89 = 5.2". Compare this number to 3.2" from Example 1.

> **Example 3** 
> Averaging group sizes of two 5 shot groups works about as well as one 10 shot group size (in both cases CV is approximately 0.19).

### CEP

If accuracy is less than ideal, then group size alone does not mean much. 2" group 2' above the target is not particularly useful. But there is a way to estimate  hit probability that does not have this problem. It works by estimating CEP rather than group size.

CEP stands for [Circular Error Probable](http://en.wikipedia.org/wiki/Circular_error_probable): minimum radius of a circle centered on the target that contains half the impacts. CEP is sometimes called R<sub>50</sub>. If we only care about precision we can center the circle about the mean, but then it won't help with hit probability.

There are several ways to estimate CEP. The easiest two are median and Rayleigh estimators. Both look at *radial miss distances* - distances from the center of the target to the center of the impact.

*Median CEP estimator* is the simplest one possible: rank order shots by radial miss distance, then take the median. For example, in a 5 shot group discard two impacts closest to the center of the target and two impacts furthest from the center of the target, then measure the distance between the center of the target and the center of remaining impact. This gives you estimated CEP.

Median estimator is non-parametric (it does not rely on assumptions about underlying distributions) and is robust (not very sensitive to outliers). It's slightly biased up, especially for small groups, but the bias is in the third significant digit so probably won't be visible in presence of mush stronger noise.

![Estimating CEP as median radial miss](cep.jpg?raw=true)

*Rayleigh CEP estimator* is a bit more work: measure all radial miss distances, take the average, then multiply it by sqrt((2 ln 4)/Pi)&nbsp;&asymp;&nbsp;0.9394. This magic number comes from the observation that mean of Rayleigh distribution (that we just estimated by averaging radial miss distances) is &sigma;&nbsp;sqrt(&pi;&nbsp;/&nbsp;2&nbsp;) and CEP is median of this distribution, or &sigma;&nbsp;sqrt(&nbsp;ln&nbsp;4&nbsp;).

|             |Median Estimator Mean|Median Estimator CV|Rayleigh Estimator Mean|Rayleigh Estimator CV|
|-------------|--------------------:|------------------:|----------------------:|--------------------:|
|3 shot group |                 1.21|               0.37|                   1.18|                 0.30|
|5 shot group |                 1.20|               0.30|                   1.18|                 0.23|
|10 shot group|                 1.19|               0.21|                   1.18|                 0.16|

In this simulation CV of Rayleigh estimator is consistently lower, but that's to be expected. Rayleigh estimator is parametric - it assumes the data follows a certain distribution, and in case of our Monte Carlo simulation that's certainly true. If shots follow a different distribution, especially one with heavy tails, the picture can be different.

*Maximum likelihood CEP estimator* is even more work: sum squares of all radial miss distances, take square root, then multiply by [ugly adjustment factor](https://en.wikipedia.org/wiki/Rayleigh_distribution#Parameter_estimation) `sqrt(ln(2)/&pi;)*power(4,N)*N!*(N-1)!/(2*N)!` that depends on number of shots N. In theory it's slightly better than Rayleigh estimator, but even more sensitive to outliers.

### Estimating R<sub>90</sub> from a Single Order Statistic

R<sup>m:n</sup> stands for "*m*th smallest miss radius in a group of n shots". For small
groups using the worst miss radius results in lowest variance, while second worst miss radius works better for larger groups. The latter is also less sensitive to fliers.

|  Shots | R<sub>90</sub>         |
|-------:|------------------------|
|       1| 3 R<sub>1:1</sub>      |
|       2| 1.732 R<sub>2:2</sub>  |
|       3| 1.414 R<sub>3:3</sub>  |
|       5| 1.172 R<sub>5:5</sub>  |
|       8| 1.281 R<sub>7:8</sub>  |
|      10| 1.187 R<sub>9:10</sub> |

### Contaminated Normal Distribution

In practice, impact coordinates do not necessarily follow normal distribution. A canonical example is contaminated normal distribution: a mixture of a standard normal distribution and a normal distribution with a different variance. It might simulate shooter errors or other rare events. In the following example 1% of shots were pulled from the distribution with 10 times higher standard deviation.

|5 shot group        |Median Estimator CV|Rayleigh Estimator CV|
|--------------------|------------------:|--------------------:|
|Standard normal     |               0.30|                 0.23|
|Contaminated normal |               0.30|                 0.48|

CV of median estimator did not budge, but CV of Rayleigh estimator doubled. The takeaway is that unless you are certain that the data follows normal distribution, it might be prudent to use a robust estimator such as the median.

### Optimal Number of Shots in Group

Assuming normal distribution, optimal number of shots per group is 6. That said, the difference between 5 and 6 is very small, and 5 is more convenient.

The following table shows CV of average group size from 2,520 shots broken down in different number of groups.

|Shots in group|Groups| Shots |   CV    |    |
|-------------:| ----:|------:|--------:|:---|
|            3 | 840  | 2,520 | 0.01281 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|
|            4 | 630  | 2,520 | 0.01221 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|
|            5 | 504  | 2,520 | 0.01204 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|
|            6 | 420  | 2,520 | 0.01197 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|
|            7 | 360  | 2,520 | 0.01201 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|
|            8 | 315  | 2,520 | 0.01208 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|
|            9 | 280  | 2,520 | 0.01217 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|
|           10 | 252  | 2,520 | 0.01226 | IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII|

In presence of outliers, such as with contaminated normal distribution, CV simply grows with number of shots in group. This happens because probability of catching an outlier in a group is proportional to number of shots in group.

### Median Group Size

Averaging works better with normal distribution, but median is better for contaminated normal.

|5 groups, 5 shots each |Average Group Size CV|Best Group Size CV|Median Group Size CV|
|-----------------------|--------------------:|-----------------:|-------------------:|
|Standard normal        |                 0.12|              0.21|                0.15|
|Contaminated normal    |                 0.35|              0.22|                0.17|

Distribution of group size is asymmetric, so median is not the same as mean. For standard normal, this difference is within 2%, but can be larger for distributions with heavier tails.

### Group Size Excluding Worst Shot

This sounds like cheating, but in reality it is a good, robust statistic (less sensitive to occasional fliers). To avoid bias, excluding the worst shot needs to be done for *all* groups, not just the ones with obvious outliers.

To compare with regular group size:

  + After excluding worst shot in a 5 shot group, multiply the result by 1.45 to get regular five-shot group size 
  + Group size after excluding the worst shot in a 10-shot group is approximately the same as regular five-shot group size

### tl,dr: Rules of Thumb

Assuming perfect zero:

  + 3 shot group size is about the same as R<sub>95</sub>, or twice the CEP
  + 5 shot group size is about the same as R<sub>99</sub>
  + CEP in cm is about the same as 5 shot group size in inches (more precisely, coefficient is 2.6 rather than 2.54)
  + R<sub>90</sub> is about 1.2 times larger than worst miss radius in a five-shot group
  + R<sub>90</sub> is about 1.2 times larger than second worst miss radius in a ten-shot group
