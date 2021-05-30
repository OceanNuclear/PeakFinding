from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
# from math import factorial
# fac = lambda a: ary([factorial(i) for i in a])
from scipy.special import factorial as fac

"""
This module explores what happens when you do weighted average of multiple independently distributed poisson distribution,
i.e. using applying a kernel on a series of neighbouring bins, all of whom follows the same poisson distribution.
    The weighting factor = the kernel[i].
"""

def mean_variance(distribution_x, distribution_P):
    mean = (distribution_x * distribution_P).sum() # scalar
    variance = ((distribution_x - mean)**2 * distribution_P).sum() # method 1
    # variance = (distribution_x**2 * P_N).sum() - mean**2 # method 2
    return mean, variance # two scalars

kernel = [0.125, 0.375, 0.375, 0.125] # this gives a factor of exactly 3.2 reduction from the variance of y to the variance of convolved distribution
# kernel = [0.25, 0.25, 0.25, 0.25] # this gives a factor of exactly 3.2 reduction from the variance of y to the variance of convolved distribution
# kernel = [0.00, 0.00, 0.50, 0.50] # this gives a factor of exactly 3.2 reduction from the variance of y to the variance of convolved distribution
# kernel = [0.00, 0.00, 0.60, 0.40] # this gives a factor of exactly 3.2 reduction from the variance of y to the variance of convolved distribution
if __name__=='__main__':
    # lambda: the free variable to change.
    N = np.arange(100) # range of all natural numbers. 1000 is close enough to infinity.
    y = sqrt(N) # sqrt(counts) representation
    y_line, c_line, y_distortion, sqrt_line = [], [], [], []

    # for lamb in np.logspace(-1, 2, 30):
    for lamb in np.logspace(-1, np.log10(20)):
        print("processing lambda=", lamb)
        gaussian_mean = N_mean = gaussian_var = N_var = lamb
        
        # lamb = 20.0 # 1.2 # 2.0 # 0.5
        P_N_precursor = fac(N) * lamb**-N
        P_N = exp(-lamb) * np.nan_to_num(1/P_N_precursor, nan=0.0, posinf=0.0, neginf=0.0) # the pmf: probability mass function

        y_mean, y_var = mean_variance(y, P_N)

        c_dist_x, c_dist_P = ary([0.0]), ary([1.0]) #convolved distribution 
        for k in kernel:
            c_dist_x = np.add.outer(c_dist_x, k*y).flatten()
            c_dist_P = np.outer(c_dist_P, P_N).flatten()
        print("Captured {} of the full probability distribution".format(c_dist_P.sum()))

        # convolved distribution's parameters: mean, variance, etc.
        c_mean, c_var = mean_variance(c_dist_x, c_dist_P)
        y_line.append((y_mean, y_var))
        c_line.append((c_mean, c_var))

        y_distortion.append((N_mean, c_mean))
        sqrt_line.append((N_mean, sqrt(N_mean)))
        print()
    plt.plot(*ary(y_distortion).T, label="actual mean")
    plt.plot(*ary(sqrt_line).T, label="sqrt approx. mean")
    plt.title("sqrt distribution mean's deviation\nfrom the sqrt mean(N) approximation")
    plt.show()
    plt.plot(*ary(y_line).T)
    plt.plot(*ary(c_line).T)
    plt.title("after pplying c_lin")
    plt.show()
"""
Conclusion:
1. For the distribution y=sqrt(counts), y_var started at 0.3783934365408028 at lamb=0.1, and asymptotically approaches around 0.25
2. averaging up samples from a few of these identical distributions (in an unweighted manner) will yield a smaller variance
3. Weighted average of several of these identical distribution
    for [a, b, c, d] where a+b+c+d == 1
    new_variance = sum(a**2 + b**2 + c**2 + d**2) * old_variance
4. an all-positive kernel is doing precisely this: a weighted average.
"""