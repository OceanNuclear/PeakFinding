import numpy as np
from numpy import pi, sqrt, exp, array as ary, log as ln
# # create a chainmul operator
# from operator import mul
# from functools import reduce
# chainmul = lambda *args: reduce(mul, args)
import scipy.stats

class ProbabilityDistribution():
    """
    ABC for probability distribution.
    """
    @property
    def mean(self):
        return self._distribution.mean()

    @property
    def variance(self):
        return self._distribution.var()

    def get_samples(self, sample_size):
        self._distribution.rvs(sample_size)

    def likelihood(self, samples):
        """
        Parameters
        ----------
        samples: the list of samples that we want the likelihood values of.
        This gives the likelihood of these samples coming from this distribution
        """
        return self._distribution.pmf(samples)

    def negative_log_likelihood(self, samples):
        lamb = self.mean


class Poisson(ProbabilityDistribution):
    """
    Wrapper around the scipy poisson
    """
    def __init__(self, lamb):
        self._distribution = scipy.stats.poisson(lamb)

class Normal(ProbabilityDistribution):
    def __init__(self, mean, variance):
        self._distribution = scipy.stats.norm(mean, sqrt(variance))

    def likelihood(self, sample):
        """
        Parameters
        """

if __name__=='__main__':
    PLOT = True
    from matplotlib import pyplot as plt
    # from math import factorial
    # fac = lambda a: ary([factorial(i) for i in a])
    from scipy.special import factorial as fac

    # lambda: the free variable to change.
    N = np.arange(1000) # range of all natural numbers. 1000 is close enough to infinity.
    y = sqrt(N) # sqrt(counts) representation

    for lamb in np.logspace(-1, 2, 30):
        gaussian_mean = N_mean = gaussian_var = N_var = lamb
        
        # lamb = 20.0 # 1.2 # 2.0 # 0.5
        P_N_precursor = fac(N) * lamb**-N
        P_N = exp(-lamb) * np.nan_to_num(1/P_N_precursor, nan=0.0, posinf=0.0, neginf=0.0) # the pmf: probability mass function

        y_mean = (y * P_N).sum()
        # print(y_mean)
        y_var = ((y - y_mean)**2 * P_N).sum() # method 1 
        # y_var = (y**2 * P_N).sum() - y_mean**2 # method 2

        print("gaussian_mean = N_mean = gaussian_var = N_var =", lamb)
        print("y_mean, y_var =", y_mean, y_var)
        print("sqrt(lamb) =", sqrt(lamb))

        if PLOT:
            ax = plt.subplot()
            bar_widths = np.diff(np.append(y, y[-1]))
            ax.bar(y, P_N, width=np.clip(0.2, None, bar_widths), align="edge", alpha=0.5, label="sqrt(counts)")

            saved_xmax = ax.get_xlim()[1]
            ax.bar(N, P_N, width=0.2, align="edge", alpha=0.5, label="counts")
            ax.set_xlim(0, np.clip(saved_xmax, N[(np.cumsum(P_N)>=0.999).argmax()+1], None)) # undo the expansion of the graph due to plotting the new pdf

            ax.set_xlabel("y values (sqrt(count) values)")
            ax.set_ylabel("probability")
            ax.legend()
            plt.show()
        print()
"""
Conclusion:
y_var started at 0.3783934365408028 at lamb=0.1, and asymptotically approaches around 0.25
"""