import numpy as np
from numpy import pi, sqrt, exp, array as ary, log as ln
from math import fsum
tau = 2*pi
# # create a chainmul operator
# from operator import mul
# from functools import reduce
# chainmul = lambda *args: reduce(mul, args)
import scipy.stats

class ProbabilityDistribution():
    """
    ABC for probability distributions that relies heavily on scipy.stats
    Classes that implement this must have a "_distribution" attribute, created from one of the scipy.stats distributions
    """
    @property
    def mean(self):
        return self._distribution.mean()

    @property
    def variance(self):
        return self._distribution.var()

class ProbabilityDistributionLikelihood(ProbabilityDistribution):
    """
    ABC for probability distribution that can use likelihood function
    """

    def get_samples(self, sample_size):
        self._distribution.rvs(sample_size)

    def naive_NLL(self, samples):
        likelihood = self.likelihood(samples)
        return -ln(sqrt(tau * self._lambda) * likelihood)

class Poisson(ProbabilityDistributionLikelihood):
    """
    Wrapper around the scipy poisson
    """
    def __init__(self, lamb):
        self._distribution = scipy.stats.poisson(lamb)
        self._lambda = lamb

    def likelihood(self, samples):
        """
        Parameters
        ----------
        samples: the list of samples that we want the likelihood values of.
        This gives the likelihood of these samples coming from this distribution
        """
        return self._distribution.pmf(samples)

    def negative_log_likelihood(self, samples):
        """
        Direct formula that calculates the negative log likelihood
        The samples provided must be integers
        """
        if self._lambda==0:
            return np.zeros(np.shape(samples))
        NLL_list = []
        const_offset = -0.5 * ln(tau) + self._lambda -0.5*ln(self._lambda)
        for N in samples:
            factorial_part = ln(np.arange(1, N+1))
            power_part = -ln(self._lambda)
            overflow_likely_part = factorial_part + power_part

            # perform a "folding" operation on the array, where the first element is added to the last, second to the second-last, etc.
            # this should yield an array of half the length if even N, half-length +1 if odd N..
            N_2 = N//2 #floor div of an integer should be an int
            first_part, second_part = overflow_likely_part[:N_2], overflow_likely_part[-N_2:]
            overflowy_part_folded_in_half = first_part + second_part[::-1]
            if (N%2)==1:
                overflowy_part_folded_in_half = np.append(overflowy_part_folded_in_half, overflow_likely_part[N_2]) # append in the exact half-way point.

            nll = const_offset + fsum(overflowy_part_folded_in_half) # the last two terms will partially cancel out each other. so they must be added together first.

            NLL_list.append(nll)
        return ary(NLL_list)

    def less_naive_negative_log_likelihood(self, samples):
        if self._lambda==0:
            return 0.0
        return self.naive_NLL(samples) - 0.5*ln(tau * self._lambda)

class Normal(ProbabilityDistributionLikelihood):
    def __init__(self, mean, variance):
        """
        Parameters
        ----------
        mean: mean of the normal distribution (the centroid)
        variance: sigma**2 of the normal distribution (standard deviation)
        """
        self._distribution = scipy.stats.norm(mean, sqrt(variance)) # norm has signature (mean, sigma)
        # therefore we need to do sqrt(variance) to get sigma

    def likelihood(self, samples):
        """
        Parameters
        ----------
        samples: the list of samples that we want the likelihood values of.
        """
        self._distribution.pdf(samples)

    def negative_log_likelihood(self, samples):
        """
        direct formula that calculates the negative log-likelihood of a normal distribution:
        chi^2 = sum_i(
                    (x[i] - distribution.mean)**2
                    _____________
                    variance
                    )
        """
        return 0.5*np.nan_to_num((samples-self.mean)**2/self.variance, nan=0.0)

class Chi2(ProbabilityDistribution):
    def __init__(self, DoF):
        """
        Usually, a function is fitted, generating a goodness-of-fit parameter known as chi2.
        The chi2 itself is also a statisical variable

        Parameters
        ----------
        DoF: degree of freedom used when fitting hte 
        """
        self._distribution = scipy.stats.chi2(DoF)

    def cdf(self, samples):
        return self._distribution.cdf(samples)

    def pdf(self, samples):
        return self._distribution.pdf(samples)        

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