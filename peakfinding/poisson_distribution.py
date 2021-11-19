import numpy as np
from numpy import pi, sqrt, array as ary, log as ln
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

    def less_naive_negative_log_likelihood(self, samples):
        if self._lambda==0:
            return 0.0
        return self.naive_NLL(samples) - 0.5*ln(tau * self._lambda)

class PoissonFast():
    def __init__(self, lamb):
        self._distribution = 'Poisson'
        self._lambda = lamb

    def negative_log_likelihood(self, samples):
        """
        Direct formula that calculates the negative log likelihood
        The samples provided must be integers

        Parameters
        ----------
        samples: array of int! Otherwise it will throw erros

        Returns
        -------
        NLL_list: array of the same shape
        """
        if self._lambda==0:
            return np.zeros(np.shape(samples))
        NLL_list = []
        const_offset = -0.5 * ln(tau) + self._lambda -0.5*ln(self._lambda)
        for N in samples:
            if np.isnan(N): # nan handling
                nll = np.nan
            else:
                factorial_part = ln(np.arange(1, N+1))
                power_part = -ln(self._lambda)
                overflow_likely_part = factorial_part + power_part

                # perform a "folding" operation on the array, where the first element is added to the last, second to the second-last, etc.
                # this should yield an array of half the length if even N, half-length +1 if odd N..
                N_2 = int(N//2) #floor div of an integer should be an int
                first_part, second_part = overflow_likely_part[:N_2], overflow_likely_part[-N_2:]
                # ******* ^ IF THIS LINE FAILS THAT MEANS YOU FAILED TO PASS IN INT AS SAMPLES. YOU MAY HAVE USED FLOATS
                overflowy_part_folded_in_half = first_part + second_part[::-1]
                if (N%2)==1: # take care of odd N case
                    overflowy_part_folded_in_half = np.append(overflowy_part_folded_in_half, overflow_likely_part[N_2]) # append in the exact half-way point.

                nll = const_offset + fsum(overflowy_part_folded_in_half) # the last two terms will partially cancel out each other. so they must be added together first.

            NLL_list.append(nll)
        return ary(NLL_list)

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

class Chi2List(list):
    def __getitem__(self, N):
        while N>=len(self):
            self.append(Chi2( len(self) ))
        if N<0:
            N = 0
        return super().__getitem__(N)