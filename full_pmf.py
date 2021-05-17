from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
from sqrt_repr import plot_sqrt
import pandas as pd
import sys, os
from scipy.special import factorial as fac

w = WINDOW_WIDTH = 4

# _log = np.vectorize(lambda x: ln(x) if x>0.0 else 0.0)

"""
To avoid over flow I stopped using cumprod and used this instead.
"""
def poisson_unknown_max_N(lamb, max_N):
    P_N = [exp(-lamb),]
    P_N.extend([ chainmul(*(exp(-lamb/N) * lamb/np.arange(1, N+1))) for N in range(1, max_N+1) ]) # this is a nested for-loop and may take a while.
    return ary(P_N)

def quantify_max_N_given_lambda(lamb):
    max_N = 10
    while not np.isclose(poisson_unknown_max_N(lamb, max_N).sum(), 1.0, rtol=0, atol=1E-8):
        print(poisson_unknown_max_N(lamb, max_N).sum(), max_N)
        max_N *= 10
    return max_N

def poisson(lamb):
    max_N = quantify_max_N_given_lambda(lamb)
    P_N = poisson_unknown_max_N(lamb, max_N)
    return P_N

def cross_entropy(samples, lamb):
    """
    calculates the cross-entropy of that from a poisson distribution
    """
    max_N = quantify_max_N_given_lambda()
    P_N = poisson(lamb, max_N)
    P_sam = ary([(sample==i).sum() for i in range(max_N)])

    return P_sam @ _log(P_N)

def self_entropy(lamb):
    """
    calculates the self-entropy expected from a Poisson distribution with mean = lamb
    """
    P_N = poisson(lamb)
    return P_N @ _log(P_N)

relative_entropy = lambda samples, lamb: cross_entropy(samples, lamb) - self_entropy(lamb)

if __name__=='__main__':
    spectrum = pd.read_csv(sys.argv[1], index_col=[0]).values.T
    E_l, E_u, counts = spectrum
    for num_window, window in enumerate( zip(*[counts[i:-w+i] for i in range(WINDOW_WIDTH)]) ):
        sample_mean = np.longdouble(np.mean(window))
        # max_N = quantify_max_N_given_lambda(sample_mean)
        print(f"{num_window=}, {sample_mean=}, {max_N=}")
        poisson_hypothesized = poisson(sample_mean)
