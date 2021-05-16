from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import numpy as np

from poisson_distribution import Poisson

if __name__=='__main__':
    WINDOW_WIDTH = 4
    spectrum = pd.read_csv(sys.argv[1], index_col=[0]).values.T
    E_l, E_u, counts = spectrum
    for num_window, window in enumerate( zip(*[counts[i:-WINDOW_WIDTH+i] for i in range(WINDOW_WIDTH)]) ):
        sample_mean = np.longdouble(np.mean(window))
        max_N = poisson_likelihood(sample_mean)
        print(f"{num_window=}, {sample_mean=}, {max_N=}")
        poisson_hypothesized = poisson(sample_mean)