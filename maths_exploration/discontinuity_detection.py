import numpy as np
from numpy import pi, sqrt, exp, array as ary, log as ln
tau = 2*pi
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys, os
from scipy.ndimage import filters, gaussian_filter1d
from scipy import signal
from peakfinding.sqrt_repr import plot_sqrt # signature: (E, sqrt(counts), ax, rewrite_yticks)
from peakfinding.spectrum import RealSpectrum

if __name__=='__main__':
    """
    Failed idea:
    Using this script, we can see that the "taking derivative" or "smoothen, then take derivative" idea fails terribly
    """

    # from collections import defaultdict
    REPLOT_TICKS = False
    
    WINDOW_WIDTHS = range(4, 40)
    spectrum = RealSpectrum.from_multiple_files(*sys.argv[1:])
    E_bound, counts = spectrum.boundaries(), spectrum.counts

    mid_E = E_bound.mean(axis=1)

    # operate on the following block:
    # KERNEL = [0.125, 0.375, 0.375, 0.125]
    KERNEL = signal.ricker(10, 3)
    sqrt_counts = sqrt(counts)
    modified_counts = filters.convolve1d(sqrt_counts, KERNEL)
    # modified_counts = gaussian_filter1d(sqrt_counts, 4)

    discontinuity = np.diff(sqrt_counts)

    fig, (ax_u, ax_l) = plt.subplots(2, 1, sharex=True)
    plot_sqrt(E_bound, counts, ax=ax_u)
    ax_l.plot( ary([mid_E[:-1], mid_E[1:]]).T.flatten(), np.repeat(discontinuity,2) )
    ax_l.set_ylabel("discontinuity")
    plt.show()