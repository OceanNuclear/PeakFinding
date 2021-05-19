from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys, os
from tqdm import tqdm

from sqrt_repr import plot_sqrt # signature: (E, sqrt(counts), ax, rewrite_yticks)
from poisson_distribution import Poisson, Normal, Chi2

class SpectrumGoodnessOfFit():
    def __init__(self, counts, window_width):
        """
        Calculates the contribution to the Poisson chi^2 when a window size of window_width is rolled across the counts array.

        Parameters
        ----------
        counts: numpy 1d array of counts in each channel.
        window_width: integer denoting the width of the window used.

        Returns
        -------
        poisson_goodness_of_fit_full_stack : array of shape = (len(counts)-window_width+1, window_width)
        """
        self.window_width = window_width
        self.chi2_cdf_converter = Chi2(window_width-1)
        
        poisson_goodness_of_fit_full_stack = []

        zip_object = []
        for i in range(window_width):
            end_index = -(window_width-1)+i
            if end_index==0:
                end_index = None
            zip_object.append(counts[i:end_index])
        spread_stack = ary(list(zip(*zip_object)))
        mean_stack = spread_stack.mean(axis=1)

        for num_window, (samples, sample_mean) in tqdm(enumerate(zip(spread_stack, mean_stack)),
                                                    total=len(counts)-window_width+1):
            poisson_hypothesized = Poisson(sample_mean)
            # calculate the contribution to the poisson chi2 equivalent quantity.
            poisson_chi2_components = poisson_hypothesized.negative_log_likelihood(samples)
            poisson_goodness_of_fit_full_stack.append(poisson_chi2_components)
        self._goodness_of_fit_values = ary(poisson_goodness_of_fit_full_stack).T

    def get_canonical_peakiness(self):
        """
        Calculates the probability that the bin is NOT a background noise sample
            among window_width neighbouring samples.
        This is the PROPER (canonical) way to calculate the peakiness.

        Returns
        -------
        self_peakiness: array of len = len(counts)-window_width+1
        """
        goodness_of_fit_sum = self._goodness_of_fit_values.sum(axis=0)
        return self.chi2_cdf_converter.cdf(goodness_of_fit_sum)

    def get_self_contributed_peakiness(self):
        """
        Experimental/beta way of calculating the probability of this being a peak itself. This seems to exaggerate the number quite a lot and allows for a

        Visual explanation of what the code is doing:
        
        for a window_width = w, (_poisson_goodness_of_fit_raw_values).T gives an array as follows:
        [
            [GoF(0:w  , 0), GoF(1:w+1, 1), ...]
            [GoF(0:w  , 1), GoF(1:w+1, 2), ...]
            ...
        ]
        where
            GoF = Goodness of fit = chi2 contribution;
            GoF(i:j, k) where i<=k<=j gives the goodness of fit value caluclated within window = count[i:j] for the k-th bin.

        Therefore in this function, we slide the first line down forward by 1, second line down forward by 2, etc:
        [
            [GoF(0:w  , 0), GoF(1:w+1, 1), GoF(2:w+2, 2)]
            [               GoF(0:w  , 1), GoF(1:w+1, 2), ...]
            ...
        ]
        
        and then vertically sum the w elements up in each column, so that that we can get the GoF(i:w+i, k) for the k-th bin.
        To handle the edge cases (leftmost w-1 and rightmost w-1), we have two approaches:
        1. Upscale the limited samples from <w to w. (by taking the aveage and then multiplying by w.)
        2. Use only the samples given, test it on the chi^2 with a reduced number of DoF (< w-1).
        I won't be using 2 because the GoF was originally generated from a distribution of DoF =w-1,
            and I don't think this new method of arrangement will change the DoF somehow.
            I'll use 1 instead.

        Returns
        -------
        a full sized array that matches shape with that of count
        """
        nan_rectangle = np.full((self.window_width, self.window_width-1), np.nan) # shape = (window_width, window_width-1)
        full_sized_goodness_of_fit = ary([np.insert(nans, ind, GoF_line) for ind, (nans, GoF_line) in enumerate(zip(nan_rectangle, self._goodness_of_fit_values))])

        goodness_of_fit_sum = np.nanmean(full_sized_goodness_of_fit, axis=0) * self.window_width
        return self.chi2_cdf_converter.cdf((goodness_of_fit_sum))

if __name__=='__main__':
    import seaborn as sns
    from math import floor
    # from collections import defaultdict
    REPLOT_TICKS = False
    
    WINDOW_WIDTHS = range(4, 40)
    spectrum = pd.read_csv(sys.argv[1], index_col=[0]).values.T
    E_l, E_u, counts = spectrum
    counts = ary(counts, dtype=int)
    E_bound = ary([E_l, E_u]).T

    mid_E = E_bound.mean(axis=1)

    results_canonical, results_self_cont = [], []
    for w in WINDOW_WIDTHS:
        print(f"Checking for peak-iness using window size = {w}")
        GoF = SpectrumGoodnessOfFit(counts, w)
        GoF.get_self_contributed_peakiness()

        results_canonical.append(np.insert([np.nan,]*w, floor(w/2), GoF.get_canonical_peakiness()))
        results_self_cont.append(GoF.get_self_contributed_peakiness())

    # fig, ((ax_u, _), (ax_l, cbar_ax)) = plt.subplots(2, 2, sharex=True, gridspec_kw={"width_ratios": (.9, .05), "wspace": .3})
    fig, (ax_u, ax_m, ax_l) = plt.subplots(3, 1, sharex=True)

    # upper plot
    plot_sqrt(counts, ax=ax_u)
    if REPLOT_TICKS:
        old_ticks = ax_u.get_xticks()
        # fix the tick problem (as we can't plot the with the ticks)
        new_ticks = []
        for x in old_ticks:
            new_ticks.append(str(mid_E[np.clip(int(x), 0, len(mid_E)-1)]))
        ax_u.set_xticklabels(new_ticks)
        ax_u.set_xlabel(r"$E_\gamma$ (eV)")

    # lower plot
    sns.heatmap(results_self_cont, yticklabels=WINDOW_WIDTHS, ax=ax_m, cbar=False, vmin=0.0, vmax=1.0)
    ax_m.set_title("Likelihood of being a peak\nby self-contribution method")
    ax_m.set_ylabel("Size of hypothesis window")

    sns.heatmap(results_canonical, yticklabels=WINDOW_WIDTHS, ax=ax_l, cbar=False, vmin=0.0, vmax=1.0)
    ax_l.set_title("Likelihood of being a peak\nby the canonical method")
    ax_l.set_ylabel("Size of hypothesis window")

    plt.show()