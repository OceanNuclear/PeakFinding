from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys, os
from tqdm import tqdm
from collections import defaultdict

from sqrt_repr import plot_sqrt # signature: (E, sqrt(counts), ax, rewrite_yticks)
from poisson_distribution import Poisson, Normal, Chi2

if __name__=='__main__':
    import seaborn as sns
    WINDOW_WIDTHS = range(4, 40)
    spectrum = pd.read_csv(sys.argv[1], index_col=[0]).values.T
    E_l, E_u, counts = spectrum
    counts = ary(counts, dtype=int)
    E_bound = ary([E_l, E_u]).T

    poisson_goodness_of_fit, GoF_x = defaultdict(list), defaultdict(list)
    mid_E = E_bound.mean(axis=1)

    for w in WINDOW_WIDTHS:
        print(f"Checking for peak-iness using window size = {w}")
        chi2_cdf_converter = Chi2(w) # all channels captured by the window is hypothesized to be identical Poisson distributions

        nan_blank = [np.nan,]*(w//2)
        poisson_goodness_of_fit[w].extend(nan_blank)
        for num_window, samples in tqdm(enumerate(zip(*[counts[i:-w+i] for i in range(w)])), total=len(counts)-w):
            sample_mean = np.mean(samples)
            poisson_hypothesized = Poisson(sample_mean)
            gaussian_approximated = Normal(sample_mean, sample_mean)

            poisson_goodness_of_fit[w].append(poisson_hypothesized.negative_log_likelihood(samples).sum())
            # GoF_x[w].append( np.mean(mid_E[num_window:num_window+w]) ) # broken/ incompatible with the current method.
        poisson_goodness_of_fit[w].extend(nan_blank)
        if (w%2)==1:
            poisson_goodness_of_fit[w].append(np.nan)

    # fig, ((ax_u, _), (ax_l, cbar_ax)) = plt.subplots(2, 2, sharex=True, gridspec_kw={"width_ratios": (.9, .05), "wspace": .3})
    fig, (ax_u, ax_l) = plt.subplots(2, 1, sharex=True)

    # upper plot
    plot_sqrt(counts, ax=ax_u)
    old_ticks = ax_u.get_xticks()
    new_ticks = []
    for x in old_ticks:
        new_ticks.append(str(mid_E[np.clip(int(x), 0, len(mid_E)-1)]))
    ax_u.set_xticklabels(new_ticks)
    ax_u.set_xlabel(r"$E_\gamma$ (eV)")

    # middle plot
    # for w in WINDOW_WIDTHS:
    #   ax_m.plot(GoF_x[w], poisson_goodness_of_fit[w], label="Poisson goodness of fit")
    # ax_m.legend()

    # lower plot
    peakiness = ary([chi2_cdf_converter.cdf(poisson_goodness_of_fit[w]) for w in WINDOW_WIDTHS])
    sns.heatmap(peakiness, yticklabels=WINDOW_WIDTHS, ax=ax_l, cbar=False, vmin=0.0, vmax=1.0)
    ax_l.set_title("Likelihood of being a peak")
    ax_l.set_ylabel("Size of checking window")
    plt.show()