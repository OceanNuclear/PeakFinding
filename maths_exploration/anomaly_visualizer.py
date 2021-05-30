from peakfinding.anomaly_detection import *
import seaborn as sns
from math import floor
import pandas as pd
import sys
from matplotlib import pyplot as plt

# from collections import defaultdict
REPLOT_XTICKS = False

WINDOW_WIDTHS = range(4, 40)
spectrum = pd.read_csv(sys.argv[1], index_col=[0]).values.T
E_l, E_u, counts = spectrum
counts = ary(counts, dtype=int)
E_bound = ary([E_l, E_u]).T

mid_E = E_bound.mean(axis=1)

results_canonical, results_self_cont, GoF_collection = [], [], {}
for w in WINDOW_WIDTHS:
    print(f"Checking for peak-iness using window size = {w}")
    GoF = SpectrumGoodnessOfFit(counts, w)

    GoF_collection[w] = GoF
    results_canonical.append(np.insert([np.nan,]*(w-1), floor(w/2), GoF.get_canonical_peakiness()))
    results_self_cont.append(GoF.get_self_contributed_peakiness())

fig, (ax_u, ax_m, ax_l) = plt.subplots(3, 1, sharex=True)

# upper plot
plot_sqrt(counts, ax=ax_u)
ax_u.set_xlabel("")
if REPLOT_XTICKS:
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

ax_l.set_xlabel(r"$E_\gamma$ (keV)")
plt.show()

# peakiness = np.nanmean(results_canonical, axis=0)
fig, (ax_u, ax_l) = plt.subplots(2, 1, sharex=True)
plot_sqrt(E_bound, counts, ax=ax_u)
ax_l.plot(E_bound.flatten(), np.repeat(np.nanmean(results_canonical, axis=0), 2), label="canonical mean")
ax_l.set_xlabel(r"$E_\gamma$ (keV)")
ax_l.set_ylabel("Probabilty of being NOT noise")
plt.show()