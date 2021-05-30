from numpy import array as ary, log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos
tau = 2*pi
import numpy as np
from matplotlib import pyplot as plt
from peakfinding.poisson_distribution import Poisson, Normal
"""
This module is written about for comparing the
- negative log likelihood of the normal distribution (chi^2)
against
- negative log likelihood of the Poisson distribution

A video will be made to demonstrate the difference between the whether using the normal distribution likelihood function instead of the Poisson distribution function likelihood to calculate the NLE
"""

EXPLICIT_COMPARISON = False

if __name__=='__main__':
    import matplotlib.animation as manimation
    if EXPLICIT_COMPARISON:
        video_file_title = "negative log likelihood difference"
    else:
        video_file_title = "negative log likelihood"
    writer = manimation.writers['ffmpeg'](fps=15, metadata={"title":video_file_title, "comment":"from lambda = 0.1 to 100", "artist":"Ocean Wong"})
    grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
    fig, ax = plt.subplots()

    N = np.arange(1000)
    y = sqrt(N)
    bar_widths = np.diff(np.append(y, y[-1]))

    with writer.saving(fig, video_file_title.replace(" ", "_")+".mp4", 300):
        for lamb in np.logspace(0.1, 3):
            pois = Poisson(lamb)
            norm = Normal(lamb, lamb)

            log_unlog_nll = pois.less_naive_negative_log_likelihood(N)
            manual_nll = pois.negative_log_likelihood(N)
            gaussian_approximation = norm.negative_log_likelihood(N)

            if not EXPLICIT_COMPARISON:
                # simply display all of them together.
                ax.plot(y, log_unlog_nll, linewidth=0.8, alpha=0.5, linestyle="--", label="naively calculated")
                ax.plot(y, manual_nll, linewidth=0.8, alpha=0.5, linestyle="-.", label="sum of logs")
                ax.plot(y, gaussian_approximation, linewidth=0.8, alpha=0.5, linestyle=":", label="gaussian approximation")
            else:
                ax.plot(y, gaussian_approximation - manual_nll, label="approx. deviation")
                ax.set_ylim(-1, 1)

            fig.suptitle("Negative log likelihood functions comparison\n" + f"Poisson mean = lambda = {lamb}")
            print(f"{lamb=}")
            ax.set_xlabel("sqrt(counts)")
            ax.set_ylabel("NLL")
            ax.legend()

            writer.grab_frame()
            ax.clear()
    fig.clf()