from peakfinding.spectrum import RealSpectrumInteractive
import sys
import matplotlib.pyplot as plt
import os

import matplotlib.animation as manimation

fig, ax = plt.subplots()
def animate(i):
    ax.clear()
    spectrum = RealSpectrumInteractive.from_multiple_files(sys.argv[i+1])
    fig.suptitle(os.path.basename(sys.argv[i+1]))
    if hasattr(spectrum, "live_time") and hasattr(spectrum, "wall_time"):
        ax.set_title("live = {}s, real = {}s".format(spectrum.live_time, spectrum.wall_time),)
    spectrum.plot_sqrt_scale(ax=ax, rewrite_yticks=True)
    return ax

anim = manimation.FuncAnimation(fig, animate, len(sys.argv)-1, interval=500)
plt.show()
