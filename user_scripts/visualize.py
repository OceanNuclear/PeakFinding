from peakfinding.spectrum import RealSpectrumInteractive, RealSpectrum
import sys
import matplotlib.pyplot as plt

print("plotting a sum of", *sys.argv[1:])
spectrum = RealSpectrumInteractive.from_multiple_files(*sys.argv[1:])

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
spectrum.show_sqrt_scale(ax=ax1)
fig.suptitle("Gamma spectrum data")
ax1.set_title("sqrt-scale")
spectrum.show_log_scale(ax=ax2)
ax2.set_title("log-scale")
spectrum.show_log_scale(ax=ax3)
ax3.set_yscale("linear")
ax3.set_title("linear-scale")
ax1.set_xlabel("E (keV)")
plt.show()