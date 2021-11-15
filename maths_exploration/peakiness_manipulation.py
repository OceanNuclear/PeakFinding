from peakfinding.curvature_detection import *
import sys
import matplotlib.pyplot as plt

spectrum = RealSpectrumPeakFinder.from_multiple_files(*sys.argv[1:])
spectrum.show_sqrt_scale()
print("Type in the calibration coefficients (comma separated) ; or press enter to fit the peaks manually.")
values = input()
if values.strip()=="":
    spectrum.fit_fwhm_cal_interactively()
else:
    spectrum.fwhm_cal = ary([float(i) for i in values.split(",")])

# spectrum.fwhm_cal = ary([1.70101954, 0.02742255])

fig, (ax_u, ax_l) = plt.subplots(2, 1, sharex=True)
spectrum.plot_sqrt_scale(ax=ax_u)
# ax_l.plot(spectrum.boundaries.flatten(), np.repeat(spectrum.apply_function_on_peakiness(np.nanmin) , 2), label="min")
ax_l.plot(spectrum.boundaries.flatten(), np.repeat(spectrum.apply_function_on_peakiness(np.nanmean), 2), label="mean")
ax_l.plot(spectrum.boundaries.flatten(), np.repeat(spectrum.get_peakiness(), 2), label="self")
# thresholds
# ax_l.plot([spectrum.boundaries.min(), spectrum.boundaries.max()], np.repeat(0.95, 2), lw=0.5)
ax_l.plot([spectrum.boundaries.min(), spectrum.boundaries.max()], np.repeat(0.9, 2), lw=0.5)
ax_l.plot([spectrum.boundaries.min(), spectrum.boundaries.max()], np.repeat(0.8, 2), lw=0.5)
ax_l.plot([spectrum.boundaries.min(), spectrum.boundaries.max()], np.repeat(0.6, 2), lw=0.5)
ax_l.plot([spectrum.boundaries.min(), spectrum.boundaries.max()], np.repeat(0.5, 2), lw=0.5)
ax_l.set_title("Possible thresholds above which it should be considered as peak")
ax_l.legend()

fig, (ax_u2, ax_l2) = plt.subplots(2, 1, sharex=True)
spectrum.plot_sqrt_scale(ax=ax_u2)
ax_l2.plot(spectrum.boundaries.flatten(), np.repeat(spectrum.apply_function_on_peakiness(np.nanmean), 2), label="mean")
ax_l2.plot(spectrum.boundaries.flatten(), np.repeat(spectrum.apply_function_on_peakiness(np.nanmax) , 2), label="max")
ax_l2.plot(spectrum.boundaries.flatten(), np.repeat(spectrum.get_peakiness(), 2), label="self")
ax_l2.plot([spectrum.boundaries.min(), spectrum.boundaries.max()], np.repeat(0.5, 2), lw=0.5)
ax_l2.plot([spectrum.boundaries.min(), spectrum.boundaries.max()], np.repeat(0.3, 2), lw=0.5)
ax_l2.set_title("Possible threshold below which it should be considered as noise")
plt.legend()
plt.show()

"""
Conclusion:
Use nanmean>=0.9 or nanmin>=0.9 for peaks discovery;
Use nanmax<=0.8 for noise floor determination.
"""
"""
Current plan:
1. Use nanmin>=0.9
2. Use nanmean>=0.9
3. Use nanmin>=0.8
4. Use nanmean>=0.8

5. Use nanmean>=0.9 for more lenience?
"""