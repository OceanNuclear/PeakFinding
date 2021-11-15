from peakfinding.curvature_detection import RealSpectrumPeakFinder, threshold_curve_function_generator
import pandas as pd
from numpy import array as ary
import sys
import matplotlib.pyplot as plt
import numpy as np

assert not sys.argv[-1].endswith(".Spe"), "Must provide a file name (that doesn't include '.csv') as the output location to save the .csv to."
spectrum = RealSpectrumPeakFinder.from_multiple_files(*sys.argv[1:-1])
spectrum.show_sqrt_scale()
print("Type in the calibration coefficients (comma separated) ; or press enter to fit the peaks manually.")
values = input()
spectrum.fwhm_cal = ary([1.5853449048830446, 0.0020776090022188573])
# if values.strip()=="":
#     spectrum.fit_fwhm_cal_interactively()
# else:
#     spectrum.fwhm_cal = ary([float(i) for i in values.split(",")])


DEMONSTRATE_DECISION_MAKING = True
if DEMONSTRATE_DECISION_MAKING:
    # calculate the required values first.
    curvature_coef = spectrum.calculate_sqrt_curvature()
    self_peakiness = spectrum.get_peakiness()
    peakiness = spectrum.apply_function_on_peakiness(np.nanmean)
    peak_indices = spectrum.default_peak_identifier()

    import inspect
    curvature_threshold, peakiness_threshold = inspect.signature(spectrum.default_peak_identifier).parameters.values()

    # actual spectrum (ax_u)
    fig, (ax_u, ax_m, ax_l) = plt.subplots(3, 1, sharex=True)
    spectrum.plot_sqrt_scale(ax=ax_u)
    spectrum.highlight_peaks(ax_u, peak_indices)

    # curvature (ax_m)
    ax_m.plot(spectrum.boundaries.flatten(), np.repeat(curvature_coef, 2))
    threshold_func = threshold_curve_function_generator(curvature_threshold.default, *spectrum.fwhm_cal)
    ax_m.plot(spectrum.boundaries.flatten(), threshold_func(spectrum.boundaries.flatten()) )
    ax_m.set_ylim(-5, -0.05)
    ax_m.set_title("Curvatrue")

    # peakiness (ax_l)
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(peakiness, 2))
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(peakiness_threshold.default, spectrum.boundaries.flatten().shape))
    ax_l.set_title("Peakiness")

    plt.show()

peak_indices = spectrum.default_peak_identifier()
l_edge, r_edge = spectrum.boundaries[peak_indices][:, 0, 0], spectrum.boundaries[peak_indices][:, 1, 1]
mean_bin_index = peak_indices.mean(axis=1).astype(int)
counts = [spectrum.counts[l_ind:r_ind].sum() for l_ind, r_ind in peak_indices]

df = pd.DataFrame(ary([l_edge, r_edge, counts]).T, index=mean_bin_index, columns=["left", "right", "counts included"])

ax, line = spectrum.plot_sqrt_scale()
spectrum.highlight_peaks(ax, peak_indices)
plt.show()

while True:
    answer = input("Acceptable? (y/n) (y=save to file {})\n".format(sys.argv[-1])).lower()
    if answer=="y":
        df.to_csv(sys.argv[-1], index_label="bin")
        break
    elif answer=="n":
        pass
        break