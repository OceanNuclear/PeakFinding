from peakfinding.curvature_detection import *
import sys
import matplotlib.pyplot as plt

DEMONSTRATE_DECISION_MAKING = True

spectrum = RealSpectrumPeakFinder.from_multiple_files(*sys.argv[1:])
spectrum.show_sqrt_scale()
print("Type in the calibration coefficients (comma separated) ; or press enter to fit the peaks manually.")
values = input()
if values.strip()=="":
    spectrum.add_fwhm_cal_interactively()
else:
    spectrum.fwhm_cal = ary([float(i) for i in values.split(",")])

# spectrum.fwhm_cal = ary([1.70101954, 0.02742255])

import inspect
if DEMONSTRATE_DECISION_MAKING:
    # calculate the required values first.
    curvature_coef = spectrum.calculate_sqrt_curvature()
    mean_peakiness = spectrum.apply_function_on_peakiness(np.nanmean)
    max_peakiness = spectrum.apply_function_on_peakiness(np.nanmax)
    peakiness = spectrum.get_peakiness()
    curvature_threshold, peakiness_threshold, _ = inspect.signature(spectrum.default_peak_identifier).parameters.values()

    # actual spectrum (ax_u)
    fig, (ax_u, ax_m, ax_l) = plt.subplots(3, 1, sharex=True)
    spectrum.plot_sqrt_scale(ax=ax_u)
    spectrum.highlight_peaks(ax_u, spectrum.default_peak_identifier(), alpha=0.5)
    spectrum.highlight_peaks(ax_u, spectrum.faster_peak_identifier(), color="C2", alpha=0.5)
    spectrum.highlight_peaks(ax_u, spectrum.default_peak_identifier(curvature_threshold=-2.1, peakiness_function=np.nanmax), color="C3", alpha=0.5)

    # curvature (ax_m)
    threshold_func = threshold_curve_function_generator(curvature_threshold.default, *spectrum.fwhm_cal)
    ax_m.plot(spectrum.boundaries.flatten(), threshold_func(spectrum.boundaries.flatten()), label="threshold 1")
    ax_m.plot(spectrum.boundaries.flatten(), np.repeat(curvature_coef, 2))
    threshold_func = threshold_curve_function_generator(-2.1, *spectrum.fwhm_cal)
    ax_m.plot(spectrum.boundaries.flatten(), threshold_func(spectrum.boundaries.flatten()), label="threshold 2", color="C3")
    ax_m.legend()
    ax_m.set_ylim(-5, -0.1)
    ax_m.set_title("Curvatrue")

    # peakiness (ax_l)
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(peakiness_threshold.default, len(spectrum.boundaries.flatten())), label="threshold")
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(mean_peakiness, 2), label="self-peakiness mean values")
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(peakiness, 2), label="true peakiness values")
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(max_peakiness, 2), label="self-peakiness max values")
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(0.6, len(spectrum.boundaries.flatten())), label="threshold for max")
    ax_l.set_title("Peakiness")
    ax_l.legend()

    plt.show()

else:
    ax, line = spectrum.plot_sqrt_scale()
    peak_indices = spectrum.default_peak_identifier()
    spectrum.highlight_peaks(ax, peak_indices)
    plt.show()