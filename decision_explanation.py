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
    peakiness = spectrum.get_peakiness()
    peak_indices = spectrum.default_peak_identifier()
    curvature_threshold, peakiness_threshold = inspect.signature(spectrum.default_peak_identifier).parameters.values()


    # actual spectrum (ax_u)
    fig, (ax_u, ax_m, ax_l) = plt.subplots(3, 1, sharex=True)
    spectrum.plot_sqrt_scale(ax=ax_u)
    spectrum.plot_identified_peaks(ax_u, peak_indices)

    # curvature (ax_m)
    ax_m.plot(spectrum.boundaries.flatten(), np.repeat(curvature_coef, 2))
    threshold_func = threshold_curve_function_generator(curvature_threshold.default, *spectrum.fwhm_cal)
    ax_m.plot(spectrum.boundaries.flatten(), threshold_func(spectrum.boundaries.flatten()) )
    ax_m.set_ylim(-5, -0.1)
    ax_m.set_title("Curvatrue")

    # peakiness (ax_l)
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(peakiness, 2))
    ax_l.plot(spectrum.boundaries.flatten(), np.repeat(peakiness_threshold.default, len(spectrum.boundaries.flatten())))
    ax_l.set_title("Peakiness")

    plt.show()

else:
    ax, line = spectrum.plot_sqrt_scale()
    peak_indices = spectrum.default_peak_identifier()
    spectrum.plot_identified_peaks(ax, peak_indices)

    plt.show()