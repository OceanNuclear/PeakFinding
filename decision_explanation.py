from peakfinding.curvature_detection import threshold_curve_function_generator, RealSpectrumPeakFinderFromNoiseFloor
from numpy import array as ary
import numpy as np
import inspect, sys
import matplotlib.pyplot as plt

OLD_DEMONSTRATION = False

if OLD_DEMONSTRATION:
    from peakfinding.curvature_detection import RealSpectrumPeakFinderDuplicatePeakiness
    spectrum = RealSpectrumPeakFinderDuplicatePeakiness.from_multiple_files(*sys.argv[1:])
else:
    spectrum = RealSpectrumPeakFinderFromNoiseFloor.from_multiple_files(*sys.argv[1:])
spectrum.show_sqrt_scale()
print("Type in the calibration coefficients (comma separated); Infer the calibration coefficients from the metadata ('infer'); or press enter to fit the peaks manually.")
values = input()
if values.strip()=="":
    spectrum.fit_fwhm_cal_interactively()
elif values=="infer":
    raise NotImplementedError("Do not recommend using infer anymore as the FWHM values set at the moment of data acquisition is >0.")
    fwhm_preferences = ("shape_cal", "fwhm_cal")
    for fwhm_attr in fwhm_preferences:
        if hasattr(spectrum, fwhm_attr):
            spectrum.fwhm_cal = getattr(spectrum, fwhm_attr)
            break
    else:
        raise NameError("File(s) have no corresponding metadata:\n"+str(fwhm_preferences))
else:
    spectrum.fwhm_cal = ary([float(i) for i in values.split(",")])

if OLD_DEMONSTRATION:
    # calculate the required values first.
    curvature_values = spectrum.calculate_sqrt_curvature()
    mean_peakiness = spectrum.apply_function_on_full_peakiness(np.nanmean)
    max_peakiness = spectrum.apply_function_on_full_peakiness(np.nanmax)
    true_peakiness3 = spectrum.get_peakiness(3.0).copy()
    del spectrum.peakiness
    true_peakiness425 = spectrum.get_peakiness(4.25).copy()
    peakiness = spectrum.get_peakiness()
    curvature_threshold, peakiness_threshold, _ = inspect.signature(spectrum.default_peak_identifier).parameters.values()

    # actual spectrum (ax_u)
    fig, (ax_u, ax_m, ax_l) = plt.subplots(3, 1, sharex=True)
    spectrum.plot_sqrt_scale(ax=ax_u)
    # spectrum.highlight_peaks(ax_u, spectrum.default_peak_identifier(), alpha=0.5)
    spectrum.highlight_peaks(ax_u, spectrum.faster_peak_identifier(), color="C2", alpha=0.5)
    # spectrum.highlight_peaks(ax_u, spectrum.default_peak_identifier(curvature_threshold=-1.5, peakiness_function=np.nanmax), color="C3", alpha=0.5)

    # curvature (ax_m)
    default_threshold_func = threshold_curve_function_generator(curvature_threshold.default, *spectrum.fwhm_cal)
    ax_m.plot(spectrum.boundaries().flatten(), default_threshold_func(spectrum.boundaries().flatten()), label="threshold 1")
    ax_m.plot(spectrum.boundaries().flatten(), np.repeat(curvature_values, 2))
    new_threshold_func = threshold_curve_function_generator(-1.5, *spectrum.fwhm_cal)
    ax_m.plot(spectrum.boundaries().flatten(), new_threshold_func(spectrum.boundaries().flatten()), label="threshold 2", color="C3")
    ax_m.legend()
    ax_m.set_ylim(-5, -0.1)
    ax_m.set_title("Curvatrue")

    # peakiness (ax_l)
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(peakiness_threshold.default, len(spectrum.boundaries().flatten())), label="threshold")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(mean_peakiness, 2), label="self-peakiness mean values")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(true_peakiness3, 2), label="true peakiness (3) values")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(true_peakiness425, 2), label="true peakiness (4.25) values")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(max_peakiness, 2), label="self-peakiness max values")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(0.6, len(spectrum.boundaries().flatten())), label="threshold for max")
    ax_l.set_title("Peakiness")
    ax_l.legend()

    plt.show()

else:
    fig, (ax_u, ax_m, ax_l) = plt.subplots(3, 1, sharex=True)
    fig.tight_layout()
    # calculate the required values first.
    curvature_values = spectrum.calculate_sqrt_curvature()
    peakiness = spectrum.get_peakiness()
    curvature_threshold, peakiness_threshold, _ = inspect.signature(spectrum.peak_identifier).parameters.values()

    # actual spectrum plot (ax_u)
    spectrum.plot_sqrt_scale(ax=ax_u)
    spectrum.determine_noise_floor()
    ax_u.plot(spectrum.boundaries().flatten(), np.repeat(np.sqrt(spectrum.raw_noise_floor), 2), label="raw_noise_floor")
    ax_u.plot(spectrum.boundaries().flatten(), np.repeat(np.sqrt(spectrum.noise_floor), 2), label="noise_floor")
    spectrum.highlight_peaks(ax_u, spectrum.peak_identifier(), alpha=0.6)
    ax_u.legend()
    # spectrum.highlight_peaks(ax_u, spectrum.faster_peak_identifier(), color="C2", alpha=0.5)
    # spectrum.highlight_peaks(ax_u, spectrum.default_peak_identifier(curvature_threshold=-1.5, peakiness_function=np.nanmax), color="C3", alpha=0.5)

    # curvature plot (ax_m)
    default_threshold_func = threshold_curve_function_generator(curvature_threshold.default, *spectrum.fwhm_cal)
    ax_m.plot(spectrum.boundaries().flatten(), default_threshold_func(spectrum.boundaries().flatten()), label="default_threshold_func")
    ax_m.plot(spectrum.boundaries().flatten(), np.repeat(curvature_values, 2))
    ax_m.legend()
    ax_m.set_ylim(-5, -0.1)
    ax_m.set_title("Curvatrue")

    # peakiness plot (ax_l)
    exclusive_peakiness_default = spectrum.peakiness_from_bg_noise.copy()
    del spectrum.peakiness_from_bg_noise

    spectrum.infer_peakiness_from_background(negative_deviation_dilution_factor=0.5)
    exclusive_peakiness_05 = spectrum.peakiness_from_bg_noise.copy()
    del spectrum.peakiness_from_bg_noise

    spectrum.infer_peakiness_from_background(negative_deviation_dilution_factor=0)
    exclusive_peakiness_0 = spectrum.peakiness_from_bg_noise.copy()
    del spectrum.peakiness_from_bg_noise

    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(spectrum.peakiness, 2), label="probability of being not noise")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(exclusive_peakiness_default, 2), label="exclusive_peakiness_default")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(exclusive_peakiness_05, 2), label="exclusive_peakiness_suppressed_05")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(exclusive_peakiness_0, 2), label="exclusive_peakiness_suppressed_0")
    # if hasattr(spectrum, "peakiness"): del spectrum.peakiness
    # true_peakiness2 = spectrum.get_peakiness(2.0).copy()
    # ax_l.plot(spectrum.boundaries().flatten(), np.repeat(true_peakiness2, 2), label="true peakiness (3) values")

    # if hasattr(spectrum, "peakiness"): del spectrum.peakiness
    # true_peakiness3 = spectrum.get_peakiness(3.0).copy()
    # ax_l.plot(spectrum.boundaries().flatten(), np.repeat(true_peakiness3, 2), label="true peakiness (3) values")

    # if hasattr(spectrum, "peakiness"): del spectrum.peakiness
    # true_peakiness425 = spectrum.get_peakiness(4.25).copy()
    # ax_l.plot(spectrum.boundaries().flatten(), np.repeat(true_peakiness425, 2), label="true peakiness (4.25) values")

    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(0.5, len(spectrum.boundaries().flatten())), label="threshold = 0.5")
    ax_l.plot(spectrum.boundaries().flatten(), np.repeat(peakiness_threshold.default, len(spectrum.boundaries().flatten())), label="default peakiness threshold relative to the background.")
    ax_l.set_title("Peakiness")
    ax_l.legend()

    plt.show()
