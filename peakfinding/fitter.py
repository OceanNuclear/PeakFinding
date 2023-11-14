import numpy as np
from scipy.optimize import curve_fit

# TODO: Write a class that encapulates the functionality of /home/ocean/Documents/PhD/ExperimentalVerification/analysis/get_peaks.py


def fit_peaks(x_values, y_values, guess_peak_centroids, guess_sigma_values):
    """
    Parameters
    ----------
    x_values : the mean energy of each bin within the ROI.
    y_values : the count within each bin within the ROI.
    guess_peak_centroids : a list of N expected peak centroid locations, with length = N = num_peaks.
    fixed_sigma_values : a list of expected sigma values of each peak, with length = N = num_peaks.
    """
    num_peaks = len(guess_peak_centroids)
    lower_limits, upper_limits = [], []  # list to contain the expected minium and maximul values.
    guess_peak_params = []  # the guess parameters for each peak (amplitude and centroid location)

    # guess the background
    slope, level = np.polyfit([x_values[0], x_values[-1]], [y_values[0], y_values[1]], 1)
    total_counts = np.sum(y_values)
    num_channels = len(x_values)
    x_span = max(x_values) - min(x_values)
    # when packed as a triangle,
    # the max height channel has counts = total_area/num_channels * 2
    # the min height channel has counts = 0
    max_cnt_chn_cnt = 2 * total_counts / num_channels
    # the limits for bg_level
    lower_limits.append(-max_cnt_chn_cnt * (max(x_values) / x_span) + 2 * total_counts / num_channels)
    upper_limits.append(max_cnt_chn_cnt * (max(x_values) / x_span))
    # the limits for bg_slope
    lower_limits.append(-max_cnt_chn_cnt / x_span)  # the limits for slope
    upper_limits.append(max_cnt_chn_cnt / x_span)

    # use the prexisting calibration equation to find the expected widths for all of these peaks.
    for i, (centroid, sigma) in enumerate(zip(guess_peak_centroids, guess_sigma_values)):
        guess_peak_params.append(total_counts / num_peaks)  # guess amplitude
        lower_limits.append(0.0)
        upper_limits.append(total_counts / 0.8)  # some peaks might not be fully included in the ROI;
        # so the counts within ROI may only include as low as 80% of the counts.
        guess_peak_params.append(centroid)
        lower_limits.append((centroid + guess_peak_centroids[i - 1]) / 2 if (i - 1) >= 0 else min(x_values))
        upper_limits.append((centroid + guess_peak_centroids[i + 1]) / 2 if (i + 1) < num_peaks else max(x_values))
        guess_peak_params.append(sigma)
        lower_limits.append(min(guess_sigma_values) / 2)
        upper_limits.append(
            max(guess_sigma_values) * 1.5 * 2
        )  # account for unexpectedly wide peaks this way, e.g. the 511 keV peak.

    popt, pcov = curve_fit(
        multiple_gaussians_with_bg,
        x_values,
        y_values,
        sigma=np.clip(np.sqrt(y_values), 0.5, None),
        # Note: This sigma refers to "how much the fit is allowed to vertically deviate from the data",
        # and is different from the sigma used everywhere else in this program to describe the width of the peaks.
        # The np.clip is used to make sure it doesn't reach 0, since sigma->0 leads to infinite chi2 curvature which cannot be optimized.
        p0=[level, slope, *guess_peak_params],
        bounds=(lower_limits, upper_limits),
        method="trf",  # default optimization algorithm for bounded problem. Can change to 'dogbox' to see how it performs.
    )
    return popt, pcov


def multiple_gaussians_with_bg(x, bg_level, bg_slope, *peak_params):
    """
    Parameters
    ----------
    x : the mean energy of each bin to be fitted.
    bg_level: the offset of the background
    bg_slope : the gradient
    peak_params : a vector whose length is a multiple of 3.
            element 0, 3, 6, 9... describes the amplitude of peak 1, 2, 3, 4...
            element 1, 4, 7, 10... describes the amplitude of peak 1, 2, 3, 4...
            element 2, 5, 8, 11... describes the amplitude of peak 1, 2, 3, 4...
            Therefore peak_params = [amp1, cent1, sigma1, amp2, cent2, sigma2, ....]
    Return
    ------
    A sigma.
    """
    assert len(peak_params) % 3 == 0, "Each peak must be described by exactly 3 parameters: amplitude, centroid, sigma."
    fitted_curve = bg_level + bg_slope * x

    for amplitude, centroid, sigma in zip(peak_params[::3], peak_params[1::3], peak_params[2::3]):
        fitted_curve += amplitude * np.exp(-((x - centroid) ** 2) / (2 * sigma**2))
    return fitted_curve


def fit_peaks_fix_sigma(x_values, y_values, guess_peak_centroids, fixed_sigma_values):
    """
    Parameters
    ----------
    x_values : the mean energy of each bin within the ROI.
    y_values : the count within each bin within the ROI.
    guess_peak_centroids : a list of N expected peak centroid locations.
    fixed_sigma_values : either a list of expected sigma values of each peak, with length = N = num_peaks;
                        or a single scalar value, which means we assume all peaks to have the same sigma.
    """
    num_peaks = len(guess_peak_centroids)
    lower_limits, upper_limits = [], []  # list to contain the expected minium and maximul values.
    reduced_guess_peak_params = []  # the guess parameters for each peak (amplitude and centroid location)
    if np.isscalar(fixed_sigma_values):  # if it's a single scalar describing the sigma of all peaks,
        fixed_sigma_values = [
            fixed_sigma_values for _ in guess_peak_centroid
        ]  # then duplicate this scalar into a list of the correct length

    # guess the background
    slope, level = np.polyfit([x_values[0], x_values[-1]], [y_values[0], y_values[1]], 1)
    total_counts = np.sum(y_values)
    num_channels = len(x_values)
    x_span = max(x_values) - min(x_values)
    # when packed as a triangle,
    # the max height channel has counts = total_area/num_channels * 2
    # the min height channel has counts = 0
    max_cnt_chn_cnt = 2 * total_counts / num_channels
    # the limits for bg_level
    lower_limits.append(-max_cnt_chn_cnt * (max(x_values) / x_span) + 2 * total_counts / num_channels)
    upper_limits.append(max_cnt_chn_cnt * (max(x_values) / x_span))
    # the limits for bg_slope
    lower_limits.append(-max_cnt_chn_cnt / x_span)  # the limits for slope
    upper_limits.append(max_cnt_chn_cnt / x_span)

    # use the prexisting calibration equation to find the expected widths for all of these peaks.
    for i, centroid in enumerate(guess_peak_centroids):
        reduced_guess_peak_params.append(total_counts / num_peaks)
        lower_limits.append(0.0)
        upper_limits.append(total_counts / 0.8)  # some peaks are not fully included in the ROI;
        # so the counts within ROI may only include as low as 80% of the counts.
        reduced_guess_peak_params.append(centroid)
        lower_limits.append((centroid + guess_peak_centroids[i - 1]) / 2 if (i - 1) >= 0 else min(x_values))
        upper_limits.append((centroid + guess_peak_centroids[i + 1]) / 2 if (i + 1) < num_peaks else max(x_values))

    # I could've asked curve_fit to use smaller number of
    fit_func_with_reduced_number_of_arguments = multiple_gaussians_with_bg_and_fixed_sigma(*fixed_sigma_values)
    # But I want to reduce the effect of the curse of dimensionality, so I reduced it to the
    popt, pcov = curve_fit(
        fit_func_with_reduced_number_of_arguments,
        x_values,
        y_values,
        sigma=np.sqrt(
            y_values
        ),  # Note: This sigma refers to "how much the fit is allowed to vertically deviate from the data",
        # and is different from the sigma used everywhere else in this program to describe the width of the peaks.
        p0=[level, slope, *reduced_guess_peak_params],
        bounds=(lower_limits, upper_limits),
        method="trf",  # default optimization algorithm for bounded problem. Can change to 'dogbox' to see how it performs.
    )
    return popt


def multiple_gaussians_with_bg_and_fixed_sigma(*sigmas):
    """
    A wrapper function that bakes in the sigma of all peaks into multiple_gaussians_with_bg.

    Returns
    -------
    A function that takes a reduced number of arguments compared to multiple_gaussians_with_bg.
    """

    def fit_func_with_reduced_number_of_arguments(x, bg_level, bg_slope, *reduced_peak_params):
        # reconstruct the peak_params by inserting the sigma as every third element.
        peak_params = intercalate_sigma_into_peak_params(reduced_peak_params, sigmas)
        # then call the multiple_gaussians_with_bg.
        return multiple_gaussians_with_bg(x, bg_level, bg_slope, *peak_params)

    return fit_func_with_reduced_number_of_arguments


def intercalate_sigma_into_peak_params(reduced_peak_params, sigmas):
    """
    Insert the sigmas back into reduced_peak_params to form peak_params.

    To describe N peaks,
    Input
    -----
    reduced_peak_params : A vector of length = 2*N.
                        Every odd element describe an amplitude,
                        every even element describes the centroid of a peak.
    sigmas : A vector of length = N.
            Each element describes the sigma of a single peak.

    Returns
    -------
    peak_params : A vector of length 3*N.
        element 0, 3, 6, 9... describes the amplitude of peak 1, 2, 3, 4...
        element 1, 4, 7, 10... describes the amplitude of peak 1, 2, 3, 4...
        element 2, 5, 8, 11... describes the amplitude of peak 1, 2, 3, 4...
    """
    assert len(reduced_peak_params) // 2 == len(sigmas), f"Expected to have Number of peaks = len(sigmas) = {len(sigmas)}. "
    assert (
        len(reduced_peak_params) % 2 == 0
    ), "Exactly two arguments must be provided to describe each peak: amplitude and centroid."
    peak_params = []

    for amplitude, centroid, sigma in zip(reduced_peak_params[::2], reduced_peak_params[1::2], sigmas):
        peak_params.append(amplitude)
        peak_params.append(centroid)
        peak_params.append(sigma)
    return peak_params
