import numpy as np
from numpy import pi, sqrt, exp, array as ary
from peakfinding.spectrum import RealSpectrumInteractive
from peakfinding.poisson_distribution import PoissonFast, Chi2List
from tqdm import tqdm
from collections import namedtuple
import warnings
"""
Only downside of this module:
The following factors were haphazardly introduced (though not without reason):
    - smear_factor, which makes the  the weight window used to calculate the noise floor bigger.
    - curvature_threshold, which rejects peaks whose fitted curvature is above a this threshold.
All other aspects of this module is derived in a mathematically rigorous sense.

TODO:
1. Modify show_sqrt_scale and show_log_scale to make peak appear more obviously
2. See if it's worth removing the codes that generates and saves the following objects as attributes:
    - self.peakiness
    - self.raw_noise_floor
    - self.peakiness_from_bg_noise
    - self.fitted_curvature
    And instead generate them on the fly whenever needed.
"""
__all__ = ["RealSpectrumCurvature",
            "threshold_curve_function_generator",
            "RealSpectrumLikelihoodSimple",
            "RealSpectrumLikelihoodWithNoiseFlooring",
            "RealSpectrumPeakFinder",
            "RealSpectrumPeakFinderFromNoiseFloor",
            ]

def _first_n_notnan_val_of_array(array, n):
    return array[np.isfinite(array)][:n]

def _bool_array_to_peak_indices(bool_array):
    """
    Parameters
    ----------
    Input: 1 boolean array

    Returns
    -------
    array of shape (N,2) indicating start and end of N peaks.
    """
    # turn boolean array (obtained by applying the threshold) into index pairs indicating the start and end of peaks.
    bool_array = ary([False]+list(bool_array)+[False], dtype=bool)
    # wrap it with False's so even if it starts/ends with True it'll be properly recorded
    peak_start_stops = np.diff(bool_array.astype(int))
    peak_ledge_indices = np.argwhere(peak_start_stops==1).flatten()
    peak_redge_indices = np.argwhere(peak_start_stops==-1).flatten()
    if peak_redge_indices[-1] == len(bool_array):
        peak_redge_indices[-1] = len(bool_array)-1 # clip the right-most edge index into range.
    
    return ary([peak_ledge_indices, peak_redge_indices]).T

class RealSpectrumWithChi2(RealSpectrumInteractive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._Chi2_dists = Chi2List()

class RealSpectrumCurvature(RealSpectrumWithChi2):
    """
    self.calculate_sqrt_curvature doesn't take much time.
    Therefore we don't save its output into self.curvature, and can keep the multiple inheritance class (RealSpectrumPeakFinder) below simple.
    """
    def _calculate_sqrt_curvature_chi2_and_probabilities(self):
        """
        (This method should only be used for debugging.)
        Fit the sqrt(counts) within a window to a 2nd order polynomial expression.
        Returns: all of the debugging information
        -------
        curvature_coef : p2 of the 2nd order polynomial fit coefficients [p2, p1, p0] on that window.
        chi2_values : chi^2 obtained by fitting that particular window
        probability_of_having_such_chi2_or_less : probability of getting such chi2.
        """
        # lists to contain them
        curvature_coef, chi2_values, probability_of_having_such_chi2_or_less = [], [], []
        y = sqrt(self.counts)
        print("Getting peakiness values...")
        for window in tqdm(self.get_windows()):
            if window.sum() < 3:
                curvature_coef.append(np.nan)
                chi2_values.append(np.nan)
                probability_of_having_such_chi2_or_less.append(np.nan)
            else:
                # perform deg=2 polynomial fit on this window
                results = np.polyfit(self.boundaries().mean(axis=1)[window], y[window], 2, full=True)
                # unpack fit results
                p2, p1, p0 = results[0]
                residuals = results[1]
                cdf_value = self._Chi2_dists[window.sum()-3].cdf(residuals) # 3 DoF removed because we're fitting 3 coefficients.

                # add into the lists created above
                curvature_coef.append(p2)
                chi2_values.append(residuals)
                probability_of_having_such_chi2_or_less.append(cdf_value)

        return ary(curvature_coef), ary(chi2_values), ary(probability_of_having_such_chi2_or_less)

    # def get_sqrt curvature # get the other two coefficients too!

    def calculate_sqrt_curvature(self):
        """
        Fit the sqrt counts 
        """
        # create lists to contain for the following vectors
        y = sqrt(self.counts)
        curvature_coef = []
        if not hasattr(self, "fitted_curvature")
            print("Calculating curvature...")
            for window in tqdm(self.get_windows()):
                if window.sum() < 3:
                    curvature_coef.append(np.nan)
                else:
                    # perform deg=2 polynomial fit on this window
                    p2, p1, p0 = np.polyfit(self.boundaries().mean(axis=1)[window], y[window], 2, full=False)

                    # add these results to the final output
                    curvature_coef.append(p2)
            self.fitted_curvature = ary(curvature_coef)

        return self.fitted_curvature

    def apply_threshold_on_curvature(self, threshold=-2.5): # unit of threshold = counts (self.bound_units)^-2
        """
        Identify peaks by claiming (regions with curvature <= threshold) == peaks.

        The best scalar value, found by trial and error, is -0.69;
        But a better thresholding method is to use a variable threshold: one that decreases with window size.

        Therefore a function is called to calculate this energy dependent threshold.

        Parameters
        ----------
        threshold: multiplier onto the following function:
                threshold_values = threshold * 1/sqrt(A + B*E)

        if self.bound_units == "keV", then
        unit of threshold = counts keV^-2
        unit of A: dimensionless
        unit of B: keV^-1
        A, B are the coefficients fitted by FWHM(E) = sqrt(A + B*E)

        Returns
        -------
        boolean array denoting whether this is or isn't a peak.
        """
        # calculate curvature
        curvature = self.calculate_sqrt_curvature()

        # apply threshold on curvature datae
        threshold_function = threshold_curve_function_generator(threshold, *self.fwhm_cal)
        threshold_values = threshold_function(self.boundaries().mean(axis=1))
        bool_array = curvature<=threshold_values
        return bool_array

def threshold_curve_function_generator(numerator, *coeffs):
    """
    The idea is that A, B are the fwhm_cal;
    while numerator is just a multiplier to scale the entire function by.

    Therefore the threshold would be
          numerator
        -------------
        sqrt(A + B*E)
    Where the unit of the numerator is keV
        (independent of the gain of the amplifier,
        so the same numerator can be reused even if you've changed the amplifier gain).
    """
    def equation(E):
        denominator = ary([c * E**n for n, c in enumerate(coeffs)]).sum(axis=0)

        return numerator/denominator
    return equation

class RealSpectrumLikelihoodSimple(RealSpectrumWithChi2):
    """
    self.peakiness refers to the Chi2(window_size).cdf(PoissonFast.negative_log_likelihood(samples)) of the sample window associated with that bin.
    It's generated by self.get_peakiness() and is saved as an attribute to speed up further computation down the line.
    """

    def _Prob_slice_is_not_noise(self, samples):
        """
        Calculates the probability of a sample being not noise.
        """
        mean = np.mean(samples)
        # hypothesis
        poisson_dist = PoissonFast(mean) # null hypothesis, i.e. hypothesized distribution
        # perform p-value test, by calculating (1-pvalue):
        chi2 = poisson_dist.negative_log_likelihood(samples).sum()

        chi2_distribution = self._Chi2_dists[len(samples)-1] # faster to call than to create own chi2 distributions
        # number of free parameters = 1; number of variables to fit = len(samples)
        probability_not_noise = chi2_distribution.cdf(chi2) # this gives 1-pvalue

        return probability_not_noise

    def get_peakiness(self, window_size_multiplier=1.0):
        """
        Calculates the peakiness of the entire spectrum, 
        one scalar for each bin, 
        forming a vector of the same shape as the spectrum.
        """
        if not hasattr(self, "peakiness"):
            print("Calculating the (true) peakiness of each bin...")
            peakiness = []
            for window in tqdm(self.get_windows(width_multiplier=window_size_multiplier)):
                samples = self.counts[window]
                peakiness.append(self._Prob_slice_is_not_noise(samples))
            self.peakiness = ary(peakiness)
        return self.peakiness

_gaussian = lambda x, sigma, mu: exp(-((x-mu)/sigma)**2/2)

class RealSpectrumLikelihoodWithNoiseFlooring(RealSpectrumLikelihoodSimple):
    def determine_noise_floor(self, max_pvalue=0.5, smear_factor=2.35*2):
        """
        Determine 
        Output
        ------
        (Also sets self.raw_noise_floor and self.noise_floor)
        self.raw_noise_floor : the raw count values of bins which are considered as noise.
        self.noise_floor : the noise
        """
        L = len(self.counts)
        if not hasattr(self, "raw_noise_floor"):
            # print("Finding the raw_noise_floor values...")
            is_noise = self.get_peakiness(window_size_multiplier=1.0) <= max_pvalue
            peaks_replaced_by_nans = np.where(is_noise, self.counts, np.nan)
            self.raw_noise_floor = peaks_replaced_by_nans
        """
        # Find all the local minima.
        # Accept the local minima as a replacement for the noise
        #   if it's close enough to the interpolated noise value there.
        """
        print("Calculating the final noise_floor...")
        raw_noise_values = np.broadcast_to(self.raw_noise_floor, (L, L))
        valid_positions = np.isfinite(raw_noise_values) # record the positions of NOT nans

        # by weighted average
        # make a weight vector that's almost twice the length of the spectrum.

        energies = self.boundaries().mean(axis=1)
        sigma_at_E = self.get_width_at(energies)
        weight_vector_decay_rate = 1/2.35*smear_factor * sigma_at_E
        weight_matrix = np.empty([L, L])

        for i, (s, mu) in enumerate(zip(weight_vector_decay_rate, energies)):
            weight_matrix[i] = _gaussian(energies, s, mu)

        final_w = np.where(valid_positions, weight_matrix, np.nan)
        final_w_sum = np.nansum(final_w, axis=1)

        weighted_sum_mat = raw_noise_values * final_w
        weighted_sum = np.nansum(weighted_sum_mat, axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            noise_floor = np.true_divide(weighted_sum, final_w_sum)
        self.noise_floor = noise_floor

        assert np.isfinite(noise_floor).all()
        return self.noise_floor

    def infer_peakiness_from_background(self, negative_deviation_dilution_factor=0.0):
        """
        For each bin, determine the window size, and call that 'n'.
        Then, search left and right of that window for the first n noise values on each side.
        Use the mean of that as the background noise_floor.

        negative_deviation_dilution_factor: allowed values: between 1 to -1.
            The smaller it is, the more strongly it dilutes the negatively deviating sample's contribution to the chi^2.
            It does so by changing the shape of the NLE function
                from an approximately symmetric (upward) parabola into
                a parabola with depressed left branch.
            This allows us to focus more on the effect of positively deviating values.
        """
        if not hasattr(self, "peakiness_from_bg_noise"):
            peakiness_from_bg_noise = []
            self.determine_noise_floor()
            peak_windows = self.get_windows(width_multiplier=1.0)

            print("Calculating peakiness_from_bg_noise...")
            for peak_bins, bg_level in tqdm(zip(peak_windows, self.noise_floor), total=peak_windows.shape[0]):
                samples = self.counts[peak_bins]

                # create null hypothesis
                poisson_dist = PoissonFast(bg_level)

                # hijack chi2 distribution's cdf to calculate the p-value, again.
                chi2_array = poisson_dist.negative_log_likelihood(samples)
                # apply the dilution factor onto any negatively-deviating points,
                # so that negatively deviating datapoints contribute less to pushing up the chi2.
                # The chi2 being very low is great.
                chi2_array[samples<bg_level] = negative_deviation_dilution_factor*chi2_array[samples<bg_level]
                chi2 = chi2_array.sum()
                chi2_distribution = self._Chi2_dists[len(samples)] # faster to call than to create own chi2 distributions
                # DoF = len(samples), because NO part of the calculation of bg_level involved 'samples'.
                # Therefore there are zero fitted parameters.
                probability_is_peak = chi2_distribution.cdf(chi2)
                peakiness_from_bg_noise.append(probability_is_peak)

            self.peakiness_from_bg_noise = ary(peakiness_from_bg_noise)
        return self.peakiness_from_bg_noise

Region = namedtuple("Region", ["left_edge", "right_edge", "peak_tips"])

class RealSpectrumPeakFinder(RealSpectrumCurvature):
    def highlight_peaks(self, ax, peaks_identified, plot_scale="sqrt", color="C1", **plot_kwargs):
        """
        To be used after a self.plot_sqrt_repr 
        """
        if plot_scale=="sqrt":
            y = sqrt(self.counts)
        else:
            y = self.counts

        for region in peaks_identified:
            if isinstance(region, np.ndarray) and region.shape==(2,):# peak is a numpy array with shape=(2,)
                l_index, r_index = region
                ax.fill_between(self.boundaries()[l_index:r_index].flatten(), np.repeat(y[l_index:r_index], 2), color=color, **plot_kwargs)
            elif isinstance(region, Region): # it's a namedtuple with left_edge, right_edge and peak_tips as fields.
                l_index, r_index = region.left_edge, region.right_edge
                peak_tips = region.peak_tips
                ax.fill_between(self.boundaries()[l_index:r_index].flatten(), np.repeat(y[l_index:r_index], 2), color=color, **plot_kwargs)

                x_values = np.broadcast_to(self.boundaries()[peak_tips].mean(axis=1), (2, len(peak_tips)))
                y_values = ary([np.zeros(len(y))[peak_tips], y[peak_tips]])
                ax.plot(x_values, y_values, color='black', **plot_kwargs)
        return ax

class RealSpectrumPeakFinderFromNoiseFloor(RealSpectrumPeakFinder, RealSpectrumLikelihoodWithNoiseFlooring):
    def peak_identifier(self, curvature_threshold=-4.0, peakiness_threshold=0.5, negative_deviation_dilution_factor=0.0):
        """
        curvature_threshold is only used for deciding where the peak tips lie, 
        """
        curvature_bool_array = self.apply_threshold_on_curvature(threshold=curvature_threshold)
        self.infer_peakiness_from_background(negative_deviation_dilution_factor=negative_deviation_dilution_factor)

        # use curvature
        curvature_vector = self.calculate_sqrt_curvature()
        diff = np.diff(curvature_vector)
        fDiff, bDiff = np.hstack([diff, [np.nan]]), np.hstack([[np.nan], diff])
        local_mins_in_curvature_plot = np.logical_and(curvature_bool_array, np.logical_and(fDiff>0, bDiff<0))

        peakiness_bool_array = self.peakiness_from_bg_noise >= peakiness_threshold

        regions = []
        # for l_index, r_index in _bool_array_to_peak_indices(np.logical_and(curvature_bool_array, peakiness_bool_array)):
        for l_index, r_index in _bool_array_to_peak_indices(peakiness_bool_array):
            bracketing_dummy_bool_vector = np.zeros(len(curvature_vector), dtype=bool)
            bracketing_dummy_bool_vector[l_index:r_index] = True
            peaks_as_bool = np.logical_and(bracketing_dummy_bool_vector, local_mins_in_curvature_plot)

            peak_locations = np.argwhere(peaks_as_bool)[:,0]
            regions.append(Region(l_index, r_index, peak_locations))
        return regions
