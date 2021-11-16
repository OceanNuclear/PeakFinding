import numpy as np
from numpy import pi, sqrt, exp, array as ary
from peakfinding.spectrum import RealSpectrumInteractive
from peakfinding.poisson_distribution import Chi2, Poisson
from tqdm import tqdm
from collections import namedtuple
import warnings

__all__ = ["RealSpectrumCurvature",
            "threshold_curve_function_generator",
            "RealSpectrumLikelihoodSimple",
            "RealSpectrumLikelihoodWithNoiseFlooring",
            # "RealSpectrumLikelihoodDuplicatePeakiness",
            "RealSpectrumPeakFinder",
            "RealSpectrumPeakFinderFromNoiseFloor",
            # "RealSpectrumPeakFinderDuplicatePeakiness",
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
    bool_array = np.insert(bool_array, 0, False) # extend it by 1 so that you know there wasn't a peak at T=0
    peak_start_stops = np.diff(bool_array.astype(int))
    peak_ledge_indices = np.argwhere(peak_start_stops==1).flatten()
    peak_redge_indices = np.argwhere(peak_start_stops==-1).flatten()

    return ary([peak_ledge_indices, peak_redge_indices]).T

class RealSpectrumCurvature(RealSpectrumInteractive):
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
        
        for window in tqdm(self.get_windows()):
            if window.sum() < 3:
                curvature_coef.append(np.nan)
                chi2_values.append(np.nan)
                probability_of_having_such_chi2_or_less.append(np.nan)
            else:
                # perform deg=2 polynomial fit on this window
                results = np.polyfit(self.boundaries.mean(axis=1)[window], y[window], 2, full=True)
                # unpack fit results
                p2, p1, p0 = results[0]
                residuals = results[1]
                cdf_value = Chi2(window.sum()-3).cdf(residuals) # 3 DoF removed because we're fitting 3 coefficients.

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
        for window in tqdm(self.get_windows()):
            if window.sum() < 3:
                curvature_coef.append(np.nan)
            else:
                # perform deg=2 polynomial fit on this window
                p2, p1, p0 = np.polyfit(self.boundaries.mean(axis=1)[window], y[window], 2, full=False)

                # add these results to the final output
                curvature_coef.append(p2)

        return ary(curvature_coef)

    def apply_threshold_on_curvature(self, threshold=-2.5): # unit of threshold = counts keV^-2
        """
        Identify peaks by claiming (regions with curvature <= threshold) == peaks.

        The best scalar value, found by trial and error, is -0.69;
        But a better thresholding method is to use a variable threshold: one that decreases with window size.

        Therefore a function is called to calculate this energy dependent threshold.

        Parameters
        ----------
        threshold: multiplier onto the following function:
                threshold_values = threshold * 1/sqrt(A + B*E)

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
        threshold_values = threshold_function(self.boundaries.mean(axis=1))
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

class RealSpectrumLikelihoodSimple(RealSpectrumInteractive):
    """
    self.peakiness refers to the Chi2(window_size).cdf(Poisson.negative_log_likelihood(samples)) of the sample window associated with that bin.
    It's generated by self.get_peakiness() and is saved as an attribute to speed up further computation down the line.
    """
    @staticmethod
    def _Prob_slice_is_not_noise(samples):
        """
        Calculates the probability of a sample being not noise.
        """
        mean = np.mean(samples)
        # hypothesis
        poisson_dist = Poisson(mean) # null hypothesis, i.e. hypothesized distribution
        # perform p-value test, by calculating (1-pvalue):
        chi2 = poisson_dist.negative_log_likelihood(samples).sum()

        chi2_distribution = Chi2(len(samples)-1) # number of free parameters = 1; number of variables to fit = len(samples)
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

class RealSpectrumLikelihoodDuplicatePeakiness(RealSpectrumLikelihoodSimple):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise DeprecationWarning(f"This class ({self.__class__}) has been deemed less promising in its peak identification abilities. Use at own risk.")

    def get_full_sized_peakiness(self):
        """
        Spread out the peakiness value of each window across the entire window,
        rather than assigning it to the window itself.

        """
        windows = self.get_windows()
        peakiness = (windows.astype(float).T * self.get_peakiness(1.0)).T
        return np.where(windows, peakiness, np.nan)

    def apply_function_on_full_peakiness(self, function):
        """Taking nanmean, nanmean, nanmax along axis=1 yields different versions of the peakiness values."""
        peakiness_matrix = self.get_full_sized_peakiness()
        return function(peakiness_matrix, axis=0)

    def apply_threshold_on_full_peakiness(self, threshold=0.9, function=np.nanmean):
        function_output = self.apply_function_on_full_peakiness(function)
        return function_output >= threshold

class RealSpectrumLikelihoodWithNoiseFlooring(RealSpectrumLikelihoodSimple):
    def determine_noise_floor(self, max_pvalue=0.5):
        """
        Determine 
        Output
        ------
        (Also sets self.raw_noise_floor and self.noise_floor)
        self.raw_noise_floor :
        self.noise_floor : the noise
        """
        if not hasattr(self, "raw_noise_floor"):
            # print("Finding the raw noise floor...")
            is_noise = self.get_peakiness(window_size_multiplier=1.0) <= max_pvalue
            peaks_replaced_by_nans = np.where(is_noise, self.counts, np.nan)
            self.raw_noise_floor = peaks_replaced_by_nans

        # noise_window = self.get_windows(width_multiplier=3.0)
        # raw_noise_values = np.where(noise_window, self.raw_noise_floor, np.nan)
        # counts_values = np.where(self.get_windows(width_multiplier=1.0), self.counts, np.nan)

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        #     approved_noise = np.nanmean(raw_noise_values, axis=1)
        #     min_counts = np.nanmin(counts_values, axis=1)

        if not hasattr(self, "noise_floor"):
            # refine this noise floor so that no nans are left
            print("Refining the noise floor values...")
            noise_window = self.get_windows(width_multiplier=1.0)

            background_noise_level = []
            for peak_bins in tqdm(noise_window):
                left_index, right_index = np.argwhere(peak_bins)[[0,-1], 0]
                left_side, right_side = self.raw_noise_floor[:left_index], self.raw_noise_floor[right_index+1:] # find the end point
                backgrounds = np.hstack([_first_n_notnan_val_of_array(left_side[::-1], peak_bins.sum()),
                                        _first_n_notnan_val_of_array(right_side, peak_bins.sum())])
                # test to make sure that there's not a significant overestimation of the background:
                background_noise_level.append(backgrounds.mean())
            self.noise_floor = ary(background_noise_level)
        return self.noise_floor

    def infer_peakiness_from_background(self, negative_deviation_dilution_factor=1.0):
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

            for peak_bins, bg_level in tqdm(zip(peak_windows, self.noise_floor), total=peak_windows.shape[0]):
                samples = self.counts[peak_bins]

                # create null hypothesis
                poisson_dist = Poisson(bg_level)

                # hijack chi2 distribution's cdf to calculate the p-value, again.
                chi2_array = poisson_dist.negative_log_likelihood(samples)
                chi2_array[samples<bg_level] = negative_deviation_dilution_factor*chi2_array[samples<bg_level]
                chi2 = chi2_array.sum()
                chi2_distribution = Chi2(len(samples))
                # DoF = len(samples), because NO part of the calculation of bg_level involved 'samples'.
                # Therefore there are zero fitted parameters.
                probability_is_peak = chi2_distribution.cdf(chi2)
                peakiness_from_bg_noise.append(probability_is_peak)

            self.peakiness_from_bg_noise = ary(peakiness_from_bg_noise)
        return self.peakiness_from_bg_noise

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
                ax.fill_between(self.boundaries[l_index:r_index].flatten(), np.repeat(y[l_index:r_index], 2), color=color, **plot_kwargs)
            else: # it's a namedtuple with left_edge, right_edge and peak_tips as fields.
                l_index, r_index = region.left_edge, region.right_edge
                peak_tips = region.peak_tips
                ax.fill_between(self.boundaries[l_index:r_index].flatten(), np.repeat(y[l_index:r_index], 2), color=color, **plot_kwargs)

                x_values = np.broadcast_to(self.boundaries[peak_tips].mean(axis=1), (2, len(peak_tips)))
                y_values = ary([np.zeros(len(y))[peak_tips], y[peak_tips]])
                ax.plot(x_values, y_values, color='black', **plot_kwargs)
        return ax

class RealSpectrumPeakFinderDuplicatePeakiness(RealSpectrumPeakFinder, RealSpectrumLikelihoodDuplicatePeakiness):
    def faster_peak_identifier(self, curvature_threshold=-2.0, peakiness_threshold=0.7):
        """
        First apply the curvature thresholding to find the peaks.
        Then within each peak, find the likelihood.
        if none of the bins within said peak has a high enough likelihood, then reject it.
        Parameters
        ----------
        curvature_threshold: threshold parameter inputted into the RealSpectrumCurvature.apply_threshold_on_curvature method.
        peakiness_threshold: threshold parameter inputted into the RealSpectrumLikelihood.apply_threshold_on_full_peakiness method.
            For both of these paramteres: a default best threshold is already provided if they're omitted in the call signature.

        Returns
        -------
        array of shape (N,2) indicating start and end of all N peaks that passes through both thresholding methods.
        """
        print("Calculating curvatures in the spectrum, then applying a threshold to cut out peaks:")
        curvature_bool_array = self.apply_threshold_on_curvature(threshold=curvature_threshold)
        peak_indices = _bool_array_to_peak_indices(curvature_bool_array)

        print("Calculating likelihood of each peak containing not-noise...")
        new_peak_indices = []
        windows = self.get_windows()

        for l_index, r_index in peak_indices:
            for centroid_ind in range(l_index, r_index):
                window = windows[centroid_ind]
                peakiness = self._Prob_slice_is_not_noise(self.counts[window])
                # If ANY of these points reaches a peakiness above , then accept it, and move onto the next peak.
                if peakiness >= peakiness_threshold:
                    new_peak_indices.append([l_index, r_index])
                    break

        return ary(new_peak_indices)

    def default_peak_identifier(self, curvature_threshold=-2.0, peakiness_threshold=0.7, peakiness_function=np.nanmean):
        """
        Slower than faster_peak_identifier becdause both the curvatures and peakiness are calculted for the entire spectrum.

        See the docstring for faster_peak_identifier for Parameters and Returns

        The difference being that ONLY bins that break through both thresholds simulatenously will be considered part of a peak
        """
        print("Calculating curvatures in the spectrum, then applying a threshold to cut out peaks:")
        curvature_bool_array = self.apply_threshold_on_curvature(threshold=curvature_threshold)

        print("Calculating likelihood of each peak containing not-noise...")
        peakiness_bool_array = self.apply_threshold_on_full_peakiness(threshold=peakiness_threshold, function=peakiness_function)

        return _bool_array_to_peak_indices(np.logical_and(curvature_bool_array, peakiness_bool_array))

Region = namedtuple("Region", ["left_edge", "right_edge", "peak_tips"])

class RealSpectrumPeakFinderFromNoiseFloor(RealSpectrumPeakFinder, RealSpectrumLikelihoodWithNoiseFlooring):
    def peak_identifier(self, curvature_threshold=-2.0, peakiness_threshold=0.99):
        """
        curvature_threshold is only used for deciding where the peak tips lie, 
        """
        curvature_bool_array = self.apply_threshold_on_curvature(threshold=curvature_threshold)
        self.infer_peakiness_from_background()

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
            bracketing_dummy_bool_vector[l_index:r_index+1] = True
            peaks_as_bool = np.logical_and(bracketing_dummy_bool_vector, local_mins_in_curvature_plot)

            peak_locations = np.argwhere(peaks_as_bool)[:,0]
            regions.append(Region(l_index, r_index, peak_locations))
        return regions
