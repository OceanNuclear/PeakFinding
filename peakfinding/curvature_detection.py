import numpy as np
from numpy import pi, sqrt, exp, array as ary, log as ln
from peakfinding.spectrum import RealSpectrumInteractive
from peakfinding.poisson_distribution import Chi2, Poisson
from tqdm import tqdm

class RealSpectrumCurvature(RealSpectrumInteractive):
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
            if window.sum() <= 3:
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
        unit of B: 
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

def threshold_curve_function_generator(numerator, A, B):
    """
    The idea is that A, B are the fwhm_cal;
    while numerator is just a multiplier to scale the entire function by.
    """
    return lambda E: numerator/sqrt(A * B*E)

class RealSpectrumLikelihood(RealSpectrumInteractive):
    @staticmethod
    def get_peakiness_of_slice(samples):
        """
        Calculates the probability of a sample being not noise.
        """
        mean = np.mean(samples)
        poisson_dist = Poisson(mean)
        chi2 = poisson_dist.negative_log_likelihood(samples).sum()

        chi2_distribution = Chi2(len(samples)-1)
        probability_not_noise = chi2_distribution.cdf(chi2)

        return probability_not_noise

    def get_peakiness(self):
        """
        Calculates the peakiness of the entire spectrum
        """
        if not hasattr(self, "peakiness"):
            peakiness = []
            for window in tqdm(self.get_windows()):
                samples = self.counts[window]
                peakiness.append(self.get_peakiness_of_slice(samples))
            self.peakiness = ary(peakiness)
        return self.peakiness

    def apply_threshold_on_peakiness(self, threshold=0.99):
        """
        Apply peakiness on the entire array.
        """
        return self.get_peakiness() >= threshold

    def apply_threshold_on_noisiness(self, threshold=0.3):
        """
        Calculates whether each part of the spectrum is noise or not.
        What get_peakiness actually gets is the likelihood of a window containing non-noise.
        So get_peakiness can also be interpreted as non-noisiness.

        Logic: For each bin,
            if ANY window including this bin has a non-noisiness level <= threshold, 
                i.e. 1 - (non-noisiness) >= 1 - threshold,
                i.e. noisiness >= 1 - threshold,
            Then we consider this bin as containing noise.
        """
        windows = self.get_windows()
        truth_vector = self.get_peakiness()<= threshold

        return (windows.T * truth_vector).T.any(axis=1)

    def full_sized_peakiness(self):
        windows = self.get_windows()
        peakiness = (windows.astype(float).T * self.get_peakiness()).T
        return np.where(windows, peakiness, np.nan)

class RealSpectrumPeakFinder(RealSpectrumCurvature, RealSpectrumLikelihood):
    def bool_array_to_peak_indices(self, bool_array):
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

    def default_peak_identifier(self, curvature_threshold=-2.0, peakiness_threshold=0.9):
        """
        First apply the curvature thresholding to find the peaks.
        Then within each peak, find the likelihood.
        if the likelihood is never high enough, then reject it.
        Parameters
        ----------
        curvature_threshold: threshold parameter inputted into the RealSpectrumCurvature.apply_threshold_on_curvature method.
        peakiness_threshold: threshold parameter inputted into the RealSpectrumLikelihood.apply_threshold_on_peakiness method.
            For both of these paramteres: a default best threshold is already provided if they're omitted in the call signature.

        Returns
        -------
        array of shape (N,2) indicating start and end of all N peaks that passes through both thresholding methods.
        """
        print("Calculating curvatures in the spectrum, then applying a threshold to cut out peaks:")
        curvature_bool_array = self.apply_threshold_on_curvature(threshold=curvature_threshold)
        peak_indices = self.bool_array_to_peak_indices(curvature_bool_array)

        print("Calculating likelihood of each peak containing not-noise...")
        new_peak_indices = []
        windows = self.get_windows()

        for l_index, r_index in peak_indices:
            for centroid_ind in range(l_index, r_index):
                window = windows[centroid_ind]
                peakiness = self.get_peakiness_of_slice(self.counts[window])
                # If ANY of these points reaches a peakiness above , then accept it, and move onto the next peak.
                if peakiness >= peakiness_threshold:
                    new_peak_indices.append([l_index, r_index])
                    break

        return ary(new_peak_indices)

    def slower_peak_identifier(self, curvature_threshold=-2.0, peakiness_threshold=0.9):
        """
        Slower than default_peak_identifier becdause both the curvatures and peakiness are calculted for the entire spectrum.
        default is not only faster, but also better.

        See the docstring for default_peak_identifier for Parameters and Returns
        """
        print("Calculating curvatures in the spectrum, then applying a threshold to cut out peaks:")
        curvature_bool_array = self.apply_threshold_on_curvature(threshold=curvature_threshold)

        print("Calculating likelihood of each peak containing not-noise...")
        peakiness_bool_array = self.apply_threshold_on_peakiness(threshold=peakiness_threshold)

        return self.bool_array_to_peak_indices(np.logical_and(curvature_bool_array, peakiness_bool_array))

    def plot_identified_peaks(self, ax, peak_indices, plot_scale="sqrt", **plot_kwargs):
        """
        To be used after a self.plot_sqrt_repr 
        """
        if plot_scale=="sqrt":
            y = sqrt(self.counts)
        else:
            y = self.counts

        for l_index, r_index in peak_indices:
            ax.fill_between(self.boundaries[l_index:r_index].flatten(), np.repeat(y[l_index:r_index], 2), color="C1", **plot_kwargs)
        return ax

if __name__=='__main__':
    from matplotlib import pyplot as plt
    import sys
    DEMONSTRATE_DECISION_MAKING = True

    spectrum = RealSpectrumPeakFinder.from_multiple_files(*sys.argv[1:])
    spectrum.show_sqrt_scale()
    spectrum.add_fwhm_cal_interactively()
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