import numpy as np
from numpy import pi, sqrt, exp, array as ary, log as ln
tau = 2*pi
from numpy import cos, sin, arccos
from matplotlib import pyplot as plt
import re
import sys
import datetime as dt
import warnings
import functools
from operator import add as add

class Histogram():
    """
    Parameters
    ----------
    counts: an array of ints, each representing the count within a bin.
    boundaries: 2d array of shape (len(counts), 2),
        the 2 values representing the lower and upper boundaries of the bin.
    bound_units: the units of the x axis of the hisogram.
    """
    def __init__(self, counts : np.ndarray, boundaries : np.ndarray, bound_units : str):
        self.counts = counts
        self.boundaries = boundaries
        self.bound_units = bound_units

    def plot_log_scale(self, ax=None, **kwargs):
        if not ax:
            ax = plt.subplot()
        line, = ax.semilogy(self.boundaries.flatten(), np.repeat(self.counts, 2), **kwargs)
        ax.set_xlabel(self.bound_units)
        ax.set_ylabel("counts")
        return ax, line
        
    def plot_sqrt_scale(self, ax=None, rewrite_yticks=False, **kwargs):
        if not ax:
            ax = plt.subplot()
        transformed_cnts = sqrt(self.counts)
        line, = ax.plot(self.boundaries.flatten(), np.repeat(transformed_cnts, 2), **kwargs)
        ax.set_xlabel(self.bound_units)
        # y labelling
        if rewrite_yticks:
            ax.set_ylabel("counts")
            ax.set_yticklabels(np.sign(yticks:=ax.get_yticks()) * yticks**2)
        else:
            ax.set_ylabel("sqrt(counts)")
        return ax, line

class TimeSeries(Histogram):
    def __add__(self, ts2):
        counts = np.hstack(self.counts, ts2.counts)
        boundaries = np.vstack(self.boundaries, ts2.boundaries)
        bound_units = self.bound_units
        return self.__class__(counts, boundaries, bound_units)

class RealSpectrum(Histogram):
    def __init__(self, counts, boundaries, bound_units, wall_time:float, **init_dict):
        """
        Creates a Real Spectrum, tailored to gamma spectra, but can be used for other particle count vs energy spectra.

        wall_time: float scalar denoting the duration of this spectrum
        """
        super().__init__(counts, boundaries, bound_units)
        self.wall_time = wall_time # duration in seconds seconds
        self.__dict__.update(init_dict)

    def __add__(self, rs2): # rs2 = RealSpectrum 2
        warnings.warn("Adding spectra together.\nMake sure you don't repeat the same spectrum, and add chronologically left-to-right.", UserWarning)
        init_dict = rs2.__dict__.copy() 
        init_dict.update(self.__dict__) # overwrite with properties of self whenever there's a duplication.
        init_dict.pop("counts")
        init_dict.pop("boundaries")
        init_dict.pop("bound_units")
        init_dict.pop("wall_time")

        # defined in super().__init__:
        counts = rs2.counts + self.counts
        boundaries = self.boundaries
        bound_units = self.bound_units

        # special cases:
        wall_time = self.wall_time + rs2.wall_time

        if "date_end_mea" in init_dict.keys():
            if hasattr(rs2, "date_end_mea"):
                init_dict["date_end_mea"] = ts2.date_end_mea
            else:
                init_dict["date_end_mea"] = None
        if "live_time" in init_dict.keys():
            if hasattr(self, "live_time") and hasattr(rs2, "live_time"):
                init_dict["live_time"] = self.live_time + rs2.live_time

        return self.__class__(counts, boundaries, bound_units, wall_time, **init_dict)

    @staticmethod
    def calibration_equation(cls, calibration_constants):
        return np.poly1d(calibration_constants[::-1])

    @classmethod
    def from_Spe(cls, file_name):
        """
        Reads from an .Spe file and creates an object.
        """
        init_dict = {}
        # default unit is bins
        with open(file_name) as f:
            text = f.read()
        for block in text.split("$"):
            lines = block.split("\n") # remove the last \n
            block_identifier = lines[0].strip(":")

            if block_identifier=="DATA": # All .Spe files must have the data block
                min_chan, max_chan = regex_num(lines[1], int)
                counts = ary([float(i) for i in lines[2+min_chan:3+max_chan]])

            elif block_identifier=="MEAS_TIM": # measurement time
                live_time, wall_time = [float(i) for i in lines[1].split()]
                init_dict["live_time"] = live_time

            elif block_identifier.endswith("CAL"): # calibration
                num_constants = int(lines[1])
                calibration_constants = lines[2].split()
                init_dict[block_identifier.lower()] = [float(i) for i in calibration_constants[:num_constants]]
                if len(calibration_constants)>num_constants:
                    init_dict[block_identifier.lower() + "_unit"]= calibration_constants[num_constants:]

            elif block_identifier=="ENER_FIT": # energy fit (which is specific to maestro software, non-standard last I checked)
                init_dict["energy_fit"] = [float(i) for i in lines[1].split()]

            elif block_identifier in ("DATE_END_MEA", "DATE_MEA"):
                init_dict[block_identifier.lower()] = _to_datetime(lines[1])

            else: # all other types of blocks
                init_dict[block_identifier.lower()] = "\n".join(lines[1:])
                # raise KeyError("Invalid BLOCK identifier found: {}".format(block_identifier))

        bin_boundaries = ary([np.arange(max_chan+1), np.arange(1, max_chan+2)]).T

        # check for the clibration equation
        if "mca_cal" in init_dict or "energy_fit" in init_dict:
            if "mca_cal" in init_dict:
                calibration_eqn = cls.calibration_equation(cls,init_dict["mca_cal"])
            elif "energy_fit" in init_dict:
                calibration_eqn = cls.calibration_equation(cls,init_dict["energy_fit"])
            boundaries = calibration_eqn(bin_boundaries)
            bound_units = init_dict.get("mca_cal_unit", ["keV",])[0]

        else:# no calibration equation available
            boundaries = bin_boundaries
            bound_units = "bins"
            
        return cls(counts, boundaries, bound_units, wall_time, **init_dict)

    @classmethod
    def from_Spes(cls, *filenames):
        return functools.reduce(add, [cls.from_Spe(fname) for fname in filenames])

    def to_Spe(self, file_path):
        # identify what blocks are writable (all cases that aren't special cases will be handled differently)
        self_dict = self.__dict__
        with open(filenames, "w") as f:
            # the first thing is to write measurement time
            if "live_time" in self.__dict__ and "wall_time" in self.__dict__:
                f.write("$MEAS_TIM:\n{} {}\n".format(self.wall_time, self.live_time))

            for k, v in self.__dict__.items():
                if k in ("live_time", "wall_time", "boundaries", "bound_units"):
                    continue # take care of these later
                    f.write("${}:\n".format(k.upper()))

                elif k.upper().endswith("CAL"):
                    f.write(_format_key(k))
                    f.write(_format_CAL(v)+"\n")

                elif k.upper() == "counts":
                    f.write(_format_key(k))
                    f.write("{} {}".format(0, len()-1))
                    f.write("\n".join([str(i).ljust() for i in v]))
                    f.write("\n")

                elif k.upper() in ("DATE_END_MEA", "DATE_MEA"):
                    f.write(_format_key(k))
                    f.write(dt.datetime.strptime(k, "%M/%D/%Y %h:%m:%s\n"))

                elif k == "energy_fit":
                    f.write("$ENER_FIT\n")
                    f.write(_format_CAL(v)[2:])

                else:
                    f.write(_format_key(k))
                    f.write(v+"\n")

        raise NotImplementedError("Too busy; will write this later")

    def to_csv(self, file_path):
        import pandas as pd
        df = pd.DataFrame(ary([self.counts, *self.boundaries.T]).T, columns=["lenergy", "uenergy", "count"])
        df.to_csv(file_path, index_label="channel")
        return

def _format_key(key):
    return "${}:\n".format(key)

def _format_CAL(coefficients):
    formatter_str = "{}\n{}"
    return formatter_str.format(len(coefficients), " ".join([str(i) for i in coefficients]))

class RealSpectrumInteractive(RealSpectrum):
    def __init__(self, counts, boundaries, bound_units, wall_time, **init_dict):
        super().__init__(counts, boundaries, bound_units, wall_time, **init_dict)
        self._clicked_and_dragged = []

    def show_log_scale(self, ax=None, **kwargs):
        """
        Identical documentation as show_sqrt_scale
        """
        ax, line = super().plot_log_scale(ax, **kwargs)
        self._setup_fig(ax.figure)
        plt.show()
        self._teardown_fig()
        return

    def show_sqrt_scale(self, ax=None, **kwargs):
        """
        Connect button clicks on the plot to other useful stuff.
        Parameters
        ----------
        same as RealSpectrum.plot_sqrt_scale

        Returns
        -------
        nothing, as this method only returns when we close the plot.
        """

        ax, line = super().plot_sqrt_scale(ax, False, **kwargs)
        self.fig = ax.figure
        self._setup_fig(ax.figure)
        ax.set_ylabel("counts")
        yticks = ax.get_yticks()
        yticks = round_to_nearest_sq_int(yticks)
        ax.set_yticklabels(["{:d}".format(int(np.round(i))) for i in np.sign(yticks)*(yticks)**2])
        plt.show()
        self._teardown_fig()
        return

    def _on_press(self, event):
        """
        gets the information about a button press down event.
        Other usable attributes about the events are:
        'button', 'canvas', 'dblclick', 'guiEvent', 'inaxes', 'key', 'lastevent', 'name', 'step'
        """
        canvas = event.canvas
        if canvas.manager.toolbar._active is None:
            print("pressed down at x={}, y={}".format(event.xdata, event.ydata))
            if event.inaxes:
                self._clicked_and_dragged.append([event.xdata, event.ydata])
        elif canvas.manager.toolbar._active in ("PAN", "ZOOM"):
            self._event_ax = event.inaxes
        return event

    def _on_release(self, event):
        ax = event.inaxes
        canvas = event.canvas
        if canvas.manager.toolbar._active is None:
            print("released at x={}, y={}".format(event.xdata, event.ydata))
            if (len(self._clicked_and_dragged)%2)==1: # odd number of entries in the _clicked_and_dragged list
                self._clicked_and_dragged.append([event.xdata, event.ydata])
            print()
        elif canvas.manager.toolbar._active in ("PAN", "ZOOM"):
            # if False:
            #     yticks = ax.get_yticks()
            #     ax.set_yticklabels(["{:.1f}".format(i) for i in np.sign(yticks)*(yticks)**2])
            # else:
            ylim = self._event_ax.get_ylim()
            yticks = np.linspace(*ylim, 10)
            yticks = round_to_nearest_sq_int(yticks)
            self._event_ax.set_yticks(yticks)
            self._event_ax.set_yticklabels(["{:d}".format(int(np.round(i))) for i in np.sign(yticks)*(yticks)**2])
            delattr(self, "_event_ax")
        return event

    def _setup_fig(self, fig):
        self.fig = fig
        self.on_press_connection = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.on_release_connection = self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _teardown_fig(self):
        self.fig.canvas.mpl_disconnect(self.on_press_connection)
        self.fig.canvas.mpl_disconnect(self.on_release_connection)
        delattr(self, "fig")
        delattr(self, "on_press_connection")
        delattr(self, "on_release_connection")

    def __del__(self):
        if hasattr(self, "fig"):
            try:
                self._teardown_fig()
            except:
                pass

    def add_resolution_coefficients(self, peak_min1, peak_max1, *peak2_minmax):
        """
        Given min-max energies of ONE or TWO peaks, generate the resolution curve's coefficients
        Resolution curve equation:
        FWHM = Full width half-maximum = FWHM_overall
        (unless otherwise specified, FWHM always refers to the overall FWHM)
        
             FWHM  √(A + B*E)
        R(E)=----= ----------
              E        E
        A ≈ 0.1-10
        B ≈ 0.001 - 0.01

        # we can explain this this A+B*E equation as follows:
        variance_{a+b} = variance_a + variance_b
        sigma_{a+b}^2  = sigma_a^2  + sigma_b^2
        (2.35 FWHM){a+b}^2 = (2.35 FWHM)a^2 + (2.35 FWHM)b^2
        FWHM_{a+b}^2   = FWHM_a^2   + FWHM_b^2
        FWHM_{total}^2 = FWHM_{Poisson}^2 + FWHM_{noise}^2

        FWHM_stat = FWHM_Possion : statiscial constribution
        FWHM_others : other constributions, including noise, drift, etc.

        ∵ FWHM_stat = √N = √(charge_of_pulse/e) # e = electron charge
        caveat : FWHM_stat_HPGe (for HPGe detector specifically) : F(Fano factor) * FWHM_stat
        ∵ FWHM_overall = √(FWHM_stat^2 + FWHM_others^2) # where others = constant
        ∴ FWHM_overall = √(  √(B*√E)^2 + A            )
        # in other words, if A and B are fitted correctly,
          B = (F * FWHM_stat            )^2
            = (F * √N                   )^2
            = (F * √(charge_of_pulse/e) )^2

          A = FWHM_others^2

        Paramters (all are float scalars)
        ---------
        peak_min1: left side of the first peak
        peak_min2: right side of the first peak
        peak2_minmax: if provided, expands to the left and right sides of the second peak.

        Returns
        -------
        resolution_coefficient : [A, B] in the equation above.
        """
        E1 = (peak_min1 + peak_max1)/2 # centroid energy for peak 1
        w1 = peak_max1 - peak_min1 # width of peak 1
        if len(peak2_minmax)==2:
            peak_min2, peak_max2 = peak2_minmax
            E2 = (peak_min2 + peak_max2)/2 # centroid 2
            w2 = peak_max2 - peak_min2 # width 2
            # solve by FWHM **2 = A + B*E:
            resolution_coeffs = (np.linalg.inv([[1, E1], [1, E2]]) @ ary([w1**2, w2**2]))
            # matrix inversion to find solution to  simultaneous equation
        else:
            # assume A = 0,
            resolution_coeff_2 = w1**2/E1
            # B = FWHM^2/B
            resolution_coeffs = ary([0, resolution_coeff_2]) # A is assumed zero by default
        self.resolution_coefficients = resolution_coeffs
        return self.resolution_coefficients

    def add_resolution_coefficients_interactively(self, plot_scale="sqrt"):
        """
        Plot the spectrum on a matplotlib figure, on which the user can click and drag to define one or two peaks.
        If >2 clicks were detected, then only the last two will made.

        Paramters
        ---------
        scale: scale to show the spectrum plot in. Options are: "sqrt" (default), "log".
        """
        print("Click and drag across the peak(s) (left to right) that you'd like to fit;")
        print("Only the last two click-and-dragged peaks will be used as the data.")
        self._clicked_and_dragged = [] # clear the list
        getattr(self, "show_{}_scale".format(plot_scale))()
        self.add_resolution_coefficients( *ary(self._clicked_and_dragged)[-4:, 0] )

    def get_width_at(self, peak_E):
        assert hasattr(self, "resolution_coefficients"), "Must run one of the add_resolution_coefficients* method first."

def round_to_nearest_sq_int(yticks):
    rounded_values = np.round(yticks).astype(int)

    sq_values = np.sign(yticks)*(yticks)**2
    sq_rounded_values = np.round(sq_values)
    sq_rounded_values_no_repeat = ary(sorted(set(sq_rounded_values)))
    return np.sign(sq_rounded_values_no_repeat) * sqrt(abs(sq_rounded_values_no_repeat))

def regex_num(line, dtype=int):
    return [dtype(i) for i in re.findall(r"[\w]+", line)]

def _to_datetime(line):
    """
    Convert .Spe file line metadata into date-time data.
    """
    month, day, year, hour, minute, second = regex_num(line, int)
    return dt.datetime(year, month, day, hour, minute, second)

if __name__=='__main__':
    spectrum = RealSpectrumInteractive.from_Spes(*sys.argv[1:])
    spectrum.show_sqrt_scale()