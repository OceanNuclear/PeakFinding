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

            else: # all other types of blocks
                if block_identifier in ("DATE_END_MEA", "DATE_MEA"):
                    init_dict[block_identifier.lower()] = _to_datetime(lines[1])
                else:
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

class RealSpectrumTunable(RealSpectrum):
    def __init__(self, counts, boundaries, bound_units, wall_time, **init_dict):
        super().__init__(counts, boundaries, bound_units, wall_time, **init_dict)
        assert self.bound_units=="keV"

    def plot_log_scale(self, ax=None, **kwargs):
        ax, line = super().plot_log_scale(ax, **kwargs)
        fig = ax.figure
        fig.canvas.mpl_connect("button_press_event", self.on_click)
        return ax, line

    def _on_click(self, ):

def regex_num(line, dtype=int):
    return [dtype(i) for i in re.findall(r"[\w]+", line)]

def _to_datetime(line):
    """
    Convert .Spe file line metadata into date-time data.
    """
    month, day, year, hour, minute, second = regex_num(line, int)
    return dt.datetime(year, month, day, hour, minute, second)

if __name__=='__main__':
    spectrum = RealSpectrum.from_Spes(*sys.argv[1:])
    ax, line = spectrum.plot_sqrt_scale()
    fig = ax.figure
    connection = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    fig.canvas.mpl_disconnect(connection)
