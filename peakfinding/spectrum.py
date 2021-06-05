import numpy as np
from numpy import pi, sqrt, exp, array as ary, log as ln
tau = 2*pi
from numpy import cos, sin, arccos
from matplotlib import pyplot as plt
import datetime as dt
import warnings, functools, os
import re
from operator import add as add
from collections import OrderedDict, namedtuple
from itertools import zip_longest
from dataclasses import dataclass

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
            old_yticks = ax.get_yticks()
            ax.set_yticklabels(np.sign(old_yticks) * old_yticks**2)
        else:
            ax.set_ylabel("sqrt(counts)")
        return ax, line

class TimeSeries(Histogram):
    def __add__(self, ts2):
        counts = np.hstack(self.counts, ts2.counts)
        boundaries = np.vstack(self.boundaries, ts2.boundaries)
        bound_units = self.bound_units
        return self.__class__(counts, boundaries, bound_units)

@dataclass
class IECPeak:
    """Data class storing one peak specified in an IEC file"""
    energy : float
    FWHM : float

def overwrite_protection(method_before_decoration):
    """
    Decorator to interact with the user already exists.
    """
    def method_after_decoration(self, *args, **kwargs):
        if os.path.exists(args[0]):
            print("File {} already exists!".format(args[0]))
            if input("Overwrite? (y/n)").lower()!="y":
                return None
        return method_before_decoration(self, *args, **kwargs)
    return method_after_decoration

class RealSpectrum(Histogram):
    def __init__(self, counts, boundaries, bound_units, live_time:float, **init_dict):
        """
        Creates a Real Spectrum, tailored to gamma spectra, but can be used for other particle count vs energy spectra.

        live_time: float scalar denoting the duration of this spectrum
        """
        super().__init__(counts, boundaries, bound_units)
        self.live_time = live_time # duration in seconds
        self.__dict__.update(init_dict)

    def __add__(self, rs2): # rs2 = RealSpectrum 2
        warnings.warn("Adding spectra together.\nMake sure you don't repeat the same spectrum, and add chronologically left-to-right.", UserWarning)
        init_dict = rs2.__dict__.copy() 
        init_dict.update(self.__dict__) # overwrite with properties of self whenever there's a duplication.
        init_dict.pop("counts")
        init_dict.pop("boundaries")
        init_dict.pop("bound_units")
        init_dict.pop("live_time")

        # defined in super().__init__:
        counts = rs2.counts + self.counts
        boundaries = self.boundaries
        bound_units = self.bound_units

        # special cases:
        live_time = self.live_time + rs2.live_time

        if "date_end_mea" in init_dict.keys():
            if hasattr(rs2, "date_end_mea"):
                init_dict["date_end_mea"] = ts2.date_end_mea
            else:
                init_dict["date_end_mea"] = None
        if "wall_time" in init_dict.keys():
            if hasattr(self, "wall_time") and hasattr(rs2, "wall_time"):
                init_dict["wall_time"] = self.wall_time + rs2.wall_time

        return self.__class__(counts, boundaries, bound_units, live_time, **init_dict)

    @staticmethod
    def calibration_equation(calibration_constants):
        return np.poly1d(calibration_constants[::-1])

    @classmethod
    def from_csv(cls, file_path):
        """
        Assume it reads from a file similar to spectrum*.csv in
        https://github.com/fispact/peakingduck/tree/master/reference
        i.e.
        channel,lenergy,uenergy,count
        0,0.00000000e+00,1.98091030e-01,0.00000000e+00
        1,1.98091030e-01,4.00114130e-01,0.00000000e+00
        2,4.00114130e-01,6.02137230e-01,0.00000000e+00
        ...
        So no metadata about the count time will be provided, and no metadata about 
        """
        import pandas as pd
        df = pd.read_csv(file_path, index_col=[0])
        boundaries = df[df.columns[:2]].values
        counts = df[df.columns[2]].values.astype(int)
        live_time = np.nan
        return cls(counts, boundaries, "keV", live_time)

    @classmethod
    def from_IEC(cls, file_path, read_calibration_equation=True):
        """
        Largely taken and adapted from Steve Bradnam's ADRIANA toolbox utility.
        Credits to him.
        I can't find the documentations for .IEC spectra file format standard in a reasonable time on Google.
        I give up and just followed the code
        """
        counts, channel_index = [], []
        sample_peaks_energy, sample_peaks_FWHM = [], []
        init_dict = {}
        with open(file_path) as f:

            READ_PEAKS, READ_INTO_DATA = False, False
            for lineno, line in enumerate(f):
                line = line[4:] # skip the first four letter because it's always A004.
                # I have literally no idea why it's that though.
                if lineno == 1:
                    live_time, wall_time, num_bins = [float(i) for i in line.split()]
                    # I assume wall_time is on the left and live_time is on the right.
                    num_bins = int(num_bins)
                    init_dict["wall_time"] = wall_time
                elif lineno == 2:
                    init_dict["date_mea"] = dt.datetime.strptime(line.strip(), "%d/%m/%y %H:%M:%S")
                elif lineno == 3:
                    init_dict["mca_cal"] = [float(i) for i in line.split()] # p0, p1, p2, p3
                elif lineno == 4:
                    init_dict["fwhm_cal"] = [float(i) for i in line.split()] # p0, p1, p2, p3

                # toggling "read-into" flags
                if "SPARE" in line.strip():
                    READ_PEAKS, READ_INTO_DATA = True, False
                    # These are probably just highlighted/fitted peaks that was noted on the program.
                    # Or some kind of counts vs t variation record.
                    # Or something else that I'm not entirely sure because I don't have the doucmetation for .IEC files.
                elif "USERDEFINED" in line.strip():
                    READ_PEAKS, READ_INTO_DATA = False, True

                elif READ_PEAKS:
                    data_in_line = [float(i) for i in line.split()]
                    sample_peaks_energy.extend(data_in_line[::2])
                    sample_peaks_FWHM.extend(data_in_line[1::2])
                elif READ_INTO_DATA:
                    # channel_index.append( int(line.split()[0]) )
                    counts.extend([ int(i) for i in line.split()[1:] ])

                elif line.strip=="": # toggle both to off when the block ends and we hit an empty line.
                    READ_PEAKS, READ_INTO_DATA = False, False

        init_dict["sample_peaks"] = [IECPeak(energy, FWHM) for energy, FWHM in zip(sample_peaks_energy, sample_peaks_FWHM)]
        counts = ary(counts[:num_bins], dtype=int)
        # channel_index = ary(channel_index) # not that useful TBH, just gonna ignore it.

        # default : saving as bin boundaries
        boundaries = ary([np.arange(num_bins), np.arange(1, num_bins+1)]).T
        bound_units = "bin"

        # if read_calibration_equation: convert the bin boundaries values into energy.
        if init_dict["mca_cal"]==[0, 0, 0, 0]:
            read_calibration_equation = False
        if read_calibration_equation:
            boundaries = cls.calibration_equation(init_dict["mca_cal"])(boundaries)
            bound_units = "keV"

        return cls(counts, boundaries, bound_units, live_time, **init_dict)

    @classmethod
    def from_Spe(cls, file_path):
        """
        Reads from an .Spe file and creates an object.
        """
        init_dict = {}
        # default unit is bins
        with open(file_path) as f:
            text = f.read()
        for block in text.split("$"):
            lines = block.split("\n") # remove the last \n
            block_identifier = lines[0].strip(":")

            if block_identifier.startswith("DATE"):
                init_dict[block_identifier.lower()] = _to_datetime(lines[1])

            elif block_identifier=="MEAS_TIM": # measurement time
                live_time, wall_time = [float(i) for i in lines[1].split()]
                init_dict["wall_time"] = wall_time

            elif block_identifier=="DATA": # All .Spe files must have the data block
                min_chan, max_chan = regex_num(lines[1], int)
                if min_chan!=0:
                    raise FutureWarning("lowest channel number isn't 0, this can cause error in the future!")
                counts = ary([float(i) for i in lines[2+min_chan:3+max_chan]], dtype=int)

            elif block_identifier in ("TUBE_CURRENT", "DATE_IRRAD"):
                int_vector = regex_num(line[1], int)
                init_dict[block_identifier.lower()] = int_vector

            elif block_identifier=="ENER_FIT": # energy fit (which is specific to maestro software, non-standard last I checked)
                init_dict["energy_fit"] = [float(i) for i in lines[1].split()]

            elif block_identifier.endswith("CAL"): # calibration
                num_constants = int(lines[1])
                calibration_constants = lines[2].split()
                init_dict[block_identifier.lower()] = [float(i) for i in calibration_constants[:num_constants]]
                if len(calibration_constants)>num_constants:
                    init_dict[block_identifier.lower() + "_unit"]= calibration_constants[num_constants:]

            elif len(block_identifier)>0: # all other types of blocks (except empty ones)
                init_dict[block_identifier.lower()] = "\n".join(lines[1:-1])
                # the last line is always an empty line, caused by splitting at "$" and then "\n",
                # which are often characters next to each other.

        bin_boundaries = ary([np.arange(max_chan+1), np.arange(1, max_chan+2)]).T
        # bin_boundaries = ary([np.arange(max_chan+1), np.arange(1, max_chan+2)]).T - 0.5
        # it's not clear whether we should subtract 0.5 to make the centroid of the first bin 0 or not.
        # I'll assume not.

        # check for the clibration equation
        if "mca_cal" in init_dict or "energy_fit" in init_dict:
            if "mca_cal" in init_dict:
                calibration_eqn = cls.calibration_equation(init_dict["mca_cal"])
            elif "energy_fit" in init_dict:
                calibration_eqn = cls.calibration_equation(init_dict["energy_fit"])
            boundaries = calibration_eqn(bin_boundaries)
            bound_units = init_dict.get("mca_cal_unit", ["keV",])[0]

        else:# no calibration equation available
            boundaries = bin_boundaries
            bound_units = "bins"
            
        return cls(counts, boundaries, bound_units, live_time, **init_dict)

    @classmethod
    def from_multiple_files(cls, *filenames):
        """
        Create a Spectrum object from one of the 3 accepted file types:
        .csv, .IEC, .Spe
        e.g. RealSpectrum.from_multiple_files("spectrum1.csv", "spectrum2.csv", "spectrum3.IEC")
        """
        spectra_created = [getattr(cls, "from_{}".format(fname.split(".")[-1]))(fname) for fname in filenames]
        return functools.reduce(add, spectra_created)

    @overwrite_protection
    def to_Spe(self, file_path, mimic_MAESTRO=True):
        """
        Documentation for the standards about the .Spe file is found here:
        https://inis.iaea.org/collection/NCLCollectionStore/_Public/32/042/32042415.pdf
        p.31

        This method is slightly bodged,
        as I forced a certain blocks to be written first when mimic_MAESTRO = True;
        but when mimic_MAESTRO = False I just write the blocks without order.
        The writing logic that handles writing different blocks can be found in the with statement after the else.

        Parameters
        ----------
        file_path: name of the output file (should end in .Spe)
        mimic_MAESTRO: boolean for whether the blocks should be arranged in exactly the same order as a MAESTRO .Spe file is.
        """
        # CAREFULLY write the top half of the file, which includes $MEA_TIM that needs to be treated carefully.
        if mimic_MAESTRO:
            self_dict = OrderedDict()
            with open(file_path, "w") as f:
                # $SPEC_ID
                f.write(_format_Spe_key("spec_id"))
                f.write(self.__dict__.get("spec_id", "")+"\n")
                # $SPEC_REM
                f.write(_format_Spe_key("spec_rem"))
                f.write(self.__dict__.get("spec_rem", "\n\n")+"\n")
                # $DATE_MEA
                f.write(_format_Spe_key("date_mea"))
                f.write(_format_Spe_DATE(self.__dict__.get("date_mea", dt.datetime.now())) )
                # $MEAS_TIM
                live_time, wall_time = self.__dict__.get("live_time", np.nan), self.__dict__.get("wall_time", np.nan)
                f.write("$MEAS_TIM:\n{} {}\n".format(live_time, wall_time))

            # all other keys that are also block_identifier used in a MAESTRO .Spe file
            other_MAESTRO_keys = ["counts", "roi", "presets", "ener_fit", "mca_cal", "shape_cal"]
            for key in other_MAESTRO_keys:
                if hasattr(self, key):
                    self_dict[key] = self.__dict__[key]

            # update all other keys
            for key in self.__dict__.keys():
                # be careful not to include the keys corresponding to blocks that are already written.
                if key not in ("live_time", "wall_time", "spec_rem", "spec_id", "date_mea"):
                    self_dict[key] = self.__dict__[key]
        else:
            # just dump in every attribute without caring about the order.
            self_dict = self.__dict__
            # clear the file
            with open(file_path, "w") as f:
                pass # write nothing, empty file.
        # handle every allowed block under the sun. (or rather, under the IAEA convention (aforementioned file standard))
        with open(file_path, "a+") as f:
            # the first thing is to write measurement time, because this one is hard to handle.
            if "wall_time" in self.__dict__ and "live_time" in self_dict:
                f.write("$MEAS_TIM:\n{} {}\n".format(self.wall_time, self.live_time))

            for k, v in self_dict.items():
                if k in ("wall_time", "live_time", "boundaries", "bound_units"):
                    continue # live/wall time has already been taken care of;
                    # and boundaries (and units) aren't recorded in the .Spe files to begin with.

                elif k.startswith("date_"): # all date format metadata
                    f.write(_format_Spe_key(k))
                    f.write(_format_Spe_DATE(v))

                elif k == "counts": # data
                    f.write(_format_Spe_key("data"))
                    f.write("{} {}\n".format(0, len(v)-1))
                    counts_max_width = max(len(str(i)) for i in v)
                    f.write("\n".join(str(i).rjust(max(8, counts_max_width)) for i in v)+ "\n")

                elif k == "energy_fit": # non-standard energy calibration constant used by Maestro
                    f.write("$ENER_FIT\n")
                    f.write(_format_Spe_CAL(v)[2:]+ "\n") # doesn't need the "2\n" at the beginning.

                elif k.endswith("cal"): # all standard calibration coefficients
                    f.write(_format_Spe_key(k))
                    f.write(_format_Spe_CAL(v)+ "\n")

                elif not k.startswith("_"): # all other text information
                    if isinstance(v, list):
                        f.write(_format_Spe_key(k))
                        f.write(" ".join(map(str, v))+ "\n")
                    elif isinstance(v, str):
                        f.write(_format_Spe_key(k))
                        f.write(v+ "\n")
                    else:
                        warnings.warn("Attribute {} is unexpected and thus is ignored.".format(k), RuntimeWarning)

    @overwrite_protection
    def to_csv(self, file_path):
        import pandas as pd
        df = pd.DataFrame(ary([self.counts, *self.boundaries.T]).T, columns=["lenergy", "uenergy", "count"])
        df.to_csv(file_path, index_label="channel")
        return

    @overwrite_protection
    def to_IEC(self, file_path):
        with open(file_path, "w") as f:
            # header
            f.write(_format_IEC_line("     peakfinding   1   1     0"))
            # live, real time and number of channels
            f.write(_format_IEC_line("{:14}{:14}{:6}".format(
                self.__dict__.get("wall_time", np.nan),
                self.__dict__.get("live_time", np.nan),
                len(self.counts)
                ))
            )
            # date of acquisition
            f.write(_format_IEC_line(self.__dict__.get("date_mea", dt.datetime.now()).strftime("%d/%m/%y %H:%M:%S")))
            # energy calibration coefficients
            if hasattr(self, "mca_cal") and len(self.mca_cal)>0:
                f.write(_format_IEC_vector_line([*self.mca_cal, *[0 for _ in range( 4-len(self.mca_cal)) ]], 14))
            else:
                f.write(_format_IEC_line()) #otherwise write an empty line
            # FWHM calibration coefficients
            if hasattr(self, "fwhm_cal") and len(self.fwhm_cal)>0:
                f.write(_format_IEC_vector_line([*self.fwhm_cal, *[0 for _ in range( 4-len(self.fwhm_cal) )]], 14)[:-5]+"1    \n")
            else:
                f.write(_format_IEC_line()) #otherwise write an empty line
            # 4 empty lines
            f.write(_format_IEC_line())
            f.write(_format_IEC_line())
            f.write(_format_IEC_line())
            f.write(_format_IEC_line())
            f.write(_format_IEC_line("SPARE"))
            num_peak_lines = -1
            # write the sample peaks that was previously recorded
            for num_peak_lines, (peak1, peak2) in enumerate(
                zip_longest(self.__dict__.get("sample_peaks", [])[::2], self.__dict__.get("sample_peaks", [])[1::2], 
                fillvalue=IECPeak(0, 0) )):
                f.write(_format_IEC_vector_line([peak1.energy, peak1.FWHM, peak2.energy, peak2.FWHM], 16))
            while num_peak_lines<35:
                f.write(_format_IEC_vector_line([0, 0, 0, 0], 16))
                num_peak_lines += 1
            # 11 empty lines
            for _ in range(11):
                f.write(_format_IEC_line())
            # data block
            f.write(_format_IEC_line("USERDEFINED"))
            for data_line_no, vec5 in enumerate(zip_longest(
                                        self.counts[::5],
                                        self.counts[1::5],
                                        self.counts[2::5],
                                        self.counts[3::5],
                                        self.counts[4::5],
                                        fillvalue=0)):
                f.write(_format_IEC_data_line(data_line_no, vec5))
        return

def _format_IEC_line(text=" "*64):
    return ("A004"+text).ljust(68)+"\n"

def _format_IEC_vector_line(vec, num_space_per_element):
    return _format_IEC_line("".join(str(e).rjust(num_space_per_element) for e in vec))

def _format_IEC_data_line(dat_line_no, vec5):
    return "A004{:6d}{:10d}{:10d}{:10d}{:10d}{:10d}   \n".format(5*dat_line_no, *vec5)

def _format_Spe_key(key):
    return "${}:\n".format(key.upper())

def _format_Spe_CAL(coefficients):
    formatter_str = "{}\n{}"
    return formatter_str.format(len(coefficients), " ".join(map(str, coefficients)))

def _format_Spe_DATE(date):
    return date.strftime("%D %H:%M:%S\n")

class RealSpectrumInteractive(RealSpectrum):
    def __init__(self, counts, boundaries, bound_units, live_time, **init_dict):
        super().__init__(counts, boundaries, bound_units, live_time, **init_dict)
        self._clicked_and_dragged = []

    def show_log_scale(self, ax=None, execute_before_showing=None, **kwargs):
        """
        Parameters
        ----------
        ax : an optional matpotlib axes object to draw over
        execute_before_showing : a function to execute before showing.
                                must have no calls ignature of NO parameters.
        Returns
        -------
        nothing, as this method only returns when we close the plot
        """
        ax, line = super().plot_log_scale(ax=ax, **kwargs)

        if callable(execute_before_showing): # if this is a function
            execute_before_showing()
        self._setup_fig(ax.figure, False)
        plt.show()
        self._teardown_fig()
        return

    def show_sqrt_scale(self, ax=None, execute_before_showing=None, **kwargs):
        """
        Connect button clicks on the plot to other useful stuff.
        Parameters
        ----------
        ax : an optional matpotlib axes object to draw over
        execute_before_showing : a function to execute before showing.
                                must have no calls ignature of NO parameters.
        an example execute_before_showing function is as follows:
            def func():
                ax.set_title("Some title")
                ax.plot([-1, 1000], [0.9, 0.9])
                return

        Returns
        -------
        nothing, as this method only returns when we close the plot.
        """

        ax, line = super().plot_sqrt_scale(ax=ax, rewrite_yticks=False, **kwargs)

        # functionality specific to sqrt_scale
        self.fig = ax.figure
        ax.set_ylabel("counts")
        yticks = ax.get_yticks()
        yticks = round_to_nearest_sq_int(yticks)
        ax.set_yticklabels("{:d}".format(int(np.round(i))) for i in np.sign(yticks)*(yticks)**2)

        if callable(execute_before_showing): # if this is a function
            execute_before_showing()        
        self._setup_fig(ax.figure, True)
        plt.show()
        self._teardown_fig()
        return

    def _on_press(self, event):
        """
        'event' contains information about the button press down event.
            Other usable attributes about the events are:
            'button', 'canvas', 'dblclick', 'guiEvent', 'inaxes', 'key', 'lastevent', 'name', 'step'
        """
        canvas = event.canvas
        toolbar = canvas.manager.toolbar
        # There are two versions of NavigationToolbar2,
        # The latter version of which       denotes cursor state as toolbar.mode="";
        # while the former version of which denotes cursor state as toolbar._active=None.
        is_cursor, is_zoom_pan = check_matplotlib_toolbar_tool_used(toolbar)

        if is_cursor:
            if event.inaxes:
                # print and store the within-bound clicks
                print("pressed down at x={}, y={}".format(event.xdata, event.ydata))
                self._clicked_and_dragged.append([event.xdata, event.ydata])
        elif is_zoom_pan:
            # store the axis where the press-down event happened
            self._event_ax = event.inaxes
        return event

    def _on_release_callback_function_generator(self, sqrt_scale):
        """
        Function factory, creates an _on_release function which is complementary to _on_press.
        """
        def _on_release(event):
            """
            Rewrite y-axis upon self.event if it's a zoom/pan event.
            Minor issue may arise if the user decides to begin the click from within bounds, 
                but then release the click from out-of-bounds.
                This means a [None, None] will be recorded as the location of the release event,
                which will get translated into float(nan).
                This will lead to the function being unable to fit the values.
            """
            ax = event.inaxes
            canvas = event.canvas
            toolbar = canvas.manager.toolbar
            is_cursor, is_zoom_pan = check_matplotlib_toolbar_tool_used(toolbar)

            if is_cursor:
                if (len(self._clicked_and_dragged)%2)==1:
                    # odd number of entries in the _clicked_and_dragged list means the press-down event for the same button click happened within bounds.
                    print("released at x={}, y={}".format(event.xdata, event.ydata))
                    self._clicked_and_dragged.append([event.xdata, event.ydata])
                print() # print an empty line to signal to the user that the click has finished and is detected, even if it's not within bounds.
            elif is_zoom_pan:
                # If the click was started within the axes
                if sqrt_scale:
                    # calculate the y-ticks location and un-sqrt their values.
                    ylim = self._event_ax.get_ylim()
                    ylim_range = np.diff(ylim)[0]
                    yticks = np.linspace(ylim[0]+ylim_range*0.02, ylim[1]-ylim_range*0.02, 10)
                    yticks = round_to_nearest_sq_int(yticks)
                    self._event_ax.set_yticks(yticks)
                    self._event_ax.set_yticklabels("{:d}".format(int(np.round(i))) for i in np.sign(yticks)*(yticks)**2)
                delattr(self, "_event_ax")
            return event
        return _on_release

    def _setup_fig(self, fig, sqrt_scale=False):
        self.fig = fig
        self.on_press_connection = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        _on_release = self._on_release_callback_function_generator(sqrt_scale)
        self.on_release_connection = self.fig.canvas.mpl_connect("button_release_event", _on_release)

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

    def add_fwhm_cal(self, peak_min1, peak_max1, *peak2_minmax):
        """
        Given min-max energies of ONE or TWO peaks, generate the FWHM curve's coefficients
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
        peak2_minmax: if provided, expands to the left and right sides of the second (or more) peak(s).

        Returns
        -------
        fwhm_coefficient : [A, B] in the equation above.
        """
        E1 = (peak_min1 + peak_max1)/2 # centroid energy for peak 1
        w1 = abs(peak_max1 - peak_min1) # width of peak 1
        if len(peak2_minmax)==0:
            print("1 peak was located. Fully-determined fitting 1 DoF (FWHM=sqrt(B*E))...")
            # assume A = 0,
            fwhm_coeff_2 = w1**2/E1
            # B = FWHM^2/B
            fwhm_coeffs = ary([0, fwhm_coeff_2]) # A is assumed zero by default
        elif len(peak2_minmax)==2:
            print("2 peak were located. Fully-determined fitting 2 DoF (FWHM=sqrt(A+B*E))...")
            peak_min2, peak_max2 = peak2_minmax
            E2 = (peak_min2 + peak_max2)/2 # centroid 2
            w2 = abs(peak_max2 - peak_min2) # width 2
            # solve by FWHM **2 = A + B*E:
            fwhm_coeffs = (np.linalg.inv([[1, E1], [1, E2]]) @ ary([w1**2, w2**2]))
            # matrix inversion to find solution to  simultaneous equation
        else:
            print("3+ peak were located. Over-determined fitting 1 DoF (FWHM=sqrt(A+B*E))...")
            # Use more than 2 data point to perform 2 degrees of freedom fitting.
            # i.e. overdetermined fitting.

            # assert statement
            template = "\nleft-edge x value of peak {num}, right edge x of peak {num}, "
            explanation = template.format(num=1)+ template.format(num=2)
            assert len(peak2_minmax)%2==0, "Must enter peak limits in pairs:({}\netc.)".format(explanation.format())

            # get the peak_limits array, shape it into (-1, 2)
            peak_limits = np.insert(peak2_minmax, 0, [peak_min1, peak_max1])
            peak_limits = peak_limits.reshape([-1, 2])

            # get the w vector and E vector, both of which are 1D
            w = abs(peak_limits[:, 1] - peak_limits[:, 0])
            E = peak_limits.mean(axis=1)

            # perform pseudo-inversion
            invertible_matrix = ary([np.ones(len(E)), E]).T
            fwhm_coeffs = np.linalg.pinv(invertible_matrix) @ w

        self.fwhm_cal = fwhm_coeffs
        return self.fwhm_cal

    def get_width_at(self, E):
        assert hasattr(self, "fwhm_cal"), "Must run one of the add_fwhm_cal* method first."
        inside_sqrt_func = np.poly1d(self.fwhm_cal[::-1])
        width_at_E = sqrt(inside_sqrt_func(E))
        return width_at_E

    def add_fwhm_cal_interactively(self, plot_scale="sqrt"):
        """
        Plot the spectrum on a matplotlib figure, on which the user can click and drag to define one or two peaks.
        If >2 clicks were detected, then only the last two will made.

        Paramters
        ---------
        scale: scale to show the spectrum plot in. Options are: "sqrt" (default), "log".
        """
        print("Click and drag across the peak(s) that you'd like to fit;")
        print("Only the last two click-and-dragged peaks will be used as the data.")
        while True:
            self._clicked_and_dragged = [] # clear the list
            print("\nPlease click and drag across at least one peak:")
            ax = plt.subplot()
            ax.set_title("Drag cursor across 1 or 2 peaks (left to right)")
            getattr(self, "show_{}_scale".format(plot_scale))(ax=ax)
            x_coordinates = ary(self._clicked_and_dragged, dtype=float).reshape([-1,2])[:, 0] # select the x coordinates
            mouse_press_down_up_pair = x_coordinates.reshape([-1, 2])
            valid_x_coords = mouse_press_down_up_pair[np.isfinite(mouse_press_down_up_pair).all(axis=1)]
            
            # isfinite handle the cases that we don't have enough
            if len(valid_x_coords)==0:
                print("Not enough/invalid peaks selected! Please try again:")
            else:
                break # exit the loop

        self.add_fwhm_cal( *valid_x_coords.flatten() )
        print("FWHM coefficients are found as", ", ".join(map(str, self.fwhm_cal)))

    def get_windows(self):
        """
        Uses the FWHM equation to calculate the width of peak to be expected at each energy,
        thus returning the window that fits the various widths 

        Returns
        -------
        a mask of shape [len(self.counts), len(self.counts)]
        """
        assert hasattr(self, "fwhm_cal"), "Must run one of the add_fwhm_cal* method first before we can calculate the window sizes."
        E_mid = self.boundaries.mean(axis=1)

        windows = []
        for mid_E in self.boundaries.mean(axis=1): # calculate the mean energy of that bin.
            lower_lim, upper_lim = mid_E - self.get_width_at(mid_E)/2, mid_E + self.get_width_at(mid_E)/2
            above_lower_lim = (self.boundaries>=lower_lim).any(axis=1)
            below_upper_lim = (self.boundaries<=upper_lim).any(axis=1)
            window = np.logical_and(above_lower_lim, below_upper_lim)
            windows.append(window)
        return ary(windows)

mpl_active_tool = namedtuple("mpl_active_tool", ("is_cursor", "is_zoom_pan"))
def check_matplotlib_toolbar_tool_used(toolbar):
    if hasattr(toolbar, "_active"):
        is_cursor = toolbar._active is None
        is_zoom_pan = toolbar._active in ("PAN", "ZOOM")
    elif hasattr(toolbar, "mode"):
        is_cursor = toolbar.mode == ""
        is_zoom_pan = "zoom" in toolbar.mode
    # should only raise an error if check_matplotlib_toolbar_tool_used() is broken.
    assert (int(is_cursor) + int(is_zoom_pan)) == 1, "Expected the tool to either be cursor, pan, or zoom."
    return mpl_active_tool(is_cursor, is_zoom_pan)

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