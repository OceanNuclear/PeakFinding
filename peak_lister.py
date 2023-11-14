from peakfinding.curvature_detection import (
    RealSpectrumPeakFinderFromNoiseFloor,
    threshold_curve_function_generator,
)
import pandas as pd
from numpy import array as ary
import sys
import matplotlib.pyplot as plt
import numpy as np

assert not sys.argv[-1].endswith(
    ".Spe"
), "Must provide a file name (that doesn't include '.csv') as the output location to save the .csv to."
spectrum = RealSpectrumPeakFinderFromNoiseFloor.from_multiple_files(*sys.argv[1:-1])
spectrum.show_sqrt_scale()
print(
    "Type in the calibration coefficients (comma separated) ; or press enter to fit the peaks manually."
)
values = input()
spectrum.fwhm_cal = ary([0.3524835463387513, 0.002308561222141246])
# 1.5853449048830446, 0.0020776090022188573
# if values.strip()=="":
#     spectrum.fit_fwhm_cal_interactively()
# else:
#     spectrum.fwhm_cal = ary([float(i) for i in values.split(",")])


DEMONSTRATE_DECISION_MAKING = True
if DEMONSTRATE_DECISION_MAKING:
    pass
raise NotImplementedError(
    "Need to copy from /home/ocean/Documents/PhD/ExperimentalVerification/data/get_peaks.py"
)
while True:
    answer = input(
        "Acceptable? (y/n) (y=save to file {})\n".format(sys.argv[-1])
    ).lower()
    if answer == "y":
        df.to_csv(sys.argv[-1], index_label="bin")
        break
    elif answer == "n":
        pass
        break
