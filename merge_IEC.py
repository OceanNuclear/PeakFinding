from peakfinding.spectrum import RealSpectrumInteractive, RealSpectrum
import sys

print(*sys.argv[1:-1])
print(sys.argv[-1]+".Spe")
"""
Note: inside ipython, the using * (wildcard) in sys.argv will give an UNSORTED (disorderd!) list of the files grepped by wildcard.
But outside of ipython sys.argv will give a sorted sys.argv.
Therefore you're encouraged to not use * in ipython.
"""
spectrum = RealSpectrumInteractive.from_multiple_files(*sys.argv[1:-1])
spectrum.show_sqrt_scale()
spectrum.to_IEC(sys.argv[-1]+".IEC")