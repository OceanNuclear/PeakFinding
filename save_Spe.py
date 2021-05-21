from peakfinding.spectrum import RealSpectrumInteractive, RealSpectrum
import sys

if __name__=='__main__':
    spectrum = RealSpectrum.from_Spes(*sys.argv[1:-1])
    spectrum.to_Spe(sys.argv[-1]+".Spe")
