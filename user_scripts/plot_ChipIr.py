from peakfinding.spectrum import RealSpectrumInteractive, RealSpectrum
import sys
import matplotlib.pyplot as plt

if __name__=='__main__':
    spectrum = RealSpectrum.from_multiple_files(*sys.argv[1:-1])
    # spectrum.to_Spe(sys.argv[-1]+".Spe")
    ax, line = spectrum.plot_sqrt_scale()
    ax.set_xlim(20, 2922)
    #hard coded magic number for one specific script; but I'll accept it because this is a script
    ax.figure.set_size_inches(11, 4)
    ax.set_title(sys.argv[-1].split("/")[-1] + " "+ str(spectrum.wall_time)
        + "s\nstarted at " + spectrum.date_mea.strftime("%Y-%m-%d %H:%M:%S"))
    plt.savefig(sys.argv[-1]+".png", dpi=200, bbox_inches=None)
    ax.figure.set_size_inches(24, 18)
    plt.savefig(sys.argv[-1]+".svg", dpi=200, bbox_inches=None)