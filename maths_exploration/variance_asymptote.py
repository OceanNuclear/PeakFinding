from peakfinding.poisson_distribution import *
from numpy import exp

"""
Prints the mean and variance of sqrt(poisson) distribution.

"""

if __name__=='__main__':
    PLOT = False
    from matplotlib import pyplot as plt
    # from math import factorial
    # fac = lambda a: ary([factorial(i) for i in a])
    from scipy.special import factorial as fac

    # lambda: the free variable to change.
    N_max = 2000
    N = np.arange(N_max) # range of all natural numbers. 1000 is close enough to infinity.
    y = sqrt(N) # sqrt(counts) representation

    sqrt_lambda, mean, variance = [], [], []

    for lamb in (np.linspace(0, 10, 400)**2).tolist():
    # can't go far beyond 10 because that would distribute too much probability mass beyond N_max = 2000
        gaussian_mean = N_mean = gaussian_var = N_var = lamb
        
        # lamb = 20.0 # 1.2 # 2.0 # 0.5
        P_N_precursor = fac(N) * lamb**-N
        P_N = exp(-lamb) * np.nan_to_num(1/P_N_precursor, nan=0.0, posinf=0.0, neginf=0.0) # the pmf: probability mass function

        y_mean = (y * P_N).sum()
        # print(y_mean)
        y_var = ((y - y_mean)**2 * P_N).sum() # method 1 
        # y_var = (y**2 * P_N).sum() - y_mean**2 # method 2

        print("gaussian_mean = N_mean = gaussian_var = N_var =", lamb)
        print("y_mean, y_var =", y_mean, y_var)
        print("sqrt(lamb) =", sqrt(lamb))
        sqrt_lambda.append(sqrt(lamb))
        mean.append(y_mean)
        variance.append(y_var)

        if PLOT:
            ax = plt.subplot()
            bar_widths = np.diff(np.append(y, y[-1]))
            ax.bar(y, P_N, width=np.clip(0.2, None, bar_widths), align="edge", alpha=0.5, label="sqrt(counts)")

            saved_xmax = ax.get_xlim()[1]
            ax.bar(N, P_N, width=0.2, align="edge", alpha=0.5, label="counts")
            ax.set_xlim(0, np.clip(saved_xmax, N[(np.cumsum(P_N)>=0.999).argmax()+1], None)) # undo the expansion of the graph due to plotting the new pdf

            ax.set_xlabel("y values (sqrt(count) values)")
            ax.set_ylabel("probability")
            ax.legend()
            plt.show()
        print()

    # plt.plot(sqrt_lambda, mean, label="Mean(sqrt(counts))")
    plt.plot(mean, variance, label="vs mean(sqrt(counts))")
    plt.plot(sqrt_lambda, variance, label="vs sqrt(mean(counts))")
    plt.plot([min(sqrt_lambda), max(sqrt_lambda)], [0.25, 0.25], label="asymptote")
    plt.legend()
    plt.title("Variance variation wrt. sqrt(lambda)")
    plt.ylabel("Variance of distribution of sqrt(counts)")
    plt.xlabel("sqrt(counts)")
    # draw the asymptote: prove that it approaches the asymptote 
    plt.show()
"""
Conclusion:
y_var started at 0.3783934365408028 at lamb=0.1, and asymptotically approaches around 0.25
"""