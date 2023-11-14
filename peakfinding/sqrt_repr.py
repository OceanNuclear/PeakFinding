from numpy import array as ary
import numpy as np
from matplotlib import pyplot as plt

__all__ = ["plot_sqrt", "plot_log"]

"""This is the question I want to be asking: why aren't we using the sqrt y style of plotting???"""


def plot_sqrt(*args, ax=None):
    """
    plot an n bin gamma spectrum in sqrt representation.
    Parameters
    ----------
    [optional] boundary_2d : numpy array of shape (n, 2) denoting the bin edges (upper edge, lower edge)
    counts : numpy array of shape (n) denoting the means
    ax: Axes on which to plot the graph if one already exists. Otherwise a new Axes will be created.
    rewrite_yticks: Whether to (True) rewrite the ylabels so they reflect counts;
                            or to (False) keep the y label as sqrt(counts).
                            Note that this option comes with the caveat of needing extra flux space.
    """
    assert len(args) <= 2, "Call signature = plot_sqrt([boundary_2d], counts, ax=None, rewrite_yticks=False)"
    if ax is None:
        ax = plt.subplot()

    has_x = len(args) == 2
    if has_x:
        boundary_2d, counts = args
        ax.set_xlabel(r"$E_{\gamma}$ (keV)")
    else:
        counts = args[0]
        boundary_2d = ary([np.arange(0, len(counts)), np.arange(1, len(counts) + 1)]).T
        ax.set_xlabel("bins")
    ax.set_yscale(
        "function",
        functions=[lambda x: np.sign(x) * np.sqrt(np.abs(x)), lambda x: np.sign(x) * np.square(np.abs(x))],
    )
    (line,) = ax.plot(boundary_2d.flatten(), np.repeat(counts, 2))
    ax.set_ylabel("counts")
    ax.set_title("sqrt-y plot of gamma spec")
    return ax, line


def plot_log(*args, ax=None):
    """
    plot an n-bin gamma spectrum in log-log representation
    ax: Axes on which to plot the graph if one already exists. Otherwise a new Axes will be created.
    """
    assert len(args) <= 2, "Call signature = plot_sqrt([boundary_2d], counts, ax=None, rewrite_yticks=False)"
    if ax is None:
        ax = plt.subplot()

    has_x = len(args) == 2
    if has_x:
        boundary_2d, counts = args
        ax.set_xlabel("gamma E (keV)")
    else:
        counts = args[0]
        boundary_2d = ary([np.arange(0, len(counts)), np.arange(1, len(counts) + 1)]).T
        ax.set_xlabel("bins")
    (line,) = ax.semilogy(boundary_2d.flatten(), np.repeat(counts, 2))
    ax.set_ylabel("counts")
    ax.set_title("log plot of gamma spec")
    return ax, line
