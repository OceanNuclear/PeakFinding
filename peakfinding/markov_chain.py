import numpy as np
from numpy import pi, sqrt, exp, array as ary, log as ln
tau = 2*pi
from numpy import cos, sin, arccos
from matplotlib import pyplot as plt
from tqdm import tqdm
from poisson_distribution import Poisson, Chi2
from sqrt_repr import plot_sqrt

chi2_1_cdf = Chi2(1).cdf

def slide_spectrum(counts, window_width):
    """
    horizontally slide the 
    Parameters
    ----------
    same as get_markov_matrix

    Returns
    -------
    spread_stack: array of shape (window_width-1, counts+window_width-1)
    for a window_size of 3, counts=c, spreadstack would be
    [
    [nan , c[0], c[1], ... , c[-2]],
    # [c[0], c[1], c[2], ... , c[-1]],
    [c[1], c[2], c[3], ... , nan ],
    ]
    Note that the middle line is exactly the same as the counts itself so it's been removed
    """
    excess = window_width//2
    spread_stack = [ [np.nan]*np.clip(-i, 0, None) + counts[np.clip(i, 0, None):len(counts)+i].tolist() + [np.nan]*np.clip(i, 0, None) for i in range(-excess, excess+1) ]
    collated_spectrum = ary(spread_stack[:excess] + spread_stack[-excess:])
    return ary(collated_spectrum)

def get_markov_matrix(counts, window_width):
    assert (window_width%2)==1, "Window size must be an odd integer >1."
    collated_spectrum = slide_spectrum(counts, window_width)
    _len = len(counts)

    transition_p_matrix = np.zeros((_len, _len))

    excess = window_width//2
    for ind, samples in tqdm(enumerate(collated_spectrum.T), total=len(collated_spectrum.T)):
        poisson_hypothesized = Poisson(counts[ind])

        negative_log_likelihood = poisson_hypothesized.negative_log_likelihood(samples)
        if poisson_hypothesized._lambda>0:
            probability_of_not_noise = chi2_1_cdf(negative_log_likelihood)
        else:
            # one for count>0 and zero for count==0.
            probability_of_not_noise = ary(negative_log_likelihood>0, dtype=float) 

        P_is_noise = (1 - probability_of_not_noise)
        voting_preference = P_is_noise/(2*excess)
        offset = (1 - np.nansum(voting_preference))/(2*excess)
        # evenly distribute the remaining transition probability using the offset
        transition_probabilities = voting_preference + offset

        left_transition_prob = transition_probabilities[np.clip(0, excess-ind, _len):excess]
        right_transition_prob = transition_probabilities[excess:np.clip(2*excess, 0, _len+excess-ind-1)]

        ind_left, ind_right = np.clip(ind-excess, 0, _len), np.clip(ind+excess+1, 0, _len)
        transition_p_matrix[ind, ind_left:ind] = left_transition_prob # probability of voting to the left
        # For this implementation, peaks will NOT hold onto their vote, unless they're edge cases
        transition_p_matrix[ind, ind] = 1 - left_transition_prob.sum() - right_transition_prob.sum() # probability of keeping the vote
        transition_p_matrix[ind, ind+1:ind_right] = right_transition_prob # probability of voting to the right

    # decreasing_arange = -np.arange(_len)
    # transition_p_matrix[np.diag_indices(_len)] = np.max([decreasing_arange+excess, decreasing_arange[::-1]+excess, np.zeros(_len)], axis=0)/(excess*2)
    return transition_p_matrix.T

visualization_function = lambda votes_held: np.tanh(votes_held)

if __name__=='__main__':
    import pandas as pd
    import sys
    import seaborn as sns
    spectrum = pd.read_csv(sys.argv[1], index_col=[0]).values.T
    E_l, E_u, counts = spectrum
    counts = ary(counts, dtype=int)
    E_bound = ary([E_l, E_u]).T
    transition_matrix = get_markov_matrix(counts, 21)
    
    window_width = 21

    import matplotlib.animation as manimation
    writer = manimation.writers['ffmpeg'](fps=3, metadata={"title":"markov chain peak identification algorithm", "comment":"re-invented by myself, by assuming even chance of being spread across", "artist":"Ocean Wong"})

    fig, (ax_u, ax_l) = plt.subplots(2,1, sharex=True)
    fig.suptitle("Markov Chain Peak identification")
    plot_sqrt(E_bound, counts, ax=ax_u)

    with writer.saving(fig, "../Video/"+"Markov_chain.mp4", 300):
        markov_probability = np.ones(len(counts))
        for _ in range(30):
            ax_l.plot(E_bound.flatten(), np.repeat(markov_probability, 2))
            ax_l.set_ylabel("Noisiness")
            ax_l.set_title("Likelihood of it being just noise")
            # ax_l.set_ylim(-0.05, 1.05)

            writer.grab_frame()
            ax_l.clear()
            markov_probability = transition_matrix @ markov_probability
    fig.clf()


    """
    Conclusion:
    This is a very weak method; needs at least 2000 iteration for a window size of 21. I won't be using it.
    This is the best that I can do.
    """