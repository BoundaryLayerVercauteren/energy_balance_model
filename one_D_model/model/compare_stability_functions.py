import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sdeint
import seaborn as sns

import one_D_model.model.solve_SDE_stoch_stab_function as stoch_stab_function

plt.style.use('science')

# Set font sizes for plots
SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def define_vandewiel_short_tail_stab_function(Ri, alpha=5):
    """Calculate value of short-tail stability function for given Richardson number (and alpha). The function was
    originally defined in Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature
    inversions: A conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073,
    https://doi.org/10.1175/JAS-D-16-0180.1.

    Args:
        Ri (float): Richardson number.
        alpha (float): Parameter for stability function.

    Returns:
        (float): value of short-tail stability function
    """
    return np.exp(-2 * alpha * Ri - (alpha * Ri) ** 2)


def define_vandewiel_long_tail_stab_function(Ri, alpha=5):
    """Calculate value of long-tail stability function for given Richardson number (and alpha). The function was
    originally defined in Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature
    inversions: A conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073,
    https://doi.org/10.1175/JAS-D-16-0180.1.

    Args:
        Ri (float): Richardson number.
        alpha (float): Parameter for stability function.

    Returns:
        (float): value of long-tail stability function
    """
    return np.exp(-2 * alpha * Ri)


def define_vandewiel_cutoff_stab_function(Ri, alpha=5):
    """Calculate value of cutoff stability function for given Richardson number (and alpha). The function was
    originally defined in Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature
    inversions: A conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073,
    https://doi.org/10.1175/JAS-D-16-0180.1.

    Args:
        Ri (float): Richardson number.
        alpha (float): Parameter for stability function.

    Returns:
        (float): value of cutoff stability function
    """
    cutoff = 1 / alpha
    if Ri <= cutoff:
        return 1 - 2 * alpha * Ri
    else:
        return 0.0


def make_comparison(sol_directory_path):
    """
    Create plot to compare different stablity functions.

    Args:
        sol_directory_path (str): Path were figure should be saved.
    """
    # Define range of Richardson numbers at which function will be evaluated
    richardson_num = np.round(np.linspace(0, 0.5, 100), 4)

    # Calculate stability function values for a range of Richardson numbers
    vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)
    vec_vandewiel_long_tail_stab_func = np.vectorize(define_vandewiel_long_tail_stab_function)
    vec_vandewiel_cutoff_stab_func = np.vectorize(define_vandewiel_cutoff_stab_function)

    # Create plot
    color = matplotlib.cm.get_cmap('cmc.batlow', 3).colors
    markers = ['v', '*', '^']

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(richardson_num, vec_vandewiel_short_tail_stab_func(richardson_num), label='short tail',
             color=color[0], marker=markers[0], markevery=10)
    ax1.plot(richardson_num, vec_vandewiel_long_tail_stab_func(richardson_num), label='long tail',
             color=color[1], marker=markers[1], markevery=10)
    # ax1.plot(richardson_num, vec_vandewiel_cutoff_stab_func(richardson_num), label='cutoff',
    #          color=color[2], marker=markers[2], markevery=100)

    ax1.set_xlabel(r'$R_b$')
    ax1.set_ylabel(r'$f_{stab}$')
    #ax1.set_xscale('log')

    plt.legend()

    plt.savefig(sol_directory_path + 'stability_functions.pdf', bbox_inches='tight', dpi=300)
