"""Script to compare  deterministic stability function (see section 2.2.)."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from one_D_model.utils.set_plotting_style import configure_plotting_style


# Set plotting style
configure_plotting_style('full_page_width')


def define_vandewiel_short_tail_stab_function(Ri, alpha=5):
    """Calculate value of short-tail stability function for given Richardson number (and alpha). The function was
    originally defined in Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature
    inversions: A conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073,
    https://doi.org/10.1175/JAS-D-16-0180.1."""
    return np.exp(-2 * alpha * Ri - (alpha * Ri) ** 2)


def define_vandewiel_long_tail_stab_function(Ri, alpha=5):
    """Calculate value of long-tail stability function for given Richardson number (and alpha). The function was
    originally defined in Van de Wiel, B. J. H., and Coauthors, 2017: Regime transitions in near-surface temperature
    inversions: A conceptual model. Journal of the Atmospheric Sciences, 74, 1057–1073,
    https://doi.org/10.1175/JAS-D-16-0180.1."""
    return np.exp(-2 * alpha * Ri)


def make_comparison(sol_directory_path):
    """Create plot to compare different stability functions."""
    # Define range of Richardson numbers at which function will be evaluated
    richardson_num = np.round(np.linspace(0, 0.5, 100), 4)

    # Calculate stability function values for a range of Richardson numbers
    vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)
    vec_vandewiel_long_tail_stab_func = np.vectorize(define_vandewiel_long_tail_stab_function)

    # Create plot
    color = matplotlib.cm.get_cmap('cmc.batlow', 3).colors
    markers = ['v', '*', '^']

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(richardson_num, vec_vandewiel_short_tail_stab_func(richardson_num), label='short tail',
             color=color[0], marker=markers[0], markevery=10)
    ax1.plot(richardson_num, vec_vandewiel_long_tail_stab_func(richardson_num), label='long tail',
             color=color[1], marker=markers[1], markevery=10)

    ax1.set_xlabel(r'$R_b$')
    ax1.set_ylabel(r'$f_{stab}$')

    plt.legend()

    plt.savefig(sol_directory_path + 'stability_functions.pdf', bbox_inches='tight', dpi=300)
