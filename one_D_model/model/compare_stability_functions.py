import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sdeint

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


def define_SDE_stoch_stab_function(phi_stoch, Ri):
    return (1 / 3600) * (
            1 + stoch_stab_function.kappa(Ri) * phi_stoch - stoch_stab_function.upsilon(Ri) * phi_stoch ** 2)


def solve_SDE_stoch_stab_function(Ri_span):
    phi_0 = 1 / define_vandewiel_short_tail_stab_function(Ri_span[0])

    t_span = np.linspace(0.0, 100.0, 100)
    sim_span = np.arange(0, 100, 1)

    solution = np.empty((len(sim_span), len(t_span)))
    expected_value = np.empty((len(Ri_span), len(t_span)))
    pdf = np.empty((len(Ri_span), len(t_span)))

    for Ri_idx, Ri in enumerate(Ri_span):
        for sim_idx in sim_span:
            # Define functions for SDE
            _f = lambda phi, t: define_SDE_stoch_stab_function(phi, Ri)
            _G = lambda phi, t: (1 / np.sqrt(3600)) * (stoch_stab_function.sigma(Ri, 0) * phi)
            solution[sim_idx, :] = 1/sdeint.itoint(_f, _G, phi_0, t_span).flatten()
        #pdf[Ri_idx, :] = norm.pdf(solution)
        expected_value[Ri_idx, :] = np.min(solution, axis=0)

    return expected_value, t_span


def make_comparison(params):
    richardson_num = np.linspace(10 ** (-4), 10 ** (2), 100)

    # vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)
    # vec_vandewiel_long_tail_stab_func = np.vectorize(define_vandewiel_long_tail_stab_function)
    # #vec_vandewiel_cutoff_stab_func = np.vectorize(define_vandewiel_cutoff_stab_function)
    #
    # # Create plot
    # color = matplotlib.cm.get_cmap('cmc.batlow', 6).colors
    # markers = ['v', '*', '^', 's', 'p', '.']
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax1 = fig.add_subplot(1, 1, 1)
    #
    # ax1.plot(richardson_num, vec_vandewiel_short_tail_stab_func(richardson_num), label='short tail',
    #          color=color[0], marker=markers[0], markevery=10)
    # ax1.plot(richardson_num, vec_vandewiel_long_tail_stab_func(richardson_num), label='long tail',
    #          color=color[1], marker=markers[1], markevery=10)
    # # ax1.plot(richardson_num, vec_vandewiel_cutoff_stab_func(richardson_num), label='cutoff',
    # #          color=color[2], marker=markers[2], markevery=100)
    # richardson_num=richardson_num.reshape(len(richardson_num),1)
    # #ax1.plot(np.tile(richardson_num, (1, np.shape(stoch_stab_function_values)[1])), stoch_stab_function_values, label='1/stochastic', markevery=100) #, color=color[2], marker=markers[2]
    #
    # ax1.set_xlabel('Ri')
    # ax1.set_ylabel(r'$f_{stab}$')
    # ax1.set_xscale('log')
    #
    # plt.legend()
    #
    # plt.savefig(params.sol_directory_path + 'stability_functions.png', bbox_inches='tight', dpi=300)

    stoch_stab_function_exp_values, time = solve_SDE_stoch_stab_function(richardson_num)
    vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)
    vandewiel_short_tail = np.tile(vec_vandewiel_short_tail_stab_func(richardson_num), (len(time), 1))

    X, Y = np.meshgrid(richardson_num, time)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))

    img = ax.plot_surface(X, Y, stoch_stab_function_exp_values.T, alpha=0.8, cmap=matplotlib.cm.coolwarm)
    ax.plot_surface(X, Y, vandewiel_short_tail, alpha=0.8, color='green')

    fig.colorbar(img)

    ax.set_xlabel('Ri')
    ax.set_ylabel('time')
    ax.set_zlabel(r'$\phi$')

    plt.savefig(params.sol_directory_path + 'stoch_stability_function_min.png', bbox_inches='tight', dpi=300)
