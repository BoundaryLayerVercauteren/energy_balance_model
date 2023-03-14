import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sdeint
from scipy.stats import norm

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
    """Note the parameters for phi are estimated from data based on a sampling frequency with units cycles/hour.
    Therefore, they need to be transformed with 1/3600 to have the resolution in seconds."""
    phi_0 = 1 / define_vandewiel_short_tail_stab_function(Ri_span[0])

    t_span = np.linspace(0.0, 1000.0, 100)
    sim_span = np.arange(0, 1000, 1)

    solution = np.empty((len(sim_span), len(t_span)))
    expected_value = np.empty((len(Ri_span), len(t_span)))

    for Ri_idx, Ri in enumerate(Ri_span):
        for sim_idx in sim_span:
            # Define functions for SDE
            _f = lambda phi, t: define_SDE_stoch_stab_function(phi, Ri)
            _G = lambda phi, t: (1 / np.sqrt(3600)) * (stoch_stab_function.sigma(Ri, 0) * phi)
            solution[sim_idx, :] = sdeint.itoint(_f, _G, phi_0, t_span)[:, 0]

        expected_value[Ri_idx, :] = np.mean(solution, axis=0)

    return expected_value, t_span


def define_poisson_stab_function(Ri_range, critical_Ri=0.25):
    # Define Poisson process
    poisson_process = np.random.poisson(lam=1.0, size=len(Ri_range)) * 0.1
    # Define list to hold results for stability function
    stab_function = []
    # Calculate stochastic stability function for all Richardson numbers
    for idx, Ri_val in enumerate(Ri_range):
        if Ri_val <= critical_Ri:
            stab_function.append(define_vandewiel_short_tail_stab_function(Ri_val))
        else:
            stab_function_val = define_vandewiel_short_tail_stab_function(Ri_val) + poisson_process[idx]
            # Limit stability function to 1
            if stab_function_val > 1:
                stab_function_val = 1
            stab_function.append(stab_function_val)
    return stab_function


def make_comparison(params):
    richardson_num = np.linspace(10 ** (-5), 10 ** (1), 100)

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

    # stoch_stab_function_exp_values, time = solve_SDE_stoch_stab_function(richardson_num)
    vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)
    # vandewiel_short_tail = np.tile(vec_vandewiel_short_tail_stab_func(richardson_num), (len(time), 1))
    #
    # X, Y = np.meshgrid(richardson_num, time)
    #
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 5))
    #
    # img = ax.plot_surface(X, Y, stoch_stab_function_exp_values.T, alpha=0.8, cmap=matplotlib.cm.coolwarm)
    # ax.plot_surface(X, Y, vandewiel_short_tail, alpha=0.8, color='green')
    #
    # fig.colorbar(img)
    #
    # ax.set_xlabel('Ri')
    # ax.set_ylabel('time')
    # ax.set_zlabel(r'$\phi$')

    # plt.savefig(params.sol_directory_path + 'stoch_stability_function_mean.png', bbox_inches='tight', dpi=300)

    # fig, ax = plt.subplots(figsize=(5, 5))
    # # ones = np.repeat(1.0, len(richardson_num))
    # # inverse_vandewiel_short_tail = vec_vandewiel_short_tail_stab_func(richardson_num)
    # # ax.plot(richardson_num, np.divide(ones, inverse_vandewiel_short_tail, out=np.zeros_like(ones), where=inverse_vandewiel_short_tail!=0), label='short tail')
    # ax.plot(richardson_num, vec_vandewiel_short_tail_stab_func(richardson_num), label='short tail')
    # ax.plot(richardson_num, 1/stoch_stab_function_exp_values[:, -1], label='stoch. stab. function ')
    # # ax.plot(richardson_num, 1 / stoch_stab_function_exp_values[:, -2], label='stoch. stab. function')
    # # ax.plot(richardson_num, 1 / stoch_stab_function_exp_values[:, -3], label='stoch. stab. function')
    # ax.set_xscale('log')
    # plt.legend()
    # ax.set_xlabel('Ri')
    #
    # plt.savefig(params.sol_directory_path + 'stoch_stability_function_mean_last.png', bbox_inches='tight', dpi=300)

    vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(richardson_num, vec_vandewiel_short_tail_stab_func(richardson_num), label='short tail')
    ax.plot(richardson_num, define_poisson_stab_function(richardson_num), label='stoch. stab. function')
    #ax.set_xscale('log')
    plt.legend()
    ax.set_xlabel('Ri')
    plt.savefig(params.sol_directory_path + 'stoch_stability_function_poisson.png', bbox_inches='tight', dpi=300)

    # Get mean over several simulations of stab. function
    num_sim = 1000
    stoch_stab_res = np.zeros((num_sim, len(richardson_num)))
    for sim_idx in range(0,num_sim):
        stoch_stab_res[sim_idx,:] = define_poisson_stab_function(richardson_num)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(stoch_stab_res.flatten())
    ax.set_xlabel('f')
    plt.savefig(params.sol_directory_path + 'stoch_stability_function_hist_poisson.png', bbox_inches='tight', dpi=300)