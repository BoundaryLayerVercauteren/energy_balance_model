import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sdeint

import one_D_model.model.solve_SDE_stoch_stab_function as stoch_stab_function

plt.style.use('science')

# set font sizes for plots
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
    return np.exp(-2 * alpha * Ri - (alpha * Ri) ** 2)


def define_vandewiel_long_tail_stab_function(Ri, alpha=5):
    return np.exp(-2 * alpha * Ri)


def define_vandewiel_cutoff_stab_function(Ri, alpha=5):
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

    t_span = np.linspace(0.0, 1.0, 2)

    solution = np.zeros((len(Ri_span),10))

    for sim_idx in np.arange(0,10,1):
        for idx, Ri in enumerate(Ri_span):
            # Define functions for SDE
            _f = lambda phi, t: define_SDE_stoch_stab_function(phi, Ri)
            _G = lambda phi, t: (1 / np.sqrt(3600)) * (stoch_stab_function.sigma(Ri,0) * phi)
            solution[idx,sim_idx] = sdeint.itoint(_f, _G, phi_0, t_span)[1]

    return solution


def make_comparison(params):
    richardson_num = np.linspace(10**(-4), 10**(2), 1000)

    vec_vandewiel_short_tail_stab_func = np.vectorize(define_vandewiel_short_tail_stab_function)
    vec_vandewiel_long_tail_stab_func = np.vectorize(define_vandewiel_long_tail_stab_function)
    #vec_vandewiel_cutoff_stab_func = np.vectorize(define_vandewiel_cutoff_stab_function)

    #stoch_stab_function_values = solve_SDE_stoch_stab_function(richardson_num)

    # Create plot
    color = matplotlib.cm.get_cmap('cmc.batlow', 6).colors
    markers = ['v', '*', '^', 's', 'p', '.']

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.plot(richardson_num, vec_vandewiel_short_tail_stab_func(richardson_num), label='short tail',
             color=color[0], marker=markers[0], markevery=10)
    ax1.plot(richardson_num, vec_vandewiel_long_tail_stab_func(richardson_num), label='long tail',
             color=color[1], marker=markers[1], markevery=10)
    # ax1.plot(richardson_num, vec_vandewiel_cutoff_stab_func(richardson_num), label='cutoff',
    #          color=color[2], marker=markers[2], markevery=100)
    richardson_num=richardson_num.reshape(len(richardson_num),1)
    #ax1.plot(np.tile(richardson_num, (1, np.shape(stoch_stab_function_values)[1])), stoch_stab_function_values, label='1/stochastic', markevery=100) #, color=color[2], marker=markers[2]

    ax1.set_xlabel('Ri')
    ax1.set_ylabel(r'$f_{stab}$')
    ax1.set_xscale('log')

    plt.legend()

    plt.savefig(params.sol_directory_path + 'stability_functions.png', bbox_inches='tight', dpi=300)
