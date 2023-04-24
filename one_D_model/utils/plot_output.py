import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import cmcrameri.cm as cmc
import dataclasses

from one_D_model.model import solve_ODE

plt.style.use('science')

# set font sizes for plots
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def make_2D_plot(params, x, y, file_name, xlabel='t [h]', ylabel=r'$\Delta T$ [K]', ylim=(0, 40)):
    color = matplotlib.cm.get_cmap('cmc.batlow', 1).colors
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.axhline(y=24, color='r')
    ax1.axhline(y=4, color='r')
    ax1.axhline(y=12, color='r', linestyle='--')
    ax1.plot(x, y, color=color)

    ax1.set_ylim(ylim)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(params.sol_directory_path + file_name, bbox_inches='tight', dpi=300)
    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def make_2D_multi_line_plot(params, x, y_array, labels, file_name, xlabel='u [m/s]', ylabel=r'$\Delta T_{eq}$ [K]',
                            ylim=(0, 30), ylabel2='u [m/s]'):
    color = matplotlib.cm.get_cmap('cmc.batlow', np.shape(y_array)[1] + 1).colors
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax2 = ax1.twinx()

    ax1.axhline(y=24, color='r', lw=2)
    ax1.axhline(y=4, color='r', lw=2)
    ax1.axhline(y=12, color='r', linestyle=':', lw=2)

    ax1.plot(x, y_array[:, 0], label=labels[0], color=color[0], lw=2)
    ax2.plot(x, y_array[:, 1], label=labels[1], color=color[1], alpha=0.6)
    average_time = 600
    ax2.plot(x[average_time - 1:], np.convolve(y_array[:, 1], np.ones(average_time), 'valid') / average_time,
             label=labels[1], color=color[1], alpha=1, lw=2)

    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_visible(False)

    ax1.set_ylim(ylim)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color=color[0])
    ax1.tick_params(axis="y", labelcolor=color[0])

    ax2.set_ylabel(ylabel2, color=color[1])
    ax2.tick_params(axis="y", labelcolor=color[1])

    # plt.legend()
    plt.savefig(params.sol_directory_path + file_name, bbox_inches='tight', dpi=300)
    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def make_distribution_plot(values, params, file_name, xlabel):
    fig = plt.figure(figsize=(5, 5))

    plt.axvline(x=24, color='r')
    plt.axvline(x=4, color='r')
    plt.axvline(x=12, color='r', linestyle='--')
    plt.hist(values, 100)

    plt.xlabel(xlabel)
    plt.ylabel(r'Density of $\Delta T$')
    plt.savefig(params.sol_directory_path + file_name, bbox_inches='tight', dpi=300)
    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def plot_potentials(param_class):
    delta_T_range = np.arange(0, 30, 0.5)
    u_list_st = [5.3, 5.6, 5.9]
    u_list_lt = [4.87, 4.89, 4.9]

    color = matplotlib.cm.get_cmap('cmc.batlow', 4).colors
    markers = ['v', 's', 'p']

    # copy dataclass to prevent overwriting original
    param_copy = dataclasses.replace(param_class)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for idx, u_elem in enumerate(u_list_st):
        param_copy.stab_func_type = 'short_tail'
        potential_st = solve_ODE.calculate_potential(delta_T_range, u_elem, param_copy)
        ax[0].plot(delta_T_range, - potential_st, label='u = ' + str(u_elem), color=color[idx], marker=markers[idx],
                   markevery=5)

    ax[0].set_xlabel('$\Delta T$ [K]')
    ax[0].set_ylabel('V [$K^2$/s]')
    # ax[0].set_ylim(-0.25, 0.15)
    ax[0].legend()
    ax[0].set_title('a)', loc='left')

    for idx, u_elem in enumerate(u_list_lt):
        param_copy.stab_func_type = 'long_tail'
        potential_lt = solve_ODE.calculate_potential(delta_T_range, u_elem, param_copy)
        ax[1].plot(delta_T_range[8:44], - potential_lt[8:44], label='u = ' + str(u_elem), color=color[idx],
                   marker=markers[idx], markevery=5)

    ax[1].set_xlabel('$\Delta T$ [K]')
    ax[1].set_ylabel('V [$K^2$/s]')
    # ax[1].set_ylim(-0.25/2, 0.15/2)
    ax[1].legend()
    ax[1].set_title('b)', loc='left')

    plt.savefig(param_class.sol_directory_path + 'potentials.png', bbox_inches='tight', dpi=300)


def plot_potentials_and_output_distribution(param_class, delta_T_data):
    delta_T_range = np.arange(0, 30, 0.5)
    u_list_st = [5.3, 5.6, 5.9]
    u_list_lt = [4.87, 4.89, 4.9]

    color = matplotlib.cm.get_cmap('cmc.batlow', 4).colors
    markers = ['s', 'p', 'v']

    # copy dataclass to prevent overwriting original
    param_copy = dataclasses.replace(param_class)

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(right=0.75)

    ax2 = ax1.twinx()

    if param_copy.stab_func_type == 'short_tail':
        p0 = ax2.hist(delta_T_data, 100, color=color[1], alpha=0.5, zorder=0)
        for idx, u_elem in enumerate(u_list_st):
            potential_st = solve_ODE.calculate_potential(delta_T_range, u_elem, param_copy)
            p1, = ax1.plot(delta_T_range, - potential_st, label='u = ' + str(u_elem), color=color[idx],
                           marker=markers[idx], markevery=5, zorder=10)
        title = 'a)'

    elif param_copy.stab_func_type == 'long_tail':
        p2 = ax2.hist(delta_T_data, 100, color=color[1], alpha=0.5, zorder=0)
        for idx, u_elem in enumerate(u_list_lt):
            potential_lt = solve_ODE.calculate_potential(delta_T_range, u_elem, param_copy)
            p3, = ax1.plot(delta_T_range, - potential_lt, label='u = ' + str(u_elem), color=color[idx],
                           marker=markers[idx], markevery=5, zorder=10)
        title = 'b)'

    ax1.set_xlabel(r'$\Delta T$ [K]')
    ax1.set_ylabel('V [$K^2$/s]')
    ax2.set_ylabel(r'Density of $\Delta T$', color=color[1], alpha=1)
    ax2.tick_params(axis="y", labelcolor=color[1])

    ax2.set_ylim((0.0, 4.5 * 10 ** 6))
    ax1.set_title(title, loc='left')
    ax1.legend()

    plt.savefig(param_class.sol_directory_path + 'data_dist_potentials_' + param_copy.stab_func_type + '.png',
                bbox_inches='tight', dpi=300)
