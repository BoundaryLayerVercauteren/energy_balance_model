import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from one_D_model.model import solve_ODE

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


def make_2D_plot(params, x, y, file_name, xlabel='t [h]', ylabel=r'$\Delta T$ [K]'):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x, y, color='black')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.savefig(params.sol_directory_path + file_name, bbox_inches='tight', dpi=300)


def make_2D_multi_line_plot(params, x, y_array, labels, file_name, xlabel='u [m/s]', ylabel=r'$\Delta T_{eq}$ [K]'):
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    for idx in range(np.shape(y_array)[1]):
        ax1.plot(x, y_array[:, idx], label=labels[idx])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.legend()
    plt.savefig(params.sol_directory_path + file_name, bbox_inches='tight', dpi=300)


def make_distribution_plot(values, params, file_name, xlabel):
    fig = plt.figure(figsize=(5, 5))
    plot = sns.displot(data=values, color='blue', kde=True, legend=False)
    plot.set(xlabel=xlabel, ylabel='Density')
    sns.despine(fig=None, ax=None, top=False, right=False, left=False, bottom=False, offset=None, trim=False)
    plt.savefig(params.sol_directory_path + file_name, bbox_inches='tight', dpi=300)
    # To clear memory
    plt.cla()  # Clear the current axes.
    plt.clf()  # Clear the current figure.
    plt.close('all')  # Closes all the figure windows.


def plot_potentials(param_class):
    delta_T_range = np.arange(0, 30, 0.5)
    u_list = [5.6]

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    for u_elem in u_list:
        param_class.stab_func_type = 'short_tail'
        potential_st = solve_ODE.calculate_potential(delta_T_range, u_elem, param_class)
        ax1.plot(delta_T_range, - potential_st, label='short tail')

        param_class.stab_func_type = 'long_tail'
        potential_lt = solve_ODE.calculate_potential(delta_T_range, u_elem, param_class)
        ax1.plot(delta_T_range, - potential_lt, label='long tail')

    ax1.set_xlabel('$\Delta T$ [K]')
    ax1.set_ylabel('V []')
    plt.legend(loc="upper left")
    plt.savefig(param_class.sol_directory_path + 'potentials.png', bbox_inches='tight', dpi=300)
