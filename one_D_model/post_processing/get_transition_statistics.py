import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import cmcrameri.cm as cmc

import sys
import os

SCRIPT_DIR = '/mnt/c/Users/amandink/OneDrive - Universitetet i Oslo/01_PhD/02_Code/van_de_Wiel_model/DomeC'
sys.path.append(os.path.dirname(SCRIPT_DIR))

from DomeC import analyze_dome_c_data

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

location_unstable_eq = 12


def get_time_spend_in_regimes(values):
    time_in_weakly = []
    time_in_very = []

    for row in values:
        time_in_weakly.append((row < location_unstable_eq).sum())
        time_in_very.append((row > location_unstable_eq).sum())

    if np.all(np.isclose(time_in_very, 0.0)):
        idx_max_time_very = np.nan
    else:
        idx_max_time_very = np.nanargmax(time_in_very)

    return np.nanmean(time_in_weakly), np.nanmax(time_in_weakly), np.nanmean(time_in_very), np.nanmax(
        time_in_very), idx_max_time_very, sum(elem != 0 for elem in time_in_very), sum(
        elem != 0 for elem in time_in_weakly)


def count_how_many_sim_crashed(values):
    number_crashes = 0

    for row in values:
        if any(np.isnan(row)):
            number_crashes += 1
    return number_crashes


def split_into_consecutive_ts(val, stepsize=1):
    array = np.split(val, np.where(np.diff(val) != stepsize)[0] + 1)
    if len(array) == 1:
        array = array[0]
    return array


def get_time_series_with_most_transitions(values):
    number_trans = []
    start_very_stable = []
    start_weakly_stable = []
    for idx, row in enumerate(values):

        idx_very_stable = np.where(row > location_unstable_eq)[0]
        idx_weakly_stable = np.where(row < location_unstable_eq)[0]

        cons_idx_in_very_stable = split_into_consecutive_ts(idx_very_stable)
        cons_idx_in_weakly_stable = split_into_consecutive_ts(idx_weakly_stable)

        if row[0] > location_unstable_eq:
            cons_idx_in_very_stable = cons_idx_in_very_stable[1:]

        if len(cons_idx_in_very_stable) > 0:
            # Check if the first element is a list
            if isinstance(cons_idx_in_very_stable[0], np.ndarray):
                start_very_stable = [elem[0] for elem in cons_idx_in_very_stable]
            else:
                start_very_stable = [cons_idx_in_very_stable[0]]

        if len(cons_idx_in_weakly_stable) > 0:
            # Check if the first element is a list
            if isinstance(cons_idx_in_weakly_stable[0], np.ndarray):
                start_weakly_stable = [elem[0] for elem in cons_idx_in_weakly_stable]
            else:
                start_weakly_stable = [cons_idx_in_weakly_stable[0]]

        # Make sure that at least 1 hour is between transitions
        # Small bumps during a transition should not count
        elem_to_be_removed_weakly = []
        elem_to_be_removed_very = []
        for elem_idx in np.arange(0, len(start_weakly_stable) - 1):
            if start_weakly_stable[elem_idx + 1] - start_weakly_stable[elem_idx] < 360:
                elem_to_be_removed_weakly.append(start_weakly_stable[elem_idx])
                for elem_v in start_very_stable:
                    if elem_v < start_weakly_stable[elem_idx + 1] and elem_v > start_weakly_stable[elem_idx]:
                        elem_to_be_removed_very.append(elem_v)
        start_weakly_stable = [elem for elem in start_weakly_stable if elem not in elem_to_be_removed_weakly]

        for elem_idx in np.arange(0, len(start_very_stable) - 1):
            if start_very_stable[elem_idx + 1] - start_very_stable[elem_idx] < 360:
                elem_to_be_removed_very.append(start_very_stable[elem_idx])
        start_very_stable = [elem for elem in start_very_stable if elem not in elem_to_be_removed_very]

        # if len(idx_weakly_stable) != 0:
        #     try:
        #         fig, ax = plt.subplots(figsize=(15, 5))
        #         ax.plot(row, color='black')
        #         ax.axhline(y=12, color='blue')
        #         ax.scatter(start_weakly_stable, row[start_weakly_stable], color='red')
        #         ax.scatter(start_very_stable, row[start_very_stable], color='green')
        #         #ax.set_xlim(27000,30000)
        #         fig.savefig(output_directory + f'{idx}_test.pdf', bbox_inches='tight', dpi=300)
        #         plt.cla()  # Clear the current axes.
        #         plt.clf()  # Clear the current figure.
        #         plt.close('all')  # Closes all the figure windows.
        #         #exit()
        #     except Exception:
        #         pass

        if start_weakly_stable or start_very_stable:
            num_trans = len(start_weakly_stable) + len(start_very_stable) - 1
        else:
            num_trans = 0

        number_trans.append(num_trans)

    print(np.nanmax(number_trans))
    return np.nanargmax(number_trans), np.nanmean(number_trans)


def get_time_of_first_transition(values):
    """A transition is defined when delta T = 14 (i.e. the unstable point)"""
    first_trans = []

    for row in values:
        row_diff = row - location_unstable_eq
        first_trans.append(np.argmax(np.round(row_diff, 0) == 0))

    return np.mean(first_trans)


def get_transition_statistics(data_values):
    mean_in_w, max_in_w, mean_in_v, max_in_v, idx_max_v, count_ts_in_v, count_ts_in_w = get_time_spend_in_regimes(
        data_values)
    mean_first_trans = get_time_of_first_transition(data_values)
    num_crashes = count_how_many_sim_crashed(data_values)
    idx_max_trans, mean_trans = get_time_series_with_most_transitions(data_values)
    return mean_in_w, max_in_w, mean_in_v, max_in_v, idx_max_v, count_ts_in_v, count_ts_in_w, mean_first_trans, \
           num_crashes, idx_max_trans, mean_trans


# Define directory where simulation output is saved
# output_directory = 'output/1000_sim_short_tail_stab_func/very_weakly/'
# output_directory = 'output/1000_sim_short_tail_u/'
output_directory = 'output/1000_sim_short_tail_internal_var/sigma_0_2/'
# output_directory = 'output/1000_sim_short_tail_stab_func_multi_noise/sigma_0_1/'
# output_directory = 'output/1000_sim_short_stail_u_internal/sigma_u_0_015_sigma_i_0_12/'

# Get all result files in given directory
# deltaT_file_name = '/SDE_stab_func_sol_delta_T.npy'
# deltaT_file_name = '/SDE_u_sol_delta_T.npy'
deltaT_file_name = '/SDE_sol_delta_T.npy'
# deltaT_file_name = '/SDE_stab_func_poisson_sol_delta_T.npy'
# deltaT_file_name = '/SDE_stab_func_multi_noise_sol_delta_T.npy'
# deltaT_file_name = '/SDE_u_internal_var_sol_delta_T.npy'

u_file_name = 'SDE_u_sol_param.npy'

subdirectories = [x[0] for x in os.walk(output_directory) if 'simulations' in x[0]]
output_files = [subdir + deltaT_file_name for subdir in subdirectories]
perturbations_labels = {'sigma_s_2_0': 2.0, 'sigma_s_1_5': 1.5, 'sigma_s_0_0': 0.0, 'sigma_s_1_0': 1.0,
                        'sigma_s_minus_0_07': -0.07, 'sigma_s_minus_0_1': -0.1, 'sigma_s_minus_1_0': -1.0,
                        'sigma_0_1': 0.1, 'sigma_0_2': 0.2, 'sigma_0_3': 0.3, 'sigma_0_4': 0.4, 'sigma_0_5': 0.5,
                        'sigma_0_6': 0.6, 'sigma_0_7': 0.7, 'sigma_0_8': 0.8, 'sigma_0_9': 0.9, 'sigma_1_0': 1.0,
                        'sigma_0_01': 0.01, 'sigma_0_001': 0.001, 'sigma_0_02': 0.02, 'sigma_0_03': 0.03,
                        'sigma_1_0': 1.0, 'sigma_2_0': 2.0, 'sigma_3_0': 3.0, 'sigma_4_0': 4.0, 'sigma_5_0': 5.0,
                        'sigma_0_005': 0.005, 'sigma_0_08': 0.08, 'lambda_0_1': 0.1, 'lambda_0_2': 0.2,
                        'lambda_0_3': 0.3, 'lambda_0_4': 0.4, 'lambda_0_5': 0.5, 'lambda_0_6': 0.6, 'lambda_0_7': 0.7}

# Define empty dictionary to store results
mean_time_weakly = {}
mean_time_very = {}
max_time_weakly = {}
max_time_very = {}
mean_time_until_trans = {}
num_ts_in_very = {}
num_ts_in_weakly = {}
num_crashes_per_sim = {}
mean_trans_per_sim = {}

fig1, ax1 = plt.subplots(3, 5, figsize=(25, 15))
axes1 = ax1.ravel()
fig2, ax2 = plt.subplots(3, 5, figsize=(25, 15))
axs2 = ax2.ravel()

fig3, ax3 = plt.subplots(figsize=(5, 5))
xtick_lab = []

# DomeC_statistics = analyze_dome_c_data.get_Dome_C_statistics(2013)
# ax3.errorbar(0, DomeC_statistics[0], DomeC_statistics[1], linestyle='None', capsize=3, marker='o', lw=3, color='orange')
xtick_lab.append('Dome C')

for idx, file in enumerate(output_files):
    perturb_strength = np.nan
    # Load data
    try:
        data = np.load(file)
    except OSError as e:
        continue

    # Find perturbation strength for current simulation/ output
    for key in perturbations_labels:
        if key in file:
            perturb_strength = str(perturbations_labels[key])
        else:
            perturb_strength = str('nan')

    if perturb_strength == '0.08' and '_u_' in file:
        chosen_data = data
    elif perturb_strength == '0.2' and '_u_' not in file:
        chosen_data = data
    elif perturb_strength == 'nan':
        chosen_data = data

    # Get transition statistics for current simulation
    mean_time_weakly[perturb_strength], max_time_weakly[perturb_strength], mean_time_very[perturb_strength], \
    max_time_very[perturb_strength], idx_max, num_ts_in_very[perturb_strength], num_ts_in_weakly[perturb_strength], \
    mean_time_until_trans[perturb_strength], num_crashes_per_sim[perturb_strength], \
    idx_max_transitions, mean_trans_per_sim[perturb_strength] = get_transition_statistics(data)

    print(f'For sigma/lambda={perturb_strength} the average number of transitions per run is: '
          f'{mean_trans_per_sim[perturb_strength]}.')

    # Plot time series which has the longest occupation time in the very stable regime
    if idx_max_transitions is not np.nan:
        axes1[idx].plot(data[idx_max_transitions, :])
        axes1[idx].title.set_text(f'Perturbation strength: {perturb_strength}')
        axes1[idx].set_xlabel('time [s]')
        axes1[idx].set_ylabel(r'$\Delta T$')

        # if '_u_' in file:
        #     data_u = np.load(file.rsplit('/', 1)[0] + '/' + u_file_name)
        #     axes1[idx].plot(data_u[idx_max_transitions, :], color='g')
        #
        #     curr_mean = np.nanmean(data_u.flatten())
        #     ax3.errorbar(idx + 1, curr_mean, np.nanstd(data_u.flatten()), linestyle='None',
        #                  capsize=3, marker='o', lw=3, color='black')
        #     erb_dome_c = ax3.errorbar(idx + 1, curr_mean, DomeC_statistics[1], capsize=3, color='orange')
        #     erb_dome_c[-1][0].set_linestyle('dashed')
        #     xtick_lab.append(perturb_strength)

    # Plot histogram of
    axs2[idx].hist(data.flatten(), 100)
    axs2[idx].set_xlim(data.min(), data.max())
    axs2[idx].set_ylabel(r'Density of $\Delta T$')
    axs2[idx].title.set_text(f'Perturbation strength: {perturb_strength}')
    axs2[idx].axvline(x=24, color='r')
    axs2[idx].axvline(x=4, color='r')
    axs2[idx].axvline(x=12, color='r', linestyle='--')

# print(float(xtick_lab[0]))
# print(float(xtick_lab[4]))
ax3.set_xticks(range(len(xtick_lab)), xtick_lab)
ax3.set_xlim(-0.5, 3.5)
ax3.set_ylim(4.5, 7)
ax3.set_xlabel(r'$\sigma_u$')
ax3.set_ylabel('u [m/s]')

fig1.savefig(output_directory + 'max_transitions.pdf', bbox_inches='tight', dpi=300)
fig2.savefig(output_directory + 'distributions.pdf', bbox_inches='tight', dpi=300)
fig3.savefig(output_directory + 'mean_variance.pdf', bbox_inches='tight', dpi=300)

# To clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close('all')  # Closes all the figure windows.

# Plot statistics
num_time_steps = np.shape(data)[1]

color = matplotlib.cm.get_cmap('cmc.batlow', 4).colors

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
axes = ax.ravel()
box1 = axes[0].boxplot((chosen_data < location_unstable_eq).sum(axis=1) / num_time_steps * 100, positions=[0],
                       patch_artist=True)
box1['boxes'][0].set(facecolor=color[1])#, alpha=0.5)
box1['medians'][0].set_color('black')
box2 = axes[0].boxplot((chosen_data > location_unstable_eq).sum(axis=1) / num_time_steps * 100, positions=[1],
                       patch_artist=True)
box2['boxes'][0].set(facecolor=color[2])#, alpha=0.5)
box2['medians'][0].set_color('black')

axes[0].set_xticks([0, 1], ['weakly', 'very'])
axes[0].set_ylabel('% of time spend in regime')
axes[0].set_title('a)', loc='left')

# colors = plt.cm.Greys(np.linspace(0, 0.6, np.shape(chosen_data)[0]))
# ax[0].set_prop_cycle('color', colors)
# ax[0].plot(np.linspace(0, 24, 24 * 3600), chosen_data.T)
# ax[0].plot(np.linspace(0, 24, 24 * 3600), np.mean(chosen_data, axis=0), color='red')


axes[1].axvspan(chosen_data.min(), 12, facecolor=color[1])#, alpha=0.5)
axes[1].axvspan(12, chosen_data.max(), facecolor=color[2])#, alpha=0.5)
axes[1].hist(chosen_data.flatten(), 100, color=color[0])
axes[1].set_xlim(chosen_data.min(), chosen_data.max())
axes[1].set_xlabel(r'$\Delta T$')
axes[1].set_ylabel(r'Density of $\Delta T$')
axes[1].axvline(x=24, color='r')
axes[1].axvline(x=4, color='r')
axes[1].axvline(x=12, color='r', linestyle='--')
axes[1].set_title('b)', loc='left')

plt.tight_layout()

plt.savefig(output_directory + 'time_in_regime.pdf', bbox_inches='tight', dpi=300)
exit()
# Transform values to percentages
num_sim = np.shape(data)[0]

mean_time_weakly_perc = {k: v / num_time_steps * 100 if v != 0 else 0 for k, v in mean_time_weakly.items()}
mean_time_very_perc = {k: v / num_time_steps * 100 if v != 0 else 0 for k, v in mean_time_very.items()}
num_crashes_per_sim_perc = {k: v / num_sim * 100 if v != 0 else 0 for k, v in num_crashes_per_sim.items()}
mean_time_until_trans_perc = {k: v / num_time_steps * 100 if v != 0 else 0 for k, v in mean_time_until_trans.items()}
max_time_weakly_perc = {k: v / num_time_steps * 100 if v != 0 else 0 for k, v in max_time_weakly.items()}
max_time_very_perc = {k: v / num_time_steps * 100 if v != 0 else 0 for k, v in max_time_very.items()}
num_ts_in_very_perc = {k: v / num_sim * 100 if v != 0 else 0 for k, v in num_ts_in_very.items()}
num_ts_in_weakly_perc = {k: v / num_sim * 100 if v != 0 else 0 for k, v in num_ts_in_weakly.items()}

# Plot statistics
fig, ax = plt.subplots(3, 4, figsize=(20, 15))
axs = ax.ravel()

axs[0].bar(mean_time_weakly_perc.keys(), mean_time_weakly_perc.values(), color='g')
axs[0].title.set_text('Mean time in weakly stable regime (%)')
axs[0].set_xlabel('perturbation strength')
# axs[0].set_ylim((0, 86400))

axs[1].bar(mean_time_very_perc.keys(), mean_time_very_perc.values(), color='b')
axs[1].title.set_text('Mean time in very stable regime (%)')
axs[1].set_xlabel('perturbation strength')
# axs[1].set_ylim((0, 300))

axs[2].bar(num_crashes_per_sim_perc.keys(), num_crashes_per_sim_perc.values(), color='orange')
axs[2].title.set_text('Percentage of crashes per 1000 simulations')
axs[2].set_xlabel('perturbation strength')

axs[3].bar(mean_time_until_trans_perc.keys(), mean_time_until_trans_perc.values(), color='orange')
axs[3].title.set_text('Mean time until first transition (%)')
axs[3].set_xlabel('perturbation strength')

axs[4].bar(max_time_weakly_perc.keys(), max_time_weakly_perc.values(), color='g')
axs[4].title.set_text('Max time in weakly stable regime (%)')
axs[4].set_xlabel('perturbation strength')
# axs[2].set_ylim((0, 86400))

axs[5].bar(max_time_very_perc.keys(), max_time_very_perc.values(), color='b')
axs[5].title.set_text('Max time in very stable regime (%)')
axs[5].set_xlabel('perturbation strength')
# axs[3].set_ylim((0, 300))

axs[6].bar(num_ts_in_very_perc.keys(), num_ts_in_very_perc.values(), color='b')
axs[6].title.set_text('Perc. of time series with very stable regime')
axs[6].set_xlabel('perturbation strength')

axs[7].bar(num_ts_in_weakly_perc.keys(), num_ts_in_weakly_perc.values(), color='g')
axs[7].title.set_text('Perc. of time series with weakly stable regime')
axs[7].set_xlabel('perturbation strength')

axs[8].bar(mean_trans_per_sim.keys(), mean_trans_per_sim.values(), color='orange')
axs[8].title.set_text('Average number of transitions')
axs[8].set_xlabel('perturbation strength')

plt.savefig(output_directory + 'transition_statistics_0_8.pdf', bbox_inches='tight', dpi=300)
