import os
import numpy as np
import matplotlib.pyplot as plt


def get_time_spend_in_regimes(values):
    time_in_weakly = []
    time_in_very = []

    for row in values:
        time_in_weakly.append((row < 14).sum())
        time_in_very.append((row > 14).sum())

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
    return np.split(val, np.where(np.diff(val) != stepsize)[0] + 1)


def get_time_series_with_most_transitions(values):
    number_trans = []
    start_very_stable = []
    start_weakly_stable = []
    for row in values:
        idx_very_stable = np.where(row > 14)[0]
        idx_weakly_stable = np.where(row < 14)[0]

        cons_idx_in_very_stable = split_into_consecutive_ts(idx_very_stable)
        cons_idx_in_weakly_stable = split_into_consecutive_ts(idx_weakly_stable)

        if len(cons_idx_in_very_stable) > 0:
            start_very_stable = [elem[0] for elem in cons_idx_in_very_stable]

        if len(cons_idx_in_weakly_stable) > 0:
            start_weakly_stable = [elem[0] for elem in cons_idx_in_weakly_stable]

        if len(start_weakly_stable) > 0 or len(start_very_stable) > 0:
            num_trans = len(start_weakly_stable) + len(start_very_stable) - 1
        else:
            num_trans = 0

        number_trans.append(num_trans)

    return np.nanargmax(number_trans), np.nanmean(number_trans)


def get_time_of_first_transition(values):
    """A transition is defined when delta T = 14 (i.e. the unstable point)"""
    first_trans = []

    for row in values:
        row_diff = row - 14
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
output_directory = 'output/1000_sim_short_tail_u/'
# output_directory = 'output/1000_sim_short_tail_internal_var/'

# Get all result files in given directory
# deltaT_file_name = '/SDE_stab_func_sol_delta_T.npy'
deltaT_file_name = '/SDE_u_sol_delta_T.npy'
# deltaT_file_name = '/SDE_sol_delta_T.npy'

u_file_name = 'SDE_u_sol_param.npy'

subdirectories = [x[0] for x in os.walk(output_directory) if 'simulations' in x[0]]
output_files = [subdir + deltaT_file_name for subdir in subdirectories]
perturbations_labels = {'sigma_s_2_0': 2.0, 'sigma_s_1_5': 1.5, 'sigma_s_0_0': 0.0, 'sigma_s_1_0': 1.0,
                        'sigma_s_minus_0_07': -0.07, 'sigma_s_minus_0_1': -0.1, 'sigma_s_minus_1_0': -1.0,
                        'sigma_0_1': 0.1, 'sigma_0_2': 0.2, 'sigma_0_3': 0.3, 'sigma_0_4': 0.4, 'sigma_0_5': 0.5,
                        'sigma_0_6': 0.6, 'sigma_0_7': 0.7, 'sigma_0_8': 0.8, 'sigma_0_9': 0.9, 'sigma_1_0': 1.0}

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

# fig1, ax1 = plt.subplots(2, 5, figsize=(25, 10))
# axes1 = ax1.ravel()
# fig2, ax2 = plt.subplots(2, 5, figsize=(25, 10))
# axs2 = ax2.ravel()

for idx, file in enumerate(output_files):
    # Load data
    data = np.load(file)

    # Find perturbation strength for current simulation/ output
    for key in perturbations_labels:
        if key in file:
            perturb_strength = str(perturbations_labels[key])

    # Get transition statistics for current simulation
    mean_time_weakly[perturb_strength], max_time_weakly[perturb_strength], mean_time_very[perturb_strength], \
    max_time_very[perturb_strength], idx_max, num_ts_in_very[perturb_strength], num_ts_in_weakly[perturb_strength], \
    mean_time_until_trans[perturb_strength], num_crashes_per_sim[perturb_strength], \
    idx_max_transitions, mean_trans_per_sim[perturb_strength] = get_transition_statistics(data)

    # Plot time series which has the longest occupation time in the very stable regime
#     if idx_max_transitions is not np.nan:
#         axes1[idx].plot(data[idx_max_transitions, :])
#         axes1[idx].title.set_text(f'Perturbation strength: {perturb_strength}')
#         axes1[idx].set_xlabel('time [s]')
#         axes1[idx].set_ylabel(r'$\Delta T$')
#
#         if '_u_' in file:
#             data_u = np.load(file.rsplit('/', 1)[0] + '/' + u_file_name)
#             axes1[idx].plot(data_u[idx_max_transitions, :], color='g')
#
#     # Plot histogram of
#     axs2[idx].hist(data.flatten(), 100)
#     axs2[idx].set_ylabel(r'Density of $\Delta T$')
#     axs2[idx].title.set_text(f'Perturbation strength: {perturb_strength}')
#     axs2[idx].axvline(x=24, color='r')
#     axs2[idx].axvline(x=4, color='r')
#     axs2[idx].axvline(x=12, color='r', linestyle='--')
#
# fig1.savefig(output_directory + 'max_transitions.png', bbox_inches='tight', dpi=300)
# fig2.savefig(output_directory + 'distributions.png', bbox_inches='tight', dpi=300)
# To clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close('all')  # Closes all the figure windows.

# Transform values to percentages
num_time_steps = np.shape(data)[1]
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

plt.savefig(output_directory + 'transition_statistics.png', bbox_inches='tight', dpi=300)
