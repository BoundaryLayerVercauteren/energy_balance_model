import numpy as np

from process_dome_c_data import prepare_dome_c_data
import matplotlib.pyplot as plt


def find_data_in_u_bin(data, bin_mean=4.5, bin_var=1.5):
    data.loc[(data['U2[m s-1]'] < (bin_mean - bin_var)) | (data['U2[m s-1]'] > (bin_mean + bin_var))] = np.nan
    return data


def find_longest_continous_time_series(data):
    data_values = data['tempInv [K]'].values
    # Mask values
    masked_values = np.concatenate(([True], np.isnan(data_values), [True]))
    # Find start-stop limits (adjust index to fit input dataframe)
    ss = np.flatnonzero(masked_values[1:] != masked_values[:-1]).reshape(-1, 2)  # + data.index.min()
    # Find index of 10 longest time series
    ts_length_start_stop = []
    for row in ss:
        ts_length_start_stop.append((row[1] - row[0], row))
    # Sort time series start stop values by length
    ts_length_start_stop = sorted(ts_length_start_stop, key=lambda x: x[0])
    # Get 10 longest time series
    start_stop_longest_ts = ts_length_start_stop[-20:]
    return start_stop_longest_ts


def check_for_transition(data, start_top_cons):
    start_stop_with_trans = []
    for idx in np.arange(0, len(start_top_cons)):
        curr_data = data['tempInv [K]'].iloc[longest_cons_deltaT[idx][1][0]:longest_cons_deltaT[idx][1][1]]
        if curr_data.max() >= 10 and curr_data.min() <= 5:
            start_stop_with_trans.append((longest_cons_deltaT[idx][1][0], longest_cons_deltaT[idx][1][1]))
    return start_stop_with_trans


if __name__ == "__main__":
    # Prepare data
    _, data_season = prepare_dome_c_data()
    # Select data where u is in specific bin
    data_in_u_bin = find_data_in_u_bin(data_season.copy())
    # Select 10 longest time series with no missing delta T (returns start and stop index of time series)
    longest_cons_deltaT = find_longest_continous_time_series(data_in_u_bin)
    # Only select time series with a transition
    start_stop_idx_with_trans = check_for_transition(data_in_u_bin, longest_cons_deltaT)

    # # Plot histogram
    # fig = plt.figure(figsize=(5, 5))
    # plt.hist(data_in_u_bin['tempInv [K]'], 10)
    # plt.ylabel(r'Density of $\Delta T$')
    # plt.xlabel(r'$\Delta T$ [K]')
    # plt.savefig('hist_domeC_u_bin.png', bbox_inches='tight', dpi=300)
    #
    # # Plot data over time
    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(data_in_u_bin['Local Time (UTC+8h)'], data_in_u_bin['tempInv [K]'])
    # plt.ylabel(r'$\Delta T$ [K]')
    # plt.ylabel(r'time')
    # # plt.xticks(rotation=90)
    # plt.savefig('domeC_u_bin_over_t.png', bbox_inches='tight', dpi=300)
    #
    # # Plot data over u
    # fig = plt.figure(figsize=(10, 5))
    # plt.axvline(x=5.6, color='r')
    # plt.axvline(x=5.1, color='r', linestyle='dashed')
    # plt.axvline(x=6.1, color='r', linestyle='dashed')
    # plt.scatter(data_season['U2[m s-1]'], data_season['tempInv [K]'], s=5)
    # plt.ylabel(r'$\Delta T$ [K]')
    # plt.xlabel(r'u [m/s]')
    # plt.savefig('domeC_deltaT_over_u.png', bbox_inches='tight', dpi=300)

    # # Plot data over time
    # for idx in np.arange(0, len(start_stop_idx_with_trans)):
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.plot(data_in_u_bin['Local Time (UTC+8h)'].iloc[
    #              start_stop_idx_with_trans[idx][0]:start_stop_idx_with_trans[idx][1]],
    #              data_in_u_bin['tempInv [K]'].iloc[start_stop_idx_with_trans[idx][0]:start_stop_idx_with_trans[idx][1]])
    #     plt.ylabel(r'$\Delta T$ [K]')
    #     plt.xlabel(r'time')
    #     plt.savefig('domeC_u_bin_longest_ts_over_t_' + str(idx) + '.png', bbox_inches='tight', dpi=300)
    #
    # # Plot data over u
    # for idx in np.arange(0, len(start_stop_idx_with_trans)):
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.scatter(
    #         data_in_u_bin['U2[m s-1]'].iloc[start_stop_idx_with_trans[idx][0]:start_stop_idx_with_trans[idx][1]],
    #         data_in_u_bin['tempInv [K]'].iloc[start_stop_idx_with_trans[idx][0]:start_stop_idx_with_trans[idx][1]])#, s=5)
    #     # plt.axvline(x=5.6, color='r')
    #     # plt.axvline(x=5.1, color='r', linestyle='dashed')
    #     # plt.axvline(x=6.1, color='r', linestyle='dashed')
    #     plt.ylabel(r'$\Delta T$ [K]')
    #     plt.xlabel(r'u [m/s]')
    #     plt.savefig('domeC_u_bin_longest_ts_over_u_' + str(idx) + '.png', bbox_inches='tight', dpi=300)

    # Save one u time series to use as input for model
    index = 7
    np.savetxt(r'u_values.txt', data_in_u_bin['U2[m s-1]'].iloc[start_stop_idx_with_trans[index][0]:start_stop_idx_with_trans[index][1]].values)
    np.savetxt(r'delta_theta_values.txt', data_in_u_bin['tempInv [K]'].iloc[start_stop_idx_with_trans[index][0]:start_stop_idx_with_trans[index][1]].values)
