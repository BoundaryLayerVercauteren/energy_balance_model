import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from process_dome_c_data import prepare_dome_c_data
import matplotlib.pyplot as plt
import matplotlib
import cmcrameri.cm as cmc


def find_data_in_u_bin(data, bin_mean=5.6, bin_var=1.0):
    if 'U2[m s-1]' in data.columns:
        data.loc[(data['U2[m s-1]'] < (bin_mean - bin_var)) | (data['U2[m s-1]'] > (bin_mean + bin_var))] = np.nan
    else:
        data.loc[(data['U_9m [m/s]'] < (bin_mean - bin_var)) | (
                    data['U_9m [m/s]'] > (bin_mean + bin_var)), 'U_9m [m/s]'] = np.nan
    return data


def find_longest_continous_time_series(data, year):
    # Select rows which are in given year
    data.loc[data['year'] != year, 'tempInv [K]'] = np.nan
    # Select rows where U is not NaN
    if 'U2[m s-1]' in data.columns:
        data.loc[data['U2[m s-1]'].isnull(), 'tempInv [K]'] = np.nan
    else:
        data.loc[data['U_9m [m/s]'].isnull(), 'tempInv [K]'] = np.nan
    data_values = data['tempInv [K]'].values
    # Mask values
    masked_values = np.concatenate(([True], np.isnan(data_values), [True]))
    # Find start-stop limits (adjust index to fit input dataframe)
    ss = np.flatnonzero(masked_values[1:] != masked_values[:-1]).reshape(-1, 2)  # + data.index.min()
    # Find length of time series
    ts_length_start_stop = []
    for row in ss:
        ts_length_start_stop.append((row[1] - row[0], row))
    # Sort time series start stop values by length
    ts_length_start_stop = sorted(ts_length_start_stop, key=lambda x: x[0])
    # Get 10 longest time series
    start_stop_longest_ts = ts_length_start_stop[-10:]

    return start_stop_longest_ts, ts_length_start_stop


def check_for_transition(data, longest_cons_deltaT):
    start_stop_with_trans = []
    for idx in np.arange(0, len(longest_cons_deltaT)):
        curr_data = data['tempInv [K]'].iloc[longest_cons_deltaT[idx][1][0]:longest_cons_deltaT[idx][1][1]]
        if curr_data.max() >= 10 and curr_data.min() <= 5:
            start_stop_with_trans.append((longest_cons_deltaT[idx][1][0], longest_cons_deltaT[idx][1][1]))
    return start_stop_with_trans


if __name__ == "__main__":
    # Prepare data
    data_season = prepare_dome_c_data()

    # Fill jumps in time with nan
    # Be aware that this changes the index
    data_season = data_season.set_index('timestamp').resample('30Min').mean().reset_index()
    jump = np.abs(data_season.timestamp - data_season.timestamp.shift(1))

    # Add year column
    data_season['year'] = pd.DatetimeIndex(data_season['timestamp']).year

    # Select data where u is in specific bin
    data_in_u_bin = find_data_in_u_bin(data_season.copy())

    # # Plot data over time for every year
    # g = sns.FacetGrid(data=data_in_u_bin, col='year', col_wrap=4, height=4, aspect=3/2, sharey=True, sharex=False)
    # g.map_dataframe(sns.lineplot, x='timestamp', y='tempInv [K]')
    # plt.savefig('DomeC/figures/domeC_u_binned_deltaT_over_time.png', bbox_inches='tight', dpi=300)

    # # Plot histogram for every year
    # facet_kws = {'sharey': False, 'sharex': True}
    # sns.displot(data=data_in_u_bin, x='tempInv [K]', col='year', col_wrap=4, height=4, aspect=1, facet_kws=facet_kws,
    #             kde=True)
    # plt.savefig('DomeC/figures/hist_deltaT_domeC_u_bin.png', bbox_inches='tight', dpi=300)

    # Select longest times series with transitions for specified years and plot them
    years = [2010, 2012, 2014, 2016, 2017, 2018, 2019]
    ss_idx_with_trans = np.zeros((len(years), 20), dtype="f,f")
    for year_idx, year in enumerate(years):
        # Select longest time series (plural) with no missing delta T (returns start and stop index of time series)
        longest_cons_deltaT, all_cons_deltaT = find_longest_continous_time_series(data_in_u_bin.copy(), year)

        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(data_in_u_bin.loc[data_in_u_bin['year'] == year]['timestamp'],
                 data_in_u_bin.loc[data_in_u_bin['year'] == year]['tempInv [K]'], color='black')

        for elem_idx, elem in enumerate(longest_cons_deltaT):
            ax1.plot(data_in_u_bin.iloc[elem[1][0]:elem[1][1]]['timestamp'],
                     data_in_u_bin.iloc[elem[1][0]:elem[1][1]]['tempInv [K]'], color='red')

        # Only select time series with a transition
        trans_ts = check_for_transition(data_in_u_bin, longest_cons_deltaT)

        for elem_idx, elem in enumerate(trans_ts):
            ss_idx_with_trans[year_idx, elem_idx] = elem
            ax1.plot(data_in_u_bin.iloc[elem[0]:elem[1]]['timestamp'],
                     data_in_u_bin.iloc[elem[0]:elem[1]]['tempInv [K]'], color='green')

        plt.savefig('DomeC/figures/domeC_u_binned_deltaT_over_time_marked_trans_' + str(year) + '.png',
                    bbox_inches='tight', dpi=300)

    # Select times series with transitions separately for specified years and plot them over time
    for year_idx, year in enumerate(years):
        fig, axs = plt.subplots(2,3, figsize=(10, 5))
        fig.suptitle(str(year), fontsize=16)
        fig.subplots_adjust(hspace=.5, wspace=.001)
        axs = axs.ravel()
        for elem_idx, elem in enumerate(ss_idx_with_trans[year_idx]):
            if elem[0] != 0.0 and elem[1] != 0.0:
                axs[elem_idx].plot(data_in_u_bin.iloc[int(elem[0]):int(elem[1])]['timestamp'],
                                   data_in_u_bin.iloc[int(elem[0]):int(elem[1])]['tempInv [K]'], color='blue')
                if 'U2[m s-1]' in data_in_u_bin.columns:
                    axs[elem_idx].plot(data_in_u_bin.iloc[int(elem[0]):int(elem[1])]['timestamp'],
                                       data_in_u_bin.iloc[int(elem[0]):int(elem[1])]['U2[m s-1]'], color='green')
                else:
                    axs[elem_idx].plot(data_in_u_bin.iloc[int(elem[0]):int(elem[1])]['timestamp'],
                                       data_in_u_bin.iloc[int(elem[0]):int(elem[1])]['U_9m [m/s]'], color='green')
                axs[elem_idx].tick_params(labelrotation=90)

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        plt.savefig('DomeC/figures/domeC_u_binned_deltaT_over_time_sep_trans_' + str(year) + '.png',
                    bbox_inches='tight', dpi=300)

    # Select times series with transitions separately for specified years and plot histogram of them
    for year_idx, year in enumerate(years):
        fig, axs = plt.subplots(2,3, figsize=(10, 5))
        fig.suptitle(str(year), fontsize=16)
        fig.subplots_adjust(hspace=.5, wspace=.001)
        axs = axs.ravel()
        for elem_idx, elem in enumerate(ss_idx_with_trans[year_idx]):
            if elem[0] != 0.0 and elem[1] != 0.0:
                deltaT_val = data_in_u_bin.iloc[int(elem[0]):int(elem[1])]['tempInv [K]']
                axs[elem_idx].hist(deltaT_val, bins=20, density=True)
                sns.kdeplot(deltaT_val, ax=axs[elem_idx])

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

        plt.savefig('DomeC/figures/domeC_u_binned_deltaT_hist_sep_trans_' + str(year) + '.png',
                    bbox_inches='tight', dpi=300)
    exit()
    # Save selected time series in file

    np.savetxt('DomeC/data/deltaT_values_2018_3.txt',
               data_in_u_bin.iloc[int(ss_idx_with_trans[3][3][0]):int(ss_idx_with_trans[3][3][1])]['tempInv [K]'].values)
    np.savetxt('DomeC/data/deltaT_values_2019_2.txt',
               data_in_u_bin.iloc[int(ss_idx_with_trans[4][2][0]):int(ss_idx_with_trans[4][2][1])]['tempInv [K]'].values)

    if 'U2[m s-1]' in data_in_u_bin.columns:
        np.savetxt('DomeC/data/u_values_2018_3.txt',
                   data_in_u_bin.iloc[int(ss_idx_with_trans[3][3][0]):int(ss_idx_with_trans[3][3][1])][
                       'U2[m s-1]'].values)
        np.savetxt('DomeC/data/u_values_2019_2.txt',
                   data_in_u_bin.iloc[int(ss_idx_with_trans[4][2][0]):int(ss_idx_with_trans[4][2][1])][
                       'U2[m s-1]'].values)
    else:
        np.savetxt('DomeC/data/u_values_2018_3.txt',
                   data_in_u_bin.iloc[int(ss_idx_with_trans[3][3][0]):int(ss_idx_with_trans[3][3][1])]['U_9m [m/s]'].values)
        np.savetxt('DomeC/data/u_values_2019_2.txt',
                   data_in_u_bin.iloc[int(ss_idx_with_trans[4][2][0]):int(ss_idx_with_trans[4][2][1])]['U_9m [m/s]'].values)
