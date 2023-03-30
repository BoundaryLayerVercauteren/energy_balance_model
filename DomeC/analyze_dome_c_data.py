import pandas as pd
import numpy as np
from DomeC.process_dome_c_data import prepare_dome_c_data


def find_data_in_u_bin(data, bin_mean=5.5, bin_var=1.5):
    if 'U2[m s-1]' in data.columns:
        data.loc[(data['U2[m s-1]'] < (bin_mean - bin_var)) | (data['U2[m s-1]'] > (bin_mean + bin_var))] = np.nan
    else:
        data.drop(data[(data['U_9m [m/s]'] < (bin_mean - bin_var)) | (data['U_9m [m/s]'] > (bin_mean + bin_var))].index,
                  inplace=True)
    return data


def calculate_standard_deviation(values):
    return values['U_9m [m/s]'].std()


def calculate_variance(values):
    return values['U_9m [m/s]'].var()


def calculate_mean(values):
    return values['U_9m [m/s]'].mean()


def get_yearly_statistics(data):
    statistics = {}
    for cur_year in data['year'].unique():
        cur_year_data = data.loc[data['year'] == cur_year]
        statistics[cur_year] = (np.round(calculate_mean(cur_year_data), 2),
                                np.round(calculate_standard_deviation(cur_year_data), 2),
                                np.round(calculate_variance(cur_year_data), 2))

    return statistics


def get_Dome_C_statistics(year=2017):
    # Prepare data
    data_season = prepare_dome_c_data()
    # Add year column
    data_season['year'] = pd.DatetimeIndex(data_season['timestamp']).year
    # print('The (mean, standard deviation, variance) for all years are:')
    # get_yearly_statistics(data_season)
    # # Select data where u is in specific bin
    data_in_u_bin_4_7 = find_data_in_u_bin(data_season.copy(), bin_mean=5.5, bin_var=1.5)
    all_statistics = get_yearly_statistics(data_in_u_bin_4_7)

    # # Note: var(u) = sigma**2/(2*r)
    # print(f'The (mean, variance) of u in the model are: {(5.6, (0.08 ** 2) / (2 * 0.005))}')
    return all_statistics[year]