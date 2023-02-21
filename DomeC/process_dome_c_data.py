import pandas as pd
import numpy as np
#import one_D_model.utils.plot_output as plot


def load_data(file_name):
    # load data and skip comments in header
    if 'ascii' in file_name:
        values = pd.read_csv(file_name, sep='\t', skiprows=32, index_col=False, na_values='NaN')
    elif 'tab' in file_name:
        values = pd.read_csv(file_name, sep='\t', skiprows=26, index_col=False, na_values='NaN')
    return values


def rename_columns(values, file_name):
    if 'ascii' in file_name:
        values = values.rename(columns={'Local Time (UTC+8h)': 'timestamp'})
    elif 'tab' in file_name:
        values = values.rename(columns={'Date/Time local (local time (UTC+08))': 'timestamp',
                                        'TTT [°C] (at 3m mean height above groun...)': 'T_3m [oC]',
                                        'TTT [°C] (at 9m mean height above groun...)': 'T_9m [oC]',
                                        'TTT [°C] (at 17m mean height above grou...)': 'T_17m [oC]',
                                        'TTT [°C] (at 25m mean height above grou...)': 'T_25m [oC]',
                                        'TTT [°C] (at 32m mean height above grou...)': 'T_32m [oC]',
                                        'TTT [°C] (at 40m mean height above grou...)': 'T_40m [oC]',
                                        'ff [m/s] (at 3m mean height above groun...)': 'U_3m [m/s]',
                                        'ff [m/s] (at 9m mean height above groun...)': 'U_9m [m/s]',
                                        'ff [m/s] (at 17m mean height above grou...)': 'U_17m [m/s]',
                                        'ff [m/s] (at 25m mean height above grou...)': 'U_25m [m/s]',
                                        'ff [m/s] (at 32m mean height above grou...)': 'U_32m [m/s]',
                                        'ff [m/s] (at 40m mean height above grou...)': 'U_40m [m/s]'})
    return values


def make_data_type_numeric(values):
    # change data type
    return pd.to_numeric(values, errors='coerce')


def convert_temp_unit(values):
    # convert celsius to kelvin
    return values + 273.15


def convert_to_datetime_format(values):
    # convert local time to date format
    try:
        values = pd.to_datetime(values, format='%Y-%m-%d_%H:%M:%S')
    except Exception:
        values = pd.to_datetime(values, format='%Y-%m-%dT%H:%M:%S')
    return values


def select_time_period(values, period):
    # select only data for given month interval
    if len(period) == 2:
        return values.loc[values['timestamp'].dt.month.between(period[0], period[1]), values.columns]
    else:
        return values[(values['timestamp'] >= '2017-07-01 00:00:00') & (values['timestamp'] <= '2017-07-15 23:50:00')]


def calculate_rad_force(values):
    # calculate radiative forcing
    values['radForce'] = values['SWdn[W m-2]'] - values['SWup[W m-2]'] + values['LWdn[W m-2]']
    return values


def calculate_temp_inv(values):
    # Calculate temperature difference
    if 'T2[K]' in values.columns:
        values['tempInv [K]'] = values['T2[K]'] - values['TS[K]']
    else:
        values['tempInv [K]'] = values['T_9m [K]'] - values['T_3m [K]']
    return values


def prepare_dome_c_data():
    """prepare data so that it can be used."""
    # Load data
    #input_file = 'DomeC/data/10min/DC_2017_10min.ascii'
    input_file_temp = 'DomeC/data/30min/Genthon-etal_2021_DomeC-Temp.tab'
    input_file_wind = 'DomeC/data/30min/Genthon-etal_2021_DomeC-WindSpeed.tab'
    # data = load_data(input_file)
    data_temp = load_data(input_file_temp)
    data_wind = load_data(input_file_wind)
    data = data_temp.join(data_wind.set_index('Date/Time local (local time (UTC+08))'), on='Date/Time local (local time (UTC+08))')
    # Rename columns
    data = rename_columns(data, input_file_temp)
    # Convert column content to correct format
    for i in data.columns[1:]:
        data[i] = pd.to_numeric(data[i], errors='coerce')
        if 'oC' in i:
            data[i] = convert_temp_unit(data[i])
            data = data.rename(columns={i: i.replace('oC', 'K')})
    # Convert timestamp column to correct format
    data.iloc[:, 0] = convert_to_datetime_format(data.iloc[:, 0])
    # # calculate radiative forcing
    # data = calculate_rad_force(data)
    # Calculate values for temperature inversion
    data = calculate_temp_inv(data)
    # # select subset where the forcing is less than a given value and from a specific time period
    # data.loc[data['radForce'] >= 80, data.columns] = np.nan
    sub_data = data.copy()
    sub_data_season = select_time_period(sub_data, [3,9])

    return sub_data_season


def main():
    # Prepare data
    data_season = prepare_dome_c_data()
    # Plot temperature inversion over time
    #plot.make_2D_plot(params, data_day['Local Time (UTC+8h)'], data_day['tempInv [K]'], 'dome_c_day.png', xlabel='t [d]', ylabel=r'$\Delta T$ [K]')
    # Plot distribution for whole season
    #plot.make_distribution_plot(data_season['tempInv [K]'], params, 'dome_c_season_distribution.png', r'$\Delta T$ [K]')

    return data_season


if __name__ == "__main__":
    main()
