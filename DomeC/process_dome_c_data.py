import pandas as pd

import one_D_model.utils.plot_output as plot


def load_data(file_name):
    # load data and skip comments in header
    return pd.read_csv(file_name, sep='\t', skiprows=32, index_col=False, na_values='NaN')


def make_data_type_numeric(values):
    # change data type
    return pd.to_numeric(values, errors='coerce')


def convert_temp_unit(values):
    # convert celsius to kelvin
    return values + 273.15


def convert_to_data_format(values):
    # convert local time to date format
    return pd.to_datetime(values, format='%Y-%m-%d_%H:%M:%S')


def select_time_period(values, period):
    # select only data for given month interval
    #return values.loc[values['Local Time (UTC+8h)'].dt.month.between(period[0], period[1]), values.columns]
    return values[(values['Local Time (UTC+8h)'] >= '2017-07-01 00:00:00') & (values['Local Time (UTC+8h)'] <= '2017-07-15 23:50:00')]


def calculate_rad_force(values):
    # calculate radiative forcing
    values['radForce'] = values['SWdn[W m-2]'] - values['SWup[W m-2]'] + values['LWdn[W m-2]']
    return values


def calculate_temp_inv(values):
    # calculate temperature difference
    values['tempInv [K]'] = values['T2[K]'] - values['TS[K]']
    return values


def prepare_dome_c_data():
    """prepare data so that it can be used."""
    # load data
    input_file = 'DomeC/DC_2017_10min.ascii'
    data = load_data(input_file)
    # convert column content to correct format
    for i in data.columns[1:]:
        data[i] = pd.to_numeric(data[i], errors='coerce')
        if 'oC' in i:
            data[i] = convert_temp_unit(data[i])
            data = data.rename(columns={i: i.replace('oC', 'K')})
    data.iloc[:, 0] = convert_to_data_format(data.iloc[:, 0])
    # calculate radiative forcing
    data = calculate_rad_force(data)
    # calculate values for temperature inversion
    data = calculate_temp_inv(data)
    # select subset where the forcing is less than a given value and from a specific time period
    #data.loc[data['radForce'] >= 80, data.columns] = np.nan
    sub_data = data.copy()
    period = [8, 9]
    sub_data = select_time_period(sub_data, period)

    return sub_data.copy()


def main(params):
    # Prepare data
    data = prepare_dome_c_data()
    # Plot temperature inversion over time
    plot.make_2D_plot(params, data['Local Time (UTC+8h)'], data['tempInv [K]'], 'dome_c.png', xlabel='t [d]', ylabel=r'$\Delta T$ [K]')
