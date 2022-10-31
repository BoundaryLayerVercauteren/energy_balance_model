import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import os

from one_D_model.utils import plot_output as plot

# Define directory where simulation output is saved
output_directory = 'output/new_sim/simulations/'

# Load data
SDE_sol_delta_T = np.load(output_directory + 'SDE_sol_delta_T.npy')

SDE_u_sol_delta_T = np.load(output_directory + 'SDE_u_sol_delta_T.npy')
SDE_u_sol_u = np.load(output_directory + 'SDE_u_sol_param.npy')

SDE_Qi_sol_delta_T = np.load(output_directory + 'SDE_Qi_sol_delta_T.npy')
SDE_Qi_sol_Qi = np.load(output_directory + 'SDE_Qi_sol_param.npy')

SDE_lambda_sol_delta_T = np.load(output_directory + 'SDE_lambda_sol_delta_T.npy')
SDE_lambda_sol_lambda = np.load(output_directory + 'SDE_lambda_sol_param.npy')

SDE_stab_func_sol_delta_T = np.load(output_directory + 'SDE_stab_func_sol_delta_T.npy')
SDE_stab_func_sol_sf = np.load(output_directory + 'SDE_stab_func_sol_param.npy')

# SDE_z0_sol_delta_T = np.load(output_directory + 'SDE_z0_sol_delta_T.npy', allow_pickle=True)
# SDE_z0_sol_z0 = np.load(output_directory + 'SDE_z0_sol_param.npy', allow_pickle=True)
# SDE_z0_sol_time = np.load(output_directory + 'SDE_z0_sol_time.npy', allow_pickle=True)


# Load parameters
@dataclass_json
@dataclass
class Parameters:
    t_start: float
    t_end_h: float
    t_end: float
    dt: float
    num_steps: float
    num_simulation: int


with open(output_directory + 'parameters.json', 'r') as file:
    param_data = file.read()

params = Parameters.from_json(param_data)

params.sol_directory_path = output_directory.rsplit('/', 2)[0] + '/'
print(params.sol_directory_path)
params.t_span_h = np.linspace(params.t_start, params.t_end_h, params.num_steps)

# Make directory for visualizations
if not os.path.exists(params.sol_directory_path + 'visualizations/'):
    os.makedirs(params.sol_directory_path + 'visualizations/')

# Make distribution plots
plot.make_distribution_plot(np.array(SDE_sol_delta_T).flatten(), params, 'visualizations/SDE_sol_delta_T_distribution.png',
                            r'$\Delta T$ [K]')
plot.make_distribution_plot(SDE_u_sol_delta_T.flatten(), params, 'visualizations/SDE_u_sol_delta_T_distribution.png',
                            r'$\Delta T$ [K]')
plot.make_distribution_plot(SDE_Qi_sol_delta_T.flatten(), params, 'visualizations/SDE_Qi_sol_delta_T_distribution.png',
                            r'$\Delta T$ [K]')
plot.make_distribution_plot(SDE_lambda_sol_delta_T.flatten(), params, 'visualizations/SDE_lambda_sol_delta_T_distribution.png',
                            r'$\Delta T$ [K]')
plot.make_distribution_plot(SDE_stab_func_sol_delta_T.flatten(), params, 'visualizations/SDE_stab_func_sol_delta_T_distribution.png',
                            r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_z0_sol_delta_T.flatten(), params, 'visualizations/SDE_z0_sol_delta_T_distribution.png',
#                             r'$\Delta T$ [K]')

# Plot ten time series for every simulation type
if params.num_simulation > 10:
    vis_idx = np.linspace(0, params.num_simulation - 1, 10).astype(int)
    for idx in vis_idx:
        plot.make_2D_plot(params, params.t_span_h, SDE_sol_delta_T[idx][:].flatten(),
                          'visualizations/SDE_sol_delta_T_over_time_sim' + str(idx) + '.png', xlabel='t [h]',
                          ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_u_sol_delta_T[idx, :], SDE_u_sol_u[idx, :]]).T,
                                     [r'$\Delta T$', 'u'], 'visualizations/SDE_u_sol_u_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_Qi_sol_delta_T[idx, :], 1/10*SDE_Qi_sol_Qi[idx, :]]).T,
                                     [r'$\Delta T$', r'$0.1*Q_i$'],
                                     'visualizations/SDE_Qi_sol_Qi_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_lambda_sol_delta_T[idx, :], SDE_lambda_sol_lambda[idx, :]]).T,
                                     [r'$\Delta T$', r'$\lambda$'],
                                     'visualizations/SDE_lambda_sol_lambda_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_stab_func_sol_delta_T[idx, :], SDE_stab_func_sol_sf[idx, :]]).T,
                                     [r'$\Delta T$', r'$\phi$'],
                                     'visualizations/SDE_stab_func_sol_sf_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        # plot.make_2D_multi_line_plot(params, params.t_span_h,
        #                              np.array([SDE_z0_sol_delta_T[idx, :], 100*SDE_z0_sol_z0[idx, :]]).T,
        #                              [r'$\Delta T$', r'$100*z_0$'],
        #                              'visualizations/SDE_z0_sol_z0_delta_T_over_time_sim' + str(idx) + '.png',
        #                              xlabel='t [h]',
        #                              ylabel=r'$\Delta T$ [K]')
