import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json

from one_D_model.utils import plot_output as plot

# Load data
output_directory = 'output/1000_sim/'

SDE_sol_delta_T = np.load(output_directory + 'SDE_sol_delta_T.npy')

SDE_u_sol_delta_T = np.load(output_directory + 'SDE_u_sol_delta_T.npy')
SDE_u_sol_u = np.load(output_directory + 'SDE_u_sol_u.npy')

SDE_Qi_sol_delta_T = np.load(output_directory + 'SDE_Qi_sol_delta_T.npy')
SDE_Qi_sol_Qi = np.load(output_directory + 'SDE_Qi_sol_Qi.npy')

SDE_lambda_sol_delta_T = np.load(output_directory + 'SDE_lambda_sol_delta_T.npy')
SDE_lambda_sol_lambda = np.load(output_directory + 'SDE_lambda_sol_lambda.npy')

SDE_stab_func_sol_delta_T = np.load(output_directory + 'SDE_stab_func_sol_delta_T.npy')
SDE_stab_func_sol_sf = np.load(output_directory + 'SDE_stab_func_sol_sf.npy')

SDE_z0_sol_delta_T = np.load(output_directory + 'SDE_z0_sol_delta_T.npy', allow_pickle=True)
SDE_z0_sol_z0 = np.load(output_directory + 'SDE_z0_sol_z0.npy', allow_pickle=True)
SDE_z0_sol_time = np.load(output_directory + 'SDE_z0_sol_time.npy', allow_pickle=True)


# Load parameters
@dataclass_json
@dataclass
class Parameters:
    t_start: float
    t_end_h: float
    t_end: float
    dt: float
    num_steps: float
    #t_span: np.ndarray
    #t_span_h: np.ndarray
    num_simulation: int


with open(output_directory + 'parameters.json', 'r') as file:
    param_data = file.read()

params = Parameters.from_json(param_data)

params.sol_directory_path = output_directory
params.t_span_h = np.linspace(params.t_start, params.t_end_h, params.num_steps)

# Make distribution plots
# plot.make_distribution_plot(np.array(SDE_sol_delta_T).flatten(), params, 'SDE_sol_delta_T_distribution.png',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_u_sol_delta_T.flatten(), params, 'SDE_u_sol_delta_T_distribution.png',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_Qi_sol_delta_T.flatten(), params, 'SDE_Qi_sol_delta_T_distribution.png',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_lambda_sol_delta_T.flatten(), params, 'SDE_lambda_sol_delta_T_distribution.png',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_stab_func_sol_delta_T.flatten(), params, 'SDE_stab_func_sol_delta_T_distribution.png',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_z0_sol_delta_T.flatten(), params, 'SDE_z0_sol_delta_T_distribution.png',
#                             r'$\Delta T$ [K]')

# Plot ten time series
if params.num_simulation > 10:
    vis_idx = np.linspace(0, params.num_simulation - 1, 10).astype(int)
    for idx in vis_idx:
        plot.make_2D_plot(params, params.t_span_h, SDE_sol_delta_T[idx][:].flatten(),
                          'SDE_sol_delta_T_over_time_sim' + str(idx) + '.png', xlabel='t [h]',
                          ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_u_sol_delta_T[idx, :], SDE_u_sol_u[idx, :]]).T,
                                     [r'$\Delta T$', 'u'], 'SDE_u_sol_u_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_Qi_sol_delta_T[idx, :], SDE_Qi_sol_Qi[idx, :]]).T,
                                     [r'$\Delta T$', r'$Q_i$'],
                                     'SDE_Qi_sol_Qi_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_lambda_sol_delta_T[idx, :], SDE_lambda_sol_lambda[idx, :]]).T,
                                     [r'$\Delta T$', r'$\lambda$'],
                                     'SDE_lambda_sol_lambda_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_stab_func_sol_delta_T[idx, :], SDE_stab_func_sol_sf[idx, :]]).T,
                                     [r'$\Delta T$', r'$\phi$'],
                                     'SDE_stab_func_sol_sf_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, SDE_z0_sol_time[idx, :],
                                     np.array([SDE_z0_sol_delta_T[idx, :], SDE_z0_sol_z0[idx, :]]).T,
                                     [r'$\Delta T$', r'$z_0$'],
                                     'SDE_z0_sol_z0_delta_T_over_time_sim' + str(idx) + '.png',
                                     xlabel='t [h]',
                                     ylabel=r'$\Delta T$ [K]')
