import numpy as np
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import os

from one_D_model.utils import plot_output as plot

# Define directory where simulation output is saved
output_directory = 'output/sigma_0_04/simulations/'
#
# from one_D_model.model import run_SDE_model
# import os
#
# _, _, files = next(os.walk(output_directory+ 'temporary/'))
# num_sim = len(files)
# run_SDE_model.combine_npy_files(num_sim, 'output/20230322_1422_31/', 'SDE_stab_func_poisson_sol_delta_T')

# Load data
#SDE_sol_delta_T = np.load(output_directory + 'SDE_sol_delta_T.npy')
# #
SDE_u_sol_delta_T = np.load(output_directory + 'SDE_u_sol_delta_T.npy')
SDE_u_sol_u = np.load(output_directory + 'SDE_u_sol_param.npy')
# SDE_u_sol_delta_T = np.load(output_directory + 'SDE_u_internal_var_sol_delta_T.npy')
# SDE_u_sol_u = np.load(output_directory + 'SDE_u_internal_var_sol_param.npy')
#
# SDE_Qi_sol_delta_T = np.load(output_directory + 'SDE_Qi_sol_delta_T.npy')
# SDE_Qi_sol_Qi = np.load(output_directory + 'SDE_Qi_sol_param.npy')
#
# SDE_lambda_sol_delta_T = np.load(output_directory + 'SDE_lambda_sol_delta_T.npy')
# SDE_lambda_sol_lambda = np.load(output_directory + 'SDE_lambda_sol_param.npy')
#
# SDE_stab_func_sol_delta_T = np.load(output_directory + 'SDE_stab_func_sol_delta_T.npy')
# SDE_stab_func_sol_sf = np.load(output_directory + 'SDE_stab_func_sol_param.npy')
#
# SDE_stab_func_sol_poisson_delta_T = np.load(output_directory + 'SDE_stab_func_poisson_sol_delta_T.npy')

# SDE_z0_sol_delta_T = np.load(output_directory + 'SDE_z0_sol_delta_T.npy', allow_pickle=True)
# SDE_z0_sol_z0 = np.load(output_directory + 'SDE_z0_sol_param.npy', allow_pickle=True)
#
# SDE_stab_func_sol_delta_T = np.load(output_directory + 'SDE_stab_func_multi_noise_sol_delta_T.npy')
# SDE_stab_func_sol_sf = np.load(output_directory + 'SDE_stab_func_multi_noise_sol_param.npy')


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
    stab_func_type: str
    Lambda: float
    Q_i: float
    z0: float
    zr: float
    grav: float
    Tr: float
    alpha: float
    kappa: float
    cv: float
    rho: float
    cp: float


with open(output_directory + 'parameters.json', 'r') as file:
    param_data = file.read()

params = Parameters.from_json(param_data)

params.sol_directory_path = output_directory.rsplit('/', 2)[0] + '/'
params.t_span_h = np.linspace(params.t_start, params.t_end_h, params.num_steps)

# Make directory for visualizations
if not os.path.exists(params.sol_directory_path + 'visualizations/'):
    os.makedirs(params.sol_directory_path + 'visualizations/')

# Make distribution plots
# plot.make_distribution_plot(np.array(SDE_sol_delta_T).flatten(), params, 'visualizations/SDE_sol_delta_T_distribution.pdf',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_u_sol_delta_T.flatten(), params, 'visualizations/SDE_u_sol_delta_T_distribution.pdf',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_Qi_sol_delta_T.flatten(), params, 'visualizations/SDE_Qi_sol_delta_T_distribution.pdf',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_lambda_sol_delta_T.flatten(), params, 'visualizations/SDE_lambda_sol_delta_T_distribution.pdf',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_stab_func_sol_delta_T.flatten(), params, 'visualizations/SDE_stab_func_sol_delta_T_distribution.pdf',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_stab_func_sol_poisson_delta_T.flatten(), params, 'visualizations/SDE_stab_func_sol_delta_T_distribution.pdf',
#                             r'$\Delta T$ [K]')
# plot.make_distribution_plot(SDE_z0_sol_delta_T.flatten(), params, 'visualizations/SDE_z0_sol_delta_T_distribution.pdf',
#                             r'$\Delta T$ [K]')

# Overlay potential plot with distribution plot
#plot.plot_potentials_and_output_distribution(params, SDE_sol_delta_T.flatten())
# exit()
# Plot ten time series for every simulation type
if params.num_simulation >= 10:
    vis_idx = np.linspace(0, params.num_simulation - 1, 10).astype(int)
    for idx in vis_idx:
        # plot.make_2D_plot(params, params.t_span_h, SDE_sol_delta_T[idx][:].flatten(),
        #                   'visualizations/SDE_sol_delta_T_over_time_sim' + str(idx) + '.pdf', xlabel='t [h]',
        #                   ylabel=r'$\Delta T$ [K]')

        plot.make_2D_multi_line_plot(params, params.t_span_h,
                                     np.array([SDE_u_sol_delta_T[idx, :], SDE_u_sol_u[idx, :]]).T,
                                     [r'$\Delta T$', 'U'], 'visualizations/SDE_u_sol_u_delta_T_over_time_sim' + str(idx) + '.pdf',
                                     xlabel='t [h]', ylabel=r'$\Delta T$ [K]', ylabel2='U [m/s]')
        #
        # plot.make_2D_multi_line_plot(params, params.t_span_h,
        #                              np.array([SDE_Qi_sol_delta_T[idx, :], 1/10*SDE_Qi_sol_Qi[idx, :]]).T,
        #                              [r'$\Delta T$', r'$0.1*Q_i$'],
        #                              'visualizations/SDE_Qi_sol_Qi_delta_T_over_time_sim' + str(idx) + '.pdf',
        #                              xlabel='t [h]',
        #                              ylabel=r'$\Delta T$ [K]')
        #
        # plot.make_2D_multi_line_plot(params, params.t_span_h,
        #                              np.array([SDE_lambda_sol_delta_T[idx, :], SDE_lambda_sol_lambda[idx, :]]).T,
        #                              [r'$\Delta T$', r'$\lambda$'],
        #                              'visualizations/SDE_lambda_sol_lambda_delta_T_over_time_sim' + str(idx) + '.pdf',
        #                              xlabel='t [h]',
        #                              ylabel=r'$\Delta T$ [K]')
        #
        # plot.make_2D_multi_line_plot(params, params.t_span_h,
        #                              np.array([SDE_stab_func_sol_delta_T[idx, :], SDE_stab_func_sol_sf[idx, :]]).T,
        #                              [r'$\Delta T$', r'$\phi$'],
        #                              f'visualizations/SDE_stab_func_sol_sf_delta_T_over_time_sim{idx}.pdf',
        #                              xlabel='t [h]', ylabel=r'$\Delta T$ [K]', ylabel2=r'$\phi$')
        #
        # plot.make_2D_plot(params, params.t_span_h, SDE_stab_func_sol_poisson_delta_T[idx][:].flatten(),
        #                   'visualizations/SDE_sol_delta_T_over_time_sim' + str(idx) + '.pdf', xlabel='t [h]',
        #                   ylabel=r'$\Delta T$ [K]')

        # plot.make_2D_multi_line_plot(params, params.t_span_h,
        #                              np.array([SDE_z0_sol_delta_T[idx, 1:], 100*SDE_z0_sol_z0[idx, 1:]]).T,
        #                              [r'$\Delta T$', r'$100*z_0$'],
        #                              'visualizations/SDE_z0_sol_z0_delta_T_over_time_sim' + str(idx) + '.pdf',
        #                              xlabel='t [h]',
        #                              ylabel=r'$\Delta T$ [K]')
