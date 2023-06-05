"""
Script to visualize model output when the model was run with a time dependent wind velocity and a perturbed
stability function.

Note: command to run script: python -m one_D_model.post_processing.plot_solution_with_variable_u
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from one_D_model.model import solve_ODE

# Set plotting style
plt.style.use('science')

# Set font sizes for plots
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define directory where simulation output is saved
output_directory = 'output/1000_sim_short_tail_stab_func_multi_noise_var_u/'
sde_directory_vw = output_directory + 'sigma_0_1_start_very/simulations/'
sde_directory_wv = output_directory + 'sigma_0_1_start_weakly/simulations/'

# Load data
SDE_stab_func_sol_delta_T_vw = np.load(sde_directory_vw + 'SDE_stab_func_multi_noise_sol_delta_T.npy')
SDE_stab_func_sol_sf_vw = np.load(sde_directory_vw + 'SDE_stab_func_multi_noise_sol_param.npy')

SDE_stab_func_sol_delta_T_wv = np.load(sde_directory_wv + 'SDE_stab_func_multi_noise_sol_delta_T.npy')
SDE_stab_func_sol_sf_wv = np.load(sde_directory_wv + 'SDE_stab_func_multi_noise_sol_param.npy')

ode_data_vw = np.load(output_directory + 'ode_vw/ODE_sol_delta_T.npy').flatten()
ode_data_wv = np.load(output_directory + 'ode_wv/ODE_sol_delta_T.npy').flatten()

# Replace first hour with nan
SDE_stab_func_sol_delta_T_vw[:3600, :] = np.nan
SDE_stab_func_sol_sf_vw[:3600, :] = np.nan
SDE_stab_func_sol_delta_T_wv[:3600, :] = np.nan
SDE_stab_func_sol_sf_wv[:3600, :] = np.nan
ode_data_vw[:3600] = np.nan
ode_data_wv[:3600] = np.nan


# Load Parameters from file
@dataclass_json
@dataclass
class Parameters:
    t_start: float
    t_end: float
    t_end_h: float
    num_steps: float
    delta_T_0: float
    stab_func_type: str
    Lambda: float
    Q_i: float
    U: float
    z0: float
    zr: float
    grav: float
    Tr: float
    alpha: float
    kappa: float
    cv: float
    rho: float
    cp: float


with open(sde_directory_vw + 'parameters.json', 'r') as file:
    param_data = file.read()

params = Parameters.from_json(param_data)
params.t_span = np.linspace(params.t_start, params.t_end_h, params.num_steps)

# Load wind velocity values
params.u_range_vw = np.loadtxt(sde_directory_vw + 'u_range.txt')
params.u_range_wv = np.loadtxt(sde_directory_wv + 'u_range.txt')

# Find indices of transition region (for short tail)
idx_start_trans_region_vw = (np.abs(params.u_range_vw - 5.6)).argmin()
idx_end_trans_region_vw = (np.abs(params.u_range_vw - 5.9)).argmin()
idx_start_trans_region_wv = (np.abs(params.u_range_wv - 5.6)).argmin()
idx_end_trans_region_wv = (np.abs(params.u_range_wv - 5.9)).argmin()

# Calculate richardson number for every time step
Rb_vw = params.zr * (params.grav / params.Tr) * (SDE_stab_func_sol_delta_T_vw / (params.u_range_vw ** 2))
Rb_wv = params.zr * (params.grav / params.Tr) * (SDE_stab_func_sol_delta_T_wv / (params.u_range_wv ** 2))

# Calculate region in which perturbation was added to phi for first simulations
idx_start_perturbation_vw = 0
idx_end_perturbation_vw = (np.abs(np.mean(Rb_vw, axis=0) - 0.25)).argmin()
idx_start_perturbation_wv = (np.abs(np.mean(Rb_wv, axis=0) - 0.25)).argmin()
idx_end_perturbation_wv = -1

# Make 3 panel plot
fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex='col', sharey='row')
# Remove horizontal space between axes
fig.subplots_adjust(hspace=0, wspace=0)
# --------------------------------------------------------------------------------------------------
# First panel: plot of the u velocity forcing
ax1 = ax[0, 0].twinx()

plt_perturb_region = ax[0, 0].axvspan(params.t_span[idx_start_perturbation_vw], params.t_span[idx_end_perturbation_vw],
                                      alpha=0.3, color='blue')
plt_trans_region = ax[0, 0].axvspan(params.t_span[idx_start_trans_region_vw], params.t_span[idx_end_trans_region_vw],
                                    alpha=0.3, color='red')
plt_u = ax[0, 0].plot(params.t_span, params.u_range_vw, color='red')

colors = plt.cm.Greys(np.linspace(0.1, 0.7, int(np.shape(SDE_stab_func_sol_delta_T_vw)[0] / 2)))
ax1.set_prop_cycle('color', colors)
plt_Rb_all_sim = ax1.plot(params.t_span, Rb_vw[::2, :].T)

plt_Rb_one_sim = ax1.plot(params.t_span, Rb_vw[0, :], color='blue')
plt_Rb_mean = ax1.plot(params.t_span, np.mean(Rb_vw, axis=0), color='green')

ax[0, 0].set_ylabel('u [m/s]', color='red')
# ax[0, 0].set_xlabel('time [h]')
ax[0, 0].tick_params(axis="y", labelcolor='red')
# ax2.set_ylabel(r'$R_b$', color='blue')
# ax2.tick_params(axis="y", labelcolor='blue')

ax[0, 0].set_title('a)', loc='left')
ax1.set_yticklabels([])
# ----------------------------------

ax2 = ax[0, 1].twinx()
# ax2.get_shared_y_axes().join(ax1, ax2)

plt_perturb_region = ax[0, 1].axvspan(params.t_span[idx_start_perturbation_wv], params.t_span[idx_end_perturbation_wv],
                                      alpha=0.3, color='blue')
plt_trans_region = ax[0, 1].axvspan(params.t_span[idx_start_trans_region_wv], params.t_span[idx_end_trans_region_wv],
                                    alpha=0.3, color='red')
plt_u = ax[0, 1].plot(params.t_span, params.u_range_wv, color='red')

ax2.set_prop_cycle('color', colors)
plt_Rb_all_sim = ax2.plot(params.t_span, Rb_wv[::2, :].T)
plt_Rb_one_sim = ax2.plot(params.t_span, Rb_wv[0, :], color='blue')
plt_Rb_mean = ax1.plot(params.t_span, np.mean(Rb_wv, axis=0), color='green')

# ax[0, 1].set_ylabel('u [m/s]', color='green')
# ax[0, 0].set_xlabel('time [h]')
# ax[0, 1].tick_params(axis="y", labelcolor='green')
ax2.set_ylabel(r'$R_b$', color='blue')
ax2.tick_params(axis="y", labelcolor='blue')

ax[0, 1].legend(handles=[plt_perturb_region, plt_trans_region, plt_u[0], plt_Rb_all_sim[0], plt_Rb_one_sim[0], plt_Rb_mean[0]],
                labels=['perturbation region', 'bistable region', 'forcing', r'$R_b$: 500 model runs',
                        r'$R_b$: 1 model run', 'mean'], facecolor='white', edgecolor="black",
                frameon=True, prop={'size': SMALL_SIZE / 2})
ax[0, 1].get_legend().legendHandles[3].set_color('grey')
ax[0, 1].set_title('d)', loc='left')
# --------------------------------------------------------------------------------------------------
# Second panel: plot of delta T over time
ax[1, 0].set_prop_cycle('color', colors)
plt_sims_delta_T_all = ax[1, 0].plot(params.t_span, SDE_stab_func_sol_delta_T_vw[::2, :].T)
plt_sims_delta_T_one = ax[1, 0].plot(params.t_span, SDE_stab_func_sol_delta_T_vw[0, :], color='blue', marker='v',
                                     markevery=0.05, markersize=3)
plt_det_sol = ax[1, 0].plot(params.t_span, ode_data_vw, color='orange', marker='o', markevery=0.05, markersize=3)
plt_sims_delta_T_mean = ax[1, 0].plot(params.t_span, np.mean(SDE_stab_func_sol_delta_T_vw, axis=0), color='green',
                                      marker='s', markevery=0.05, markersize=3)

ax[1, 0].set_ylabel(r'$\Delta T$ [K]')
# ax[1, 0].set_xlabel('time [h]')
ax[1, 0].set_title('b)', loc='left')
# -----------------------------------------------------
ax[1, 1].set_prop_cycle('color', colors)
plt_sims_delta_T_all = ax[1, 1].plot(params.t_span, SDE_stab_func_sol_delta_T_wv[::2, :].T)
plt_sims_delta_T_one = ax[1, 1].plot(params.t_span, SDE_stab_func_sol_delta_T_wv[0, :], color='blue', marker='v',
                                     markevery=0.05, markersize=3)
plt_det_sol = ax[1, 1].plot(params.t_span, ode_data_wv, color='orange', marker='o', markevery=0.05, markersize=3)
plt_sims_delta_T_mean = ax[1, 1].plot(params.t_span, np.mean(SDE_stab_func_sol_delta_T_wv, axis=0), color='green',
                                      marker='s', markevery=0.05, markersize=3)

# ax[1, 0].set_ylabel(r'$\Delta T$ [K]')
# ax[1, 0].set_xlabel('time [h]')
ax[1, 1].legend(handles=[plt_sims_delta_T_all[0], plt_sims_delta_T_one[0], plt_sims_delta_T_mean[0], plt_det_sol[0]],
                labels=['500 model runs', '1 model run', 'mean', 'ODE'], facecolor='white',
                edgecolor="black", frameon=True, prop={'size': SMALL_SIZE / 2})
ax[1, 1].get_legend().legendHandles[0].set_color('grey')
ax[1, 1].set_title('e)', loc='left')

# --------------------------------------------------------------------------------------------------
# Third panel: plot of perturbed stability function over time
ax[2, 0].set_prop_cycle('color', colors)
plt_sims_sf_all = ax[2, 0].plot(params.t_span, SDE_stab_func_sol_sf_vw[::2, :].T)
plt_sims_sf_one = ax[2, 0].plot(params.t_span, SDE_stab_func_sol_sf_vw[0, :], color='blue', marker='v', markevery=0.05,
                                markersize=3)
plt_sims_sf_mean = ax[2, 0].plot(params.t_span, np.mean(SDE_stab_func_sol_sf_vw, axis=0), color='green', marker='s',
                                 markevery=0.05, markersize=3)
ax[2, 0].set_ylabel(r'$\phi$')
ax[2, 0].set_xlabel('time [h]')
ax[2, 0].set_title('c)', loc='left')
# -----------------------------------------------------
ax[2, 1].set_prop_cycle('color', colors)
plt_sims_sf_all = ax[2, 1].plot(params.t_span, SDE_stab_func_sol_sf_wv[::2, :].T)
plt_sims_sf_one = ax[2, 1].plot(params.t_span, SDE_stab_func_sol_sf_wv[0, :], color='blue', marker='v', markevery=0.05,
                                markersize=3)
plt_sims_sf_mean = ax[2, 1].plot(params.t_span, np.mean(SDE_stab_func_sol_sf_wv, axis=0), color='green', marker='s',
                                 markevery=0.05, markersize=3)
# ax[2, 0].set_ylabel(r'$\phi$')
ax[2, 1].set_xlabel('time [h]')
ax[2, 1].legend(handles=[plt_sims_sf_all[0], plt_sims_sf_one[0], plt_sims_sf_mean[0]],
                labels=['500 model runs', '1 model run', 'mean'], facecolor='white',
                edgecolor="black", frameon=True, prop={'size': SMALL_SIZE / 2})
ax[2, 1].get_legend().legendHandles[0].set_color('grey')
ax[2, 1].set_title('f)', loc='left')
# --------------------------------------------------------------------------------------------------
# plt.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()
plt.savefig(output_directory + 'solution_with_time_dependent_u.pdf', bbox_inches='tight', dpi=300)

# To clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close('all')  # Closes all the figure windows.
