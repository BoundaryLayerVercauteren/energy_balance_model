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
sde_directory = output_directory + 'sigma_0_1/simulations/'

# Load data
SDE_stab_func_sol_delta_T = np.load(sde_directory + 'SDE_stab_func_multi_noise_sol_delta_T.npy')
SDE_stab_func_sol_sf = np.load(sde_directory + 'SDE_stab_func_multi_noise_sol_param.npy')
ode_data = np.load(output_directory + 'ode/ODE_sol_delta_T.npy').flatten()

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


with open(sde_directory + 'parameters.json', 'r') as file:
    param_data = file.read()

params = Parameters.from_json(param_data)
params.t_span = np.linspace(params.t_start, params.t_end_h, params.num_steps)

# Load wind velocity values
params.u_range = np.loadtxt(sde_directory + 'u_range.txt')
params.u_range_o = np.loadtxt(output_directory + 'ode/u_range.txt')

# Find indices of transition region (for short tail)
idx_start_trans_region = (np.abs(params.u_range - 5.6)).argmin()
idx_end_trans_region = (np.abs(params.u_range - 5.9)).argmin()

# Calculate richardson number for every time step
Rb = params.zr * (params.grav / params.Tr) * (SDE_stab_func_sol_delta_T / (params.u_range ** 2))

# Calculate region in which perturbation was added to phi for first simulations
idx_start_perturbation = np.argmax(Rb[0, :] > 0.25)
idx_end_perturbation = np.argmin(Rb[0, :] > 0.25)

# Make 3 panel plot
fig, ax = plt.subplots(3, 1, figsize=(10, 10))

# First panel: plot of the u velocity forcing
ax2 = ax[0].twinx()

plt_perturb_region = ax[0].axvspan(params.t_span[idx_start_perturbation], params.t_span[idx_end_perturbation],
                                   alpha=0.3, color='blue')
plt_trans_region = ax[0].axvspan(params.t_span[idx_start_trans_region], params.t_span[idx_end_trans_region], alpha=0.3,
                                 color='green')

plt_u = ax[0].plot(params.t_span, params.u_range, color='green')
ax[0].set_ylabel('u [m/s]', color='green')
ax[0].set_xlabel('time [h]')
ax[0].tick_params(axis="y", labelcolor='green')

colors = plt.cm.Greys(np.linspace(0.1, 0.7, int(np.shape(SDE_stab_func_sol_delta_T)[0] / 2)))
ax2.set_prop_cycle('color', colors)
plt_Rb_all_sim = ax2.plot(params.t_span, Rb[::2, :].T)

plt_Rb_one_sim = ax2.plot(params.t_span, Rb[0, :], color='blue')
ax2.set_ylabel(r'$R_b$', color='blue')
ax2.tick_params(axis="y", labelcolor='blue')

ax[0].legend(handles=[plt_perturb_region, plt_trans_region, plt_u[0], plt_Rb_all_sim[0], plt_Rb_one_sim[0]],
             labels=['perturbation region', 'bistable region', 'forcing', r'$R_b$: 500 model runs',
                     r'$R_b$: 1 model run'], facecolor='white', loc="upper right", edgecolor="black",
             frameon=True)
ax[0].get_legend().legendHandles[3].set_color('grey')
ax[0].set_title('a)', loc='left')

# Second panel: plot of delta T over time
ax[1].set_prop_cycle('color', colors)
plt_sims_delta_T_all = ax[1].plot(params.t_span, SDE_stab_func_sol_delta_T[::2, :].T)
plt_sims_delta_T_one = ax[1].plot(params.t_span, SDE_stab_func_sol_delta_T[0, :], color='blue', marker='v', markevery=0.05, markersize=3)
plt_det_sol = ax[1].plot(params.t_span, ode_data, color='orange', marker='o', markevery=0.05, markersize=3)
plt_sims_delta_T_mean = ax[1].plot(params.t_span, np.mean(SDE_stab_func_sol_delta_T, axis=0), color='red', marker='s', markevery=0.05, markersize=3)

ax[1].set_ylabel(r'$\Delta T$ [K]')
ax[1].set_xlabel('time [h]')
ax[1].legend(handles=[plt_sims_delta_T_all[0], plt_sims_delta_T_one[0], plt_sims_delta_T_mean[0], plt_det_sol[0]],
             labels=['500 model runs', '1 model run', 'mean', 'ODE'], facecolor='white',
             loc="upper right", edgecolor="black", frameon=True)
ax[1].get_legend().legendHandles[0].set_color('grey')
ax[1].set_title('b)', loc='left')

# Third panel: plot of perturbed stability function over time
ax[2].set_prop_cycle('color', colors)
plt_sims_sf_all = ax[2].plot(params.t_span, SDE_stab_func_sol_sf[::2, :].T)
plt_sims_sf_one = ax[2].plot(params.t_span, SDE_stab_func_sol_sf[0, :], color='blue', marker='v', markevery=0.05, markersize=3)
plt_sims_sf_mean = ax[2].plot(params.t_span, np.mean(SDE_stab_func_sol_sf, axis=0), color='red', marker='s', markevery=0.05, markersize=3)
ax[2].set_ylabel(r'$\phi$')
ax[2].set_xlabel('time [h]')
ax[2].legend(handles=[plt_sims_sf_all[0], plt_sims_sf_one[0], plt_sims_sf_mean[0]],
             labels=['500 model runs', '1 model run', 'mean'], facecolor='white', loc="upper right", edgecolor="black",
             frameon=True)
ax[2].get_legend().legendHandles[0].set_color('grey')
ax[2].set_title('c)', loc='left')

fig.tight_layout()
plt.savefig(output_directory + 'solution_with_time_dependent_u.pdf', bbox_inches='tight', dpi=300)

# To clear memory
plt.cla()  # Clear the current axes.
plt.clf()  # Clear the current figure.
plt.close('all')  # Closes all the figure windows.
