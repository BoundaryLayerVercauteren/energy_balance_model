import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

plt.style.use('science')

# set font sizes for plots
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
output_directory = 'output/1000_sim_short_tail_stab_func_multi_noise_var_u/sigma_0_1/simulations/'

# Load data
SDE_stab_func_sol_delta_T = np.load(output_directory + 'SDE_stab_func_multi_noise_sol_delta_T.npy')
SDE_stab_func_sol_sf = np.load(output_directory + 'SDE_stab_func_multi_noise_sol_param.npy')

# Define time span
t_span = np.linspace(0, 24, 24 * 3600)

# Define u
u = np.loadtxt(output_directory + 'u_range.txt')

# Find indices of transition region (for short tail)
idx_start_trans_region = (np.abs(u - 5.6)).argmin()
idx_end_trans_region = (np.abs(u - 5.9)).argmin()

# Make 3 panel plot
fig, ax = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2, 2]})

ax[0].axvspan(t_span[idx_start_trans_region], t_span[idx_end_trans_region], alpha=0.3, color='red',
              label='transition region')
ax[0].plot(t_span, u)
ax[0].set_ylabel('u [m/s]')
ax[0].set_xlabel('time [h]')
ax[0].legend(facecolor='white', loc="upper right", edgecolor="black", frameon=True)

colors = plt.cm.Greys(np.linspace(0, 0.6, np.shape(SDE_stab_func_sol_delta_T)[0]))
ax[1].set_prop_cycle('color', colors)
plt_sims_delta_T_all = ax[1].plot(t_span, SDE_stab_func_sol_delta_T[:, :].T)
plt_sims_delta_T_one = ax[1].plot(t_span, SDE_stab_func_sol_delta_T[0, :], color='blue')
plt_sims_delta_T_mean = ax[1].plot(t_span, np.mean(SDE_stab_func_sol_delta_T, axis=0), color='red')
ax[1].set_ylabel(r'$\Delta T$ [K]')
ax[1].set_xlabel('time [h]')
ax[1].legend(handles=[plt_sims_delta_T_all[0], plt_sims_delta_T_one[0], plt_sims_delta_T_mean[0]],
             labels=['100 model runs', '1 model run', 'mean'], facecolor='white', loc="upper right", edgecolor="black",
             frameon=True)

ax[2].set_prop_cycle('color', colors)
plt_sims_sf_all = ax[2].plot(t_span, SDE_stab_func_sol_sf[:, :].T)
plt_sims_sf_one = ax[2].plot(t_span, SDE_stab_func_sol_sf[0, :], color='blue')
plt_sims_sf_mean = ax[2].plot(t_span, np.mean(SDE_stab_func_sol_sf, axis=0), color='red')
ax[2].set_ylabel(r'$\phi$')
ax[2].set_xlabel('time [h]')
ax[2].legend(handles=[plt_sims_sf_all[0], plt_sims_sf_one[0], plt_sims_sf_mean[0]],
             labels=['100 model runs', '1 model run', 'mean'], facecolor='white', loc="upper right", edgecolor="black",
             frameon=True)

fig.tight_layout()
plt.savefig(output_directory + 'solution_with_time_dependent_u.png', bbox_inches='tight', dpi=300)
