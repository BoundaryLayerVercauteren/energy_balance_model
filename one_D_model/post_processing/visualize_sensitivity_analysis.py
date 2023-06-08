import json
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

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
output_directory = 'output/sensitivity_study/internal_variability/'
subset_data_size = 500
trans_percentage = 0.8
index = 3

# Load results from sensitivity analysis
with open(f'{output_directory}very_weakly/average_transitions_{subset_data_size}_{index}.json') as file:
    result_vw = json.load(file)
with open(f'{output_directory}weakly_very/average_transitions_{subset_data_size}_{index}.json') as file:
    result_wv = json.load(file)


# Sort order by wind velocity value and then sigma(s)
def sort_list_of_list(input):
    input.sort(key=itemgetter(2))
    input.sort(key=itemgetter(1))
    input.sort(key=itemgetter(0))
    return input


result_vw = sort_list_of_list(result_vw)
result_wv = sort_list_of_list(result_wv)


u_range_vw = np.unique([cur_u[0] for cur_u in result_vw])
u_range_wv = np.unique([cur_u[0] for cur_u in result_wv])


def get_minimal_sigma_with_trans_for_every_u(trans_statistics, u_range):
    if 'internal_variability' in output_directory:
        first_sigma_with_enough_trans = []
        for u in u_range:
            # Find indices of the parameters which correspond to the current u
            cor_idx = np.where([cur_u[0] for cur_u in trans_statistics] == u)[0]

            # Find out how many transitions (on average) took place for the simulations corresponding to the current u
            cor_average_num_trans = np.array([trans_statistics[i][3] for i in cor_idx])
            idx_first_sigma_with_enough_trans = cor_idx[np.argmax(cor_average_num_trans >= trans_percentage)]
            first_sigma_with_enough_trans.append(trans_statistics[idx_first_sigma_with_enough_trans][1])
    return first_sigma_with_enough_trans

# Calculate richardson number for every time step
Rb_vw = 10.0 * (9.81 / 243.0) * (24 / (u_range_vw ** 2))
Rb_wv = 10.0 * (9.81 / 243.0) * (4 / (u_range_vw ** 2))

min_sigma_vw = get_minimal_sigma_with_trans_for_every_u(result_vw, u_range_vw)
min_sigma_wv = get_minimal_sigma_with_trans_for_every_u(result_wv, u_range_wv)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.axvspan(5.31, 5.89, alpha=0.3, color='red', label='transition region')
ax.plot(u_range_vw, min_sigma_vw, color='blue')
ax.scatter(u_range_vw, min_sigma_vw, color='blue', label='vSBL-wSBL')
ax.plot(u_range_wv, min_sigma_wv, color='green')
ax.scatter(u_range_wv, min_sigma_wv, color='green', label='wSBL-vSBL')
ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.set_axisbelow(True)
ax.grid()
ax.legend(loc='upper right', facecolor='white', edgecolor="black", frameon=True)
ax.set_xlabel(r'u [m/s]')
ax.set_ylabel(r'$\sigma_i$')

fig.tight_layout()
plt.savefig(f'{output_directory}transition_statistics_{trans_percentage}_{subset_data_size}_{index}.pdf', bbox_inches='tight', dpi=300)


# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#
# ax.plot(u_range_vw, Rb_vw, color='blue', label=r'$R_b(\Delta T=24K$', lw=2)
# ax.plot(u_range_vw, Rb_wv, color='red', label=r'$R_b(\Delta T=4K$', lw=2)
#
# ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
#
# ax.set_axisbelow(True)
# ax.grid(axis='x')
#
# ax.legend(loc='upper right', facecolor='white', edgecolor="black", frameon=True)
#
# ax.set_xlabel(r'u [m/s]')
# ax.set_ylabel(r'$R_b$')
#
# fig.tight_layout()
# plt.savefig(f'{output_directory}richardson_number.pdf', bbox_inches='tight', dpi=300)