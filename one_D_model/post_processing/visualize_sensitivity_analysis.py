import json
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np

# Set plotting style
plt.style.use("science")

# Set font sizes for plots
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define directory where simulation output is saved
output_directory = "output/sensitivity_study/internal_variability/"  # u_and_model/'  #
subset_data_size = 500
trans_percentage = 0.8
index = ""

# Load results from sensitivity analysis
with open(
    f"{output_directory}very_weakly/average_transitions_{subset_data_size}_{index}.json"
) as file:
    result_vw = json.load(file)
with open(
    f"{output_directory}weakly_very/average_transitions_{subset_data_size}_{index}.json"
) as file:
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
    if (
        "internal_variability" in output_directory
        or "stability_function" in output_directory
    ):
        first_sigma_with_enough_trans = []
        for u in u_range:
            # Find indices of the parameters which correspond to the current u
            cor_idx = np.where([cur_u[0] for cur_u in trans_statistics] == u)[0]
            # Find out how many transitions (on average) took place for the simulations corresponding to the current u
            cor_average_num_trans = np.array([trans_statistics[i][3] for i in cor_idx])
            idx_first_sigma_with_enough_trans = cor_idx[
                np.argmax(cor_average_num_trans >= trans_percentage)
            ]
            if (
                trans_statistics[idx_first_sigma_with_enough_trans][3]
                < trans_percentage
            ):
                first_sigma_with_enough_trans.append(np.nan)
            else:
                first_sigma_with_enough_trans.append(
                    trans_statistics[idx_first_sigma_with_enough_trans][1]
                )
    else:
        sigma_u = [0.0, 0.01, 0.04]
        first_sigma_with_enough_trans = np.empty((len(u_range), len(sigma_u)))
        for idx_u, u in enumerate(u_range):
            for idx_sigma_u, sigma_u_val in enumerate(sigma_u):
                cor_idx_u = np.where([cur_u[0] for cur_u in trans_statistics] == u)[0]
                cor_idx_sigma = []
                for idx, elem in enumerate(trans_statistics):
                    if elem[2] == sigma_u_val:
                        cor_idx_sigma.append(idx)
                cor_idx = np.intersect1d(cor_idx_u, cor_idx_sigma)
                cor_average_num_trans = np.array(
                    [trans_statistics[i][3] for i in cor_idx]
                )
                if len(cor_average_num_trans) >= 1:
                    idx_first_sigma_with_enough_trans = cor_idx[
                        np.argmax(cor_average_num_trans >= trans_percentage)
                    ]
                else:
                    first_sigma_with_enough_trans[idx_u, idx_sigma_u] = np.nan
                    continue
                if (
                    trans_statistics[idx_first_sigma_with_enough_trans][3]
                    < trans_percentage
                ):
                    first_sigma_with_enough_trans[idx_u, idx_sigma_u] = np.nan
                else:
                    first_sigma_with_enough_trans[
                        idx_u, idx_sigma_u
                    ] = trans_statistics[idx_first_sigma_with_enough_trans][1]
    return first_sigma_with_enough_trans


min_sigma_vw = np.array(get_minimal_sigma_with_trans_for_every_u(result_vw, u_range_vw))
min_sigma_wv = np.array(get_minimal_sigma_with_trans_for_every_u(result_wv, u_range_wv))

det_equilibria = np.loadtxt(
    "output/sensitivity_study/equilibria_det_model/x_values.txt",
    delimiter=",",
    dtype=float,
).flatten()
det_equilibria_u = np.loadtxt(
    "output/sensitivity_study/equilibria_det_model/u_values.txt",
    delimiter=",",
    dtype=float,
).flatten()

if "stability_function" in output_directory:
    Ri_vw = (np.array(10.0 * (9.81 / 243.0) * (24 / u_range_vw)) > 0.25).nonzero()[0]
    Ri_wv = (np.array(10.0 * (9.81 / 243.0) * (4 / u_range_wv)) > 0.25).nonzero()[0]
else:
    Ri_vw = np.array([0])
    Ri_wv = np.array([0])

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax2 = ax.twinx()

ax2.plot(
    det_equilibria_u,
    det_equilibria,
    color="red",
    label="equilibria det. model",
    linestyle="--",
    alpha=0.5,
)
ax2.set_ylabel(r"$\Delta T$ [K]", color="red")
ax2.tick_params(axis="y", labelcolor="red")
# ax2.get_yaxis().set_ticks([])

ax.axvspan(5.31, 5.89, alpha=0.3, color="red", label="bistable region")
# ax.plot(np.NaN, np.NaN, '-', color='none', label=' ')

ax.plot(u_range_vw, min_sigma_vw, color="blue")
# ax.plot(u_range_vw, min_sigma_vw[:, 0], color='cyan')
# ax.plot(u_range_vw, min_sigma_vw[:, 1], color='royalblue')
# ax.plot(u_range_vw, min_sigma_vw[:, 2], color='midnightblue')
# ax.plot(u_range_vw[Ri_vw], min_sigma_vw[Ri_vw], color='blue', lw=4)

ax.scatter(u_range_vw, min_sigma_vw, color="blue", label="vSBL-wSBL", marker="v")
# ax.scatter(u_range_vw, min_sigma_vw[:, 0], color='cyan', label=r'$\sigma_U$ = 0: vSBL-wSBL', marker="v")
# ax.scatter(u_range_vw, min_sigma_vw[:, 1], color='royalblue', label=r'$\sigma_U$ = 0.01: vSBL-wSBL', marker="^")
# ax.scatter(u_range_vw, min_sigma_vw[:, 2], color='midnightblue', label=r'$\sigma_U$ = 0.04: vSBL-wSBL', marker="<")
# #
# ax.plot(np.NaN, np.NaN, '-', color='none', label=' ')

ax.plot(u_range_wv, min_sigma_wv, color="green")
# ax.plot(u_range_wv, min_sigma_wv[:, 0], color='lime', linestyle=':')
# ax.plot(u_range_wv, min_sigma_wv[:, 1], color='limegreen', linestyle=':')
# ax.plot(u_range_wv, min_sigma_wv[:, 2], color='darkgreen', linestyle=':')
# ax.plot(u_range_wv[Ri_wv], min_sigma_wv[Ri_wv], color='green', lw=4)

ax.scatter(u_range_wv, min_sigma_wv, color="green", label="wSBL-vSBL")
# ax.scatter(u_range_wv, min_sigma_wv[:, 0], color='lime', label=r'$\sigma_U$ = 0: wSBL-vSBL', marker=".")
# ax.scatter(u_range_wv, min_sigma_wv[:, 1], color='limegreen', label=r'$\sigma_U$ = 0.01: wSBL-vSBL', marker="s")
# ax.scatter(u_range_wv, min_sigma_wv[:, 2], color='darkgreen', label=r'$\sigma_U$ = 0.04: wSBL-vSBL', marker="o")
# #
# ax.plot(np.NaN, np.NaN, '-', color='none', label=' ')

ax.set_xlim((4.5, 6.9))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.04))
ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
ax.set_axisbelow(True)
ax.grid()

fig.legend(
    facecolor="white", edgecolor="black", frameon=True, bbox_to_anchor=(0.9, 0.6)
)
# fig.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor='white', edgecolor="black", frameon=True)
# ax.set_facecolor('whitesmoke')

ax.set_xlabel(r"U [$\mathrm{ms^{-1}}$]")
ax.set_ylabel(r"$\sigma_{i,min}$")

fig.tight_layout()
plt.savefig(
    f"{output_directory}transition_statistics_{trans_percentage}_{subset_data_size}_{index}.pdf",
    bbox_inches="tight",
    dpi=300,
)
