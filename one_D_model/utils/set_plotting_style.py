from matplotlib import pyplot as plt


def configure_plotting_style(
    figure_type, small_size=None, medium_size=None, large_size=None
):
    plt.style.use("science")
    # set font sizes for plots
    if figure_type == "full_page_width":
        small_size = 11
        medium_size = 12
        large_size = 15
    elif figure_type == "half_page_width":
        small_size = 14
        medium_size = 15
        large_size = 19

    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=large_size)  # fontsize of the figure title
