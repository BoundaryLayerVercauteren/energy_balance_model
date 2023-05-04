from matplotlib import pyplot as plt


def configure_plotting_style(figure_type):
    plt.style.use('science')
    # set font sizes for plots
    if figure_type == 'full_page_width':
        SMALL_SIZE = 11
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 15
    elif figure_type == 'half_page_width':
        SMALL_SIZE = 14
        MEDIUM_SIZE = 15
        BIGGER_SIZE = 19

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
