"""Script to perform bifurcation analysis for the model (eq. 2)."""
import traceback

import matplotlib
import numpy as np
import PyDSTool as dst
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

from one_D_model.utils.set_plotting_style import configure_plotting_style

# Set font sizes for plots
small_font_size = 13
medium_font_size = 15
large_font_size = 18
configure_plotting_style(None, small_size=small_font_size, medium_size=medium_font_size, large_size=large_font_size)
plt.rc('figure', titlesize=large_font_size)


def setup_for_equilibrium_analysis(param):
    """Set all relevant parameters and define model to find equilibrium points model (eq. 2)."""
    # Set name of the model
    DSargs = dst.args(name='temperature inversion strength model')

    # Define model parameters
    DSargs.pars = {'Q_i': param.Q_i, 'Lambda': param.Lambda, 'U': param.U, 'rho': param.rho, 'c_p': param.cp,
                   'k': param.kappa, 'g': param.grav, 'T_r': param.Tr, 'z_r': param.zr, 'z_0': param.z0,
                   'alpha': param.alpha}

    # Define stability function for the model (see section 2.2.)
    # Note: x is Delta T
    if param.stab_func_type == 'short_tail':
        DSargs.fnspecs = {'f': (['x'], 'exp(-2 * alpha * z_r * g/T_r * x/U**2 - (alpha * z_r * g/T_r * x/U**2)**2)')}
    elif param.stab_func_type == 'long_tail':
        DSargs.fnspecs = {'f': (['x'], 'exp(-2 * alpha * z_r * g/T_r * x/U**2)')}

    # Define model (eq. 2)
    # Note: w is a dummy variable
    DSargs.varspecs = {'x': 'Q_i - Lambda*x - rho * c_p * (k/log(z_r/z_0))**2 * U * x * f(x)',
                       'w': 'x-w'}
    # Define initial conditions
    DSargs.ics = {'x': 0, 'w': 0}

    # Set the range of integration.
    DSargs.tdomain = [0, 10]

    return DSargs


def plot_solution_trajectories(args, param):
    """Set trajectories of solutions with different initial conditions."""
    ode = dst.Generator.Vode_ODEsystem(args)
    init_cond = np.linspace(0, 30, 30)
    plt.figure(figsize=(5, 5))
    for i, x0 in enumerate(init_cond):
        ode.set(ics={'x': x0})
        tmp = ode.compute('pol%3i' % i).sample()
        plt.plot(tmp['t'], tmp['x'])
    plt.xlabel('$t$ [s]')
    plt.ylabel(r'$\Delta T$ [K]')
    plt.title(ode.name + ' multi ICs')
    plt.savefig(param.sol_directory_path + 'sol_trajectories.pdf', bbox_inches='tight', dpi=300)


def make_bifurcation_diagram(DSargs, ax, title, data_val=None, save_values=False):
    """Perform bifurcation analysis and plot results."""
    color = matplotlib.cm.get_cmap('cmc.batlow', 2).colors

    ode = dst.Generator.Vode_ODEsystem(DSargs)
    ode.set(pars={'U': 0.01})  # Lower bound of the control parameter 'U'
    ode.set(ics={'x': 23.3})

    # Set parameters for bifurcation analysis
    PC = dst.ContClass(ode)  # Set up continuation class
    PCargs = dst.args(name='EQ1', type='EP-C')
    PCargs.freepars = ['U']  # control parameter(s) (it should be among those specified in DSargs.pars)
    PCargs.MaxNumPoints = 150  # The following 3 parameters are set after trial-and-error
    PCargs.MaxStepSize = 1.0
    PCargs.MinStepSize = 1e-5
    PCargs.StepSize = 2e-2
    PCargs.LocBifPoints = 'LP'  # detect limit points / saddle-node bifurcations
    PCargs.SaveEigen = True  # to tell unstable from stable branches
    PC.newCurve(PCargs)
    PC['EQ1'].forward()

    try:
        # Get bifurcation points
        bif_point_1 = [PC['EQ1'].getSpecialPoint('LP1')['U'], PC['EQ1'].getSpecialPoint('LP1')['x']]
        bif_point_2 = [PC['EQ1'].getSpecialPoint('LP2')['U'], PC['EQ1'].getSpecialPoint('LP2')['x']]

        # Get wind forcing values and corresponding equilibrium points of Delta T
        u_values = PC['EQ1'].sol['U']
        delta_T_values = PC['EQ1'].sol['x']

        # Save wind forcing values and corresponding equilibrium points of Delta T in files
        if save_values:
            np.savetxt('u_values.txt', u_values, delimiter=',')
            np.savetxt('delta_T_values.txt', delta_T_values, delimiter=',')
            return

        # Get index of u values which correspond to the two bifurcation points
        stable_points_idx = np.sort(
            np.concatenate((np.where(u_values == bif_point_1[0]), np.where(u_values == bif_point_2[0]))).flatten())

        # Add scatter plot of data values
        if data_val is not None:
            color = matplotlib.cm.get_cmap('cmc.batlow', 4).colors
            plt.scatter(data_val['U2[m s-1]'], data_val['tempInv [K]'], s=5, c=color[2])

        # Plot stable equilibria over wind forcing
        ax.plot(u_values[:stable_points_idx[0]], delta_T_values[:stable_points_idx[0]], color=color[0])
        ax.plot(u_values[stable_points_idx[1]:], delta_T_values[stable_points_idx[1]:], color=color[0])
        # Plot unstable equilibria over wind forcing
        ax.plot(u_values[stable_points_idx[0]:stable_points_idx[1]],
                delta_T_values[stable_points_idx[0]:stable_points_idx[1]], linestyle='--', color=color[0])
        # Mark bifurcation points
        ax.plot(bif_point_1[0], bif_point_1[1], marker="o", markersize=5, color=color[0])
        ax.plot(bif_point_2[0], bif_point_2[1], marker="o", markersize=5, color=color[0])

        # Mark bifurcation region
        if data_val is None:
            ax.axvspan(u_values[stable_points_idx][0], u_values[stable_points_idx][1], alpha=0.3, color='red',
                       label='bistable region')

        print(f'The unstable u region is: {u_values[stable_points_idx]}')

        # Find equilibria for u=5.6
        equilibria = delta_T_values[np.argwhere(np.around(u_values, 1) == 5.6)]
        print(f'The equilibria for u=5.6 are: {equilibria}')

    except Exception:
        print(traceback.format_exc())
        pass

    ax.set_xlim((0, 10))
    ax.set_ylim((0, 26))

    # Set labels for x and y axes
    if data_val is not None:
        # y axes labels
        ybox3 = TextArea(r'$\Delta T_{eq}$, ',
                         textprops=dict(color=color[0], size=medium_font_size, rotation=90, ha='left', va='bottom'))
        ybox2 = TextArea(r'$T_{9.4m}-T_s$ ',
                         textprops=dict(color=color[2], size=medium_font_size, rotation=90, ha='left', va='bottom'))
        ybox1 = TextArea('[K]',
                         textprops=dict(color="black", size=medium_font_size, rotation=90, ha='left', va='bottom'))
        ybox = VPacker(children=[ybox1, ybox2, ybox3], align="bottom", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc='center', child=ybox, pad=0., frameon=False, bbox_transform=ax.transAxes,
                                          borderpad=0., bbox_to_anchor=(-0.08, 0.5))
        ax.add_artist(anchored_ybox)
        # x axes labels
        xbox1 = TextArea(r'$U$, ', textprops=dict(color=color[0], size=medium_font_size, ha='left', va='bottom'))
        xbox2 = TextArea(r'$U_{8m}$ ', textprops=dict(color=color[2], size=medium_font_size, ha='left', va='bottom'))
        xbox3 = TextArea('[$\mathrm{ms^{-1}}$]',
                         textprops=dict(color="black", size=medium_font_size, ha='left', va='bottom'))
        xbox = HPacker(children=[xbox1, xbox2, xbox3], align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc='center', child=xbox, pad=0, frameon=False, bbox_transform=ax.transAxes,
                                          borderpad=0., bbox_to_anchor=(0.5, -0.09))
        ax.add_artist(anchored_xbox)
    else:
        ax.set_xlabel('U [$\mathrm{ms^{-1}}$]', fontsize=medium_font_size)
        ax.set_ylabel(r'$\Delta T_{eq}$ [K]', fontsize=medium_font_size)

    ax.legend()
    ax.set_title(title, loc='left')


def make_bifurcation_analysis(params, data=None, save_values=False):
    """Wrapper function to find equilibria and perform bifurcation analysis of the model (eq. 2)."""
    # Make setup for bifurcation analysis
    bif_args = setup_for_equilibrium_analysis(params)

    # Plot solution trajectories for different initial conditions
    plot_solution_trajectories(bif_args, params)

    # Perform bifurcation analysis and save bifurcations points and corresponding wind velocity (u) values
    if save_values:
        make_bifurcation_diagram(bif_args, None, None, None, save_values=True)

    # Plot bifurcation diagram
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if 'short' in params.stab_func_type:
        fig_label = 'a)'
    else:
        fig_label = 'b)'
    make_bifurcation_diagram(bif_args, ax, fig_label)

    plt.savefig(params.sol_directory_path + 'bifurcation_diagram_single_' + params.stab_func_type + '.png',
                bbox_inches='tight', dpi=300)

    # Plot bifurcation diagram and scatterplot of data
    if data is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig_label = 'c)'
        make_bifurcation_diagram(bif_args, ax, fig_label, data)

        plt.savefig(params.sol_directory_path + 'bifurcation_diagram_domeC_' + params.stab_func_type + '.png',
                    bbox_inches='tight', dpi=300)
