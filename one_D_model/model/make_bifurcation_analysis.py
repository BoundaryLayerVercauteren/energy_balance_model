import traceback

import PyDSTool as dst
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def make_bifurcation_analysis(param):
    # name of the model
    DSargs = dst.args(name='temperature inversion strength model')

    # parameters of Data from DomeC
    DSargs.pars = {'Q_i': param.Q_i,  # isothermal net radiation
                   'Lambda': param.Lambda,
                   'U': param.U,
                   'rho': param.rho,
                   'c_p': param.cp,
                   'k': param.kappa,
                   'g': param.grav,
                   'T_r': param.Tr,
                   'z_r': param.zr,
                   'z_0': param.z0,
                   'alpha': param.alpha}

    if param.stab_func_type == 'short_tail':
        DSargs.fnspecs = {'f': (['x'], 'exp(-2 * alpha * z_r * g/T_r * x/U**2 - (alpha * z_r * g/T_r * x/U**2)**2)')}
    elif param.stab_func_type == 'long_tail':
        DSargs.fnspecs = {'f': (['x'], 'exp(-2 * alpha * z_r * g/T_r * x/U**2)')}

    # rhs of the differential equation, including dummy variable w
    DSargs.varspecs = {'x': 'Q_i - Lambda*x - rho * c_p * (k/log(z_r/z_0))**2 * U * x * f(x)',
                       'w': 'x-w'}
    # initial conditions
    DSargs.ics = {'x': 0, 'w': 0}

    # set the range of integration.
    DSargs.tdomain = [0, 10]

    # -----------------------------------------------------------------------
    # Plot solution trajectories for different initial conditions
    # ode = dst.Generator.Vode_ODEsystem(DSargs)
    # init_cond = np.linspace(0, 30, 30)
    # fig = plt.figure(figsize=(5, 5))
    # for i, x0 in enumerate(init_cond):
    #     ode.set(ics={'x': x0})
    #     tmp = ode.compute('pol%3i' % i).sample()
    #     plt.plot(tmp['t'], tmp['x'])
    # plt.xlabel('$t$ [s]')
    # plt.ylabel(r'$\Delta T$ [K]')
    # plt.title(ode.name + ' multi ICs')
    # plt.savefig(param.sol_directory_path + 'sol_trajectories.png', bbox_inches='tight', dpi=300)

    # -----------------------------------------------------------------------
    # Plot bifurcation diagram
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    make_bifurcation_diagram(DSargs, [1.0, 2.0, 3.0], 'Lambda', r'$\lambda = $', ax[0], 'a)')
    DSargs.pars['Lambda'] = param.Lambda
    make_bifurcation_diagram(DSargs, [20, 50, 80], 'Q_i', r'$Q_i = $', ax[1], 'b)')
    DSargs.pars['Q_i'] = param.Q_i
    make_bifurcation_diagram(DSargs, [0.001, 0.01, 0.1], 'z_0', r'$z_0 = $', ax[2], 'c)')

    plt.savefig(param.sol_directory_path + 'bifurcation_diagram_' + param.stab_func_type + '.png', bbox_inches='tight', dpi=300)


def make_bifurcation_diagram(DSargs, val_list, val_name, label, ax, title):
    color = matplotlib.cm.get_cmap('cmc.batlow', len(val_list) + 1).colors

    for idx, val in enumerate(val_list):
        DSargs.pars[val_name] = val
        ode = dst.Generator.Vode_ODEsystem(DSargs)
        ode.set(pars={'U': 0.01})  # Lower bound of the control parameter 'U'
        ode.set(ics={'x': 23.3})
        PC = dst.ContClass(ode)  # Set up continuation class

        PCargs = dst.args(name='EQ1', type='EP-C')
        PCargs.freepars = ['U']  # control parameter(s) (it should be among those specified in DSargs.pars)
        PCargs.MaxNumPoints = 150  # The following 3 parameters are set after trial-and-error
        if idx == 1:
            PCargs.MaxStepSize = 0.5
        else:
            PCargs.MaxStepSize = 1
        PCargs.MinStepSize = 1e-5
        PCargs.StepSize = 2e-2
        PCargs.LocBifPoints = 'LP'  # detect limit points / saddle-node bifurcations
        PCargs.SaveEigen = True  # to tell unstable from stable branches
        PC.newCurve(PCargs)
        PC['EQ1'].forward()

        try:
            bif_point_1 = [PC['EQ1'].getSpecialPoint('LP1')['U'], PC['EQ1'].getSpecialPoint('LP1')['x']]
            bif_point_2 = [PC['EQ1'].getSpecialPoint('LP2')['U'], PC['EQ1'].getSpecialPoint('LP2')['x']]

            u_values = PC['EQ1'].sol['U']
            x_values = PC['EQ1'].sol['x']

            stable_points_idx = np.sort(np.concatenate((np.where(u_values == bif_point_1[0]), np.where(u_values == bif_point_2[0]))).flatten())

            ax.plot(u_values[:stable_points_idx[0]], x_values[:stable_points_idx[0]], label=label + str(val), color=color[idx])
            ax.plot(u_values[stable_points_idx[1]:], x_values[stable_points_idx[1]:], color=color[idx])
            ax.plot(u_values[stable_points_idx[0]:stable_points_idx[1]], x_values[stable_points_idx[0]:stable_points_idx[1]], linestyle='--', color=color[idx])
            ax.plot(bif_point_1[0], bif_point_1[1], marker="o", markersize=5, color=color[idx])
            ax.plot(bif_point_2[0], bif_point_2[1], marker="o", markersize=5, color=color[idx])
            if idx == 2:
                ax.axvline(x=5.6, color='r', label='u=5.6')

            print('For ' + val_name + '=' + str(val) + ' the unstable u region is: ' + str(u_values[stable_points_idx]))

            # Find equilibria for u=5.6
            equilibria = x_values[np.argwhere(np.around(u_values,1)==5.6)]
            print('The equilibria for u=5.6 are: ' + str(equilibria))

        except Exception:
            print(traceback.format_exc())
            pass

        # PC.display(['U', 'x'], stability=True, figure=3)

        # # Get information about the bifurcation points LP1 and LP2:
        # print(str(val_name) + '=' + str(val))
        # print(PC['EQ1'].getSpecialPoint('LP1'))
        # print(PC['EQ1'].getSpecialPoint('LP2'))

    ax.set_xlim((0, 10))
    ax.set_ylim((0, 26))
    ax.set_xlabel('u [m/s]')
    ax.set_ylabel(r'$\Delta T_{eq}$ [K]')
    ax.legend()
    ax.set_title(title, loc='left')
