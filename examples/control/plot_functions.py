import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

def plot(ref_path, data_path, algo, n_sims, n_gens, n_states):

    if n_sims == 0:
        n_sims = 1

    try:
        os.makedirs(data_path + "Plots")
    except FileExistsError:
        pass

    x_ref = np.load(ref_path + 'x_ref.npy')
    u_ref = np.load(ref_path + 'u_ref.npy')
    time = np.load(ref_path + 'time_points.npy')

    fits = np.zeros(n_sims)
    times = np.zeros(n_sims)
    best_fit = 1000
    best_sim = 0
    fit_evols = np.zeros((n_sims, n_gens))
    all_states = np.zeros((n_sims, len(time), n_states))
    all_controls = np.zeros((n_sims, len(time)))


    for sim in range(n_sims):
        with open(data_path + "best_ind_structure.txt".format(sim)) as f:
            print('{} Sim {}: '.format(algo, sim), f.read())
        fit = np.load(data_path + 'best_fitness.npy'.format(sim))
        fits[sim] = fit
        if fit < best_fit:
            best_sim = sim
            best_fit = fit
        comp_time = np.load(data_path + 'computational_time.npy'.format(sim))
        times[sim] = comp_time
        fit_evol = np.load(data_path + 'fitness_evol.npy'.format(sim))[:,0]
        fit_evols[sim,:] = fit_evol
        x = np.load(data_path + 'x.npy'.format(sim))
        all_states[sim, : :] = x
        u = np.load(data_path + 'u.npy'.format(sim))
        all_controls[sim,:] = u
    with open(data_path + "best_ind_structure.txt".format(best_sim)) as f:
        print('Best Sim {}: '.format(best_sim), f.read())
    print('Best fit ', best_fit)

    time_mean_igp = np.mean(times)
    time_std_igp = np.std(times)

    print('Computational Times\n')
    print('IGP: {}+-{}\n'.format(time_mean_igp, time_std_igp))

    points_orig = np.linspace(0, 10, 2001)
    point_target = np.linspace(0, 10, 1001)

    fig = plt.figure(0, figsize=[11, 6])
    ax = fig.add_subplot(111)
    ax.set(ylim=(-2.2, 1.2))
    plt.plot(time, x_ref[:, 0], '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_states[i, :, 0], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_states[i, :, 0], '--', color=color, alpha=alpha, zorder=1)

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend(loc=(0.15, 0.05))
    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    axins = inset_axes(ax, width=3.5, height=2.5, borderpad=1.5, loc=4)
    axins.plot(time, x_ref[:, 0], '--k', linewidth=2, label='Reference', zorder=8)


    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_states[i, :, 0], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_states[i, :, 0], '--', color=color, alpha=alpha, zorder=1)

    axins.set(xlim=(2, 6), ylim=(0.6,1.1))
    plt.savefig(data_path + 'Plots/x_pendulum.pdf', format='pdf', bbox_inches='tight')


    fig = plt.figure(1, figsize=[11, 6])
    ax = fig.add_subplot(111)
    ax.set(ylim=(-1, 3))
    plt.plot(time, x_ref[:, 1], '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_states[i, :, 1], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_states[i, :, 1], '--', color=color, alpha=alpha, zorder=1)

    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s]')
    plt.legend(loc=(0.2,0.47))
    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    axins = inset_axes(ax, width=3, height=2.5)
    axins.plot(time, x_ref[:, 1], '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_states[i, :, 1], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_states[i, :, 1], '--', color=color, alpha=alpha, zorder=1)

    axins.set(xlim=(0.4,2.5), ylim=(0.1,1.3))
    plt.savefig(data_path + 'Plots/v_pendulum.pdf', format='pdf', bbox_inches='tight')


    fig = plt.figure(2, figsize=[11, 6])
    ax = fig.add_subplot(111)
    ax.set(ylim=(2.6, 3.5))
    plt.plot(time, x_ref[:, 2], '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_states[i, :, 2], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_states[i, :, 2], '--', color=color, alpha=alpha, zorder=1)

    plt.xlabel('Time [s]')
    plt.ylabel('Angular position [rad]')
    plt.legend(loc=(0.15, 0.07))
    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    axins = inset_axes(ax, width=3, height=2.1, loc=4, borderpad=1.5)
    axins.plot(time, x_ref[:, 2], '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_states[i, :, 2], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_states[i, :, 2], '--', color=color, alpha=alpha, zorder=1)

    axins.set(xlim=(1,3), ylim=(3.14,3.25))
    plt.savefig(data_path + 'Plots/theta_pendulum.pdf', format='pdf', bbox_inches='tight')



    fig = plt.figure(3, figsize=[11, 6])
    ax = fig.add_subplot(111)
    ax.set(ylim=(-6.5, 3))
    plt.plot(time, x_ref[:, 3], '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_states[i, :, 3], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_states[i, :, 3], '--', color=color, alpha=alpha, zorder=1)

    plt.xlabel('Time [s]')
    plt.ylabel('Angular speed [rad/s]')
    plt.legend(loc=(0.17, 0.08))
    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    axins = inset_axes(ax, width=3, height=2.5, loc=4, borderpad=1.5)
    axins.plot(time, x_ref[:, 3], '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_states[i, :, 3], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_states[i, :, 3], '--', color=color, alpha=alpha, zorder=1)

    axins.set(xlim=(0.4,2), ylim=(-0.2,0.6))
    plt.savefig(data_path + 'Plots/omega_pendulum.pdf', format='pdf', bbox_inches='tight')


    fig = plt.figure(4, figsize=[11, 6])
    ax = fig.add_subplot(111)
    ax.set(ylim=(-3, 1))
    plt.plot(time, u_ref, '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            plt.plot(time, all_controls[i, :], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            plt.plot(time, all_controls[i, :], '--', color=color, alpha=alpha, zorder=1)

    plt.xlabel('Time [s]')
    plt.ylabel('Control Force [N]')
    plt.legend(loc=(0.15, 0.1))
    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    axins = inset_axes(ax, width=3, height=2.5, loc=4, borderpad=1.5)
    axins.plot(time, u_ref, '--k', linewidth=2, label='Reference', zorder=8)

    for i in range(n_sims):
        color = 'tab:blue'
        if i == best_sim:
            alpha = 1
            axins.plot(time, all_controls[i, :], color=color, label='Best IGP', zorder=9, linewidth=3)
        else:
            alpha = 0.4
            axins.plot(time, all_controls[i, :], '--', color=color, alpha=alpha, zorder=1)

    axins.set(xlim=(0,1), ylim=(-0.1,0.5))
    plt.savefig(data_path + 'Plots/u_pendulum.pdf', format='pdf', bbox_inches='tight')



    fig = plt.figure(5, figsize=[11, 6])
    ax = fig.add_subplot(111)
    #ax.set(xlim=(-1, 50), ylim=(56, 56.4))

    plt.plot(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0), label='IGP', color='tab:blue')
    plt.fill_between(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0) - np.std(fit_evols, axis=0),
                     np.mean(fit_evols, axis=0) + np.std(fit_evols, axis=0), alpha=0.2, color='tab:blue')


    plt.xlabel('Generation')
    plt.ylabel('Objective Function')

    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    axins = inset_axes(ax, width=4, height=3, borderpad=1.5)

    axins.plot(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0), label='IGP', color='tab:blue')
    axins.fill_between(np.linspace(0, 300, n_gens), np.mean(fit_evols, axis=0) - np.std(fit_evols, axis=0),
                     np.mean(fit_evols, axis=0) + np.std(fit_evols, axis=0), alpha=0.2, color='tab:blue')

    axins.set(xlim=(-1, 100), ylim=(0,50))

    plt.legend(loc='upper right')
    plt.savefig(data_path + 'Plots/fit_evol_pendulum.pdf', format='pdf', bbox_inches='tight')



    fig = plt.figure(6, figsize=[11, 6])
    ax = fig.add_subplot(111)
    plt.boxplot([fits], notch=False, showfliers=False)
    plt.plot([0.5,1,2,2.5,3,3.5], np.ones(6)*16.264779124940898, '--k', label='Reference', linewidth=2)
    plt.grid()
    plt.xticks([1], ['IGP'])
    plt.ylabel("Objective Function")
    plt.legend(loc=(0.04, 0.4))
    axins = inset_axes(ax, width=4, height=3, borderpad=1.5)
    axins.boxplot([fits], notch=False, showfliers=False)
    axins.plot([0.5, 1, 2, 2.5, 3, 3.5], np.ones(6) * 16.264779124940898, '--k', label='Reference', linewidth=2)
    plt.xticks([1], ['IGP'])
    plt.grid()

    axins.set(xlim=(1.5, 3.5), ylim=(16.1, 16.3))

    plt.savefig(data_path + 'Plots/fitness_pendulum.pdf', format='pdf', bbox_inches='tight')

    fig = plt.figure(7, figsize=[11, 6])
    ax = fig.add_subplot(111)
    plt.boxplot([times], notch=False, showfliers=False)
    plt.grid()
    plt.xticks([1], ['IGP'])
    plt.ylabel("Objective Function")
    plt.legend(loc=(0.04, 0.4))


    plt.show()

    return





