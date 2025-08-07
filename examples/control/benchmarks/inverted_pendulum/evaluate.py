from propagation_functions import propagate_forward
from scipy.integrate import simpson
import numpy as np


def evaluate_pendulum(individual, compile, **k):
    obj = k['kwargs']['plant']
    dynamics = k['kwargs']['dynamics']
    traj, contr, failure = propagate_forward(obj, individual, compile, dynamics)
    if failure is True:
        return [1e6, 0]
    else:
        g = np.zeros(obj.Npoints)
        for i in range(obj.Npoints):
            g[i] = 0.5 * ((traj[i, :]-obj.xf).T @ obj.Qz @ (traj[i, :]-obj.xf) + np.array(([[contr[i]]])).T @ obj.Qu @ np.array(([[contr[i]]])))
        int_g = simpson(g, obj.t_points)
        h = 0.5 * ((traj[-1, :]-obj.xf).T @ obj.Qt @ (traj[-1, :]-obj.xf))
        FIT = int_g + h
        return [FIT, 0]