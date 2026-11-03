import numpy as np

try:
    from ...propagation_functions import propagate_forward
except ImportError:
    from propagation_functions import propagate_forward
from scipy.integrate import simpson


def evaluate_pendulum(individual, compile, **k):
    obj = k["kwargs"]["plant"]
    dynamics = k["kwargs"]["dynamics"]
    traj, contr, failure = propagate_forward(obj, individual, compile, dynamics)
    if failure is True:
        return [1e6, 0]
    else:
        g = np.zeros(obj.Npoints)
        for i in range(obj.Npoints):
            g[i] = 0.5 * (
                (traj[i, :] - obj.xf).T @ obj.Qz @ (traj[i, :] - obj.xf)
                + float(np.asarray(obj.Qu).reshape(-1)[0]) * contr[i] ** 2
            )
        int_g = simpson(g, x=obj.t_points)
        h = 0.5 * ((traj[-1, :] - obj.xf).T @ obj.Qt @ (traj[-1, :] - obj.xf))
        FIT = int_g + h
        return [FIT, 0]
