import numpy as np

def propagate_forward(obj, GPind, compile, dynamics):
    x = np.zeros((obj.Npoints, obj.n_states))
    u = np.zeros(obj.Npoints)

    x[0, :] = obj.x0
    ufuns = compile(GPind)
    vv = x[0,:]-obj.xf
    u[0] = ufuns(*vv)
    failure = False
    for i in range(obj.Npoints - 1):
        sol_forward = RK4(obj.t_points[i], obj.t_points[i + 1], dynamics, 2, x[i, :],
                                      args=(obj, u[i]))  # evaluate states from t to t+dt
        if np.isnan(sol_forward[-1, :]).any() or np.isinf(sol_forward[-1, :]).any():
            failure = True
            break
        else:
            x[i + 1, :] = sol_forward[-1, :]
            vv = x[i+1, :] - obj.xf
            u[i + 1] = ufuns(*vv)
    return x, u, failure


def RK4(t_start, t_end, fun, Npoints, init_cond, args):
    """Runge-Kutta 4 with time integration limit. If t_elapsed > t_max_int then the integration stops"""
    t = np.linspace(t_start, t_end, Npoints)
    dt = t[1] - t[0]
    x = np.zeros((Npoints, len(init_cond)))
    x[0,:] = init_cond
    for i in range(Npoints-1):
        k1 = fun(t[i], x[i, :], *args)
        k2 = fun(t[i] + dt / 2, x[i, :] + 0.5 * dt * k1, *args)
        k3 = fun(t[i] + dt / 2, x[i, :] + 0.5 * dt * k2, *args)
        k4 = fun(t[i] + dt, x[i, :] + dt * k3, *args)
        x[i + 1, :] = x[i, :] + (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return x