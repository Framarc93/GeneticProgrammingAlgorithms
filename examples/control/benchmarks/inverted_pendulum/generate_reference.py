import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
from scipy.integrate import simpson
import Plant as PlantPendulum
from propagation_functions import RK4
from dynamics import dynamics_pendulum

if __name__=='__main__':

    obj = PlantPendulum.Pendulum()

    x = np.zeros((obj.Npoints, obj.n_states))
    u = np.zeros(obj.Npoints)

    x[0, :] = obj.x0
    vv = x[0, :] - obj.xf
    u[0] = -obj.K @ (x[0,:] - obj.xf)
    failure = False
    for i in range(obj.Npoints - 1):
        sol_forward = RK4(obj.t_points[i], obj.t_points[i + 1], dynamics_pendulum, 2, x[i, :],
                                      args=(obj, u[i]))  # evaluate states from t to t+dt
        if np.isnan(sol_forward[-1, :]).any() or np.isinf(sol_forward[-1, :]).any():
            failure = True
            break
        else:
            x[i + 1, :] = sol_forward[-1, :]
            u[i + 1] = -obj.K @ (x[i+1,:] - obj.xf)
    g = np.zeros(obj.Npoints)
    for i in range(obj.Npoints):
        g[i] = 0.5 * ((x[i, :] - obj.xf).T @ obj.Qz @ (x[i, :] - obj.xf) + np.array([[u[i]]]).T @ obj.Qu @ np.array([[u[i]]]))
    int_g = simpson(g, obj.t_points)
    h = 0.5 * ((x[-1, :] - obj.xf).T @ obj.Qt @ (x[-1, :] - obj.xf))
    FIT = int_g + h
    print('Fitness reference ', FIT)
    np.save('fitness_reference.npy', FIT)
    np.save('x_ref.npy', x)
    np.save('u_ref.npy', u)
    np.save('time_points.npy', obj.t_points)