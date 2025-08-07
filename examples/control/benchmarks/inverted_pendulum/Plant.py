import numpy as np

class Pendulum:

    def __init__(self):
        self.M = 0.1 # kg
        self.m = 0.02 # kg
        self.l = 0.1 # m
        self.g = -9.8 # m/s^2

        self.K = np.array(([-1.000000000000005,-1.419113900549633,8.131246042349595,1.223865603895364]))

        self.x0 = np.array(([-1, 0, np.pi+0.1, 0]))  # initial conditions
        self.xf = np.array(([1, 0, np.pi, 0]))
        self.Qt = np.diag([1, 1, 1, 1]) # h fitness
        self.Qz = np.diag([5, 5, 5, 5]) # g fitness
        self.Qu = np.array(([1])) # control fitness

        self.ub = np.array(([10, 10, 10, 100]))
        self.lb = np.array(([-10, -10, -10, -100]))

        self.eq_f = ['x2',
                     '(-(m**2)*(l**2)*g*cos(x3)*sin(x3)+m*(l**2)*(m*l*(x4**2)*sin(x3))+m*(l**2)*u1)/(m*(l**2)*(M+m*(1-cos(x3)**2)))',
                     'x4',
                     '((m+M)*m*g*l*sin(x3)-m*l*cos(x3)*(m*l*(x4**2)*sin(x3))-m*l*cos(x3)*u1)/(m*(l**2)*(M+m*(1-cos(x3)**2)))']
        self.eq_g = ['h*Qz1*(x1-x1r)**2 + h*Qz2*(x2-x2r)**2 + h*Qz3*(x3-x3r)**2 + h*Qz4*(x4-x4r)**2 + h*Qu*u1**2']
        self.states_dict = {'x1': 'X', 'x2': 'V', 'x3':'Theta', 'x4':'Omega'}
        self.cont_dict = {'u1': 'u1'}
        self.u_input_dict = {'x1': 'eX', 'x2': 'eV', 'x3': 'eTheta', 'x4': 'eOmega'}
        self.GPinput_dict = {'eX':'sub(X, x1r)', 'eV':'sub(V, x2r)', 'eTheta':'sub(Theta, x3r)', 'eOmega':'sub(Omega, x4r)'}
        self.str_symbols = ['m', 'l', 'M', 'g', 'Qz1', 'Qz2', 'Qz3', 'Qz4', 'h', 'Qu', 'x1r', 'x2r', 'x3r', 'x4r']
        self.val_symbols = np.array(([self.m, self.l, self.M, self.g, self.Qz[0,0], self.Qz[1,1], self.Qz[2,2],
                                      self.Qz[3,3], 1/2, self.Qu[0], self.xf[0], self.xf[1], self.xf[2], self.xf[3]]))
        self.vars_order = ['eX', 'eV', 'eTheta', 'eOmega']

        self.n_states = 4
        self.n_controls = 1

        # propagation data

        self.t0 = 0
        self.tf = 10
        self.dt = 0.005
        self.Npoints = int((self.tf - self.t0) / self.dt) + 1  # number of propagation points
        self.t_points = np.linspace(self.t0, self.tf, self.Npoints)  # time points

        # Adam parameters

        self.alfa = 0.01
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8
