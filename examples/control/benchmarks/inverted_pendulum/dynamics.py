import numpy as np

def dynamics_pendulum(t, x, obj, u):

    m = obj.m
    l = obj.l
    M = obj.M
    g = obj.g
    x1dot = x[1]
    x2dot = (-(m**2)*(l**2)*g*np.cos(x[2])*np.sin(x[2])+m*(l**2)*(m*l*(x[3]**2)*np.sin(x[2]))+m*(l**2)*u)/(m*(l**2)*(M+m*(1-np.cos(x[2])**2)))
    x3dot = x[3]
    x4dot = ((m+M)*m*g*l*np.sin(x[2])-m*l*np.cos(x[2])*(m*l*(x[3]**2)*np.sin(x[2]))-m*l*np.cos(x[2])*u)/(m*(l**2)*(M+m*(1-np.cos(x[2])**2)))

    return np.array((x1dot, x2dot, x3dot, x4dot))