"""
    This program solves the heat equation
        u_t = u_xx
    with dirichlet boundary condition
        u(0,t) = u(1,t) = 0
    with the Initial Conditions
        u(x,0) = sin( pi*x )
    over the domain x = [0, 1]
 
    using a finite difference method in space and a Forward-Euler 
    method in time. Uses the linear algebra way of doing things.
"""

from __future__ import division
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import time, sys

# CPU warmup
np.random.rand(500,500).dot(np.random.rand(500,500))

# spatial conditions
# m  = 100                     # number of inner points (grid will be (m+2)x(m+2) )
if len(sys.argv) > 1:
    N = 10**(int(sys.argv[1]))
    m = 10**(int(sys.argv[2]))
else:
    N = 10000
    m = 10000

x0 = 0                       # start
xf = 1                       # end
dx = (xf-x0)/(m+1)           # spatial step size
x  = np.linspace(x0,xf,m+2)  # x-axis points
alpha = 0                    # u(x0, t) = alpha
beta  = 0                    # u(xf, t) = beta

# temporal conditions
# N  = 100000          # number of time steps
t0 = 0            # start
tf = 1            # end
dt = (tf - t0)/N  # time step size

# coefficients
k = 0.002         # diffusion
K = k*dt/(dx**2)  # PDE coeff
dxBeta = dx*beta  # for final element of b

# initial condition function
def f(x):
    return np.sin(np.pi*x)

# create matrix A
diag_0  = [1] + m*[1 - 2*K] + [1]   # main diagonal
diag_m1 = m*[K] + [-1]              # -1 diagonal
diag_p1 = [0] + m*[K]               # +1 diagonal

A = sp.diags([diag_m1, diag_0, diag_p1], [-1, 0, 1], shape=(m+2,m+2), format="csr")

# initial solution
u = f(x)

# initialize the final solution vector (u0, ..., u_m+1)
U = np.empty((N,m+2), dtype='d')


# step through time
t0 = time.time()
for i in xrange(N):
    U[i] = u           # save old solution
    uN = A.dot(u)

    # force BCs
    uN[0]   = 0
    uN[m+1] = 0

    u = uN

tf = time.time()

print tf-t0
print U[]

sys.exit()