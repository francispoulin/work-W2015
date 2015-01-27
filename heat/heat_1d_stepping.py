#!/usr/bin/env python
# heat_1d_stepping.py
#
# This will solve the 1-D heat equation using the stepping equations
from __future__ import division
import time, sys
import numpy as np
import matplotlib.pyplot as plt


# CPU warmup
np.random.rand(500,500).dot(np.random.rand(500,500))

# read from STDIN
if len(sys.argv) > 1:
    N = 10**(int(sys.argv[1]))
    m = 2**(int(sys.argv[2])) 
else:
    N = 10000   # time steps
    m = 2048    # inner grid points

# spatial conditions
x0 = 0                       # start
xf = 1                       # end
dx = (xf-x0)/(m+1)           # spatial step size
alpha = 0                    # u(x0, t) = alpha
beta  = 0                    # u(xf, t) = beta
x  = np.linspace(x0,xf,m+2)  # x-axis points

# temporal conditions
t0 = 0            # start
tf = 300          # end
dt = (tf - t0)/N  # time step size
t  = np.linspace(t0, tf, N)

# coefficients
k = 0.0002
K = 0.0001                # PDE coeff
dxBeta = dx*beta          # for final element of b

# initial condition function
def f(x):
    return np.sin(np.pi*x)

# initial solution
u = np.array(f(x), dtype=np.float64)
un = np.empty(m+2, dtype=np.float64)

# initialize the final solution vector (u0, ..., u_m+1)
# U = np.empty((N,m+2), dtype=np.float64)
# U[0] = u

t0 = time.time()

# Loop over time
for j in range(1,N):
    un[1:m+1] = u[1:m+1] + K*(u[0:m] - 2.0*u[1:m+1] + u[2:m+2] )

    # Force Boundary Conditions
    un[0] = 0.0
    un[m+1] = 0.0

    # save solution
    # U[j] = un

    # Update solution    
    u = un

tf = time.time()

print tf-t0


# fig, ax = plt.subplots()
# ax.pcolormesh(x, t, U)
# plt.show()

#plt.clf()
#plt.plot(xf, uf,'-r')
#plt.draw()
#plt.show()
sys.exit()