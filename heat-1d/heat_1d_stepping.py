#!/usr/bin/env python
# heat_1d_stepping.py
#
# This will solve the 1-D heat equation using the stepping equations
from __future__ import division
import time, sys
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)  # make sure ENTIRE array is printed

# CPU warmup
np.random.rand(500,500).dot(np.random.rand(500,500))

# read from STDIN
if len(sys.argv) > 1:
    M = 2**(int(sys.argv[1]))
    # N = 10**(int(sys.argv[2]))
    i = int(sys.argv[3])
    writeToFile = bool(int(sys.argv[4]))
else:
    M = 256     # total grid points (inner)
    # N = 10000   # time steps
    i = None
    writeToFile = False

# spatial conditions
x0 = 0                       # start
xf = 1                       # end
dx = (xf-x0)/(M+1)           # spatial step size
alpha = 0                    # u(x0, t) = alpha
beta  = 0                    # u(xf, t) = beta
x  = np.linspace(x0,xf,M+2)  # x-axis points

# temporal conditions
N = 10000         # time steps
t0 = 0            # start
tf = 300          # end
dt = (tf - t0)/N  # time step size
t  = np.linspace(t0, tf, N)

# coefficients
k = 0.0002
K = 0.1                # PDE coeff
dxBeta = dx*beta          # for final element of b

# initial condition function
def f(x):
    return np.sin(np.pi*x)

# initial solution
u = np.array(f(x), dtype=np.float64)
un = np.empty(M+2, dtype=np.float64)

# initialize the final solution vector (u0, ..., u_m+1)
# U = np.empty((N,M+2), dtype=np.float64)
# U[0] = u

t_start = time.time()

# Loop over time
for j in range(1,N):
    un[1:M+1] = u[1:M+1] + K*(u[0:M] - 2.0*u[1:M+1] + u[2:M+2] )

    # Force Boundary Conditions
    un[0] = 0.0
    un[M+1] = 0.0

    # save solution
    # U[j] = un

    # Update solution    
    u = un

t_final = time.time()
print t_final

if writeToFile:
    # write time to a file
    F = open('./tests/ser-step/M%s.txt'%(sys.argv[1].zfill(2)), 'r+')
    F.read()
    F.write('%f\n' %(t_final - t_start))
    F.close()

if i == 0:
    G = open('./tests/ser-step/solution-M%s.txt'%(sys.argv[1].zfill(2)), 'r+')
    G.read()
    G.write('%s\n' %str(u))
    G.close()

# fig, ax = plt.subplots()
# ax.pcolormesh(x, t, U)
# plt.title('K = 0.1, M = 256, N = 10000')
# plt.show()
# plt.savefig('heat_1d_stepping.png')

sys.exit()