#!/usr/bin/env python
# heat_mpi.py
#
# This will solve the 1-D heat equation in parallel using mpi4py

import sys
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

np.set_printoptions(threshold=np.inf)  # make sure ENTIRE array is printed

rank = comm.Get_rank()   # this process' ID
p = comm.Get_size()    # number of processors

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

m = M/p    # grid points per processor

# spatial parameters
xf = 1.0
x0 = 0.0
dx = (xf - x0)/(M+1)
alpha = 0 
beta  = 0
# this takes the interval [x0,xf] and splits it equally among all processes
x = np.linspace(x0 + rank*(xf-x0)/p, x0 + (rank+1)*(xf-x0)/p, m+2)
# this takes the interval [x0, xf] and extends it across all processes (i.e. 4*[x0,xf])
# x = np.linspace(x0 + (xf-x0)*rank, (rank+1)*xf, m+2)

# temporal parameters
N = 10000
tf = 300
t0 = 0
dt = (tf - t0)/N

# coefficients
k = 0.002
# K = dt*k/dx/dx
K = 0.1

# initial condition function
def f(x):
    return np.sin(np.pi*x)

# Build the grid
u = f(x)                      # process' slice of solution
un= np.empty(m+2,dtype='d')   # process' slice of NEW solution

if rank == 0:
    xg= np.linspace(x0, xf, M+2)             # global spatial points
    uf= f(xg)[1:-1]                          # global solution points
    # U = np.empty((N,M), dtype=np.float64)
    # U[0] = uf
    t = np.linspace(t0, tf, N)
else:
    uf = None


comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

# Loop over time
for j in range(1,N):

    # Send u[1] to ID-1
    if 0 < rank:
        comm.send(u[1], dest=rank-1, tag=1)
        
    # Receive u[M+1] to ID+1
    if rank < p-1:
        u[m+1] = comm.recv(source=rank+1, tag=1)    

    # Send u[M] to ID+1
    if rank < p-1:
        comm.send(u[m], dest=rank+1, tag=2)
        
    # Receive u[0] to ID-1
    if 0 < rank: 
        u[0] = comm.recv(source=rank-1, tag=2)    

    # Update temperature
    un[1:m+1] = u[1:m+1] + K*(u[0:m] - 2.0*u[1:m+1] + u[2:m+2] )

    # Force Boundary Conditions
    if rank == 0:
        un[0] = 0.0
    elif rank == p-1:
        un[m+1] = 0.0

    # Update time and solution    
    u = un
    
    # Gather parallel vectors to a serial vector
    # comm.Gather(u[1:m+1], uf, root=0)
    # if rank == 0:
    #     U[j] = uf


comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer

if rank == 0:
    if writeToFile:
        # write time to a file
        F = open('./tests/par-step/p%d-m%s.txt' %(p, sys.argv[1].zfill(2)), 'r+')
        F.read()
        F.write('%f\n'% t_final)
        F.close()
    
    print t_final

    # write the solution to a file, but only once!
    if i == 0:
        G = open('./tests/par-spar/solution-p%d.txt'%p, 'r+')
        G.read()
        G.write('%s\n' %str(u))
        G.close()

    # fig, ax = plt.subplots()
    # ax.pcolormesh(xg[1:-1], t, U)
    # plt.title('K = 0.1, m = 256, N = 10000')
    # plt.show()
    # plt.savefig('heat_1d_stepping_mpi.png')

sys.exit()