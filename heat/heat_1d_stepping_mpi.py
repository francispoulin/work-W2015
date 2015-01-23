#!/usr/bin/env python
# heat_mpi.py
#
# This will solve the 1-D heat equation in parallel using mpi4py

import sys
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD

id = comm.Get_rank()   # this process' ID
p = comm.Get_size()    # number of processors

# Number of grid points per core
m = 10000

# Grid Parameters
x_max = 1.0
x_min = 0.0
dx = (x_max - x_min)/(p*m-1.0)   # spatial step

# Build the grid
x = np.linspace(x_min + (x_max-x_min)*id, (id+1)*x_max, m+2)  # process' slice of spatial points
u = np.zeros(m+2,dtype='d')      # process' slice of solution
un= np.zeros(m+2,dtype='d')      # process' slice of NEW solution (step)
xf= np.zeros(m*p, dtype='d')     # global spatial points
uf= np.zeros(m*p, dtype='d')     # global solution points

# Diffusion coefficient
k = 0.002

# Temporal parameters
tf = 1.0
t0 = 0
dt = 0.0001
nt = np.int(tf/dt)+1

# Define constant
K = dt*k/dx/dx

# Define Initial Conditons
u = np.sin(np.pi*x)


comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

# Loop over time
for j in range(1,nt):

    # Send u[1] to ID-1
    if 0 < id:
        comm.send(u[1], dest=id-1, tag=1)
        
    # Receive u[M+1] to ID+1
    if id < p-1:
        u[m+1] = comm.recv(source=id+1, tag=1)    

    # Send u[M] to ID+1
    if id < p-1:
        comm.send(u[m], dest=id+1, tag=2)
        
    # Receive u[0] to ID-1
    if 0 < id: 
        u[0] = comm.recv(source=id-1, tag=2)    

    # Update temperature
    un[1:m+1] = u[1:m+1] + K*(u[0:m] - 2.0*u[1:m+1] + u[2:m+2] )

    # Force Boundary Conditions
    if id == 0:
        un[0] = 0.0

    if id == p-1:
        un[m+1] = 0.0

    # Update time and solution    
    u = un
    
    # Gather parallel vectors to a serial vector
    #comm.Gather(u[1:m+1], uf, root=0)


comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer

# Plot Final Conditions    
if id == 0:
    print t_final
    print np.array(u, dtype='d')

    #plt.clf()
    #plt.plot(xf, uf,'-r')
    #plt.draw()
    #plt.show()

sys.exit()