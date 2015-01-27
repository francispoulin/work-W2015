from __future__ import division
import scipy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import time, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

rank = comm.Get_rank()   # this process' ID
p = comm.Get_size()    # number of processors

# CPU warmup
np.random.rand(500,500).dot(np.random.rand(500,500))

# read from STDIN
if len(sys.argv) > 1:
    N = 10**(int(sys.argv[1]))
    M = 10**(int(sys.argv[2]))
else:
    N = 1000   # time steps
    M = 2048    # total grid points (inner)

m = M/p    # grid points per processor

# spatial parameters
xf = 1.0
x0 = 0.0
dx = (xf - x0)/(M+1)
alpha = 0 
beta  = 0
x = np.linspace(x0, xf, M+2)

# temporal parameters
tf = 300
t0 = 0
dt = (tf - t0)/N

# coefficients
k = 0.002
# K = dt*k/dx/dx
K = 0.0001

# initial condition function
def f(x):
    return np.sin(np.pi*x)


# constructing the process' slice of the matrix, A
diags = [K, 1-2*K, K]
if rank == 0:
    As = sp.diags(diags, [-1,0,1], shape=(m+1,M+2), format='dok')
    As[0,0], As[0,1] = 1, 0
    As = sp.csr_matrix(As)

elif rank == p-1:
    As = sp.diags(diags, [rank*m,rank*m+1,rank*m+2], shape=(m+1,M+2), format='dok')
    As[m,M], As[m,M+1] = -1, 1
    As = sp.csr_matrix(As)

else:
    As = sp.diags(diags, [rank*m,rank*m+1,rank*m+2], shape=(m,M+2), format='csr')

# initial solution
u = f(x)

comm.Barrier()
t0 = MPI.Wtime()

for i in xrange(N):
    # process' slice of solution
    us = As.dot(u)

    comm.Allgather(us, u)
    u[0], u[M+1] = 0, 0

comm.Barrier()
tf = (MPI.Wtime() - t0)

if rank == 0:
    print tf