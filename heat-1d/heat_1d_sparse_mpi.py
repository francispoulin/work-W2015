from __future__ import division
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import time, sys
from mpi4py import MPI
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
x = np.linspace(x0, xf, M+2)

# temporal parameters
N = 10000
tf = 300
t0 = 0
dt = (tf - t0)/N
t = np.linspace(t0, tf, N)

# coefficients
k = 0.002
# K = dt*k/dx/dx
K = 0.1

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

mI = int(m)
count = [mI+1] + (p-2)*[mI] + [mI+1]
displs = [0] + range(mI+1, p*mI + 1, mI)


# initial solution
u = f(x)
# if rank == 0:
#     U = np.empty((N,M+2), dtype=np.float64)


comm.Barrier()
t_start = MPI.Wtime()

for i in xrange(N):
    # if rank == 0:
    #     U[i] = u

    # process' slice of solution
    us = As.dot(u)

    comm.Allgatherv([us, MPI.DOUBLE], [u, count, displs, MPI.DOUBLE ])
    u[0], u[M+1] = 0, 0

comm.Barrier()
t_final = (MPI.Wtime() - t_start)

if rank == 0:

    if writeToFile:
        # write time to a file
        F = open('./tests/par-spar/p%d-m%s.txt' %(p,sys.argv[1].zfill(2)), 'r+')
        F.read()
        F.write('%f\n'% t_final)
        F.close()
        
    print t_final

    # write the solution to a file, but only once!
    if i == 0:
        G = open('./tests/par-spar/solution-p%d.txt'%(p), 'r+')
        G.read()
        G.write('%s\n' %str(u))
        G.close()


    # fig, ax = plt.subplots()
    # ax.pcolormesh(x, t, U)
    # plt.title('K = 0.1, m = 256, N = 10000')
    # plt.show()
    # plt.savefig('heat_1d_sparse_mpi.png')