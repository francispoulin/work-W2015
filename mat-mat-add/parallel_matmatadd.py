'''
This script does matrix-matrix addition in parallel, where each matrix is NxN.
Note that it is not entirely practical, as all NxN matrices are defined globally 
(i.e. defined during each process). This was done because scattering takes more 
time, and this was only to test the timing of the operations done in parallel.
'''

import numpy as np
import time, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

np.random.seed(42)

# doubly periodic box size
# if we're doing this from the terminal, allow a parameter to be passed and set that as M
if len(sys.argv) > 1:
    M = int(sys.argv[1])
# otherwise, just use M = 3...
else:
    M = 3
N = 2**M  # dim of matrix

# Define number of processes and rank
num_processes = comm.Get_size()
rank = comm.Get_rank()
if not num_processes in [2**i for i in range(M+1)]:
    raise IOError("Number of cpus must be in ", [2**i for i in range(M+1)])

# Each cpu gets ownership of Np slices
Np = N / num_processes

# 'global' matrices (exposed to all processes)
Ag = np.random.rand(N,N)
Bg = np.ones((N,N), dtype='d')
# resultant matrix (global)
Cg = np.empty((N, N), dtype='d')

# sub-matrix for this process (Np-by-N)
A = Ag[rank*Np : (rank+1)*Np, :]
B = Bg[rank*Np : (rank+1)*Np, :]

comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

C = np.add(A,B)

comm.Gather([C, MPI.DOUBLE], [Cg, MPI.DOUBLE])

comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer

if rank == 0:
    print t_final

sys.exit()