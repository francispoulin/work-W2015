import numpy as np
import time, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

np.random.seed(42)

# grid points
if len(sys.argv) > 1:
	M = int(sys.argv[1])
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
Ag = np.ones((N,N), dtype='d')
B = np.ones((N,N), dtype='d')
# resultant matrix (global)
Cg = np.empty((N, N), dtype='d')

# sub-matrix for this process (Np-by-N)
A = Ag[rank*Np : (rank+1)*Np, :]

comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

C = np.dot(A,B)

t_final = (MPI.Wtime() - t_start)  # stop MPI timer
comm.Gather([C, MPI.DOUBLE], [Cg, MPI.DOUBLE])

if rank == 0:
	print t_final

