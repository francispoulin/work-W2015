import numpy as np
import time, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Set the size of the doubly periodic box N**2
M = 13
N = 2**M
# Set the seed as the answer to the question of life, the universe and everything
np.random.seed(42)

# Define number of processes and rank
num_processes = comm.Get_size()
rank = comm.Get_rank()
if not num_processes in [2**i for i in range(M+1)]:
    raise IOError("Number of cpus must be in ", [2**i for i in range(M+1)])

# Each cpu gets ownership of Np slices
Np = N / num_processes

x = np.random.rand(N,1)

# resultant vector
b_parallel = np.empty((N, 1))
A = np.ones((Np,N), dtype='d')

comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

b = np.dot(A, x)

comm.Gather([b, MPI.DOUBLE], [b_parallel, MPI.DOUBLE])

comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer

if rank == 0:
    print t_final

sys.exit()