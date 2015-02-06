import numpy as np
import time, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD

np.random.seed(42)

# grid points
if len(sys.argv) > 1:
	M = int(sys.argv[1])
else:
	M = 13
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
# resultant matrix (global)
Bg = np.empty((N, N), dtype='complex128')

# sub-matrix for this process (Np-by-N)
A = Ag[rank*Np : (rank+1)*Np, :]

comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

B = np.fft.fft(A)

comm.Gather([B, MPI.DOUBLE], [Bg, MPI.DOUBLE])

comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer

if rank == 0:
	print t_final

sys.exit()