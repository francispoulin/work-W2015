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
size = comm.Get_size()
rank = comm.Get_rank()
if not size in [2**i for i in range(M+1)]:
    raise IOError("Number of cpus must be in ", [2**i for i in range(M+1)])

# Each cpu gets ownership of Np slices
Np = N / size

# create the matrix and vector on the root process; scatter the matrix to each 
# process, and then broadcast the vector to all processes.
if rank == 0:
    # this 'hack' is required, as comm.scatter expects SIZE # of objects in an array,
    # but a numpy array is N objects long. thus, create a list of length SIZE and split 
    # the matrix into pieces and stuff it in there
    A = np.random.rand(N,N)
    Ag = size*[None]
    for i in xrange(size):
        Ag[i] = A[i*Np : (i+1)*Np, :]
    # print 'A: %s\n\n\n'%str(A)

    # vector is created here because the RNG was being very weird in parallel
    x = np.random.rand(N,1)
else:
    Ag = None
    x = None

Ag = comm.scatter(Ag)
x = comm.bcast(x)
# print 'A%d: %s' %(rank, str(Ag))
# print 'rank: %d, %s' %(rank, x)

# resultant vector
b_parallel = np.empty((N, 1), dtype='d')


comm.Barrier()         # start MPI timer
t_start = MPI.Wtime()

b = np.dot(Ag, x)
# print 'b%d: %s' %(rank, str(b))

comm.Gather([b, MPI.DOUBLE], [b_parallel, MPI.DOUBLE])

comm.Barrier()
t_final = (MPI.Wtime() - t_start)  # stop MPI timer

if rank == 0:
    # print b_parallel
    print t_final

sys.exit()