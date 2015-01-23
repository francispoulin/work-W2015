# testing whether or not
# $ mpirun -np [...] python script.py 
# can have command line arguments as well
import numpy as np
import time, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
args = sys.argv[1:]

print 'rank: %d'%comm.Get_rank(), sys.argv 