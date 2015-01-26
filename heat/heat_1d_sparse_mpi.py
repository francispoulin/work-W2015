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

id = comm.Get_rank()   # this process' ID
p = comm.Get_size()    # number of processors