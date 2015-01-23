import numpy as np
import time, sys

# M = 12
# Set the seed as the answer to the question of life, the universe and everything
np.random.seed(42)
# Array of times and yeah

def time_stuff():
    times = [None]*1000
    means = []

    for M in range(6,15):
        N = 2**M
        A = np.ones((N,N), dtype='d')
        x = np.random.rand(N,1)
        for i in xrange(1000):
            t0 = time.time()
            b = np.dot(A,x)
            tf = time.time()
            times[i] = tf - t0

        means.append(np.mean(times))
        # print 'M = %d : %s' %(M, str(np.mean(times)))
        print np.mean(times)

    return means

time_stuff()