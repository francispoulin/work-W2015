import numpy as np
import time, sys
# Set the seed as the answer to the question of life, the universe and everything
np.random.seed(42)

def timer():
    # Calculates the time it takes to do matrix-matrix addition in serial. The first 
    # matrix is a randomly generated one, while the second matrix is a matrix of ones.
    # Eight different sizes are inspected: 64, 128, 256, 512, 1024, 2048, 4096, 8196
    # 1000 tests are done per matrix size, and then the means of the trials are returned.
    times = [None]*100
    means = []

    for M in range(6,13):
        N = 2**M
        A = np.random.rand(N,N)
        for i in xrange(100):
            t0 = time.time()
            B = np.fft.fft(A)
            tf = time.time()
            times[i] = tf - t0

        means.append(np.mean(times))
        # print 'M = %d : %s' %(M, str(np.mean(times)))
        print np.mean(times)

    return means

timer()