import numpy as np
import time

# warm up
np.random.rand(500,500).dot(np.random.rand(500,500))

"""
# TESTING UPDATING AN ARRAY BY LOOPING THROUGH IT VS. JUST CALLING A FUNCTION ON IT

x = np.random.rand(10000)
a = np.empty(10000, dtype='d')
b = np.empty(10000, dtype='d')
r = range(10000)

# time calling a function on an array
t0 = time.time()
a = np.sin(x)
tf = time.time()
print 'Call function on array: ', tf - t0

# time calling a function on each element
t0 = time.time()
for i in r:
	b[i] = np.sin(x[i])
tf = time.time()
print 'Call function on each element: ', tf-t0


bools = 10000*[None]
for i in r:
	if a[i] == b[i]: bools[i] = True
	else: bools[i] = False

print np.all(bools)

"""

# TESTING THE INCREMENTING SPEED: a+= b  VS  a = a + b
a, b = 0, 1
t0 = time.time()
for i in xrange(1000000):
	a += b
tf = time.time()
print '+= time: ', tf-t0

a,b = 0,1
t0 = time.time()
for i in xrange(1000000):
	a = a + b
tf = time.time()
print '= time: ', tf-t0