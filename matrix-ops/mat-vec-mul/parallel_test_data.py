import numpy as np
from collections import OrderedDict

processors = [2,4]
Ms = range(6,15)
means = OrderedDict()

for p in processors:
	print p
	for M in Ms:
		filename = 'tests/p%d-m%d.txt'%(p,M)

		with open(filename) as f:
			data = [float(i) for i in f.readlines()]
			# print '%d %d: %e' %(p,M,np.mean(data))
			print np.mean(data)