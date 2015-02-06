import numpy as np
import scipy as sp
import sys

processors = [2, 4]
Ms = range(8, 13)
D1 = ['par-step']
D2 = ['ser-step']
rootDir = 'heat-2d'

# parallel files
for d in D1:
  for p in processors:
    for M in Ms:
      filename = '%s/tests/%s/p%d-M%s.txt' % (rootDir, d, p, str(M).zfill(2))

      with open(filename) as f:
        data = [float(i) for i in f.readlines()]

for d in D2:
  means = []
  for M in Ms:
    filename = '%s/tests/%s/M%s.txt' % (rootDir, d, str(M).zfill(2))

    with open(filename) as f:
      data = [float(i) for i in f.readlines()]
