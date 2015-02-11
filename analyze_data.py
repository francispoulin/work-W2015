import numpy as np

processors = [2, 4]
Ms = range(8, 11)
D1 = ['par-step']
D2 = ['ser-step']
rootDir = 'heat-2d'


def get_parallel_data(rootDir, procs, method, Ms, subDir=''):
  """
  example:
    rootDir = 'heat-2d'
    procs   = [2, 4]
    method  = 'par-step'
    subDir  = 'numba'       # for numba tests
    Ms      = [8, 9, 10]
  """
  if subDir != '':
    subDir += '/'

  result = dict([(p, []) for p in procs])

  for p in procs:
    means = []
    for M in Ms:
      filename = '%s/tests/%s%s/p%d-M%s.txt' % (rootDir, subDir, method, p, str(M).zfill(2))

      with open(filename) as f:
        data = [float(i) for i in f.readlines()]
        means.append(np.mean(data))

    result[p] = means

  return result


def get_serial_data(rootDir, method, Ms, subDir=''):
  """
  example:
    rootDir = 'heat-2d'
    method  = 'ser-step'
    subDir  = 'numba'       # for numba tests
    Ms      = [8, 9, 10]
  """
  if subDir != '':
    subDir += '/'

  means = []
  for M in Ms:
    filename = '%s/tests/%s%s/M%s.txt' % (rootDir, subDir, method, str(M).zfill(2))

    with open(filename) as f:
      data = [float(i) for i in f.readlines()]
      means.append(np.mean(data))
  print 's', means

  return means

print get_serial_data('heat-2d', 'ser-step', range(8,11), 'original')