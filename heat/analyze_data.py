import numpy as np

processors = [2, 4]
Ms = range(7, 15)

# parallel files
for d in ['par-spar', 'par-step']:
  for p in processors:
    print d, p

    means = []
    for M in Ms:
      filename = 'tests/%s/p%d-m%s.txt'%(d,p,str(M).zfill(2))

      with open(filename) as f:
        data = [float(i) for i in f.readlines()]
        means.append(np.mean(data))
        # print '%d %d: %e' %(p,M,np.mean(data))
        # print np.mean(data)
    print means

    print

for d in ['ser-spar', 'ser-step']:
  print d, 1

  means = []
  for M in Ms:
    filename = 'tests/%s/m%s.txt' %(d, str(M).zfill(2))

    with open(filename) as f:
      data = [float(i) for i in f.readlines()]
      means.append(np.mean(data))
      # print '%d: %e' %(M,np.mean(data))
  print means

  print 