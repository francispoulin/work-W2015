# remove all content from files

import subprocess
"""
for d in ['ser-spar', 'ser-step']:
    for m in xrange(7,15):
        D1 = './%s/m%s.txt' %(d, str(m).zfill(2))
        D2 = './%s/solution-m%s.txt' %(d, str(m).zfill(2))
        subprocess.call('> ' + D1, shell=True)
        subprocess.call('> ' + D2, shell=True)
"""
for d in ['par-spar', 'par-step']:
    for p in xrange(2,6,2):
        for m in xrange(7,15):
            D1 = './%s/p%d-m%s.txt' %(d, p, str(m).zfill(2))
            D2 = './%s/solution-p%d.txt' %(d, p)
            subprocess.call('> ' + D1, shell=True)
            subprocess.call('> ' + D2, shell=True)
