import os
import json
import subprocess
from numpy import *

myPath = os.path.dirname(os.path.abspath(__file__))
config = json.load(open(os.path.join(myPath, 'config.json')))
binary = config['binary']

procs = []
nIter = 1024
for i in range(6):
    p = subprocess.Popen(binary, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(array([i, random.randint(1<<30)], uint32).tobytes())
    p.stdin.write(array([0.2, 0.0, 0.4, 0.0], float32).tobytes())
    p.stdin.write(array(nIter, uint32).tobytes())
    p.stdin.flush()
    procs.append(p)

n = 2048
out = [frombuffer(p.stdout.read(n * n * 8), double) for p in procs]
density = sum(out, 0).reshape([n, n])

print(density.sum())
print(density.max())

#from pylab import *
#hist(density[425,:], 100)

myname = subprocess.check_output('hostname').decode().strip()
fname = 'density_{}_{}_{}_{}.npy'.format(os.path.basename(binary), myname, nIter, random.rand())
save(os.path.join(myPath, fname), density)
