import sys
import subprocess

from pylab import *
from numpy import *

catBin = sys.argv[1]

procs = []
for i in range(6):
    p = subprocess.Popen(catBin, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(array(i, uint32).tobytes())
    p.stdin.write(array([0.1, 0.1, 0.5, 0.5], float32).tobytes())
    p.stdin.write(array(1024, uint32).tobytes())
    p.stdin.flush()
    procs.append(p)

n = 2048
out = [frombuffer(p.stdout.read(n * n * 8), double) for p in procs]
density = sum(out, 0).reshape([n, n])
print(density)
print(density.sum())
print(density.max())
imshow(density)
