import matplotlib
matplotlib.use('Agg')
import sys
import subprocess

from pylab import *
from numpy import *
from mpi4py import MPI

catBin = sys.argv[1]

procs = []
for i in range(6):
    p = subprocess.Popen(catBin, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(array(i, uint32).tobytes())
    p.stdin.write(array([0.2, 0.0, 0.0, 1.0], float32).tobytes())
    p.stdin.write(array(512, uint32).tobytes())
    p.stdin.flush()
    procs.append(p)

n = 2048
out = [frombuffer(p.stdout.read(n * n * 8), double) for p in procs]
d = sum(out, 0).reshape([n, n])

comm = MPI.COMM_WORLD
density = zeros_like(d)
comm.Reduce(d, density)
if comm.rank == 0:
    print(density.sum())
    print(density.max())
    save('density.npy', density)
    figure(figsize=(32,32))
    imshow(density)
    axis('scaled')
    axis('off')
    grid()
    savefig('2000.png')
