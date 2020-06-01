import os
import glob
from numpy import *

myPath = os.path.dirname(os.path.abspath(__file__))

#density = 0
d1d = []
for fname in glob.glob(os.path.join(myPath, 'density_*.npy')):
    d = load(fname)
    print(fname) #, density.sum())
    d1d.append(d[:,1146:1154].sum(1))

#save('collected_density.npy', density)
save('d1d.npy', d1d)
