import os
import glob
from numpy import *

myPath = os.path.dirname(os.path.abspath(__file__))

density = 0
d1d = []
for fname in glob.glob(os.path.join(myPath, 'density_*.npy')):
    d = load(fname)
    print(fname) #, density.sum())
    density = density + d

save('collected_density.npy', density)
