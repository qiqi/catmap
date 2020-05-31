import os
import glob
from numpy import *

myPath = os.path.dirname(os.path.abspath(__file__))

density = 0
for fname in glob.glob(os.path.join(myPath, 'density_*.npy')):
    density = density + load(fname)
    print(fname, density.sum())

save('collected_density.npy', density)
