import os
import sys
import glob
from numpy import *

myPath = os.path.dirname(os.path.abspath(__file__))

density = 0
d1d = []
for fname in glob.glob(sys.argv[1]):
    d = load(fname)
    print(fname) #, density.sum())
    density = density + d

save('collected_density.npy', density)
