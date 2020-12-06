import numpy as np
from scipy.spatial.distance import cdist
import sys

def read_gro(fname):
  f = open(fname, "r")
  fl = f.readlines()
  f.close()

  xyz = np.array([line.split()[-3:] for line in fl if "MM" in line]).astype('float')

  return xyz

core_xyz = read_gro(sys.argv[1])
dists = cdist(core_xyz, core_xyz)

n_neigh = np.sum(dists < 0.34, axis=0) - 1
print(np.sum(n_neigh<12))
