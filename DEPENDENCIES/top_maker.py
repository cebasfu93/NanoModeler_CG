import numpy as np
from  scipy.spatial.distance import cdist

def get_lig_bonds(np_xyz, inp):
    n_at1, n_at2 = np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)
    n_core = len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2
    core_xyz = np_xyz[:n_core]

    lig1_bonds, lig2_bonds = [], []
    for i in range(inp.lig1_num):
        ndx0 = n_core + i*n_at1
        ndx1 = ndx0
        ndx2 = np.argsort(cdist([np_xyz[ndx0]], core_xyz))[0,0]
        bond = [ndx1, ndx2]
        lig1_bonds.append(bond)
        for j in range(n_at1-1):
            ndx1 = ndx0 + j
            ndx2 = ndx1 + 1
            bond = [ndx1, ndx2]
            lig1_bonds.append(bond)

    for i in range(inp.lig2_num):
        ndx0 = n_core + n_at1*inp.lig1_num + i*n_at2
        ndx1 = ndx0
        ndx2 = np.argsort(cdist([np_xyz[ndx0]], core_xyz))[0,0]
        bond = [ndx1, ndx2]
        lig2_bonds.append(bond)
        for j in range(n_at2-1):
            ndx1 = ndx0 + j
            ndx2 = ndx1 + 1
            bond = [ndx1, ndx2]
            lig2_bonds.append(bond)

    return (lig1_bonds, lig2_bonds)

def get_lig_angles(np_xyz, inp):
    n_at1, n_at2 = np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)
    n_core = len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2
    core_xyz = np_xyz[:n_core]

    lig1_angles, lig2_angles = [], []
    if n_at1 >= 2:
        for i in range(inp.lig1_num):
            ndx0 = n_core + i*n_at1
            ndx1 = ndx0
            ndx2 = ndx1 + 1
            ndx3 = np.argsort(cdist([np_xyz[ndx1]], core_xyz))[0,0]
            angle = [ndx1, ndx2, ndx3]
            lig1_angles.append(angle)
            for j in range(1, n_at1-1):
                ndx1 = ndx0 + j
                ndx2 = ndx1 + 1
                ndx3 = ndx1 - 1
                angle = [ndx1, ndx2, ndx3]
                lig1_angles.append(angle)

    if n_at2 >= 2:
        for i in range(inp.lig2_num):
            ndx0 = n_core + n_at1*inp.lig1_num + i*n_at2
            ndx1 = ndx0
            ndx2 = ndx1 + 1
            ndx3 = np.argsort(cdist([np_xyz[ndx0]], core_xyz))[0,0]

            angle = [ndx1, ndx2, ndx3]
            lig2_angles.append(angle)
            for j in range(1, n_at2-1):
                ndx1 = ndx0 + j
                ndx2 = ndx1 + 1
                ndx3 = ndx1 - 1
                angle = [ndx1, ndx2, ndx3]
                lig2_angles.append(angle)

    return (lig1_angles, lig2_angles)
