import numpy as np
from  scipy.spatial.distance import cdist
import logging

"""logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())
report = logging.getLogger('nanomodelercg.report')"""

def get_core_bonds(core_xyz, inp):
    core_bonds = []

    if inp.core_en:
        dists = cdist(core_xyz, core_xyz)
        if inp.core_shape != "shell":
            logging.info("\tBuilding elastic network based on first neighbours...")
            close_dists = dists <= (2*inp.bead_radius+0.01)
            for i in range(len(dists)):
                ndx1 = i*1
                close_ndxs = np.where(close_dists[i])[0]
                if len(close_ndxs) == 1:
                    dists_sorted = np.argsort(dists[i])
                    close_ndxs = dists_sorted[[1,2,3,4,5,6]]
                for ndx2 in close_ndxs:
                    if ndx2 != i and [ndx1, ndx2] not in core_bonds and [ndx2, ndx1] not in core_bonds:
                        core_bonds.append([ndx1, ndx2])

        else:
            logging.info("\tBuilding elastic network based on six nearest neighbours and one farthest neighbour...")
            dists_sorted = np.argsort(dists, axis=1)
            for i in range(len(dists)):
                ndx1 = i*1
                close_ndxs = dists_sorted[i,[1,2,3,4,5,6,-1]]
                for ndx2 in close_ndxs:
                    if ndx2 != i and [ndx1, ndx2] not in core_bonds and [ndx2, ndx1] not in core_bonds:
                        core_bonds.append([ndx1, ndx2])

    return core_bonds

def get_lig_bonds(np_xyz, inp):
    n_at1, n_at2 = np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)
    n_core = int(len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2)
    core_xyz = np_xyz[:n_core]

    lig1_bonds, lig2_bonds = [], []

    for i in range(inp.lig1_num):
        ndx0 = n_core + i*n_at1
        ndx1 = ndx0*1
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
        ndx1 = ndx0*1
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
    n_core = int(len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2)
    core_xyz = np_xyz[:n_core]

    lig1_angles, lig2_angles = [], []

    if n_at1 >= 2:
        for i in range(inp.lig1_num):
            ndx0 = n_core + i*n_at1
            ndx1 = ndx0*1
            ndx2 = ndx1 + 1
            ndx3 = np.argsort(cdist([np_xyz[ndx1]], core_xyz))[0,0]
            angle = [ndx3, ndx1, ndx2]
            lig1_angles.append(angle)
            for j in range(1, n_at1-1):
                ndx1 = ndx0 + j
                ndx2 = ndx1 + 1
                ndx3 = ndx1 - 1
                angle = [ndx3, ndx1, ndx2]
                lig1_angles.append(angle)

    if n_at2 >= 2:
        for i in range(inp.lig2_num):
            ndx0 = n_core + n_at1*inp.lig1_num + i*n_at2
            ndx1 = ndx0*1
            ndx2 = ndx1 + 1
            ndx3 = np.argsort(cdist([np_xyz[ndx1]], core_xyz))[0,0]

            angle = [ndx3, ndx1, ndx2]
            lig2_angles.append(angle)
            for j in range(1, n_at2-1):
                ndx1 = ndx0 + j
                ndx2 = ndx1 + 1
                ndx3 = ndx1 - 1
                angle = [ndx3, ndx1, ndx2]
                lig2_angles.append(angle)

    return (lig1_angles, lig2_angles)
