import numpy as np
from  scipy.spatial.distance import cdist
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

def get_core_bonds(core_xyz, inp):
    """
    Determines the atom indices of the core that are to be involved in an elastic network (nearest neighbors except for shells)
    """
    core_bonds = []

    if inp.core_en:
        dists = cdist(core_xyz, core_xyz)
        if inp.core_shape != "shell":
            logger.info("\tBuilding elastic network based on first neighbors...")
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
            logger.info("\tBuilding elastic network based on six nearest neighbours and one farthest neighbour...")
            neighboring_bonds = []
            antipodal_bonds = []
            dists_sorted = np.argsort(dists, axis=1)
            for i in range(len(dists)):
                ndx1 = i*1
                close_ndxs = dists_sorted[i,[1,2,3,4,5,6]]
                for ndx2 in close_ndxs:
                    if ndx2 != i and [ndx1, ndx2] not in core_bonds and [ndx2, ndx1] not in core_bonds:
                        neighboring_bonds.append([ndx1, ndx2])
                antipodal_ndx = dists_sorted[i,-1]
                if antipodal_ndx != i and [ndx1, antipodal_ndx] not in core_bonds and [antipodal_ndx, ndx1] not in core_bonds:
                    antipodal_bonds.append([ndx1, antipodal_ndx, "antipodal"])
            core_bonds = neighboring_bonds + antipodal_bonds

    return core_bonds

def get_lig_bonded_atoms(np_xyz, lig_ndx, close_ndxs, inp):
    """
    Determines the atom indices of the ligands that are to be involved in bonds, angles, and dihedrals
    """
    logger.info("\tAssigning bonds within the ligands...")
    lig_bonds = get_lig_bonds(np_xyz, lig_ndx, close_ndxs, inp)
    logger.info("\tAssigning angles within the ligands...")
    lig_angles = get_lig_angles(np_xyz, lig_ndx, close_ndxs, inp)
    logger.info("\tAssigning dihedrals within the ligands...")
    lig_dihedrals = get_lig_dihedrals(np_xyz, lig_ndx, close_ndxs, inp)

    return lig_bonds, lig_angles, lig_dihedrals

def get_lig_bonds(np_xyz, lig_ndx, close_ndxs, inp):
    """
    Determines the atom indices of the ligands that are to be involved in bonds
    """
    n_at1, n_at2 = np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)
    n_core = int(len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2)
    core_xyz = np_xyz[:n_core]

    lig1_bonds, lig2_bonds = [], []

    for i in range(inp.lig1_num):
        ndx0 = n_core + i*n_at1
        ndx1 = ndx0*1
        ndx2 = close_ndxs[lig_ndx[0][i]]#np.argsort(cdist([np_xyz[ndx0]], core_xyz))[0,0]
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
        ndx2 = close_ndxs[lig_ndx[1][i]]#np.argsort(cdist([np_xyz[ndx0]], core_xyz))[0,0]
        bond = [ndx1, ndx2]
        lig2_bonds.append(bond)
        for j in range(n_at2-1):
            ndx1 = ndx0 + j
            ndx2 = ndx1 + 1
            bond = [ndx1, ndx2]
            lig2_bonds.append(bond)
    return (lig1_bonds, lig2_bonds)

def get_lig_angles(np_xyz, lig_ndx, close_ndxs, inp):
    """
    Determines the atom indices of the ligands that are to be involved in angles
    """
    n_at1, n_at2 = np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)
    n_core = int(len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2)
    core_xyz = np_xyz[:n_core]

    lig1_angles, lig2_angles = [], []

    if n_at1 >= 2:
        for i in range(inp.lig1_num):
            ndx0 = n_core + i*n_at1
            ndx1 = ndx0*1
            ndx2 = ndx1 + 1
            ndx3 = close_ndxs[lig_ndx[0][i]]#np.argsort(cdist([np_xyz[ndx1]], core_xyz))[0,0]
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
            ndx3 = close_ndxs[lig_ndx[1][i]]#np.argsort(cdist([np_xyz[ndx1]], core_xyz))[0,0]

            angle = [ndx3, ndx1, ndx2]
            lig2_angles.append(angle)
            for j in range(1, n_at2-1):
                ndx1 = ndx0 + j
                ndx2 = ndx1 + 1
                ndx3 = ndx1 - 1
                angle = [ndx3, ndx1, ndx2]
                lig2_angles.append(angle)

    return (lig1_angles, lig2_angles)

def get_lig_dihedrals(np_xyz, lig_ndx, close_ndxs, inp):
    """
    Determines the atom indices of the ligands that are to be involved in dihedrals
    """
    n_at1, n_at2 = np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)
    n_core = int(len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2)
    core_xyz = np_xyz[:n_core]

    lig1_dihedrals, lig2_dihedrals = [], []

    if n_at1 >= 3:
        for i in range(inp.lig1_num):
            ndx0 = n_core + i*n_at1
            ndx1 = ndx0*1
            ndx2 = ndx1 + 1
            ndx3 = ndx1 + 2
            ndx4 = close_ndxs[lig_ndx[0][i]]#np.argsort(cdist([np_xyz[ndx1]], core_xyz))[0,0]
            dihedral = [ndx4, ndx1, ndx2, ndx3]
            lig1_dihedrals.append(dihedral)
            for j in range(n_at1-4):
                ndx1 = ndx0 + j
                ndx2 = ndx1 + 1
                ndx3 = ndx1 + 2
                ndx4 = ndx1 + 3
                dihedral = [ndx1, ndx2, ndx3, ndx4]
                lig1_dihedrals.append(dihedral)

    if n_at2 >= 3:
        for i in range(inp.lig2_num):
            ndx0 = n_core + n_at1*inp.lig1_num + i*n_at2
            ndx1 = ndx0*1
            ndx2 = ndx1 + 1
            ndx3 = ndx1 + 2
            ndx4 = close_ndxs[lig_ndx[1][i]]#np.argsort(cdist([np_xyz[ndx1]], core_xyz))[0,0]
            dihedral = [ndx4, ndx1, ndx2, ndx3]
            lig2_dihedrals.append(dihedral)
            for j in range(n_at2-4):
                ndx1 = ndx0 + j
                ndx2 = ndx1 + 1
                ndx3 = ndx1 + 2
                ndx4 = ndx1 + 3
                dihedral = [ndx1, ndx2, ndx3, ndx4]
                lig2_dihedrals.append(dihedral)

    return (lig1_dihedrals, lig2_dihedrals)
