import numpy as np
from scipy.optimize import minimize
from  scipy.spatial.distance import cdist
from DEPENDENCIES.Extras import sunflower_pts, cartesian_to_polar, polar_to_cartesian
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

def sphere_cons(xyz, rad):
    zero = np.linalg.norm(xyz) - rad
    return zero

def calc_Q(xyz, staples, ndx):
    staples[ndx] = xyz
    dists = cdist(staples, staples)
    dists = dists[dists!=0]
    Q = np.sum(np.reciprocal(dists))
    return Q

def electric_minimization(xyz):
    logger.info("\tMinimizing points on a sphere...")
    R_model = np.linalg.norm(xyz[0])
    max_iter = 100 #this value is an arbitrary value to avoid getting stuck in a loop
    for i in range(max_iter):
        iterations = []
        for j in range(len(xyz)):
            cons = {'type':'eq', 'fun':sphere_cons, 'args':[R_model]}
            res_min = minimize(calc_Q, x0=xyz[j], args=(xyz, j), constraints=cons)
            xyz[j] = res_min.x
            iterations.append(res_min.nit)
        if np.all(np.array(iterations)==1):
            logger.info("\tMinimization converged at {} steps...".format(i))
            break
        if i == (max_iter-1):
            warn_txt = "\tATTENTION. The minimization of the electric potential energy did not finish. Double check the position of the ligands in the output structure..."
            logger.warning(warn_txt)
    return xyz

def place_staples(core_xyz, inp):
    d_thres = 2*inp.bead_radius+0.01 #threshold to find neighbors to calculate normals to surface

    virtual_xyz = sunflower_pts(inp.n_tot_lig)*(inp.char_radius + 2*inp.bead_radius)
    if inp.n_tot_lig <= 20:
        logger.info("\tAnchors will be placed minimizing electric energy. This will place the ligands as far away as possible from one another...")
        virtual_xyz = electric_minimization(virtual_xyz)

    if inp.core_shape != "shell":
        core_dists = cdist(core_xyz, core_xyz)
        surface = np.sum(core_dists<=d_thres, axis=1) < inp.n_coord
    else:
        surface = np.ones(len(core_xyz), dtype='bool')
    logger.info("\tThe surface beads of the core allow a maximum of {} ligands...".format(np.sum(surface)))
    if np.sum(surface) < inp.n_tot_lig:
        err_txt = "\tThere are more ligands than surface core beads..."
        raise ValueError(err_txt)

    core_vir_dists = cdist(cartesian_to_polar(virtual_xyz)[:,1:], cartesian_to_polar(core_xyz)[:,1:])
    core_vir_dists_sort = np.argsort(core_vir_dists, axis=1)
    close_ndxs = []
    for i in range(inp.n_tot_lig):
        D = 0
        while len(close_ndxs) != i+1:
            test_ndx = core_vir_dists_sort[i,D]
            if test_ndx not in close_ndxs and surface[test_ndx]:
                #if D != 0:
                    #print(D)
                close_ndxs.append(test_ndx)
            D += 1
    closests = core_xyz[close_ndxs]

    logger.info("\tSaving normal directions to the surface at the anchoring sites...")
    normals = np.zeros((inp.n_tot_lig, 3))
    staples_xyz = np.empty((inp.n_tot_lig, 3))
    dists = cdist(closests, core_xyz)
    sort_dists = np.sort(dists, axis=1)
    for i, xyz in enumerate(closests):
        if inp.core_shape != "shell":
            neigh_ndx = np.where(dists[i] <= d_thres)[0]
            neighbors = core_xyz[neigh_ndx]
            diffs = neighbors - xyz
            normal = -1*np.mean(diffs, axis=0)
            norm = np.linalg.norm(normal)
            normals[i] = normal/norm
        else:
            normals[i] = xyz/np.linalg.norm(xyz)
        staples_xyz[i] = xyz+2*inp.bead_radius*normals[i]

    return staples_xyz, normals

def assign_morphology(staples_xyz, inp):
    indexes = list(range(inp.n_tot_lig))

    if inp.morph == 'janus_x' or inp.morph == "stripe_x":
        ax = 0
    elif inp.morph == 'janus_y' or inp.morph == "stripe_y":
        ax = 1
    elif inp.morph == 'janus_z' or inp.morph == "stripe_z":
        ax = 2

    if 'janus' in inp.morph:
        logger.info("\tDistributing ligands in a Janus configuration...")
        ax_sort = np.argsort(staples_xyz[:,ax])
        lig1_ndx = ax_sort[:inp.lig1_num]
        lig2_ndx = ax_sort[inp.lig1_num:]

    if 'stripe' in inp.morph:
        logger.info("\tDistributing ligands in a Striped configuration...")
        phis = np.arccos(np.divide(staples_xyz[:,ax], np.linalg.norm(staples_xyz, axis=1)))
        dphi = (np.pi+0.00001)/inp.stripes
        lig1_ndx = []
        lig2_ndx = []
        for i in range(inp.n_tot_lig):
            if phis[i]//dphi%2 == 0:
                lig1_ndx.append(i)
            elif phis[i]//dphi%2 == 1:
                lig2_ndx.append(i)
        inp.lig1_num = len(lig1_ndx)
        inp.lig2_num = len(lig2_ndx)

    if inp.morph == 'random':
        logger.info("\tDistributing ligands in a Random configuration...")
        np.random.seed(inp.rsd)
        np.random.shuffle(indexes)
        lig1_ndx = indexes[:inp.lig1_num]
        lig2_ndx = indexes[inp.lig1_num:]
    return (lig1_ndx, lig2_ndx)

def grow_ligands(staples_xyz, staples_normals, lig_ndx, inp):
    lig1_xyz, lig2_xyz = [], []
    logger.info("\tGrowing ligand 1 from the respective anchoring sites...")
    for ndx in lig_ndx[0]:
        dist_units = 0
        for i in range(len(inp.lig1_btypes)):
            for n_per_bead in range(inp.lig1_n_per_bead[i]):
                norma = np.linalg.norm(staples_xyz[ndx])
                xyz = staples_xyz[ndx]*(norma+2*dist_units*inp.bead_radius)/norma
                lig1_xyz.append(xyz)
                dist_units += 1

    logger.info("\tGrowing ligand 2 from the respective anchoring sites...")
    for ndx in lig_ndx[1]:
        dist_units = 0
        for i in range(len(inp.lig2_btypes)):
            for n_per_bead in range(inp.lig2_n_per_bead[i]):
                norma = np.linalg.norm(staples_xyz[ndx])
                xyz = staples_xyz[ndx]*(norma+2*dist_units*inp.bead_radius)/norma
                lig2_xyz.append(xyz)
                dist_units += 1

    return (lig1_xyz, lig2_xyz)
