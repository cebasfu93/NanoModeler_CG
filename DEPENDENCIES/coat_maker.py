import numpy as np
from scipy.optimize import minimize
from  scipy.spatial.distance import cdist
from DEPENDENCIES.Extras import sunflower_pts

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
            print("Minimization converged at: {}".format(i))
            break
        if i == (max_iter-1):
            print("The minimization of the electric potential energy did not finish")
    return xyz

def place_staples(core_xyz, inp):
    n_tot_lig = inp.lig1_num + inp.lig2_num
    virtual_xyz = sunflower_pts(n_tot_lig)*inp.core_radius + 2*inp.bead_radius
    if n_tot_lig <= 20:
        virtual_xyz = electric_minimization(virtual_xyz)

    core_vir_dists = cdist(virtual_xyz, core_xyz)
    closests = core_xyz[np.argsort(core_vir_dists, axis=1)[:,0]]
    staples_xyz = np.empty(np.shape(closests))
    for c, close in enumerate(closests):
        norma = np.linalg.norm(close)
        staples_xyz[c] = close*(norma+2*inp.bead_radius)/norma

    return staples_xyz

def assign_morphology(staples_xyz, inp):
    n_tot_lig = inp.lig1_num + inp.lig2_num
    indexes = list(range(n_tot_lig))

    if inp.morph == 'random':
        np.random.seed(inp.rsd)
        np.random.shuffle(indexes)
        lig1_ndx = indexes[:inp.lig1_num]
        lig2_ndx = indexes[inp.lig1_num:]
    if inp.morph == 'janus':
        z_sort = np.argsort(staples_xyz[:,2])
        lig1_ndx = z_sort[:inp.lig1_num]
        lig2_ndx = z_sort[inp.lig1_num:]
    if inp.morph == 'stripe':
        phis = np.arccos(np.divide(staples_xyz[:,2], np.linalg.norm(staples_xyz, axis=1)))
        dphi = (math.pi+0.00001)/inp.stripes
        lig1_ndx = []
        lig2_ndx = []
        for i in range(n_tot_lig):
            if phi(staples_xyz[i])//dphi%2 == 0:
                lig1_ndx.append(i)
            elif phi(staples_xyz[i])//dphi%2 == 1:
                lig2_ndx.append(i)
    return (lig1_ndx, lig2_ndx)

def grow_ligands(staples_xyz, lig_ndx, inp):
    lig1_xyz, lig2_xyz = [], []

    for ndx in lig_ndx[0]:
        dist_units = 0
        for i in range(len(inp.lig1_btypes)):
            for n_per_bead in range(inp.lig1_n_per_bead[i]):
                norma = np.linalg.norm(staples_xyz[ndx])
                xyz = staples_xyz[ndx]*(norma+2*dist_units*inp.bead_radius)/norma
                lig1_xyz.append(xyz)
                dist_units += 1

    for ndx in lig_ndx[1]:
        dist_units = 0
        for i in range(len(inp.lig2_btypes)):
            for n_per_bead in range(inp.lig2_n_per_bead[i]):
                norma = np.linalg.norm(staples_xyz[ndx])
                xyz = staples_xyz[ndx]*(norma+2*dist_units*inp.bead_radius)/norma
                lig2_xyz.append(xyz)
                dist_units += 1

    return (lig1_xyz, lig2_xyz)
