import numpy as np
from scipy.optimize import minimize
from  scipy.spatial.distance import cdist
from DEPENDENCIES.Extras import sunflower_pts, cartesian_to_polar, polar_to_cartesian, rot_mat
from DEPENDENCIES.ThomsonMC import ThomsonMC
from  DEPENDENCIES.transformations import *
from DEPENDENCIES.shape_normals import *
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

def place_staples(core_xyz, inp):
    """
    Return the position of the core beads where to place ligands and the vectors normal to the core's shape at such points
    """
    d_thres = 2*inp.bead_radius+0.01 #threshold to find neighbors to calculate normals to surface

    if inp.n_tot_lig <= 300 and inp.n_tot_lig > 0:
        logger.info("\tAnchors will be placed minimizing electric energy following a Monte Carlo approach. This will place the ligands as far away as possible from one another...")
        if inp.n_tot_lig == 1:
            virtual_xyz = np.array([[inp.char_radius + 2*inp.bead_radius, 0, 0]])
        elif inp.n_tot_lig == 2:
            virtual_xyz = np.array([[inp.char_radius + 2*inp.bead_radius, 0, 0], [-1*(inp.char_radius + 2*inp.bead_radius), 0, 0]])
        else:
            virtual_xyz = ThomsonMC(n=inp.n_tot_lig, mcs=1000, sigma=0.01)*(inp.char_radius + 2*inp.bead_radius)
    else:
        logger.info("\tThe number of ligands is too big to optimize their location on the core. Anchors will be placed sampling randomly the space in spherical coordinates...")
        virtual_xyz = sunflower_pts(inp.n_tot_lig)*(inp.char_radius + 2*inp.bead_radius)

    if inp.core_shape != "shell":
        core_dists = cdist(core_xyz, core_xyz)
        surface = np.sum(core_dists<=d_thres, axis=1) < inp.n_coord
    else:
        surface = np.ones(len(core_xyz), dtype='bool')
    logger.info("\tThe surface beads of the core allow a maximum of {} ligands...".format(np.sum(surface)))
    if np.sum(surface) < inp.n_tot_lig:
        inp.n_tot_lig = np.sum(surface)
        inp.lig1_num = int(inp.n_tot_lig * inp.lig1_frac)
        inp.lig2_num = inp.n_tot_lig - inp.lig1_num
        logger.warning("\tATTENTION. The grafting density is too high to meet requirements...")
        logger.warning("\t\tResetting total number of ligands to {}".format(inp.n_tot_lig))
        logger.warning("\t\tNew grafting density set to {:.3f}".format(inp.area/inp.n_tot_lig))
        logger.warning("\t\tNumber of ligands 1: {}".format(inp.lig1_num))
        logger.warning("\t\tNumber of ligands 2: {}".format(inp.lig2_num))

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

    staples_xyz = core_xyz[close_ndxs]

    logger.info("\tSaving normal directions to the surface at the anchoring sites...")
    normal_functions = {'sphere': sphere_normal,
    'ellipsoid': ellipsoid_normal,
    'cylinder': cylinder_normal,
    'rectangular prism': rectangular_prism_normal,
    'rod': rod_normal,
    'pyramid' : pyramid_normal,
    'octahedron' : octahedron_normal}
    normals = np.array([normal_functions[inp.core_shape](staple, inp) for staple in staples_xyz])

    if inp.n_tot_lig == 0:
        staples_xyz, normals = [], []
    return staples_xyz, normals, close_ndxs

def assign_morphology(staples_xyz, inp):
    """
    Assigns both ligands to the different staples according to the specified target morphology. It returns the indexes of the staples belonging to each ligand
    """
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

    elif 'stripe' in inp.morph:
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

    elif inp.morph == 'random':
        logger.info("\tDistributing ligands in a Random configuration...")
        np.random.seed(inp.rsd)
        np.random.shuffle(indexes)
        lig1_ndx = indexes[:inp.lig1_num]
        lig2_ndx = indexes[inp.lig1_num:]
    elif inp.morph == 'homogeneous':
        if inp.lig1_frac == 1.0:
            lig1_ndx = indexes
            lig2_ndx = []
        else:
            lig1_ndx = []
            lig2_ndx = indexes
    return (lig1_ndx, lig2_ndx)

def get_list_btypes(inp, lig1or2):
    """
    Makes a list with the beadtypes of any of the ligands. The first element of the list is the type assigned to the core
    """
    if lig1or2 == "1":
        lig_btypes = inp.lig1_btypes
        lig_n_per_bead = inp.lig1_n_per_bead
    elif lig1or2 == "2":
        lig_btypes = inp.lig2_btypes
        lig_n_per_bead = inp.lig2_n_per_bead

    list_btypes = [inp.core_btype]
    for i in range(len(lig_btypes)):
        list_btypes += [lig_btypes[i]]*lig_n_per_bead[i]
    return list_btypes

def grow_ligand(inp, params, lig1or2):
    """
    Generates the XYZ coordinates of a ligand along the X-axis. A core bead is used as the point (0,0,0)
    """
    n_iter = 20
    psis = np.linspace(0, 2*np.pi, n_iter)
    btypes = get_list_btypes(inp, lig1or2)
    n_at = len(btypes) #The ligand includes one bead from the core
    if params != None:
        bonds = []
        for a1, a2 in zip(btypes[:-1], btypes[1:]):
            key = "{}-{}".format(a1, a2)
            try:
                bond_eq = params.bondtypes[key][1]
            except:
                bond_eq = 2*inp.bead_radius
            bonds.append(bond_eq)
        angles = []
        for a1, a2, a3 in zip(btypes[:-2], btypes[1:-1], btypes[2:]):
            key = "{}-{}-{}".format(a1,a2,a3)
            try:
                angle = params.angletypes[key]
                angle = np.array(angle)
                angle_eq = angle[:,1][np.argmax(angle[:,2])] #takes the equilibrium value of highest constant
            except:
                angle_eq = 180.0
            angles.append(angle_eq)
    else:
        bonds = [2*inp.bead_radius]*(n_at-1)
        angles = [180.0]*(n_at-2)

    xyz = np.zeros((n_at, 3))
    xyz[1] = np.array([bonds[0],0,0])
    for i, old_b_length, new_b_length, a_length in zip(range(2,n_at), bonds[:-1], bonds[1:], angles):
        u_passive = xyz[i-1] - xyz[i-2]
        M = rot_mat([0,0,1], (180-a_length)*np.pi/180)
        v_passive = np.dot(M, u_passive)
        v_scaled = v_passive/np.linalg.norm(v_passive)*new_b_length
        best_x = v_scaled[0]
        best_psi = 0
        for psi in psis:
            M = rot_mat([1,0,0], psi)
            v_test = np.dot(M, v_scaled)
            if v_test[0] < best_x:
                best_x = v_test[0]
                best_psi = psi

        v_scaled = np.dot(rot_mat([1,0,0], best_psi), v_scaled)
        v_shifted = xyz[i-1] + v_scaled
        xyz[i] = v_shifted*1
    return xyz

def optimize_ligand_orientation(lig_shifted, other_ligands):
    """
    Explores rotations of a ligand along its PCA to find the orientation the maximizes the minimum distance with the ligands already placed
    """
    phis = np.linspace(0, 2*np.pi, 50)
    displace = lig_shifted[0]
    lig_shifted = lig_shifted - displace
    intern_pca = PCA(n_components=3)
    if len(lig_shifted) == 2:
        intern_pca_ax = lig_shifted[1] - lig_shifted[0]
        intern_pca_ax /= np.linalg.norm(intern_pca_ax)
    else:
        intern_pca.fit(lig_shifted)
        intern_pca_ax = intern_pca.components_[0]/np.linalg.norm(intern_pca.components_[0])

    if np.sum(np.mean(lig_shifted, axis=0)>=0)<2:
        intern_pca_ax=-1*intern_pca_ax
    if other_ligands == []:
        other_ligands = np.array([[0,0,0]])
    else:
        other_ligands = np.array(other_ligands) - displace
    min_dist = np.min(cdist(lig_shifted, other_ligands))
    for phi in phis:
        M = rot_mat(intern_pca_ax, phi)
        lig_test = np.dot(M,lig_shifted.T).T
        dist_test = np.min(cdist(lig_test, other_ligands))
        if dist_test > min_dist:
            min_dist = dist_test*1
            lig_shifted = lig_test*1
    lig_opt = lig_shifted + displace
    return lig_opt

def place_ligands(staples_xyz, staples_normals, lig_ndx, inp, params):
    """
    Places the ligands in their right position around the core. That is, it generates a reasonable structure for each ligand and roto-translates them
    """
    result = []
    pca = PCA(n_components=3)
    other_ligands = []
    print(lig_ndx)
    for n, ndxs in enumerate(lig_ndx, 1):
        lig_xyz = []
        if [inp.lig1_num, inp.lig2_num][n-1] != 0:
            lig_generic = grow_ligand(inp, params, str(n))
            if len(lig_generic) == 2:
                pca_ax = lig_generic[1] - lig_generic[0]
                pca_ax /= np.linalg.norm(pca_ax)
            else:
                pca.fit(lig_generic)
                pca_ax = pca.components_[0]/np.linalg.norm(pca.components_[0])
            print(pca_ax)
            #if np.sum(np.mean(lig_generic, axis=0)>=0)<2:
            if np.sum(pca_ax<0)>=2:
                pca_ax=-1*pca_ax
            print(pca_ax)
            if(np.isclose(np.abs(np.dot(pca_ax, [1,0,0])), [1], 0.01)):
                pca_ax = np.array([1,0,0])
            lig_generic = np.insert(lig_generic, 3, 1, axis=1).T
            for ndx in ndxs:
                xyz_normal_pts = np.array([staples_normals[ndx]*i for i in range(4)])#*-1  #This -1 is unclear when to put it
                xyz_generic_pts = np.array([pca_ax*i for i in range(4)])
                trans_matrix = affine_matrix_from_points(xyz_generic_pts.T, xyz_normal_pts.T, shear=False, scale=False, usesvd=True)
                lig_shifted = np.dot(trans_matrix, lig_generic).T[:,:3]
                lig_shifted += staples_xyz[ndx]
                lig_opt = optimize_ligand_orientation(lig_shifted, other_ligands)
                lig_xyz.append(lig_opt[1:]) #Discards the first bead which belongs to the core
                other_ligands += [[x,y,z] for x,y,z in zip(lig_opt[1:,0], lig_opt[1:,1], lig_opt[1:,2])]
            if lig_xyz != []:
                lig_xyz = np.concatenate(lig_xyz, axis=0)
        result.append(lig_xyz)

    return tuple(result)
