import numpy as np
from  scipy.spatial.distance import cdist
from DEPENDENCIES.Extras import polar_to_cartesian
from DEPENDENCIES.ThomsonMC import ThomsonMC
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

def sphere(block, inp):
    """
    Cuts the lattice into the shape of a sphere
    """
    logger.info("\tChopping the lattice as a sphere...")
    core = block[np.linalg.norm(block, axis=1)<= inp.core_radius]
    core = core - np.mean(core, axis=0)
    return core

def ellipsoid(block, inp):
    """
    Cuts the lattice into the shape of an ellipsoid
    """
    logger.info("\tChopping the lattice as an ellipsoid...")
    condition = (block[:,0]**2/inp.core_ellipse_axis[0]**2 + block[:,1]**2/inp.core_ellipse_axis[1]**2 + block[:,2]**2/inp.core_ellipse_axis[2]**2) <= 1.0
    core = block[condition]
    core = core - np.mean(core, axis=0)
    return core

def rectangular_prism(block, inp):
    """
    Cuts the lattice into the shape of a rectangular prism
    """
    logger.info("\tChopping the lattice as a rectangular prism...")
    condition_x = np.logical_and(block[:,0] >= -1*inp.core_rect_prism[0]/2, block[:,0] <= inp.core_rect_prism[0]/2)
    condition_y = np.logical_and(block[:,1] >= -1*inp.core_rect_prism[1]/2, block[:,1] <= inp.core_rect_prism[1]/2)
    condition_z = np.logical_and(block[:,2] >= -1*inp.core_rect_prism[2]/2, block[:,2] <= inp.core_rect_prism[2]/2)
    core = block[np.logical_and(condition_x, np.logical_and(condition_y, condition_z))]
    core = core - np.mean(core, axis=0)
    return core

def cylinder(block, inp):
    """
    Cuts the lattice into the shape of a cylinder
    """
    logger.info("\tChopping the lattice as a cylinder...")
    condition_z = np.logical_and(block[:,2] <= inp.core_cylinder[1]/2, block[:,2] >= -inp.core_cylinder[1]/2)
    condition_circle = np.linalg.norm(block[:,0:2], axis=1) <= inp.core_cylinder[0]

    core = block[np.logical_and(condition_z, condition_circle)]
    core = core - np.mean(core, axis=0)
    return core

def rod(block, inp):
    """
    Cuts the lattice into the shape of a rod
    """
    logger.info("\tChopping the lattice as a rod...")
    condition_circle = np.linalg.norm(block[:,0:2], axis=1) <= inp.core_rod_params[0]
    condition_length = np.logical_and(block[:,2]<=inp.core_rod_params[1]/2, block[:,2]>= -1*inp.core_rod_params[1]/2)
    condition_cylinder = np.logical_and(condition_circle, condition_length)
    shift_z = block*1
    shift_z[:,2] = shift_z[:,2] - inp.core_rod_params[1]/2
    condition_cap1 = np.linalg.norm(shift_z, axis=1) <= inp.core_rod_params[0]
    shift_z = block*1
    shift_z[:,2] = shift_z[:,2] + inp.core_rod_params[1]/2
    condition_cap2 = np.linalg.norm(shift_z, axis=1) <= inp.core_rod_params[0]

    core = block[np.logical_or(condition_cylinder, np.logical_or(condition_cap1, condition_cap2))]
    core = core - np.mean(core, axis=0)
    return core

def pyramid(block, inp):
    """
    Cuts the lattice into the shape of a square pyramid
    """
    logger.info("\tChopping the lattice as a square pyramid...")
    condition_base = block[:,0] >= -1*inp.core_pyramid[1]/2
    a = inp.core_pyramid[0]*1
    L = inp.core_pyramid[1]*1
    tip = np.array([L/2,0,0])
    base_pts = np.array([[-L/2,a,0],[-L/2,0,a],[-L/2,-a,0],[-L/2,0,-a]])
    n_pts = len(base_pts)
    coefs = []
    for i in range(n_pts):
        vec = np.cross(tip-base_pts[i], tip-base_pts[(i+1)%n_pts])
        vec = np.append(vec, np.dot(vec, base_pts[i]))
        coefs.append(vec)

    conditions = [condition_base]
    for coef in coefs:
        cond = np.dot(coef[:3], block.T) <= coef[3]
        conditions.append(cond)


    conditions = np.array(conditions)
    condition = np.all(conditions, axis=0)
    core = block[condition]
    dists = cdist(core, core)
    coord_numbers = np.sum(dists<=(2*inp.bead_radius+0.01), axis=1)
    ndx_alone = np.where(coord_numbers == 1)[0]
    if len(ndx_alone) != 0:
        core = np.delete(core, ndx_alone, axis=0)

    core[:,0] = core[:,0] - (np.max(core[:,0]) + np.min(core[:,0]))/2
    return core

def octahedron(block, inp):
    """
    Cuts the lattice into the shape of an octahedron
    """
    logger.info("\tChopping the lattice as an octahedron...")
    a = inp.core_octahedron*1
    tips = np.array([[0,0, a/np.sqrt(2)], [0,0, -a/np.sqrt(2)]])
    base_pts = np.array([[a/2, a/2, 0], [-a/2, a/2, 0], [-a/2, -a/2, 0], [a/2, -a/2, 0]])
    n_pts = len(base_pts)
    coefs = []
    for t, tip in enumerate(tips):
        for i in range(n_pts):
            vec = (-1)**(t)*np.cross(base_pts[i]-tip, base_pts[(i+1)%n_pts]-tip)
            vec = np.append(vec, np.dot(vec, base_pts[i]))
            coefs.append(vec)

    conditions = []
    for coef in coefs:
        cond = np.dot(coef[:3], block.T) <= coef[3] #Check this operator for lower part of the octahedron
        conditions.append(cond)

    conditions = np.array(conditions)
    condition = np.all(conditions, axis=0)

    core = block[condition]
    core = core - np.mean(core, axis=0)
    return core

def gkeka_method(a, inp):
    """
    Generates a hollow sphere of beads assuming the protocol implemented by Gkeka
    """
    rft = []
    N_count = 0
    d = np.sqrt(a)
    M_t = int(np.round(np.pi/d))
    d_t = np.pi/M_t
    d_f = a/d_t
    for m in range(M_t):
        t = np.pi*(m+0.5)/M_t
        M_f = int(np.round(2*np.pi*np.sin(t)/d_f))
        for n in range(M_f):
            f = 2*np.pi*n/M_f
            rft.append([inp.core_radius, f, t])
            N_count += 1

    rft = np.array(rft)
    gkeka_sphere = polar_to_cartesian(rft)
    return N_count, gkeka_sphere

def shell(block, inp):
    """
    Generates hollow sphere of beads.
    If there are few beads to place, a MC approach is employed to minimize the electric energy of the system.
    Else, concentric rings are built to approximate the theoretical area-per-bead
    """
    logger.info("\tConstructing hollow shell...")
    n_beads_trial = int((4*(inp.core_radius**2))/inp.bead_radius**2)
    if n_beads_trial <= 300:
        logger.info("\tCore beads will be placed minimizing electric energy following a Monte Carlo approach. This will place the beads as far away as possible from one another...")
        core = ThomsonMC(n=n_beads_trial, mcs=1000, sigma=0.01)*inp.core_radius
    else:
        logger.info("\tCore beads will be placed in concentric rings...")
        ens, diffs = [], []
        a_ini = inp.bead_radius**2
        if inp.core_radius >= 0.8 and inp.core_radius <= 1.5:
            trial_areas = np.linspace(a_ini, 10*a_ini, 300)
        elif inp.core_radius > 1.5:
            trial_areas = np.linspace(a_ini/(inp.core_radius**2), a_ini*(inp.core_radius**2), 300)
        else:
            err_txt = "Unsupported combination of build-mode and nanoparticle radius"
            raise Exception(err_txt)

        diff = 1

        for area in trial_areas:
            en, probe_sphere = gkeka_method(area, inp)
            dists = cdist(probe_sphere, probe_sphere)
            new_diff = np.abs(np.mean(np.sort(dists, axis=1)[:,1])-2*inp.bead_radius)
            ens.append(en)
            diffs.append(new_diff)
            if new_diff < diff:
                #print(en)
                diff = new_diff
                core = probe_sphere

        diffs = np.array(diffs)
        ens = np.array(ens)

    return core
