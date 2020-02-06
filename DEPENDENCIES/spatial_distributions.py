import numpy as np
from DEPENDENCIES.Extras import center
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

def primitive(inp):
    logger.info("\tConstructing lattice from primitive unit cell...")
    const = inp.bead_radius*2
    cells_per_side = int((((2*inp.char_radius)//const)+1)//2*2+1)
    N_unit_cells = cells_per_side**3
    xyz = np.array([])

    for i in range(cells_per_side):
        for j in range(cells_per_side):
            for k in range(cells_per_side):

                xyz = np.append(xyz, [i,j,k,i+1,j,k,i,j+1,k,i,j,k+1,i+1,j+1,k,i+1,j,k+1,i,j+1,k+1,i+1,j+1,k+1])

    xyz = xyz * const
    xyz = xyz.reshape((len(xyz)//3,3))
    xyz = np.unique(xyz, axis=0)
    xyz = center(xyz)
    return xyz

def bcc(inp):
    logger.info("\tConstructing lattice from BCC unit cell...")
    const = inp.bead_radius*4/np.sqrt(3)
    cells_per_side = int((((2*inp.char_radius)//const)+1)//2*2+1)
    N_unit_cells = cells_per_side**3
    xyz = np.array([])

    for i in range(cells_per_side):
        for j in range(cells_per_side):
            for k in range(cells_per_side):

                xyz = np.append(xyz, [i,j,k,i+1,j,k,i,j+1,k,i,j,k+1,i+1,j+1,k,i+1,j,k+1,i,j+1,k+1,i+1,j+1,k+1])
                xyz = np.append(xyz, [i+0.5,j+0.5,k+0.5])

    xyz = xyz * const
    xyz = xyz.reshape((len(xyz)//3,3))
    xyz = np.unique(xyz, axis=0)
    xyz = center(xyz)
    return xyz

def fcc(inp):
    logger.info("\tConstructing lattice from FCC unit cell...")
    const = inp.bead_radius*np.sqrt(8)
    cells_per_side = int((((2*inp.char_radius)//const)+1)//2*2+1)
    N_unit_cells = cells_per_side**3
    xyz = np.array([])
    Y=10
    for i in range(cells_per_side+Y):
        for j in range(cells_per_side+Y):
            for k in range(cells_per_side+Y):

                xyz = np.append(xyz, [i,j,k,i+1,j,k,i,j+1,k,i,j,k+1,i+1,j+1,k,i+1,j,k+1,i,j+1,k+1,i+1,j+1,k+1])
                xyz = np.append(xyz, [i,j+0.5,k+0.5,i+0.5,j,k+0.5,i+0.5,j+0.5,k,i+1,j+0.5,k+0.5,i+0.5,j+1,k+0.5,i+0.5,j+0.5,k+1])

    xyz = xyz * const
    xyz = xyz.reshape((len(xyz)//3,3))
    xyz = np.unique(xyz, axis=0)
    xyz = center(xyz)
    return xyz

def hcp(inp):
    logger.info("\tConstructing lattice from HCP unit cell...")
    const = inp.bead_radius * 2
    cells_per_side = int(((2*inp.char_radius)//const)//2*2)+3
    xyz = np.array([])

    for i in range(cells_per_side):
        for j in range(cells_per_side):
            for k in range(cells_per_side):
                xyz = np.append(xyz, [2*i+(j+k)%2, np.sqrt(3)*(j+k%2/3), 2*np.sqrt(6)/3*k])

    #The following loops fix the edges of the hcp cube
    i = cells_per_side
    for j in range(cells_per_side//2+1):
        for k in range(cells_per_side//2+1):
            xyz = np.append(xyz, [i*2, j*2*np.sqrt(3),k*2*2*np.sqrt(6)/3])
    for j in range(cells_per_side//2):
        for k in range(cells_per_side//2):
            xyz = np.append(xyz, [i*2, j*2*np.sqrt(3)+4*np.sqrt(3)/3, (2*k+1)*2*np.sqrt(6)/3])
    j = cells_per_side
    for i in range(cells_per_side):
        for k in range(cells_per_side//2+1):
            xyz = np.append(xyz, [i*2, j*np.sqrt(3),k*2*2*np.sqrt(6)/3])
    k = cells_per_side
    for i in range(cells_per_side):
        for j in range(cells_per_side//2):
            xyz = np.append(xyz, [i*2, j*2*np.sqrt(3),k*2*np.sqrt(6)/3])
            xyz = np.append(xyz, [2*i+1, (2*j+1)*np.sqrt(3),k*2*np.sqrt(6)/3])

    xyz = xyz*brad_opt
    xyz = xyz.reshape((len(xyz)//3,3))
    xyz = np.unique(xyz, axis=0)
    xyz = center(xyz)
    ndx_near = np.argmin(np.linalg.norm(xyz, axis=1))
    xyz = xyz - xyz[ndx_near,:]
    return xyz

def gkeka_method(a, inp):
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
    logger.info("\tConstructing hollow shell from concentric rings...")
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
