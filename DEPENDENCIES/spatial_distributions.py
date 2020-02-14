import numpy as np
from DEPENDENCIES.Extras import center, print_xyz
import itertools
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

def primitive(inp):
    """
    Replicates a primitive unit cell in all directions
    """
    logger.info("\tConstructing lattice from primitive unit cell...")
    const = inp.bead_radius*2
    cells_per_side = int((((2*inp.char_radius)//const)+1)//2*2+1)
    cells_range = list(range(cells_per_side))
    triplets = np.array(list(itertools.product(cells_range, repeat=3)))
    xyz = triplets*const
    xyz = center(xyz)
    #print_xyz(xyz, "core.xyz")
    return xyz

def bcc(inp):
    """
    Replicates a body-centered cubic unit cell in all directions
    """
    logger.info("\tConstructing lattice from BCC unit cell...")
    const = inp.bead_radius*4/np.sqrt(3)
    cells_per_side = int((((2*inp.char_radius)//const)+1)//2*2+1)
    cells_per_side = cells_per_side*2+1
    cells_range = list(range(cells_per_side))
    triplets = np.array(list(itertools.product(cells_range, repeat=3)))
    xyz = []
    for trip in triplets:
        if np.all(trip%2==0):
            xyz.append(trip)
        elif np.all(trip[:2]%2==1) and trip[2]%2==1:
            xyz.append(trip)

    xyz = np.array(xyz)*const/2
    xyz = center(xyz)
    #print_xyz(xyz, "bcc.xyz")
    return xyz

def fcc(inp):
    """
    Replicates a face-centered cubic unit cell in all directions
    """
    logger.info("\tConstructing lattice from FCC unit cell...")
    const = inp.bead_radius*np.sqrt(8)
    cells_per_side = int((((2*inp.char_radius)//const)+1)//2*2+1)
    cells_per_side = cells_per_side*2+1
    cells_range = list(range(cells_per_side))
    triplets = np.array(list(itertools.product(cells_range, repeat=3)))
    xyz = []
    for trip in triplets:
        if trip[2]%2==0:
            if np.all(trip[:2]%2==0) or np.all(trip[:2]%2!=0):
                xyz.append(trip)
        elif trip[0]%2 != trip[1]%2:
            xyz.append(trip)

    xyz = np.array(xyz)*const/2
    xyz = center(xyz)
    #print_xyz(xyz, "fcc.xyz")
    return xyz

def hcp(inp):
    """
    Replicates a hexagonal closely packed unit cell in all directions
    """
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
    Finds the best number of beads that result in the Gkeka sphere with an area per bead a close as possible as the theoretical value
    """
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
