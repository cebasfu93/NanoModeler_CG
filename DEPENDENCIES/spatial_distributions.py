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
