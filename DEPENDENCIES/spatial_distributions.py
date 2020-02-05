import numpy as np
from DEPENDENCIES.Extras import center

def primitive(inp):
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
