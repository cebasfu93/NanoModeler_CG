import numpy as np
from  scipy.spatial.distance import cdist

from DEPENDENCIES.Extras import center, cartesian_to_polar, polar_to_cartesian, sunflower_pts

def primitive_sphere(inp):
    const = inp.bead_radius*2
    cells_per_side = int((((2*inp.core_radius)//const)+1)//2*2+1)
    N_unit_cells = cells_per_side**3
    prim_block = np.array([])

    for i in range(cells_per_side):
        for j in range(cells_per_side):
            for k in range(cells_per_side):

                prim_block = np.append(prim_block, [i,j,k,i+1,j,k,i,j+1,k,i,j,k+1,i+1,j+1,k,i+1,j,k+1,i,j+1,k+1,i+1,j+1,k+1])

    prim_block = prim_block * const
    prim_block = prim_block.reshape((len(prim_block)//3,3))
    prim_block = np.unique(prim_block, axis=0)
    prim_block = center(prim_block)

    prim_sphere = prim_block[np.linalg.norm(prim_block, axis=1)<= inp.core_radius]

    return prim_sphere

def bcc_sphere(inp):
    const = inp.bead_radius*4/np.sqrt(3)
    cells_per_side = int((((2*inp.core_radius)//const)+1)//2*2+1)
    N_unit_cells = cells_per_side**3
    bcc_block = np.array([])

    for i in range(cells_per_side):
        for j in range(cells_per_side):
            for k in range(cells_per_side):

                bcc_block = np.append(bcc_block, [i,j,k,i+1,j,k,i,j+1,k,i,j,k+1,i+1,j+1,k,i+1,j,k+1,i,j+1,k+1,i+1,j+1,k+1])
                bcc_block = np.append(bcc_block, [i+0.5,j+0.5,k+0.5])

    bcc_block = bcc_block * const
    bcc_block = bcc_block.reshape((len(bcc_block)//3,3))
    bcc_block = np.unique(bcc_block, axis=0)
    bcc_block = center(bcc_block)

    bcc_sphere = bcc_block[np.linalg.norm(bcc_block, axis=1)<= inp.core_radius]

    return bcc_sphere

def fcc_sphere(inp):
    const = inp.bead_radius*np.sqrt(8)
    cells_per_side = int((((2*inp.core_radius)//const)+1)//2*2+1)
    N_unit_cells = cells_per_side**3
    fcc_block = np.array([])

    for i in range(cells_per_side):
        for j in range(cells_per_side):
            for k in range(cells_per_side):

                fcc_block = np.append(fcc_block, [i,j,k,i+1,j,k,i,j+1,k,i,j,k+1,i+1,j+1,k,i+1,j,k+1,i,j+1,k+1,i+1,j+1,k+1])
                fcc_block = np.append(fcc_block, [i,j+0.5,k+0.5,i+0.5,j,k+0.5,i+0.5,j+0.5,k,i+1,j+0.5,k+0.5,i+0.5,j+1,k+0.5,i+0.5,j+0.5,k+1])

    fcc_block = fcc_block * const
    fcc_block = fcc_block.reshape((len(fcc_block)//3,3))
    fcc_block = np.unique(fcc_block, axis=0)
    fcc_block = center(fcc_block)

    fcc_sphere=fcc_block[np.linalg.norm(fcc_block, axis=1)<= inp.core_radius]

    return fcc_sphere

def hcp_xyz(h,k,l):
    x = 2*h+(k+l)%2
    y = np.sqrt(3)*(k+l%2/3)
    z = 2*np.sqrt(6)/3*l
    return [x, y, z]

def hcp_sphere(inp):
    const = inp.bead_radius * 2
    cells_per_side = int(((2*inp.core_radius)//const)//2*2)+3
    hcp_block = np.array([])

    for i in range(cells_per_side):
        for j in range(cells_per_side):
            for k in range(cells_per_side):
                hcp_block = np.append(hcp_block, [2*i+(j+k)%2, np.sqrt(3)*(j+k%2/3), 2*np.sqrt(6)/3*k])

    #The following loops fix the edges of the hcp cube
    i = cells_per_side
    for j in range(cells_per_side//2+1):
        for k in range(cells_per_side//2+1):
            hcp_block = np.append(hcp_block, [i*2, j*2*np.sqrt(3),k*2*2*np.sqrt(6)/3])
    for j in range(cells_per_side//2):
        for k in range(cells_per_side//2):
            hcp_block = np.append(hcp_block, [i*2, j*2*np.sqrt(3)+4*np.sqrt(3)/3, (2*k+1)*2*np.sqrt(6)/3])
    j = cells_per_side
    for i in range(cells_per_side):
        for k in range(cells_per_side//2+1):
            hcp_block = np.append(hcp_block, [i*2, j*np.sqrt(3),k*2*2*np.sqrt(6)/3])
    k = cells_per_side
    for i in range(cells_per_side):
        for j in range(cells_per_side//2):
            hcp_block = np.append(hcp_block, [i*2, j*2*np.sqrt(3),k*2*np.sqrt(6)/3])
            hcp_block = np.append(hcp_block, [2*i+1, (2*j+1)*np.sqrt(3),k*2*np.sqrt(6)/3])

    hcp_block = hcp_block*brad_opt
    hcp_block = hcp_block.reshape((len(hcp_block)//3,3))
    hcp_block = np.unique(hcp_block, axis=0)
    hcp_block = center(hcp_block)
    ndx_near = np.argmin(np.linalg.norm(hcp_block, axis=1))
    hcp_block = hcp_block - hcp_block[ndx_near,:]

    hcp_sphere = hcp_block[np.linalg.norm(hcp_block, axis=1) <= inp.core_radius]

    return hcp_sphere

def gkeka_method(a):
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
            rft.append([radius_opt, f, t])
            N_count += 1

    rft = np.array(rft)
    gkeka_sphere = polar_to_cartesian(rft)
    return N_count, gkeka_sphere

def hollow_sphere(inp):
    ens, diffs = [], []
    a_ini = inp.bead_radius**2
    if inp.core_radius >= 0.8 and inp.core_radius <= 1.5:
        trial_areas = np.linspace(a_ini, 10*a_ini, 300)
    elif np_rad > 1.5:
        trial_areas = np.linspace(a_ini/(inp.core_radius**2), a_ini*(inp.core_radius**2), 300)
    else:
        print("Unsupported combination of buil-mode and nanoparticle radius")

    diff = 1

    for area in trial_areas:
        en, probe_sphere = gkeka_method(area)
        dists = cdist(probe_sphere, probe_sphere)
        new_diff = np.abs(np.mean(np.sort(dists, axis=1)[:,1])-2*inp.bead_radius)
        ens.append(en)
        diffs.append(new_diff)
        if new_diff < diff:
            #print(en)
            diff = new_diff
            hol_sphere = probe_sphere

    diffs = np.array(diffs)
    ens = np.array(ens)

    return hol_sphere

def print_xyz(coords):
    coords = coords * 10
    output = open(outname_opt, "w")
    output.write(str(len(coords)) + "\n\n")
    N_M = len(coords)-lignum_opt
    for i in range(N_M):
        output.write('MM' + '{:.3f}'.format(coords[i,0]).rjust(10) + "{:.3f}".format(coords[i,1]).rjust(10) + "{:.3f}".format(coords[i,2]).rjust(10) + "\n")
    for i in range(N_M, len(coords)):
        output.write('ST' + '{:.3f}'.format(coords[i,0]).rjust(10) + "{:.3f}".format(coords[i,1]).rjust(10) + "{:.3f}".format(coords[i,2]).rjust(10) + "\n")
    output.close()
