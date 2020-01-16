import numpy as np

class Input:
    def __init__(self,
    core_radius, core_method, bead_radius, core_bmass, core_btype, core_en_k,
    lig1_n_per_bead, lig1_num, lig1_btypes, lig1_charges, lig1_masses,
    lig2_n_per_bead, lig2_num, lig2_btypes, lig2_charges, lig2_masses,
    morph, rsd, stripes):

        self.bead_radius = bead_radius

        self.core_radius = core_radius
        self.core_method = core_method
        self.core_bmass = core_bmass
        self.core_btype = core_btype
        self.core_en_k = core_en_k
        self.lig1_num = lig1_num
        self.lig2_num = lig2_num

        self.lig1_n_per_bead = lig1_n_per_bead
        self.lig1_btypes = lig1_btypes
        self.lig1_charges = lig1_charges
        self.lig1_masses = lig1_masses

        self.lig2_n_per_bead = lig2_n_per_bead
        self.lig2_btypes = lig2_btypes
        self.lig2_charges = lig2_charges
        self.lig2_masses = lig2_masses

        self.morph = morph
        self.rsd = rsd
        self.stripes = stripes
        self.lig1_bond_k=5000
        self.lig1_angle_k=25
        self.lig2_bond_k=5000
        self.lig2_angle_k=25

def center(objeto):
    COM = np.average(objeto, axis=0)
    for i in range(len(objeto)):
        objeto[i,:] = objeto[i,:] - COM
    return objeto

def sunflower_pts(num_pts):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    xyz = np.array([x,y,z]).T
    return xyz

def cartesian_to_polar(xyz):
    ptsnew = np.zeros(np.shape(xyz))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2) #r
    ptsnew[:,1] = np.arctan2(xyz[:,1], xyz[:,0]) #phi, [-pi, pi]
    ptsnew[:,2] = np.arctan2(np.sqrt(xy), xyz[:,2]) #theta, [0, pi] For elevation angle defined from Z-axis down.
    return ptsnew

def polar_to_cartesian(rft):
    ptsnew = np.zeros(np.shape(rft))
    ptsnew[:,0] = rft[:,0]*np.cos(rft[:,1])*np.sin(rft[:,2])
    ptsnew[:,1] = rft[:,0]*np.sin(rft[:,1])*np.sin(rft[:,2])
    ptsnew[:,2] = rft[:,0]*np.cos(rft[:,2])
    return ptsnew

def merge_coordinates(core_xyz, lig_xyz):
    if lig_xyz[1] != []:
        np_xyz = np.vstack((core_xyz, lig_xyz[0], lig_xyz[1]))
    else:
        np_xyz = np.vstack((core_xyz, lig_xyz[0]))
    return np_xyz
