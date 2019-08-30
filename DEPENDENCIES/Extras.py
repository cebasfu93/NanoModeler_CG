import numpy as np

class Input:
    def __init__(self,
    core_radius, core_method, bead_radius, core_bmass, core_btype, core_en_k,
    lig1_num, lig1_btypes, lig1_charges, lig1_masses, lig1_bond_k, lig1_angle_k,
    lig2_num, lig2_btypes, lig2_charges, lig2_masses, lig2_bond_k, lig2_angle_k,
    morph, rsd, stripes):

        self.bead_radius = bead_radius

        self.core_radius = core_radius
        self.core_method = core_method
        self.core_bmass = core_bmass
        self.core_btype = core_btype
        self.core_en_k = core_en_k

        self.lig1_num = lig1_num
        self.lig1_btypes = lig1_btypes
        self.lig1_charges = lig1_charges
        self.lig1_masses = lig1_masses
        self.lig1_bond_k = lig1_bond_k
        self.lig1_angle_k = lig1_angle_k

        self.lig2_num = lig2_num
        self.lig2_btypes = lig2_btypes
        self.lig2_charges = lig2_charges
        self.lig2_masses = lig2_masses
        self.lig2_bond_k = lig2_bond_k
        self.lig2_angle_k = lig2_angle_k

        self.morph = morph
        self.rsd = rsd
        self.stripes = stripes

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
