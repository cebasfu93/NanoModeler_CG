import numpy as np

class Input:
    def __init__(self,
    core_radius, core_method, bead_radius, core_bmass, core_btype, core_en_k,
    lig1_n_per_bead, lig1_num, lig1_btypes, lig1_charges, lig1_masses,
    lig2_n_per_bead, lig2_num, lig2_btypes, lig2_charges, lig2_masses,
    morph, rsd, stripes, parameter_file):

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
        self.parameter_file = parameter_file

class Bond:
    def __init__(self, atype1, atype2, func, b0, kb):
        self.atype1 = atype1
        self.atype2 = atype2
        self.func = func
        self.b0 = b0
        self.kb = kb

class Angle:
    def __init__(self, atype1, atype2, atype3, func, th0, cth):
        self.atype1 = atype1
        self.atype2 = atype2
        self.atype3 = atype3
        self.func = func
        self.th0 = th0
        self.cth = cth

class Parameters:
    def __init__(self, parameter_file):
        fl = parameter_file.readlines()

        bondtypes_section = False
        bond_info = []
        for line in fl:
            if bondtypes_section == True:
                if ("[ " in line) and (" ]" in line):
                    break
                if ";"!=line[0] and line!="\n":
                    bond_info.append(line.split())
            if "[ bondtypes ]" in line:
                bondtypes_section = True
        bonds = {"{}-{}".format(bond[0], bond[1]) : [int(bond[2]), float(bond[3]), float(bond[4])] for bond in bond_info}
        bonds2 = {"{}-{}".format(bond[1], bond[0]) : [int(bond[2]), float(bond[3]), float(bond[4])] for bond in bond_info}
        self.bondtypes = {**bonds, **bonds2}

        angletypes_section = False
        angle_info = []
        for line in fl:
            if angletypes_section == True:
                if ("[ " in line) and (" ]" in line):
                    break
                if ";"!=line[0] and line!="\n":
                    angle_info.append(line.split())
            if "[ angletypes ]" in line:
                angletypes_section = True
        angles = {"{}-{}-{}".format(angle[0], angle[1], angle[2]) : [int(angle[3]), float(angle[4]), float(angle[5])] for angle in angle_info}
        angles2 = {"{}-{}-{}".format(angle[2], angle[1], angle[0]) : [int(angle[3]), float(angle[4]), float(angle[5])] for angle in angle_info}
        self.angletypes = {**angles, **angles2}

    def check_missing_parameters(self, inp):
        lig1_btypes_list = [inp.core_btype]
        for i in range(len(inp.lig1_btypes)):
            lig1_btypes_list += [inp.lig1_btypes[i]]*inp.lig1_n_per_bead[i]
        bond_checks = ["{}-{}".format(a1, a2) in self.bondtypes.keys() for a1, a2 in zip(lig1_btypes_list[:-1], lig1_btypes_list[1:])]
        if np.any(np.invert(bond_checks)):
            no_params_ndx = np.where(np.invert(bond_checks))[0]
            missing_pairs = ["{}-{}".format(lig1_btypes_list[ndx], lig1_btypes_list[ndx+1]) for ndx in no_params_ndx]
            raise Exception("Missing parameters for bonds: {}".format(missing_pairs))
        angle_checks = ["{}-{}-{}".format(a1, a2, a3) in self.angletypes.keys() for a1, a2, a3 in zip(lig1_btypes_list[:-2], lig1_btypes_list[1:-1], lig1_btypes_list[2:])]
        if np.any(np.invert(angle_checks)):
            no_params_ndx = np.where(np.invert(angle_checks))[0]
            missing_pairs = ["{}-{}-{}".format(lig1_btypes_list[ndx], lig1_btypes_list[ndx+1], lig1_btypes_list[ndx+2]) for ndx in no_params_ndx]
            raise Exception("Missing parameters for angles: {}".format(missing_pairs))

        if inp.lig2_num > 0:
            lig2_btypes_list = [inp.core_btype]
            for i in range(len(inp.lig2_btypes)):
                lig2_btypes_list += [inp.lig2_btypes[i]]*inp.lig2_n_per_bead[i]
            bond_checks = ["{}-{}".format(a1, a2) in self.bondtypes.keys() for a1, a2 in zip(lig2_btypes_list[:-1], lig2_btypes_list[1:])]
            if np.any(np.invert(bond_checks)):
                no_params_ndx = np.where(np.invert(bond_checks))[0]
                missing_pairs = ["{}-{}".format(lig2_btypes_list[ndx], lig2_btypes_list[ndx+1]) for ndx in no_params_ndx]
                raise Exception("Missing parameters for bonds: {}".format(missing_pairs))
            angle_checks = ["{}-{}-{}".format(a1, a2, a3) in self.angletypes.keys() for a1, a2, a3 in zip(lig2_btypes_list[:-2], lig2_btypes_list[1:-1], lig2_btypes_list[2:])]
            if np.any(np.invert(angle_checks)):
                no_params_ndx = np.where(np.invert(angle_checks))[0]
                missing_pairs = ["{}-{}-{}".format(lig2_btypes_list[ndx], lig2_btypes_list[ndx+1], lig2_btypes_list[ndx+2]) for ndx in no_params_ndx]
                raise Exception("Missing parameters for angles: {}".format(missing_pairs))

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
