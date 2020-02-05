import numpy as np

class Input:
    def __init__(self,
    bead_radius, core_radius, core_method, core_density, core_shape, core_cylinder, core_ellipse_axis, core_rect_prism, core_rod_params, core_pyramid, core_octahedron, core_btype, core_en, core_en_k,
    lig1_num, lig1_n_per_bead, lig1_btypes, lig1_charges, lig1_masses,
    lig2_num, lig2_n_per_bead, lig2_btypes, lig2_charges, lig2_masses,
    morph, rsd, stripes,
    parameter_file):

        self.bead_radius = bead_radius

        self.core_radius = core_radius
        self.core_method = core_method
        self.core_density = core_density
        self.core_shape = core_shape
        self.core_cylinder = core_cylinder
        self.core_ellipse_axis = core_ellipse_axis
        self.core_rect_prism = core_rect_prism
        self.core_rod_params = core_rod_params
        self.core_pyramid = core_pyramid
        self.core_octahedron = core_octahedron
        self.core_btype = core_btype
        self.core_en = core_en
        self.core_en_k = core_en_k

        self.lig1_num = lig1_num
        self.lig1_n_per_bead = lig1_n_per_bead
        self.lig1_btypes = lig1_btypes
        self.lig1_charges = lig1_charges
        self.lig1_masses = lig1_masses

        self.lig2_num = lig2_num
        self.lig2_n_per_bead = lig2_n_per_bead
        self.lig2_btypes = lig2_btypes
        self.lig2_charges = lig2_charges
        self.lig2_masses = lig2_masses

        self.morph = morph
        #if "stripe" in self.morph:
            #self.lig_num_tot = lig_num_tot
        #else:
            #self.lig_num_tot = self.lig1_num + self.lig2_num
        self.rsd = rsd
        self.stripes = stripes

        self.parameter_file = parameter_file

        if core_shape == "sphere" or core_shape == "shell":
            self.char_radius = self.core_radius
        elif core_shape == "ellipsoid":
            self.char_radius = np.max(self.core_ellipse_axis)
        elif core_shape == "cylinder":
            self.char_radius = np.max([self.core_cylinder[0], self.core_cylinder[1]/2])
        elif core_shape == "rectangular prism":
            self.char_radius = np.max(self.core_rect_prism)/2
        elif core_shape == "rod":
            self.char_radius = self.core_rod_params[1]/2 + self.core_rod_params[0]
        elif core_shape == "pyramid":
            self.char_radius = np.max([np.sqrt(2)*self.core_pyramid[0]/2, self.core_pyramid[1]/2])
        elif core_shape == "octahedron":
            self.char_radius = self.core_octahedron/2

        if core_method == "primitive":
            self.n_coord = 6
        elif core_method == "bcc":
            self.n_coord = 8
        elif core_method == "fcc" or core_method == "hcp":
            self.n_coord = 12

        self.vol = None

    def calculate_volume(self):
        if self.core_shape == "sphere" or self.core_shape == "shell":
            volume = 4./3*np.pi*self.core_radius**3
        elif self.core_shape == "ellipsoid":
            volume = 4./3*np.pi*np.prod(self.core_ellipse_axis)
        elif self.core_shape == "cylinder":
            volume = np.pi*self.core_cylinder[0]**2*self.core_cylinder[1]
        elif self.core_shape == "rectangular prism":
            volume = np.prod(self.rectangular_prism)
        elif self.core_shape == "rod":
            volume = np.pi*self.core_rod_params[0]**2*self.core_rod_params[1] + 4./3*np.pi*self.core_rod_params[1]**3
        elif self.core_shape == "pyramid":
            volume = self.core_pyramid[0]**2*self.core_pyramid[1]/3
        elif self.core_shape == "octahedron":
            volume = np.sqrt(2)/3*self.core_octahedron**3
        self.vol = volume

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

def hcp_xyz(h,k,l):
    x = 2*h+(k+l)%2
    y = np.sqrt(3)*(k+l%2/3)
    z = 2*np.sqrt(6)/3*l
    return [x, y, z]

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
