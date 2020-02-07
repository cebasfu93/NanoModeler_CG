import numpy as np
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

class Input:
    def __init__(self,
    bead_radius, core_radius, core_method, core_density, core_shape, core_cylinder, core_ellipse_axis, core_rect_prism, core_rod_params, core_pyramid, core_octahedron, core_btype, core_en, core_en_k,
    graft_density,
    lig1_n_per_bead, lig1_btypes, lig1_charges, lig1_masses, lig1_frac,
    lig2_n_per_bead, lig2_btypes, lig2_charges, lig2_masses,
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

        self.graft_density = graft_density

        self.lig1_n_per_bead = lig1_n_per_bead
        self.lig1_btypes = lig1_btypes
        self.lig1_charges = lig1_charges
        self.lig1_masses = lig1_masses
        self.lig1_frac = lig1_frac

        self.lig2_n_per_bead = lig2_n_per_bead
        self.lig2_btypes = lig2_btypes
        self.lig2_charges = lig2_charges
        self.lig2_masses = lig2_masses

        if self.lig1_frac == None or self.lig1_frac == 1.0:
            self.morph = "homogeneous"
        else:
            self.morph = morph
        self.rsd = rsd
        self.stripes = stripes

        self.parameter_file = parameter_file

        if self.core_shape == "sphere" or self.core_shape == "shell":
            self.char_radius = self.core_radius
        elif self.core_shape == "ellipsoid":
            self.char_radius = np.max(self.core_ellipse_axis)
        elif self.core_shape == "cylinder":
            self.char_radius = np.max([self.core_cylinder[0], self.core_cylinder[1]/2])
        elif self.core_shape == "rectangular prism":
            self.char_radius = np.max(self.core_rect_prism)/2
        elif self.core_shape == "rod":
            self.char_radius = self.core_rod_params[1]/2 + self.core_rod_params[0]
        elif self.core_shape == "pyramid":
            self.char_radius = np.max([np.sqrt(2)*self.core_pyramid[0]/2, self.core_pyramid[1]/2])
        elif self.core_shape == "octahedron":
            self.char_radius = self.core_octahedron/2

        if core_method == "primitive":
            self.n_coord = 6
        elif core_method == "bcc":
            self.n_coord = 8
        elif core_method == "fcc" or core_method == "hcp":
            self.n_coord = 12

        self.vol = None
        self.core_bmass = None
        self.area = None
        self.n_tot_lig = None
        self.lig1_num = None
        self.lig2_num = None

    def calculate_area(self):
        if self.core_shape == "sphere" or self.core_shape == "shell":
            area = 4.*np.pi*self.core_radius**2
        elif self.core_shape == "ellipsoid":
            axa, axb, axc = self.core_ellipse_axis
            area = 4*np.pi*(((axa*axb)**1.6 + (axa*axc)**1.6 + (axb*axc)**1.6)/3)**1.6
        elif self.core_shape == "cylinder":
            area = 2*np.pi*self.core_cylinder[0]*(self.core_cylinder[0]+ self.core_cylinder[1])
        elif self.core_shape == "rectangular prism":
            axa, axb, axc = self.rectangular_prism
            area = 2*(axa*axb + axa*axc + axb*axc)
        elif self.core_shape == "rod":
            area = 2*np.pi*self.core_rod_params[0]*(2*self.core_rod_params[0] + self.core_rod_params[1])
        elif self.core_shape == "pyramid":
            s = np.sqrt(self.core_pyramid[0]**2/4+self.core_pyramid[1]**2)
            area =  self.core_pyramid[0]**2 + 2*s*self.core_pyramid[1]
        elif self.core_shape == "octahedron":
            area = 2*np.sqrt(3)*self.core_octahedron**2
        return area

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
        return volume

    def characterize_core(self, core_xyz):
        logger.info("\tCharacterizing the core...")
        self.char_radius = np.max(np.linalg.norm(core_xyz, axis=1))
        logger.info("\tCalculating volume of the core...")
        self.vol = self.calculate_volume()
        logger.info("\t\tVolume of the core: {:.1f} nm3".format(self.vol))
        logger.info("\tEstimating the core's mass per bead...")
        self.core_bmass = self.core_density*self.vol*602.214/len(core_xyz) #g nm-3 to u.m.a bead-1
        logger.info("\t\tMass per core bead: {:.3f} u.m.a.".format(self.core_bmass))
        logger.info("\tCalculating surface area of the core...")
        self.area = self.calculate_area()
        logger.info("\t\tSuperficial area of the core: {:.2f} nm2".format(self.area))
        logger.info("\tCalculating total number of ligands...")
        self.n_tot_lig = int(self.area/self.graft_density)
        logger.info("\t\tTotal number of ligands: {}".format(self.n_tot_lig))
        logger.info("\tCalculating number of ligands 1...")
        self.lig1_num = int(self.n_tot_lig * self.lig1_frac)
        logger.info("\t\tNumber of ligands 1: {}".format(self.lig1_num))
        logger.info("\tCalculating number of ligands 2...")
        self.lig2_num = self.n_tot_lig - self.lig1_num
        logger.info("\t\tNumber of ligands 2: {}".format(self.lig2_num))

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
        bonds.update(bonds2)
        self.bondtypes = bonds

        angletypes_section = False
        angle_info = []
        angles = {}
        angles2 = {}
        for line in fl:
            if angletypes_section == True:
                if ("[ " in line) and (" ]" in line):
                    break
                if ";"!=line[0] and line!="\n":
                    angle_info.append(line.split())
            if "[ angletypes ]" in line:
                angletypes_section = True
        for angle in angle_info:
            a_key = "{}-{}-{}".format(angle[0], angle[1], angle[2])
            a_key_invert = "{}-{}-{}".format(angle[2], angle[1], angle[0])
            if a_key in angles.keys():
                angles[a_key] += [[int(angle[3]), float(angle[4]), float(angle[5])]]
                angles2[a_key_invert] += [[int(angle[3]), float(angle[4]), float(angle[5])]]
            else:
                angles[a_key] = [[int(angle[3]), float(angle[4]), float(angle[5])]]
                angles2[a_key_invert] = [[int(angle[3]), float(angle[4]), float(angle[5])]]
        angles.update(angles2)
        self.angletypes = angles

        dihedraltypes_section = False
        dihedral_info = []
        dihedrals = {}
        dihedrals2 = {}
        for line in fl:
            if dihedraltypes_section == True:
                if ("[ " in line) and (" ]" in line):
                    break
                if ";"!=line[0] and line!="\n":
                    dihedral_info.append(line.split())
            if "[ dihedraltypes ]" in line:
                dihedraltypes_section = True
        for dihedral in dihedral_info:
            d_key = "{}-{}-{}-{}".format(dihedral[0], dihedral[1], dihedral[2], dihedral[3])
            d_key_invert = "{}-{}-{}-{}".format(dihedral[3], dihedral[2], dihedral[1], dihedral[0])
            if d_key in dihedrals.keys():
                dihedrals[d_key] += [[int(dihedral[4]), float(dihedral[5]), float(dihedral[6]), int(dihedral[7])]]
                dihedrals2[d_key_invert] += [[int(dihedral[4]), float(dihedral[5]), float(dihedral[6]), int(dihedral[7])]]
            else:
                dihedrals[d_key] = [[int(dihedral[4]), float(dihedral[5]), float(dihedral[6]), int(dihedral[7])]]
                dihedrals2[d_key_invert] = [[int(dihedral[4]), float(dihedral[5]), float(dihedral[6]), int(dihedral[7])]]
        dihedrals.update(dihedrals2)
        self.dihedraltypes = dihedrals

    def check_bond_parameters(self, inp, lig1or2):
        lig_btypes = build_lig_btypes_list_n(inp, lig1or2)
        bond_checks = ["{}-{}".format(a1, a2) in self.bondtypes.keys() for a1, a2 in zip(lig_btypes[:-1], lig_btypes[1:])]
        if np.any(np.invert(bond_checks)):
            no_params_ndx = np.where(np.invert(bond_checks))[0]
            missing_pairs = ["{}-{}".format(lig_btypes[ndx], lig_btypes[ndx+1]) for ndx in no_params_ndx]
            warn_txt = "ATTENTION. Missing parameters for bonds: {}".format(np.unique(missing_pairs))
            logger.warning(warn_txt)

    def check_angle_parameters(self, inp, lig1or2):
        lig_btypes = build_lig_btypes_list_n(inp, lig1or2)
        angle_checks = ["{}-{}-{}".format(a1, a2, a3) in self.angletypes.keys() for a1, a2, a3 in zip(lig_btypes[:-2], lig_btypes[1:-1], lig_btypes[2:])]
        if np.any(np.invert(angle_checks)):
            no_params_ndx = np.where(np.invert(angle_checks))[0]
            missing_pairs = ["{}-{}-{}".format(lig_btypes[ndx], lig_btypes[ndx+1], lig_btypes[ndx+2]) for ndx in no_params_ndx]
            warn_txt = "ATTENTION. Missing parameters for angles: {}".format(np.unique(missing_pairs))
            logger.warning(warn_txt)

    def check_dihedral_parameters(self, inp, lig1or2):
        lig_btypes = build_lig_btypes_list_n(inp, lig1or2)
        dihedral_checks = ["{}-{}-{}-{}".format(a1, a2, a3, a4) in self.dihedraltypes.keys() for a1, a2, a3, a4 in zip(lig_btypes[:-3], lig_btypes[1:-2], lig_btypes[2:-1], lig_btypes[3:])]
        if np.any(np.invert(dihedral_checks)):
            no_params_ndx = np.where(np.invert(dihedral_checks))[0]
            missing_pairs = ["{}-{}-{}-{}".format(lig_btypes[ndx], lig_btypes[ndx+1], lig_btypes[ndx+2], lig_btypes[ndx+3]) for ndx in no_params_ndx]
            warn_txt = "ATTENTION. Missing parameters for dihedral: {}".format(np.unique(missing_pairs))
            logger.warning(warn_txt)

    def check_missing_parameters(self, inp):
        n_at1 = np.sum(inp.lig1_n_per_bead)
        n_at2 = np.sum(inp.lig2_n_per_bead)

        logger.info("\tLooking for bond parameters in ligand 1...")
        self.check_bond_parameters(inp, "1")
        if n_at1 > 2:
            logger.info("\tLooking for angle parameters in ligand 1...")
            self.check_angle_parameters(inp, "1")
        if n_at1 > 3:
            logger.info("\tLooking for dihedral parameters in ligand 1...")
            self.check_dihedral_parameters(inp, "1")

        if inp.lig2_num > 0:
            logger.info("\tLooking for bond parameters in ligand 2...")
            self.check_bond_parameters(inp, "2")
            if n_at2 > 2:
                logger.info("\tLooking for angle parameters in ligand 2...")
                self.check_angle_parameters(inp, "2")
            if n_at2 > 3:
                logger.info("\tLooking for dihedral parameters in ligand 2...")
                self.check_dihedral_parameters(inp, "2")

def build_lig_btypes_list_n(inp, lig1or2):
    if lig1or2 == "1":
        lig_btypes = inp.lig1_btypes*1
        lig_n_per_bead = inp.lig1_n_per_bead*1
    elif lig1or2 == "2":
        lig_btypes == inp.lig2_btypes*1
        lig_n_per_bead = inp.lig2_n_per_bead*1
    lig_btypes_list = [inp.core_btype]
    for i in range(len(lig_btypes)):
        lig_btypes_list += [lig_btypes[i]]*lig_n_per_bead[i]
    return lig_btypes_list

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
