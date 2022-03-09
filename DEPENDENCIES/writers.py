import numpy as np
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

def gro_writer(TMP, np_xyz, inp):
    """
    Writes a .gro file with the structure of the final nanoparticle generated
    """
    f = open(TMP+"/NP.gro", "w")
    f.write("NP.gro generated by NanoModeler_CG, t=0.0\n")
    f.write("{:d}\n".format(len(np_xyz)))

    n_at1, n_at2 = np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)
    n_core = int(len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2)
    logger.info("\tWriting the core...")
    for i in range(n_core):
        res = 1
        ndx = i
        xyz = np_xyz[ndx]
        f.write("{:>5d}{:<5}{:>5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n".format(res, "CORE", "MM", ndx+1, xyz[0], xyz[1], xyz[2]))

    logger.info("\tWriting the ligands...")
    for i in range(inp.lig1_num):
        res += 1
        for j in range(n_at1):
            ndx = n_core + i*n_at1 + j
            xyz = np_xyz[ndx]
            f.write("{:>5d}{:<5}{:>5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n".format(res, "LIG1", "A{}".format(j), ndx+1, xyz[0], xyz[1], xyz[2]))
    for i in range(inp.lig2_num):
        res += 1
        for j in range(n_at2):
            ndx = n_core + inp.lig1_num*n_at1 + i*n_at2 + j
            xyz = np_xyz[ndx]
            f.write("{:>5d}{:<5}{:>5}{:>5d}{:>8.3f}{:>8.3f}{:>8.3f}\n".format(res, "LIG2", "B{}".format(j), ndx+1, xyz[0], xyz[1], xyz[2]))

    f.write("{:>10.5f} {:>10.5f} {:>10.5f}".format(10,10,10))
    f.close()

def top_writer(TMP, np_xyz, core_bonds, lig_bonds, lig_angles, lig_dihedrals, inp, params):
    """
    Writes a .top file with all the parameters of the final generated structure
    """
    n_at1 = np.sum(inp.lig1_n_per_bead) #total number of bead in each copy of Ligand 1
    n_at2 = np.sum(inp.lig2_n_per_bead) #total number of bead in each copy of Ligand 2
    n_core = int(len(np_xyz) - inp.lig1_num*n_at1 - inp.lig2_num*n_at2) #number of beads in the core
    btypes = [inp.core_btype]*n_core
    lig1_btypes_list, lig1_charges_list, lig1_masses_list, lig2_btypes_list, lig2_charges_list, lig2_masses_list = [], [], [], [], [], []
    #Makes lists with all the types, charges, and masses of Ligand 1 and Ligand 2. This make nice iterables to loop later for writing
    for i in range(len(inp.lig1_btypes)):
        lig1_btypes_list += [inp.lig1_btypes[i]]*inp.lig1_n_per_bead[i]
        lig1_charges_list += [inp.lig1_charges[i]]*inp.lig1_n_per_bead[i]
        lig1_masses_list += [inp.lig1_masses[i]]*inp.lig1_n_per_bead[i]
    for i in range(len(inp.lig2_btypes)):
        lig2_btypes_list += [inp.lig2_btypes[i]]*inp.lig2_n_per_bead[i]
        lig2_charges_list += [inp.lig2_charges[i]]*inp.lig2_n_per_bead[i]
        lig2_masses_list += [inp.lig2_masses[i]]*inp.lig2_n_per_bead[i]
    btypes = btypes + lig1_btypes_list*inp.lig1_num + lig2_btypes_list*inp.lig2_num

    f = open(TMP+"/NP.top", "w")
    f.write(";NP.top generated by NanoModeler_CG\n")
    f.write("#include \"inputs/Martini-v2.2P.itp\"\n#include \"inputs/refPolIon.itp\"\n#include \"inputs/solvents.itp\"\n")
    logger.info("\tWriting [ moleculetype ]...")
    f.write("\n[ moleculetype ]\n")
    f.write("NP \t\t 1\n")

    logger.info("\tWriting [ atoms ]...")
    f.write("\n[ atoms ]\n")
    f.write(";   nr  type  resi  res  atom  cgnr       charge      mass      ; qtot\n")
    for i in range(n_core):
        res = 1
        at = i + 1
        f.write("{:>6d} {:>4} {:>5} {:>5} {:>5} {:>5} {:>12.6f}    {:>9.5f} ; qtot {:>6.3f}\n".format(at, inp.core_btype, res, "CORE", "MM", at, 0.0, inp.core_bmass, 0.0))
    q_tot = 0
    for i in range(inp.lig1_num):
        res += 1
        jumps = np.linspace(0, n_at1-1, n_at1, dtype='int')
        for j, btype, charge, mass in zip(jumps, lig1_btypes_list, lig1_charges_list, lig1_masses_list):
            q_tot += charge
            at = n_core + i*n_at1 + j + 1
            f.write("{:>6d} {:>4} {:>5} {:>5} {:>5} {:>5} {:>12.6f}    {:>9.5f} ; qtot {:>6.3f}\n".format(at, btype, res, "LIG1", "A{}".format(j), at, charge, mass, q_tot))
    for i in range(inp.lig2_num):
        res += 1
        jumps = np.linspace(0, n_at2-1, n_at2, dtype='int')
        for j, btype, charge, mass in zip(jumps, lig2_btypes_list, lig2_charges_list, lig2_masses_list):
            q_tot += charge
            at = n_core + inp.lig1_num*n_at1 + i*n_at2 + j + 1
            f.write("{:>6d} {:>4} {:>5} {:>5} {:>5} {:>5} {:>12.6f}    {:>9.5f} ; qtot {:>6.3f}\n".format(at, btype, res, "LIG2", "B{}".format(j), at, charge, mass, q_tot))

    logger.info("Total charge of the system: {:.3f} e...".format(q_tot))
    if q_tot != 0:
        if q_tot%int(q_tot) != 0.0:
            logger.warning("Beware, the total charge of the system is non-integer!")
    logger.info("\tWriting [ bonds ]...")
    f.write("\n[ bonds ]\n")
    f.write(";  ai    aj funct           c0           c1\n")
    for bond in core_bonds:
        at1 = bond[0] + 1
        at2 = bond[1] + 1
        b_key = "{}-{}".format(btypes[bond[0]], btypes[bond[1]])
        f.write("{:>5d} {:>5d} {:>6d} {:>13.5f} {:>13.1f} ; core EN\n".format(at1, at2, 1, 2*inp.bead_radius, inp.core_en_k))

    for l, lig_bond in enumerate(lig_bonds,1):
        n_warns = 0
        for bond in lig_bond:
            at1 = bond[0] + 1
            at2 = bond[1] + 1
            b_key = "{}-{}".format(btypes[bond[0]], btypes[bond[1]])
            try:
                b_top = params.bondtypes[b_key]
                f.write("{:>5d} {:>5d} {:>6d} {:>13.5f} {:>13.1f}\n".format(at1, at2, b_top[0], b_top[1], b_top[2]))
            except:
                n_warns += 1
        logger.warning("\t\tThe nanoparticle is missing {} bond parameters in ligands {}. See above. Params not found...".format(n_warns, l))

    logger.info("\tWriting [ angles ]...")
    f.write("\n[ angles ]\n")
    f.write(";  ai    aj    ak funct        theta          cth\n")
    for l, lig_angle in enumerate(lig_angles,1):
        n_warns = 0
        for angle in lig_angle:
            at1 = angle[0] + 1
            at2 = angle[1] + 1
            at3 = angle[2] + 1
            a_key = "{}-{}-{}".format(btypes[angle[0]], btypes[angle[1]], btypes[angle[2]])
            try:
                a_tops = params.angletypes[a_key]
                for a_top in a_tops:
                    f.write("{:>5d} {:>5d} {:>5d} {:>6d} {:>9.4e}  {:>9.4e}\n".format(at1, at2, at3, a_top[0], a_top[1], a_top[2]))
            except:
                n_warns += 1
        logger.warning("\t\tThe nanoparticle is missing {} angle parameters in ligands {}. See above. Params not found...".format(n_warns, l))

    logger.info("\tWriting [ dihedrals ]...")
    f.write("\n[ dihedrals ]\n")
    f.write(";  ai    aj    ak    al  funct        phi0        kphi          m\n")
    for l, lig_dihedral in enumerate(lig_dihedrals, 1):
        n_warns = 0
        for dihedral in lig_dihedral:
            at1 = dihedral[0] + 1
            at2 = dihedral[1] + 1
            at3 = dihedral[2] + 1
            at4 = dihedral[3] + 1
            d_key = "{}-{}-{}-{}".format(btypes[dihedral[0]], btypes[dihedral[1]], btypes[dihedral[2]], btypes[dihedral[3]])
            try:
                d_tops = params.dihedraltypes[d_key]
                for d_top in d_tops:
                    f.write("{:>5d} {:>5d} {:>5d} {:>5d} {:>6d}  {:>9.4e}  {:>9.4e}  {:>9d}\n".format(at1, at2, at3, at4, d_top[0], d_top[1], d_top[2], d_top[3]))
            except:
                n_warns += 1
        logger.warning("\t\tThe nanoparticle is missing {} dihedral parameters in ligands {}. See above. Params not found...".format(n_warns, l))

    logger.info("\tWriting [ system ]...")
    f.write("\n[ system ]\n")
    f.write("Nanoparticle prepared with NanoModeler_CG\n")

    logger.info("\tWriting [ molecules ]...")
    f.write("\n[ molecules ]\n")
    f.write("; Compound        nmols\n")
    f.write("NP \t\t 1\n")
