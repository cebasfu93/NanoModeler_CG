import logging
import shutil
import zipfile

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

__VERSION__ = "1.0.13"

def getVersion():
    return __VERSION__

def NanoModeler_CG(BEAD_RADIUS=None,
    CORE_RADIUS=None,
    CORE_METHOD=None,
    CORE_DENSITY=None,
    CORE_SHAPE=None,
    CORE_CYLINDER=None,
    CORE_ELLIPSE_AXIS=None,
    CORE_RECT_PRISM=None,
    CORE_ROD_PARAMS=None,
    CORE_PYRAMID=None,
    CORE_OCTAHEDRON_EDGE=None,
    CORE_BTYPE=None,
    CORE_EN=None,
    CORE_EN_K=None,

    GRAFT_DENSITY=None,

    LIG1_N_PER_BEAD=None,
    LIG1_BTYPES=None,
    LIG1_CHARGES=None,
    LIG1_MASSES=None,
    LIG1_FRAC=None,

    LIG2_N_PER_BEAD=[],
    LIG2_BTYPES=[],
    LIG2_CHARGES=[],
    LIG2_MASSES=[],

    MORPHOLOGY=None,
    RSEED=None,
    STRIPES=None,
    PARAMETER_FILE=None):
    """
    ----------------------
    Parameters:
    ----------------------
    BEAD_RADIUS: float
        Radius (nm) of the beads conforming the core
    CORE_RADIUS: float
        Radius (nm) of the core when the shape is set to 'sphere' or 'shell'
    CORE_METHOD: str
        Lattice to use for the building of the core. primitive, bcc, fcc, hcp, shell
    CORE_DENSITY: float
        Density (g cm-3) of the bulk material conforming the core
    CORE_SHAPE: str
        Shape to cut the lattice. sphere, cylinder, ellipsoid, rectangular prism, rod, pyramid, octahedron
    CORE_CYLINDER: list of float
        Radius and length (nm) of the core when the shape is set to 'cylinder'
    CORE_ELLIPSE_AXIS: list of float
        The x, y, and z principal axis (nm) of the core when the shape is set to ellipsoid
    CORE_RECT_PRISM: list of float
        The x, y, and z edge-lengths (nm) of the core when the shape is set to rectangular prism
    CORE_ROD_PARAMS: list of float
        Cap radius and cylinder length (nm) of the core when the shape is set to rod
    CORE_PYRAMID: list of float
        Base edge-length and height (nm) of the core when the shape is set to pyramid
    CORE_OCTAHEDRON_EDGE: float
        Edge-length (nm) of the core when the shape is set to octahedron
    CORE_BTYPE: str
        Bead type to assign to the core beads
    CORE_EN: bool
        Whether or not to include an elastic network on the core's beads
    CORE_EN_K: float
        Elastic constant (kJ nm-2 mol-1) of the cores network
    GRAFT_DENSITY: float
        Area occupied per ligand (nm2)
    LIG1_N_PER_BEAD: list of int
        Number of times to repeat each bead type of ligand 1
    LIG1_BTYPES: list of str
        Bead types (in the right order) of the beads conforming ligand 1
    LIG1_CHARGES: list of float
        Partial charges (e) assigned to each bead type present in ligand 1
    LIG1_MASSES: list of float
        Mass (a.m.u.) assigned to each bead type present in ligand 1
    LIG1_FRAC: float
        Value between 0 and 1 representing the fraction of total ligands that should be assigned to ligand 1
    LIG2_N_PER_BEAD: list of int
        Number of times to repeat each bead type of ligand 2
    LIG2_BTYPES: list of str
        Bead types (in the right order) of the beads conforming ligand 2
    LIG2_CHARGES: list of float
        Partial charges (e) assigned to each bead type present in ligand 2
    LIG2_MASSES: list of float
        Mass (a.m.u.) assigned to each bead type present in ligand 2
    MORPHOLOGY: str
        Arrangement on which to place the ligands. homogeneous, random, janus_x, janus_y, janus_z, stripe_x, stripe_y, stripe_z
    RSEED: int
        Random seed used to generate a random morphology. -1 assigns a random seed
    STRIPES: int
        Number of stripes to assign when the morphology is set to stripe*
    PARAMETER_FILE: reader
        File containing the [ bondtypes ], [ angletypes ], and [ dihedraltypes ] directives for the bonded interactions present in the system
    """

    logger.info("WELCOME TO NANOMODELER CG")
    logger.info("Importing tempfile library...")
    import tempfile
    logger.info("Importing os library...")
    import os
    logger.info("Importing numpy library...")
    import numpy as np
    logger.info("Importing scipy.spatial.distance library...")
    from scipy.spatial.distance import cdist
    logger.info("Importing sklearn.decomposition library...")
    from sklearn import decomposition
    logger.info("Importing scipy.optimize library...")
    from scipy.optimize import minimize
    logger.info("Importing transformations library...")
    import DEPENDENCIES.transformations
    logger.info("Importing private classes...")
    from DEPENDENCIES.Extras import (Input, Parameters, cartesian_to_polar,
                                     center, merge_coordinates,
                                     polar_to_cartesian, sunflower_pts)
    logger.info("Importing lattice generators...")
    from DEPENDENCIES.spatial_distributions import bcc, fcc, hcp, primitive
    logger.info("Importing shape cutters...")
    from DEPENDENCIES.core_maker import (cylinder, ellipsoid, octahedron,
                                         pyramid, rectangular_prism, rod,
                                         shell, sphere)
    logger.info("Importing coating functions...")
    from DEPENDENCIES.coat_maker import (assign_morphology, place_ligands,
                                         place_staples)
    logger.info("Importing topology builder...")
    from DEPENDENCIES.top_maker import get_core_bonds, get_lig_bonded_atoms
    logger.info("Importing writers...")
    from DEPENDENCIES.writers import gro_writer, top_writer

    logger.info("Creating temporary folder...")
    TMP = tempfile.mkdtemp(dir="./")

    logger.setLevel(logging.INFO)
    logger.handlers = []
    loggerFileHandler = logging.FileHandler(os.path.join(TMP, "report.log"), "w")
    loggerFileHandler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    loggerFileHandler.setFormatter(formatter)
    logger.addHandler(loggerFileHandler)

    logger.info("Importing parsed variables...")
    inp = Input(
    bead_radius=BEAD_RADIUS,

    core_radius=CORE_RADIUS,
    core_method=CORE_METHOD,
    core_density=CORE_DENSITY,
    core_shape=CORE_SHAPE,
    core_cylinder=CORE_CYLINDER,
    core_ellipse_axis=CORE_ELLIPSE_AXIS,
    core_rect_prism=CORE_RECT_PRISM,
    core_rod_params=CORE_ROD_PARAMS,
    core_pyramid=CORE_PYRAMID,
    core_octahedron=CORE_OCTAHEDRON_EDGE,
    core_btype=CORE_BTYPE,
    core_en=CORE_EN,
    core_en_k=CORE_EN_K,

    graft_density=GRAFT_DENSITY,

    lig1_n_per_bead=LIG1_N_PER_BEAD,
    lig1_btypes=LIG1_BTYPES,
    lig1_charges=LIG1_CHARGES,
    lig1_masses=LIG1_MASSES,
    lig1_frac = LIG1_FRAC,

    lig2_n_per_bead=LIG2_N_PER_BEAD,
    lig2_btypes=LIG2_BTYPES,
    lig2_charges=LIG2_CHARGES,
    lig2_masses=LIG2_MASSES,

    morph=MORPHOLOGY,
    rsd=RSEED,
    stripes=STRIPES,

    parameter_file=PARAMETER_FILE)
    for key, value in inp.__dict__.items():
        logger.info("\t{:<20}  {:<60}".format(key, str(value)))
    inp.characterize_nominal_values()
    packing_efficiency = {
    'primitive': 0.52,
    'bcc': 0.68,
    'fcc': 0.74,
    'hcp': 0.74,
    }
    core_packing_functions = {
    'primitive': primitive,
    'bcc': bcc,
    'fcc': fcc,
    'hcp': hcp,
    }
    core_shape_functions = {'sphere': sphere,
    'ellipsoid': ellipsoid,
    'cylinder': cylinder,
    'rectangular prism': rectangular_prism,
    'rod': rod,
    'pyramid' : pyramid,
    'octahedron' : octahedron,
    'shell': shell
    }
    # CHECK THAT THE NANOPARTICLE IS WITHIN A REASONABLE SIZE
    logger.info("\tValidating size of the final nanoparticle...")
    if inp.core_shape == 'shell':
        n_estimate_block_beads = 4 * inp.core_radius ** 2 / (inp.bead_radius ** 2)
        logger.info("\t\tEstimate number of beads in the shell: {}".format(n_estimate_block_beads))
    else:
        n_estimate_block_beads = ((packing_efficiency[inp.core_method] * 2 * inp.char_radius) ** 3) / (4 * np.pi * inp.bead_radius ** 3 / 3)
        logger.info("\t\tEstimate number of beads in the sculpting block: {}".format(n_estimate_block_beads))
    
    n_max_core_beads = 8E5
    if n_estimate_block_beads > n_max_core_beads:
        raise ValueError(f"The core requested would require a sculpting block of ca. {n_estimate_block_beads:.0f}, "
                         f"but the maximum allowed is {n_max_core_beads:.0f}. Consider reducing the core size or increasing the bead radius.")
    n_estimate_ligand_1_beads = np.sum(inp.lig1_n_per_bead) * inp.lig1_num
    logger.info("\t\tEstimate number of beads in ligand 1: {:.0f}".format(n_estimate_ligand_1_beads))
    n_estimate_ligand_2_beads = np.sum(inp.lig2_n_per_bead) * inp.lig2_num
    logger.info("\t\tEstimate number of beads in ligand 2: {:.0f}".format(n_estimate_ligand_2_beads))
    n_estimate_ligands_beads = n_estimate_ligand_1_beads + n_estimate_ligand_2_beads
    n_max_ligands_beads = 1E6
    if n_estimate_ligands_beads > n_max_ligands_beads:
        raise ValueError(f"The ligands requested would require {n_estimate_ligands_beads:.0f} beads, but the maximum allowed is {n_max_ligands_beads:.0f}. "
                         f"Consider reducing the grafting density or the number of beads per ligand.")

    #######CORE#######
    logger.info("Building lattice block...")
    if inp.core_shape != "shell":
        packed_block = core_packing_functions[inp.core_method](inp)
    else:
        packed_block = []
    logger.info("Cropping block into target shape...")
    core_xyz = core_shape_functions[inp.core_shape](packed_block, inp)

    logger.info("Describing the cut shape...")
    inp.characterize_core(core_xyz)
    if inp.char_radius <= 2 * inp.bead_radius:
        error_msg = f"Cannot build core of size {inp.char_radius:.3f} nm given a bead radius of {inp.bead_radius:.3f} nm. " \
                    "Considering making a bigger nanoparticle or using bigger beads. Exiting."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if inp.parameter_file != None:
        logger.info("User provided a topology file...")
        logger.info("Importing parameters...")
        params = Parameters(inp.parameter_file)
        logger.info("Looking for missing parameters...")
        params.check_missing_parameters(inp)
    else:
        params = None
        warn_txt = "ATTENTION. Parameter file not found. Only writing nanoparticle structure..."
        logger.warning(warn_txt)

    #######LIGANDS#######
    logger.info("Placing ligand anchoring sites...")
    staples_xyz, staples_normals, close_ndxs = place_staples(core_xyz, inp)
    logger.info("Labeling ligands to anchoring sites...")
    lig_ndx = assign_morphology(staples_xyz, inp)
    logger.info("Growing ligands...")
    lig_xyz = place_ligands(staples_xyz, staples_normals, lig_ndx, inp, params)
    logger.info("Merging core with ligands...")
    np_xyz = merge_coordinates(core_xyz, lig_xyz)
    logger.info("Writing structure file (.gro)...")
    gro_writer(TMP, np_xyz, inp)

    #######TOPOLOGY#######
    if inp.parameter_file != None:
        logger.info("Assigning bonds within the core...")
        core_bonds = get_core_bonds(core_xyz, inp)
        logger.info("Assigning bonded interactions within the ligands...")
        lig_bonds, lig_angles, lig_dihedrals = get_lig_bonded_atoms(np_xyz, lig_ndx, close_ndxs, inp)
        logger.info("Writing topology file (.top)...")
        top_writer(TMP, np_xyz, core_bonds, lig_bonds, lig_angles, lig_dihedrals, inp, params)

    loggerFileHandler.flush()
    logger.handlers.remove(loggerFileHandler)
    loggerFileHandler.close()

    logger.info("Compressing final files...")

    #files "report.log" and "NP.gro". If the user provides a parameters (.itp) file,
    #  then the output should also contain "NP.top".

    # List of files to compress to output
    copy = ["report.log", "LIG1.mol2", "LIG2.mol2", "NP.top", "NP.gro"]

    zip_path = TMP + ".zip"
    zip_tmp = zipfile.ZipFile(zip_path, "w")

    # Saves the list of output files in zip file, including the MARTINI parameters if a top file was generated
    if inp.parameter_file != None:
        martini_path = "./MARTINI"
        for _, _, fnames in os.walk(martini_path):
            for fname in fnames:
                zip_tmp.write("{}/{}".format(martini_path, fname), arcname="{}/MARTINI/{}".format(TMP, fname))

    for i in copy:
        target_file = os.path.join(TMP, i)
        if not os.path.isfile(target_file):
            logger.debug("skipping file %s", i)
            continue

        zip_tmp.write("{}/{}".format(TMP, i))

    zip_tmp.close()

    # Opens and reads the saved zip file
    zf = open(zip_path, 'rb')
    zip_data = zf.read()
    zf.close()

    logger.info("Cleaning...")

    try:
        shutil.rmtree(TMP)
    except OSError as e:
        logger.warn("An error occuring during tmp folder cleanup (e: %s)", e.strerror)

    try:
        os.remove(zip_path)
    except OSError as e:
        logger.warn("An error occuring during result file cleanup (e: %s)", e.strerror)

    logger.info("NanoModeler terminated normally.")

    return (1, zip_data)



if __name__ == "__main__":
    NanoModeler_CG(BEAD_RADIUS=1.6,

    CORE_RADIUS=4,
    CORE_METHOD="fcc", # "fcc", #"fcc",
    CORE_DENSITY=20, #19.3 g/cm3 of the material
    CORE_SHAPE="sphere",
    CORE_CYLINDER=[], # [2.5,4], #Radius and length respectively. Only read if CORE_SHAPE is "cylinder"
    CORE_ELLIPSE_AXIS=[], # [1.5,3,4.5], #Only read if CORE_SHAPE is "ellipsoid"
    CORE_RECT_PRISM=[], # [3,5,7], #Only read if CORE_SHAPE is "rectangular prism"
    CORE_ROD_PARAMS=[], # [2.5, 4], #Caps radius and cylinder length respectively. Only read if CORE_SHAPE is "rod"
    CORE_PYRAMID=[], # [5,5], #Base edge and height respectively. Only read if CORE_SHAPE is "pyramid"
    CORE_OCTAHEDRON_EDGE=None, # 6, #Edge size of a regular octahedron. Only read if CORE_SHAPE is "octahedron"
    CORE_BTYPE="C5",
    CORE_EN=False,
    CORE_EN_K=None ,#5000,

    GRAFT_DENSITY=0.3, #0.152, #0.216nm2 thiol-1

    LIG1_N_PER_BEAD=[1],
    LIG1_BTYPES=["C1"],
    LIG1_CHARGES=[1],
    LIG1_MASSES=[3],
    LIG1_FRAC=1,

    LIG2_N_PER_BEAD=[], # [3],
    LIG2_BTYPES=[], # ["C1"],
    LIG2_CHARGES=[], # [0],
    LIG2_MASSES=[], # [56],

    MORPHOLOGY=None, # 'stripe_y', #random, janus_x, janus_y, janus_z, stripe_x, stripe_y, stripe_z
    RSEED=None, # 666,# None
    STRIPES=None, # 4,

    PARAMETER_FILE= None  # open('test_params.itp', 'r')
    )
