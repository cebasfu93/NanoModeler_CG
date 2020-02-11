import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

__VERSION__ = "0.1.0"

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

    logger.info("WELCOME TO NANOMODELER CG")
    logger.info("Importing tempfile library...")
    import tempfile
    logger.info("Importing os library...")
    import os
    logger.info("Importing numpy library...")
    import numpy as np
    logger.info("Importing scipy.spatial.distance library...")
    from  scipy.spatial.distance import cdist
    logger.info("Importing scipy.optimize library...")
    from scipy.optimize import minimize
    logger.info("Importing private classes...")
    from DEPENDENCIES.Extras import Input, Parameters, center, cartesian_to_polar, polar_to_cartesian, sunflower_pts, merge_coordinates
    logger.info("Importing lattice generators...")
    from DEPENDENCIES.spatial_distributions import primitive, bcc, fcc, hcp, shell
    logger.info("Importing shape cutters...")
    from DEPENDENCIES.core_maker import sphere, ellipsoid, cylinder, rectangular_prism, rod, pyramid, octahedron
    logger.info("Importing coating functions...")
    from DEPENDENCIES.coat_maker import place_staples, assign_morphology, grow_ligands
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

    core_packing_functions = {'primitive': primitive,
    'bcc': bcc,
    'fcc': fcc,
    'hcp': hcp,
    'shell': shell
    }
    core_shape_functions = {'sphere': sphere,
    'ellipsoid': ellipsoid,
    'cylinder': cylinder,
    'rectangular prism': rectangular_prism,
    'rod': rod,
    'pyramid' : pyramid,
    'octahedron' : octahedron
    }

    #######CORE#######
    logger.info("Building lattice block...")
    packed_block = core_packing_functions[inp.core_method](inp)
    logger.info("Cropping block into target shape...")
    if inp.core_shape != 'shape':
        core_xyz = core_shape_functions[inp.core_shape](packed_block, inp)
    else:
        core_xyz = packed_block*1
    logger.info("Describing the cut shape...")
    inp.characterize_core(core_xyz)

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
    staples_xyz = place_staples(core_xyz, inp)
    logger.info("Labeling ligands to anchoring sites...")
    lig_ndx = assign_morphology(staples_xyz, inp)
    logger.info("Growing ligands...")
    lig_xyz = grow_ligands(staples_xyz, lig_ndx, inp, params)
    logger.info("Merging core with ligands...")
    np_xyz = merge_coordinates(core_xyz, lig_xyz)
    logger.info("Writing structure file (.gro)...")
    gro_writer(TMP, np_xyz, inp)

    #######TOPOLOGY#######
    if inp.parameter_file != None:
        logger.info("Assigning bonds within the core...")
        core_bonds = get_core_bonds(core_xyz, inp)
        logger.info("Assigning bonded interactions within the ligands...")
        lig_bonds, lig_angles, lig_dihedrals = get_lig_bonded_atoms(np_xyz, inp)
        logger.info("Writing topology file (.top)...")
        top_writer(TMP, np_xyz, core_bonds, lig_bonds, lig_angles, lig_dihedrals, inp, params)

    loggerFileHandler.flush()
    logger.handlers.remove(loggerFileHandler)
    loggerFileHandler.close()

    return 1


if __name__ == "__main__":
    NanoModeler_CG(BEAD_RADIUS=0.26,

    CORE_RADIUS=1.5,
    CORE_METHOD="fcc",
    CORE_DENSITY=19.3, #g/cm3 of the material
    CORE_SHAPE="sphere",
    CORE_CYLINDER=[2.5,4], #Radius and length respectively. Only read if CORE_SHAPE is "cylinder"
    CORE_ELLIPSE_AXIS=[1.5,3,4.5], #Only read if CORE_SHAPE is "ellipsoid"
    CORE_RECT_PRISM=[2,4,6], #Only read if CORE_SHAPE is "rectangular prism"
    CORE_ROD_PARAMS=[2.5, 4], #Caps radius and cylinder length respectively. Only read if CORE_SHAPE is "rod"
    CORE_PYRAMID=[5,5], #Base edge and height respectively. Only read if CORE_SHAPE is "pyramid"
    CORE_OCTAHEDRON_EDGE=6, #Edge size of a regular octahedron. Only read if CORE_SHAPE is "octahedron"
    CORE_BTYPE="C1",
    CORE_EN=True,
    CORE_EN_K=5000,

    GRAFT_DENSITY=0.152, #0.216nm2 thiol-1

    LIG1_N_PER_BEAD=[1,3,1],
    LIG1_BTYPES=["C1", "EO", "SP2"],
    LIG1_CHARGES=[0,0,0],
    LIG1_MASSES=[56,44,31],
    LIG1_FRAC=1.0,

    LIG2_N_PER_BEAD=[],
    LIG2_BTYPES=[],
    LIG2_CHARGES=[],
    LIG2_MASSES=[],

    MORPHOLOGY='random', #random, janus_x, janus_y, janus_z, stripe_x, stripe_y, stripe_z
    RSEED=666,# None
    STRIPES=4,

    PARAMETER_FILE=open('PEG.itp', 'r'))
