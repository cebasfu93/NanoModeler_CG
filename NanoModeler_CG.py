def NanoModeler_CG(BEAD_RADIUS=0.26,
    CORE_RADIUS=None,
    CORE_METHOD=None,
    CORE_DENSITY=None,
    CORE_SHAPE="sphere",
    CORE_CYLINDER=[1.5,1.5],
    CORE_ELLIPSE_AXIS=[1.5,1.5,1.5],
    CORE_RECT_PRISM=[1.5,1.5,1.5],
    CORE_ROD_PARAMS=[1.5,1.5],
    CORE_PYRAMID=[1.5,1.5],
    CORE_OCTAHEDRON_EDGE=1.5,
    CORE_BTYPE=None,
    CORE_EN=None,
    CORE_EN_K=5000,

    GRAFT_DENSITY=0.216,

    LIG1_N_PER_BEAD=[],
    LIG1_BTYPES=[],
    LIG1_CHARGES=[],
    LIG1_MASSES=[],
    LIG1_FRAC=1.0,

    LIG2_N_PER_BEAD=[],
    LIG2_BTYPES=[],
    LIG2_CHARGES=[],
    LIG2_MASSES=[],

    MORPHOLOGY='random',
    RSEED=None,
    STRIPES=1,
    PARAMETER_FILE=None):


    import tempfile
    import numpy as np
    from  scipy.spatial.distance import cdist
    from scipy.optimize import minimize
    from DEPENDENCIES.Extras import Input, Parameters, center, cartesian_to_polar, polar_to_cartesian, sunflower_pts, merge_coordinates
    from DEPENDENCIES.spatial_distributions import primitive, bcc, fcc, hcp, shell
    from DEPENDENCIES.core_maker import sphere, ellipsoid, cylinder, rectangular_prism, rod, pyramid, octahedron
    from DEPENDENCIES.coat_maker import place_staples, assign_morphology, grow_ligands
    from DEPENDENCIES.top_maker import get_core_bonds, get_lig_bonds, get_lig_angles
    from DEPENDENCIES.writers import gro_writer, top_writer

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


    TMP = tempfile.mkdtemp(dir="./")

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
    packed_block = core_packing_functions[inp.core_method](inp)
    if inp.core_shape != 'shape':
        core_xyz = core_shape_functions[inp.core_shape](packed_block, inp)
    else:
        core_xyz = packed_block*1
    inp.characterize_core(core_xyz)

    #######LIGANDS#######
    staples_xyz, staples_normals = place_staples(core_xyz, inp)
    lig_ndx = assign_morphology(staples_xyz, inp)
    lig_xyz = grow_ligands(staples_xyz, staples_normals, lig_ndx, inp)
    np_xyz = merge_coordinates(core_xyz, lig_xyz)

    gro_writer(TMP, np_xyz, inp)

    #######TOPOLOGY#######
    if inp.parameter_file != None:
        params = Parameters(inp.parameter_file)
        params.check_missing_parameters(inp)
        core_bonds = get_core_bonds(core_xyz, inp)
        lig_bonds = get_lig_bonds(np_xyz, inp)
        lig_angles = get_lig_angles(np_xyz, inp)
        top_writer(TMP, np_xyz, lig_bonds, lig_angles, inp, params)
    else:
        print("Parameter file not found. Only writing nanoparticle structure.")

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
    CORE_EN=False,
    CORE_EN_K=5000,

    GRAFT_DENSITY=0.216, #nm2 thiol-1

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
    RSEED=2,# None
    STRIPES=4,

    PARAMETER_FILE=open('PEG.itp', 'r'))
