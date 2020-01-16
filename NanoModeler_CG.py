def NanoModeler_CG(BEAD_RADIUS=0.26, CORE_RADIUS=None, CORE_METHOD=None, CORE_BMASS=72, CORE_BTYPE=None, CORE_EN_K=5000, LIG1_NUM=None, LIG2_NUM=None, LIG1_N_PER_BEAD=[], LIG1_BTYPES=[], LIG1_CHARGES=[], LIG1_MASSES=[], LIG2_N_PER_BEAD=[], LIG2_BTYPES=[], LIG2_CHARGES=[], LIG2_MASSES=[], MORPHOLOGY='random', RSEED=None, STRIPES=1, PARAMETER_FILE=None):
    import tempfile
    import numpy as np
    from  scipy.spatial.distance import cdist
    from scipy.optimize import minimize
    #import matplotlib.pyplot as plt
    from DEPENDENCIES.Extras import Input, Parameters, center, cartesian_to_polar, polar_to_cartesian, sunflower_pts, merge_coordinates
    from DEPENDENCIES.core_maker import primitive_sphere, bcc_sphere, fcc_sphere, hcp_xyz, hcp_sphere, gkeka_method, hollow_sphere
    from DEPENDENCIES.coat_maker import place_staples, assign_morphology, grow_ligands
    from DEPENDENCIES.top_maker import get_lig_bonds, get_lig_angles
    from DEPENDENCIES.writers import gro_writer, top_writer

    inp = Input(
    bead_radius=BEAD_RADIUS,

    core_radius=CORE_RADIUS,
    core_method=CORE_METHOD,
    core_bmass=CORE_BMASS,
    core_btype=CORE_BTYPE,
    core_en_k=CORE_EN_K,
    lig1_num=LIG1_NUM,
    lig2_num=LIG2_NUM,

    lig1_n_per_bead=LIG1_N_PER_BEAD,
    lig1_btypes=LIG1_BTYPES,
    lig1_charges=LIG1_CHARGES,
    lig1_masses=LIG1_MASSES,

    lig2_n_per_bead=LIG2_N_PER_BEAD,
    lig2_btypes=LIG2_BTYPES,
    lig2_charges=LIG2_CHARGES,
    lig2_masses=LIG2_MASSES,

    morph=MORPHOLOGY,
    rsd=RSEED,
    stripes=STRIPES,

    parameter_file=PARAMETER_FILE)


    TMP = tempfile.mkdtemp(dir="./")

    core_functions = {'primitive': primitive_sphere,
    'bcc': bcc_sphere,
    'fcc': fcc_sphere,
    'hcp': hcp_sphere,
    'hollow': hollow_sphere}

    #######STRUCTURE#######
    core_xyz = core_functions[inp.core_method](inp)

    staples_xyz = place_staples(core_xyz, inp)
    lig_ndx = assign_morphology(staples_xyz, inp)
    lig_xyz = grow_ligands(staples_xyz, lig_ndx, inp)
    np_xyz = merge_coordinates(core_xyz, lig_xyz)

    gro_writer(TMP, np_xyz, inp)

    #######TOPOLOGY#######
    if inp.parameter_file != None:
        params = Parameters(inp.parameter_file)
        params.check_missing_parameters(inp)
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
    CORE_BTYPE="C1",
    CORE_BMASS=711.5,
    CORE_EN_K=5000,
    LIG1_NUM=20,
    LIG2_NUM=0,

    LIG1_N_PER_BEAD=[3,2],
    LIG1_BTYPES=["C1", "Qda"],
    LIG1_CHARGES=[0,1],
    LIG1_MASSES=[72,72],

    LIG2_N_PER_BEAD=[],
    LIG2_BTYPES=[],
    LIG2_CHARGES=[],
    LIG2_MASSES=[],

    MORPHOLOGY='random',
    RSEED=1,# None
    STRIPES=1,

    PARAMETER_FILE=open('test_params.itp', 'r'))
