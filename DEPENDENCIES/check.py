import numpy as np
import logging

logger = logging.getLogger('nanomodelercg')
logger.addHandler(logging.NullHandler())

#Dictionary relating the core_shape with the attribute of Input object containing the geometric parameters of the core
core_attributes = {
'sphere' : 'core_radius',
'shell' : 'core_radius',
'ellipsoid' : 'core_ellipse_axis',
'cylinder' : 'core_cylinder',
'rectangular prism' : 'core_rect_prism',
'rod' : 'core_rod_params',
'pyramid' : 'core_pyramid',
'octahedron' : 'core_octahedron'
}

#Dictionary relating the core_shape with labels for the geometric parameters describing the core
shapes_parts = {
'sphere' : ['radius'],
'shell' : ['radius'],
'ellipsoid' : ['X-semiaxes','Y-semiaxes','Z-semiaxes'],
'cylinder' : ['radius', 'length'],
'rectangular prism' : ['X-length', 'Y-length', 'Z-length'],
'rod' : ['cap radius', 'body length'],
'pyramid' : ['base edge-length', 'height'],
'octahedron' : ['edge-length']
}

def check_core_dimensions(inp, LOWLIM=4, HIGHLIM=50):
    """
    Checks if the core's dimensions specified are too big for the server to handle
    or if they are too small for the bead radius parsed by the user.
    LOWLIM is the shortest core dimension should be at least this times the bead radius
    #HIGHLIM is the largest core dimension should be at most this times the bead radius
    """
    LOWLIM = 4
    HIGHLIM = 50
    core_lengths = getattr(inp, core_attributes[inp.core_shape])
    core_lengths = np.array(core_lengths).flatten() #Flatten makes sure that 'shell', 'sphere', and 'octahedron' (which have only one geometric parameter) behave like any other (1D array)
    current_parts = shapes_parts[inp.core_shape]

    #Checks if any of the core's geometric parameters is too big
    if np.any(core_lengths < LOWLIM*inp.bead_radius):
        bad_ndx = np.where(core_lengths < LOWLIM*inp.bead_radius)[0][0]
        bad_part = current_parts[bad_ndx]
        logger.error("The {} of the {} should be at least {} times the bead radius!".format(bad_part, inp.core_shape, LOWLIM))

    #Checks if any of the core's geometric parameters is too small given the bead radius parsed by the user
    if np.any(core_lengths > HIGHLIM*inp.bead_radius):
        bad_ndx = np.where(core_lengths > HIGHLIM*inp.bead_radius)[0][0]
        bad_part = current_parts[bad_ndx]
        logger.error("The {} of the {} should be no larger than {} times the bead radius!".format(bad_part, inp.core_shape, HIGHLIM))

def check_ligand_length(inp, THRESH=80):
    """
    Checks if the length specified for the ligands are too long for the server to handle.
    THRESH is the maximum beads in a ligand.
    The value THRESH could be made more flexible, depending on how it performs with the UI
    """
    lengths = np.array([np.sum(inp.lig1_n_per_bead), np.sum(inp.lig2_n_per_bead)])
    if np.any(lengths > THRESH):
        bad_ndx = np.where(lengths>THRESH)[0][0]
        logger.error("Ligand {} should be less than {} beads long!".format(bad_ndx+1, THRESH))
