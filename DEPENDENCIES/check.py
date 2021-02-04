import numpy as np

def check_core_minimum_dimensions(inp):
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

    core_lengths = getattr(inp, core_attributes[inp.core_shape])
    core_lengths = np.array(core_lengths).flatten()

    shapes_parts = {
    'sphere' : ['radius'],
    'shell' : ['radius'],
    'ellipsoid' : ['X-semiaxes','Y-semiaxes','Z-semiaxes'],
    'cylinder' : ['radius', 'length'],
    'rectangular prism' : ['X-length', 'Y-length', 'Z-length'],
    'rod' : ['cap radius', 'body length'],
    'pyramid' : ['base edge-length', 'height']
    }

    current_parts = shapes_parts[inp.core_shape]
    if np.any(core_lengths < 4*inp.bead_radius):
        bad_ndx = np.where(core_lengths < 4*inp.bead_radius)[0][0]
        bad_part = current_parts[bad_ndx]
        raise Exception("The {} of the {} should be at least 4 times the bead radius!".format(bad_part, inp.core_shape))
