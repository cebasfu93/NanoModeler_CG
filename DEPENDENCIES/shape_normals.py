import numpy as np

def sphere_normal(xyz, inp):
    xyz /= np.linalg.norm(xyz)
    return xyz

def ellipsoid_normal(xyz, inp):
    xyz = np.divide(xyz, np.power(inp.core_ellipse_axis,2))
    xyz /= np.linalg.norm(xyz)
    return xyz

def cylinder_normal(xyz, inp):
    return xyz

def rectangular_prism_normal(xyz, inp):
    return xyz

def rod_normal(xyz, inp):
    return xyz

def pyramid_normal(xyz, inp):
    return xyz

def octahedron_normal(xyz, inp):
    return xyz
