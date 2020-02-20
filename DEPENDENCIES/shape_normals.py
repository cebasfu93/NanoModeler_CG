import numpy as np

def sphere_normal(xyz, inp):
    normal = xyz/np.linalg.norm(xyz)
    return normal

def ellipsoid_normal(xyz, inp):
    normal = np.divide(xyz, np.power(inp.core_ellipse_axis,2))
    normal /= np.linalg.norm(normal)
    return normal

def cylinder_normal(xyz, inp):
    if (xyz[0]**2 + xyz[1]**2 + (np.abs(xyz[2])-inp.core_cylinder[1]/2)**2) < (inp.core_cylinder[0]**2):
        normal = np.array([0,0,np.sign(xyz[2])])
    else:
        normal = xyz.copy()
        normal[2] = 0.0
        normal /= np.linalg.norm(normal)
    return normal

def rectangular_prism_normal(xyz, inp):
    return xyz

def rod_normal(xyz, inp):
    return xyz

def pyramid_normal(xyz, inp):
    return xyz

def octahedron_normal(xyz, inp):
    return xyz
