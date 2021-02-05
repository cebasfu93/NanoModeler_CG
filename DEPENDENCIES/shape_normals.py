import numpy as np

def sphere_normal(xyz, inp):
    """
    Return xyz normalized
    """
    normal = xyz/np.linalg.norm(xyz)
    return normal

def ellipsoid_normal(xyz, inp):
    """
    Return xyz normalized, weighting each semiaxes properly
    """
    normal = np.divide(xyz, np.power(inp.core_ellipse_axis,2))
    normal /= np.linalg.norm(normal)
    return normal

def cylinder_normal(xyz, inp):
    """
    Returns (0,0,1) beyond the top lid, (0,0,-1) beyond the bottom lid, and the XY radial unit vector elsewhere
    """
    if (xyz[0]**2 + xyz[1]**2 + (np.abs(xyz[2])-inp.core_cylinder[1]/2)**2) < (inp.core_cylinder[0]**2):
        normal = np.array([0,0,np.sign(xyz[2])])
    else:
        normal = xyz.copy()
        normal[2] = 0.0
        normal /= np.linalg.norm(normal)
    return normal

def rectangular_prism_normal(xyz, inp):
    """
    Returns (+/-1,0,0), (0,+/-1,0) or (0,0,+/-1) depending on the surface of the rectangular prism
    """
    a, b, c = inp.core_rect_prism
    plane_normals = np.identity(3)
    pts_inplane = np.array([[a,0,0],[0,b,0],[0,0,c]])/2
    D = -1*np.array([np.dot(plane_normal, pt_inplane) for plane_normal, pt_inplane in zip(plane_normals, pts_inplane)])
    dists = np.divide(np.abs(np.dot(plane_normals,np.abs(xyz))+D), np.linalg.norm(plane_normals, axis=1))
    normals = np.zeros(3)
    normals[np.argmin(dists)] = np.sign(xyz[np.argmin(dists)])
    return normals

def rod_normal(xyz, inp):
    """
    Returns the XY radial unit vector around the cylindrical body and the normalized xyz beyond the spherical lids
    """
    if (xyz[0]**2 + xyz[1]**2 + (np.abs(xyz[2])-inp.core_rod_params[1]/2)**2) < (inp.core_rod_params[0]**2):
        normal = xyz - np.array([0,0,np.sign(xyz[2])*inp.core_rod_params[1]/2])
    else:
        normal = xyz.copy()
        normal[2] = 0.0
    normal /= np.linalg.norm(normal)
    return normal

def pyramid_normal(xyz, inp):
    """
    Returns the vector normal to a face of the pyramid
    """
    a = inp.core_pyramid[0]*1
    L = inp.core_pyramid[1]*1
    tip = np.array([L/2,0,0]) #the tip of the pyramid is on the X axis
    base_pts = np.array([[-L/2,a,0],[-L/2,0,a],[-L/2,-a,0],[-L/2,0,-a]]) #the points in the base are in the YZ plane
    n_pts = len(base_pts)
    plane_normals = []
    D = []
    for i in range(n_pts):
        vec = np.cross(tip-base_pts[i], tip-base_pts[(i+1)%n_pts]) #the base points are iterated over cyclicly. Their cross product is a vector normal to one of the pyramids faces
        vec /= np.linalg.norm(vec)
        plane_normals.append(vec)
        D.append(-1*np.dot(vec, base_pts[i]))
    plane_normals.append([-1,0,0]) #Vector normal to the pyramids base
    D.append(-1*np.dot(plane_normals[-1], base_pts[0])) #solves the equation of the plane of the pyramids surfaces
    plane_normals = np.array(plane_normals)
    D = np.array(D)
    dists = np.divide(np.abs(np.dot(plane_normals, xyz)+D), np.linalg.norm(plane_normals, axis=1)) #Calculates distance between the xyz point an the planes of the pyramid
    normals = plane_normals[np.argmin(dists)] #Stores the normal vector of the plane closest to the xyz point
    return normals

def octahedron_normal(xyz, inp):
    """
    Return the vector normal to a face of the octahedron
    """
    a = inp.core_octahedron*1
    tips = np.array([[0,0, a/np.sqrt(2)], [0,0, -a/np.sqrt(2)]]) #the tips of the octahedron are on the Z axis
    base_pts = np.array([[a/2, a/2, 0], [-a/2, a/2, 0], [-a/2, -a/2, 0], [a/2, -a/2, 0]]) #the points in the base are in the XY plane
    n_pts = len(base_pts)
    plane_normals = []
    D = []
    for t, tip in enumerate(tips):
        #the base points are iterated over cyclicly. Their cross product is a vector normal to one of the octahedron faces
        for i in range(n_pts):
            vec = (-1)**(t)*np.cross(base_pts[i]-tip, base_pts[(i+1)%n_pts]-tip) #-1**t changes the direction of the normal vector for the upper and lower halves of the octahedron
            vec /= np.linalg.norm(vec)
            plane_normals.append(vec)
            D.append(-1*np.dot(vec, base_pts[i])) #solves the equation of the plane of the pyramids surfaces
    plane_normals = np.array(plane_normals)
    D = np.array(D)
    dists = np.divide(np.abs(np.dot(plane_normals, xyz)+D), np.linalg.norm(plane_normals, axis=1)) #Calculates distance between the xyz point an the planes of the pyramid
    normals = plane_normals[np.argmin(dists)] #Stores the normal vector of the plane closest to the xyz point
    return normals
