import numpy as np


def cart2spherical(xyz):
    """
    Converts a set of Cartesian coordinates into spherical coordinates.

    Parameters
    ----------
    xyz : ndarray
        Array of shape (3, N), where the (x,y,z) coordinates are given as column vectors.

    Returns
    -------   
    ρ, θ, φ

    Notes
    -----
    Convention used:  Define theta to be the azimuthal angle in the xy-plane
    from the x-axis with 0<=theta<2pi (denoted lambda when referred to as the longitude),  
    phi to be the polar angle (also known as the zenith angle and colatitude, with phi=90 
    degrees-delta where delta is the latitude) from the positive z-axis with 0<=phi<=pi, 
    and r to be distance (radius) from a point to the origin. 

    """
    if xyz.ndim == 1:
        xyz = xyz[:, np.newaxis]

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    ρ = (x**2 + y**2 + z**2) ** 0.5
    θ = np.arctan2(y, x)
    φ = np.arccos(z / ρ)

    return ρ, θ, φ


def spherical2cart(spher_coord):
    """
        Converts a set of spherical coordinates to Cartesian coordinates.

        Parameters
        ----------
        spher_coord : ndarray
            Array of shape (3, N), where the (ρ, θ, φ) coordinates are given as column vectors.


        Notes
        -----
        Using the maths convention: azimuthal angle θ and polar angle φ

    """

    ρ = spher_coord[0]
    θ = spher_coord[1]
    φ = spher_coord[2]

    x = ρ * np.cos(θ) * np.sin(φ)
    y = ρ * np.sin(θ) * np.sin(φ)
    z = ρ * np.cos(φ)

    return x, y, z


def angle_0_to_pi(angles):
    """
    Wrap angles between 0 and pi.

    Parameters
    ----------
    angles : ndarray
        Array of angles.

    Returns
    -------
    angles : ndarray
        Array of wrapped angles.

    """
    np.putmask(angles, angles >= np.pi / 2, angles - np.pi / 2)

    return angles


def polar2cart_2D(r, θ):
    """
    Convert 2D polar coordinates to Cartesian coordinates.

    """

    x = r * np.cos(θ)
    y = r * np.sin(θ)

    return x, y


def cart2polar_2D(x, y):
    """
    Convert 2D Cartesian coordinates to polar coordinates.

    """

    r = (x ** 2 + y ** 2)**0.5
    θ = np.arctan2(y, x)

    return r, θ


def pts_on_circle(r, n=100):
    """
    Get n points on the circumference of a circle in Cartesian coordinates.

    Parameters
    ----------
    r : ndarray
        Radius of the circle.
    n : int, optional
        Number of points.

    Returns
    -------
    list of tuples 
        x and y coordinates of points.

    """
    return [(np.cos(2 * np.pi / n * p) * r, np.sin(2 * np.pi / n * p) * r)
            for p in range(0, n + 1)]


def get_box_corners(box, origin=None, tolerance=1E-10):
    """
    Get all 8 corners of parallelopipeds, each defined by three edge vectors.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3) or (3, 3)
        Array defining N parallelopipeds, each as three 3D column vectors which
        define the edges of the parallelopipeds.
    origin : ndarray of shape (3, N), optional
        Array defining the N origins of N parallelopipeds as 3D column vectors.

    Returns
    -------
    ndarray of shape (N, 3, 8)
        Returns 8 3D column vectors for each input parallelopiped.

    Examples
    --------
    >>> a = np.random.randint(-1, 4, (2, 3, 3))
    >>> a
    [[[ 0  3  1]
      [ 2 -1 -1]
      [ 1  2  0]]

     [[ 0  0  3]
      [ 1  2  0]
      [-1  1 -1]]]
    >>> geometry.get_box_corners(a)
    array([[[ 0.,  0.,  3.,  1.,  3.,  1.,  4.,  4.],
            [ 0.,  2., -1., -1.,  1.,  1., -2.,  0.],
            [ 0.,  1.,  2.,  0.,  3.,  1.,  2.,  3.]],

           [[ 0.,  0.,  0.,  3.,  0.,  3.,  3.,  3.],
            [ 0.,  1.,  2.,  0.,  3.,  1.,  2.,  3.],
            [ 0., -1.,  1., -1.,  0., -2.,  0., -1.]]])

    """

    if box.ndim == 2:
        box = box[np.newaxis]

    N = box.shape[0]

    if origin is None:
        origin = np.zeros((3, N), dtype=box.dtype)

    corners = np.zeros((N, 3, 8), dtype=box.dtype)
    corners[:, :, 1] = box[:, :, 0]
    corners[:, :, 2] = box[:, :, 1]
    corners[:, :, 3] = box[:, :, 2]
    corners[:, :, 4] = box[:, :, 0] + box[:, :, 1]
    corners[:, :, 5] = box[:, :, 0] + box[:, :, 2]
    corners[:, :, 6] = box[:, :, 1] + box[:, :, 2]
    corners[:, :, 7] = box[:, :, 0] + box[:, :, 1] + box[:, :, 2]

    corners += origin.T[:, :, np.newaxis]

    return corners


def get_box_xyz(box, origin=None, faces=False):
    """
    Get coordinates of paths which trace the edges of parallelopipeds
    defined by edge vectors and origins. Useful for plotting parallelopipeds.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3) or (3, 3)
        Array defining N parallelopipeds, each as three 3D column vectors which
        define the edges of the parallelopipeds.
    origin : ndarray of shape (3, N) or (3,)
        Array defining the N origins of N parallelopipeds as 3D column vectors.
    faces : bool, optional
        If False, returns an array of shape (N, 3, 30) where the coordinates of
        a path tracing the edges of each of N parallelopipeds are returned as
        column 30 vectors.

        If True, returns a dict where the coordinates for
        each face is a key value pair. Keys are like `face01a`, where the
        numbers refer to the column indices of the vectors in the plane of the
        face to plot, the `a` faces intersect the origin and the `b` faces are
        parallel to the `a` faces. Values are arrays of shape (N, 3, 5), which
        define the coordinates of a given face as five 3D column vectors for
        each of the N input parallelopipeds.

    Returns
    -------
    ndarray of shape (N, 3, 30) or dict of str : ndarray of shape (N, 3, 5)
    (see `faces` parameter).

    """

    if box.ndim == 2:
        box = box[np.newaxis]

    N = box.shape[0]

    if origin is None:
        origin = np.zeros((3, N), dtype=box.dtype)

    elif origin.ndim == 1:
        origin = origin[:, np.newaxis]

    if origin.shape[1] != box.shape[0]:
        raise ValueError('If `origin` is specified, there must be an origin '
                         'specified for each box.')

    c = get_box_corners(box, origin=origin)

    face01a = c[:, :, [0, 1, 4, 2, 0]]
    face01b = c[:, :, [3, 5, 7, 6, 3]]
    face02a = c[:, :, [0, 1, 5, 3, 0]]
    face02b = c[:, :, [2, 4, 7, 6, 2]]
    face12a = c[:, :, [0, 2, 6, 3, 0]]
    face12b = c[:, :, [1, 4, 7, 5, 1]]

    coords = [face01a, face01b, face02a, face02b, face12a, face12b]

    if not faces:
        xyz = np.concatenate(coords, axis=2)

    else:
        faceNames = ['face01a', 'face01b', 'face02a',
                     'face02b', 'face12a', 'face12b']
        xyz = dict(zip(faceNames, coords))

    return xyz
