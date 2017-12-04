import numpy as np
from itertools import permutations, combinations


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


def wrap_angles(angles, min_ang, max_ang, degrees=True):
    """
    Wrap angles between min_ang and max_ang.

    Parameters
    ----------
    angles : ndarray
        Array of angles.
    min_ang: float
        Minimum angle in radians or degrees.
    max_ang: float
        Maximum angle in radians or degrees.
    degrees: bool
        Unit of angles, min_ang and max_ang.

    Returns
    -------
    angles : ndarray
        Array of wrapped angles.
    TODO:
    - get rid of angle_0_to_pi
    - add radians option

    """
    if degrees:
        np.putmask(angles, angles >= (max_ang - min_ang), angles - max_ang)

        return angles
    else:
        raise ValueError('Radians option not implemented.')


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


def find_positive_int_vecs(search_size, dim=3):
    """
    Find arbitrary-dimension positive integer vectors which are
    non-collinear whose components are less than or equal to a
    given search size. Vectors with zero components are not included.

    Non-collinear here means no two vectors are related by a scaling factor.

    Parameters
    ----------
    search_size : int
        Positive integer which is the maximum vector component
    dim : int
        Dimension of vectors to search.

    Returns
    -------
    ndarray of shape (N, `dim`)

    """

    # Generate trial vectors as a grid of integer vectors
    search_ints = np.arange(1, search_size + 1)
    search_grid = np.meshgrid(*(search_ints,) * dim)
    trials = np.vstack(search_grid).reshape((dim, -1)).T

    # Multiply each trial vector by each possible integer up to
    # `search_size`:
    search_ints_rs = search_ints.reshape(-1, 1, 1)
    trial_combs = trials * search_ints_rs

    # Combine trial vectors and their associated scaled vectors:
    trial_combs_all = np.vstack(trial_combs)

    # Find unique vectors. The inverse indices`uinv` indexes
    # the set of unique vectors`u` to generate the original array `pv`:
    uniq, uniq_inv = np.unique(trial_combs_all, axis=0, return_inverse=True)

    # For a given set of (anti-)parallel vectors, we want the smallest, so get
    # their relative magnitudes. This is neccessary since `np.unique` does not
    # return vectors sorted in a sensible way if there are negative components.
    # (But we do we have negative components here?)
    uniq_mag = np.sum(uniq**2, axis=1)

    # Get the magnitudes of just the directionally-unique vectors:
    uniq_inv_mag = uniq_mag[uniq_inv]

    # Reshape the magnitudes to allow sorting for a given scale factor:
    uniq_inv_mag_rs = np.reshape(uniq_inv_mag, (search_size, -1))

    # Get the indices which sort the trial vectors
    mag_srt_idx = np.argsort(uniq_inv_mag_rs, axis=0)

    # Reshape the inverse indices
    uniq_inv_rs = np.reshape(uniq_inv, (search_size, -1))

    # Sort the inverse indices by their corresponding vector magnitudes,
    # for each scale factor:
    col_idx = np.tile(np.arange(uniq_inv_rs.shape[1]), (search_size, 1))
    uniq_inv_rs_srt = uniq_inv_rs[mag_srt_idx, col_idx]

    # Only keep inverse indices in first row which are not in any other row.
    # First row indexes lowest magnitude vectors for each scale factor.
    idx = np.setdiff1d(uniq_inv_rs_srt[0], uniq_inv_rs_srt[1:])

    # Sort kept vectors by magnitude
    final_mags = uniq_mag[idx]
    final_mags_idx = np.argsort(final_mags)

    ret = uniq[idx][final_mags_idx]

    return ret


def tile_int_vecs(int_vecs, dim):
    """
    Tile arbitrary-dimension integer vectors such that they occupy a
    half-space.

    """
    # For tiling, there will a total of 2^(`dim` - 1) permutations of the
    # original vector set. (`dim` - 1) since we want to fill a half space.
    i = np.ones(dim - 1, dtype=int)
    t = np.triu(i, k=1) + -1 * np.tril(i)

    # Get permutation of +/- 1 factors to tile initial vectors into half-space
    perms_partial_all = [j for i in t for j in list(permutations(i))]
    perms_partial = np.array(list(set(perms_partial_all)))

    perms_first_col = np.ones((2**(dim - 1) - 1, 1), dtype=int)
    perms_first_row = np.ones((1, dim), dtype=int)
    perms_non_eye = np.hstack([perms_first_col, perms_partial])
    perms = np.vstack([perms_first_row, perms_non_eye])

    perms_rs = perms[:, np.newaxis]
    tiled = int_vecs * perms_rs
    ret = np.vstack(tiled)

    return ret


def find_non_parallel_int_vecs(search_size, dim=3, tile=False):
    """
    Find arbitrary-dimension integer vectors which are non-collinear, whose
    components are less than or equal to a given search size.

    Non-collinear here means no two vectors are related by a scaling factor.
    The zero vector is excluded.

    Parameters
    ----------
    search_size : int
        Positive integer which is the maximum vector component.
    dim : int
        Dimension of vectors to search.
    tile : bool, optional
        If True, the half-space of dimension `dim` is filled with vectors,
        otherwise just the positive vector components are considered. The
        resulting vector set will still contain only non-collinear vectors.

    Returns
    -------
    ndarray of shape (N, `dim`)
        Vectors are not globally ordered.

    Notes
    -----
    Searching for vectors with `search_size` of 100 uses about 9 GB of memory.

    """

    # Find all non-parallel positive integer vectors which have no
    # zero components:
    ret = find_positive_int_vecs(search_size, dim)

    # If requested, tile the vectors such that they occupy a half-space:
    if tile and dim > 1:
        ret = tile_int_vecs(ret, dim)

    # Add in the vectors which are contained within a subspace of dimension
    # (`dim` - 1) on the principle axes. I.e. vectors with zero components:
    if dim > 1:

        # Recurse through each (`dim` - 1) dimension subspace:
        low_dim = dim - 1
        vecs_lower = find_non_parallel_int_vecs(search_size, low_dim, tile)

        # Raise vectors to current dimension with a zero component. The first
        # (`dim` - 1) "prinicple" vectors (of the form [1, 0, ...]) should be
        # considered separately, else they will be repeated.
        principle = np.eye(dim, dtype=int)
        non_prcp = vecs_lower[low_dim:]

        if non_prcp.size:

            edges_shape = (dim, non_prcp.shape[0], non_prcp.shape[1] + 1)
            vecs_edges = np.zeros(edges_shape, dtype=int)
            edges_idx = list(combinations(list(range(dim)), low_dim))

            for i in range(dim):
                vecs_edges[i][:, edges_idx[i]] = non_prcp

            vecs_edges = np.vstack([principle, *vecs_edges])

        else:
            vecs_edges = principle

        ret = np.vstack([vecs_edges, ret])

    return ret


def norm_vec(vec):
    """
    Normalise a vector

    """
    if isinstance(vec) == list:
        vec = np.array(vec)

    return vec / np.linalg.norm(vec)


def point_in_poly(poly_vert, points):
    """
    Check if points lie in or out of a polygon.

    Parameters
    ----------
    poly_vert : ndarray or list
        Vertices of polygon.
    points: ndarray of shape (N, 2)
        Points coordinates.

    Returns
    -------
    bool of shape (N, 1)

    """
    path = mpltPath.Path(poly_vert)

    return path.contains_points(points)


def get_xy_bounding_trace(x, y):

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    # Corners of bounding quad:
    trace = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min],
    ]).T

    return trace
