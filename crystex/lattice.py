import numpy as np

from crystex import numutils as nu
from crystex import coordgeometry
import warnings

def lattice_params_from_vecs(latt_vecs):
    """
    Calculates lattice parameters from a set of lattice vectors. 

    Notes
    -----
    Vectors given in Cartesian reference frame.

    """
    av = latt_vecs[0, :]
    bv = latt_vecs[1, :]
    cv = latt_vecs[2, :]

    a = np.linalg.norm(av)
    b = np.linalg.norm(bv)
    c = np.linalg.norm(cv)
    α = np.arccos(np.dot(bv, cv) / (np.linalg.norm(bv) * np.linalg.norm(cv)))
    β = np.arccos(np.dot(av, cv) / (np.linalg.norm(av) * np.linalg.norm(cv)))
    γ = np.arccos(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))

    return [a, b, c, α, β, γ]


def reciprocal_lattice_vecs(latt_vecs):
    """
    Return the reciprocal lattice vectors as column vectors.

    Parameters
    ----------
    a, b, c : ndarray of shape (3,3)
        Arrays of column representing the unit vectors of a space lattice.

    Returns
    -------
    ar, br, cr : ndarray
        Arrays of shape (3,3) representing the unit 
        vectors of the reciprocal lattice.

    Notes
    -----
    Vectors given in Cartesian reference frame.

    """
    av = latt_vecs[:, 0]
    bv = latt_vecs[:, 1]
    cv = latt_vecs[:, 2]

    V = np.dot(av, np.cross(bv, cv))
    Vr = 1 / V

    avr = np.cross(bv, cv) / V
    bvr = np.cross(cv, av) / V
    cvr = np.cross(av, bv) / V

    rec_vecs = np.array([avr, bvr, cvr]).T

    return rec_vecs


def reciprocal_lattice_params(latt_params, degrees=False):
    """
    Return the reciprocal lattice parameters from real 
    space lattice parameters.

    Parameters
    ----------
    latt_params : list
        Real space lattice parameters in order: a, b, c, α, β, γ.
    degrees : bool
        Units of angles.

    Returns
    -------
    rec_params : list
        Reciprocal lattice parameter in order: ar, br, cr, αr, βr, γr.

    """
    a, b, c, α, β, γ = latt_params

    if degrees:
        α, β, γ = [np.radians(ang) for ang in [α, β, γ]]

    V = a * b * c * (1 - np.cos(α)**2 - np.cos(β)**2 - np.cos(γ)**2
                     + 2 * np.cos(α) * np.cos(β) * np.cos(γ))**0.5

    ar = b * c * np.sin(α) / V
    br = a * c * np.sin(β) / V
    cr = a * b * np.sin(γ) / V
    αr = np.arccos((np.cos(β) * np.cos(γ) - np.cos(α)) /
                   (np.sin(β) * np.sin(γ)))
    βr = np.arccos((np.cos(α) * np.cos(γ) - np.cos(β)) /
                   (np.sin(α) * np.sin(γ)))
    γr = np.arccos((np.cos(α) * np.cos(β) - np.cos(γ)) /
                   (np.sin(α) * np.sin(β)))

    rec_params = [ar, br, cr, αr, βr, γr]

    return rec_params


def miller_brav2miller(mb_inds, idx_type='direction'):
    """
    Convert a set of Miller-Bravais to Miller indices in the hexagonal
    reference frame.

    Parameters
    ----------
    mb_ind : ndarray of shape (4, n)
        Array of column vectors equal to the Miller-Bravais indices of 
        `n` directions/planes.
    idx_type : string
        Type of index given. Options allowed: 'direction' [default] and plane'.

    Returns
    -------
    m_ind : ndarray of shape (3, n)
        Array of column vectors equal to the Miller indices.

    Notes
    -----
    The basis vectors of the hexagonal reference frame are considered to be:
    a1_v = [2, -1, -1, 0] * 6**0.5 * a 
    a2_v = [-1, 2, -1, 0] * 6**0.5 * a 
    c_v  = [0, 0, 0, 1] * c. 
    Conversion formulas found in [1].

    [1] Indices, O. M. (1963) (4), 862–866. doi.org/10.1107/S0365110X65002116.

    """
    m_inds = np.zeros((3, mb_inds.shape[1]))

    idx_types = ['direction', 'plane']

    if idx_type not in idx_types:
        raise ValueError("Invalid index type. Expected one of: %s" % idx_types)

    elif idx_type == 'direction':
        m_inds[0:2] = mb_inds[0:2] - mb_inds[2]
        m_inds[2] = mb_inds[3]

    elif idx_type == 'plane':
        m_inds[0:2] = mb_inds[0:2]
        m_inds[2] = mb_inds[3]

    return m_inds


def miller2miller_brav(m_inds, idx_type='direction'):
    """
    Convert a set of Miller to Miller-Bravais indices in the hexagonal
    reference frame.

    Parameters
    ----------
    m_ind : ndarray of shape (3, n)
        Array of column vectors equal to the Miller indices of `n` directions/planes.
    idx_type : string
        Type of index given. Options allowed: 'direction' [default] and plane'.

    Returns
    -------
    m_ind : ndarray of shape (4, n)
        Array of column vectors equal to the Miller-Bravais indices.

    Notes
    -----
    The basis vectors of the hexagonal reference frame are considered to be:
    a1_v = [2, -1, -1, 0] * 6**0.5 * a 
    a2_v = [-1, 2, -1, 0] * 6**0.5 * a 
    c_v  = [0, 0, 0, 1] * c. 
    Conversion formulas found in [1].

    [1] Indices, O. M. (1963) (4), 862–866. doi.org/10.1107/S0365110X65002116.

    """
    mb_inds = np.zeros((4, m_inds.shape[1]))

    idx_types = ['direction', 'plane']

    if idx_type not in idx_types:
        raise ValueError("Invalid index type. Expected one of: %s" % idx_types)

    elif idx_type == 'direction':
        mb_inds[0] = 2 * m_inds[0] - m_inds[1]
        mb_inds[1] = - m_inds[0] + 2 * m_inds[1]
        mb_inds[2] = - m_inds[0] - m_inds[1]
        mb_inds[3] = 3 * m_inds[2]

    elif idx_type == 'plane':
        mb_inds[0] = m_inds[0]
        mb_inds[1] = m_inds[1]
        mb_inds[2] = - m_inds[0] - m_inds[1]
        mb_inds[3] = m_inds[2]

    return mb_inds


def crystal2ortho(lattice_sys=None, a=None, b=None, c=None, α=None, β=None, γ=None,
                  degrees=False, normed=True, align='cz'):
    """
    Returns transformation matrix from a crystal to an orthonormal reference frame. 
    The opposite transformation given by the inverse matrix.

    Parameters
    ----------
    lattice_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal, 
        orthorhombic, monoclinic, triclinic.
    a, b, c : float or None, optional
        Lattice parameters representing the magnitude of each of the lattice vectors.
        If all three are None, a = b = c = 1.
    α, β, γ : float or None, optional
        Lattice parameters representing the angles between the lattice vectors. 
        If all three are None, example angles sets are used as described in Notes.
    degrees : bool, optional
        Units of `α`, `β`, `γ`. Radians by default.a
    normed : bool, optional
        Specify whether lattice vectors to be normalised to unit vectors. True 
        by default.
    align : string, optional
        Alignment option between crystal and orthonormal reference frames. 
        Three options implemented (as described in [1]): 
        - 'ax': a-axis || x-axis and c*-axis || z*-axis
        - 'by': b-axis || y-axis and a*-axis || x*-axis
        - 'cz': c-axis || z-axis and a*-axis || x*-axis [Default]
        where * corresponds to reciprocal lattice vectors. 
    Returns
    -------
    M : ndarray of shape (3,3)
        Transformation matrix from a crystal to an orthonormal reference frame.
        Acts on column vectors. The opposite transformation given by the inverse matrix.

    Notes
    -----
    If lattice parameters α, β, γ = None, None, None, default values are:
        'cubic':         α, β, γ = 90, 90, 90  (latt def)
        'hexagonal':     α, β, γ = 90, 90, 120 (latt def)
        'rhombohedral':  α, β, γ = 70, 70, 70  (arbitrary)
        'tetragonal':    α, β, γ = 90, 90, 90  (latt def)
        'orthorhombic':  α, β, γ = 90, 90, 90  (latt def)
        'monoclinic':    α, β, γ = 90, 99, 90  (α, γ: latt def, β: arbitrary )
        'triclinic':     α, β, γ = 40, 50, 100 (arbitrary)

    References
    ----------
    [1] Giacovazzo et al.(2002) Fundamentals of Crystallography. Oxf Univ Press. p. 75-76.

    Notes
    -----
    Alternative eqn for `cz` method (from Britton):
    # f = (1 - (np.cos(α))**2 - (np.cos(β))**2 - (np.cos(γ))**2 + 2*np.cos(α)*np.cos(β)*np.cos(γ))**0.5
    # ac = a * np.array([ f / np.sin(α), (np.cos(γ) - np.cos(α) * np.cos(β)) / np.sin(α), np.cos(β) ])
    # bc = b * np.array([0.0, np.sin(α), np.cos(α) ])
    # cc = np.array([0.0, 0.0, c])

    """
    params = [a, b, c, α, β, γ]

    if all([p is None for p in params[0:3]]):
        a, b, c = 1, 1, 1
    elif any([p is None for p in params[0:3]]):
        a, b, c = 1, 1, 1
        raise Warning('Not all lattice parameters `a`, `b`, `c` given.'
                      'All three will be set to one.')

    if all([p is None for p in params[3:]]):
        degrees = True
        if lattice_sys == 'cubic':
            α, β, γ = 90, 90, 90
        elif lattice_sys == 'hexagonal':
            α, β, γ = 90, 90, 120
        elif lattice_sys == 'rhombohedral':
            α, β, γ = 70, 70, 70
        elif lattice_sys == 'tetragonal':
            α, β, γ = 90, 90, 90
        elif lattice_sys == 'orthorhombic':
            α, β, γ = 90, 90, 90
        elif lattice_sys == 'monoclinic':
            α, β, γ = 90, 99, 90
        elif lattice_sys == 'triclinic':
            α, β, γ = 40, 50, 100

    if degrees:
        α, β, γ = [np.radians(x) for x in [α, β, γ]]

    if normed:
        a, b, c = [x / x for x in [a, b, c]]

    latt_params = [a, b, c, α, β, γ]
    ar, br, cr, αr, βr, γr = reciprocal_lattice_params(latt_params=latt_params)

    all_align = ['ax', 'by', 'cz']

    # Find transformation matrix
    if align not in all_align:
        raise ValueError('"{}" is not a valid align option. '
                         '`align` must be one of: {}.'.format(
                             align, all_align))

    elif align == 'cz':
        M = np.array([[1 / ar, - 1 / (np.tan(γr) * ar), a * np.cos(β)],
                      [0.0, 1 / (br * np.sin(γr)), b * np.cos(α)],
                      [0.0, 0.0, c]])

    elif align == 'ax':
        M = np.array([[a, 0.0, 0.0],
                      [b * np.cos(γ), b * np.sin(γ), 0.0],
                      [c * np.cos(β), - c * np.sin(β) * np.cos(αr), 1 / cr]])

    elif align == 'by':
        M = np.array([[a * np.sin(γ) * np.sin(βr), a * np.cos(γ), α * np.sin(γ) * np.cos(βr)],
                      [0.0, b, 0.0],
                      [0.0, c * np.cos(α),  c * np.sin(α)]])

    return nu.zero_prec(M)


def plane_normal(hkl_plane, latt_vecs=None, latt_sys=None, latt_params=None,
                 degrees=False, align='cz', normed=False):
    """
    Get the plane normal in Cartesian coordinates for a plane with Miller 
    indices (hkl).

    Parameters
    ----------
    hkl_plane : list or ndarray 
        The Miller indices of a crystallographic plane.
    latt_vecs : ndarray of shape (3, 3), optional
        Row vectors representing the lattice unit cell in a Cartesian reference frame.
        See Notes.
    latt_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal,
        orthorhombic, monoclinic, triclinic. See Notes.
    latt_params : list of lenght 6
        Lattice parameters. The fist three represent the magnitude of each of 
        the lattice vectors.
        If all three are None, a = b = c = 1.
        If all three angles are None, example angles sets are used as described in Notes.
    degrees : bool, optional
        Units of `α`, `β`, `γ`. Radians by default.
    align : string, optional
        Alignment option between crystal and orthonormal reference frames.
        Three options implemented (as described in [1]):
        - 'ax': a-axis || x-axis and c*-axis || z*-axis
        - 'by': b-axis || y-axis and a*-axis || x*-axis
        - 'cz': c-axis || z-axis and a*-axis || x*-axis [Default]
        where * corresponds to reciprocal lattice vectors.
    normed : bool
        Option to normalise the plane normal. True by default.

    Returns
    -------
    hkl_norm : ndarray

    Notes
    -----
    Specify either latt_vecs or latt_sys (and latt_params, optional).

    TODO: 
    Sort out checks for specifying lattice.
    Align option default only if latt_sys given. For latt_vecs, alignment is 
    already done.  

    """
    if isinstance(hkl_plane, list):
        hkl_plane = np.array(hkl_plane)

    if latt_vecs is not None:
        M = latt_vecs
    elif latt_params:
        params_dict = {'a': latt_params[0],
                       'b': latt_params[1],
                       'c': latt_params[2],
                       'α': latt_params[3],
                       'β': latt_params[4],
                       'γ': latt_params[5]}
        M = crystal2ortho(latt_sys, **params_dict, normed=False,
                          degrees=degrees, align=align)
    else:
        M = crystal2ortho(latt_sys, normed=False,
                          degrees=degrees, align=align)
    # print('M: ', M)
    cell_ortho = np.dot(M.T, np.eye(3))
    cell_rec = reciprocal_lattice_vecs(cell_ortho)
    # print('cell_rec: ', cell_rec)
    hkl_norm = np.dot(cell_rec, hkl_plane)
    if normed:
        hkl_norm = hkl_norm / np.linalg.norm(hkl_norm, axis=0)
    # print('hkl_norm: ', hkl_norm)
    return hkl_norm


def plane_from_normal(hkl_norm, latt_sys=None, latt_params=None, degrees=False, align='cz'):
    """
    Get the lattice plane Miller indices (hkl) from a plane normal vector in
    Cartesian coordinates.

    Parameters
    ----------
    hkl_norm : list or ndarray 
        The Cartesian coordinates for a crystallographic plane normal.
    latt_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal,
        orthorhombic, monoclinic, triclinic.
    latt_params : list of lenght 6
        Lattice parameters. The fist three represent the magnitude of each of 
        the lattice vectors.
        If all three are None, a = b = c = 1.
        If all three angles are None, example angles sets are used as described in Notes.
    degrees : bool, optional
        Units of `α`, `β`, `γ`. Radians by default.
    align : string, optional
        Alignment option between crystal and orthonormal reference frames.
        Three options implemented (as described in [1]):
        - 'ax': a-axis || x-axis and c*-axis || z*-axis
        - 'by': b-axis || y-axis and a*-axis || x*-axis
        - 'cz': c-axis || z-axis and a*-axis || x*-axis [Default]
        where * corresponds to reciprocal lattice vectors.

    Returns
    -------
    hkl_plane : ndarray 

    """
    if isinstance(hkl_norm, list):
        hkl_norm = np.array(hkl_norm)

    if latt_params:
        params_dict = {'a': latt_params[0],
                       'b': latt_params[1],
                       'c': latt_params[2],
                       'α': latt_params[3],
                       'β': latt_params[4],
                       'γ': latt_params[5]}
        M = crystal2ortho(latt_sys, **params_dict, normed=False,
                          degrees=degrees, align=align)
    else:
        M = crystal2ortho(latt_sys, normed=False,
                          degrees=degrees, align=align)
    cell_ortho = np.dot(M.T, np.eye(3))
    cell_rec = reciprocal_lattice_vecs(cell_ortho)
    hkl_plane = np.dot(np.linalg.inv(cell_rec), hkl_norm)

    return hkl_plane


def cart2miller(vec, lat, lat_params, tol, vec_type, max_mill=20, degrees_in=False,
                degrees_out=False):

    all_mill, all_ang = cart2miller_all(
        vec, lat, tol, vec_type, max_mill, degrees_in, degrees_out)

    return all_mill[:, 0], all_ang[0]


def flip_vec_to_half_space(vec):
    """
    Flip vector into a positive a-component half space.
    """
    a = vec[0, 0]
    b = vec[1, 0]
    c = vec[2, 0]

    flip = False
    if (
            (a < 0)
            or (np.isclose(a, 0) and b < 0)
            or (np.isclose(a, 0) and np.isclose(b, 0) and c < 0)
    ):
        flip = True

    if flip:
        vec *= -1


def cart2miller_all(vec, lat, tol, vec_type, max_mill=20, degrees_in=False,
                    degrees_out=False):
    """
    Convert a Cartesian vector into Miller indices representing a lattice 
    direction or plane within a particular lattice system.

    Parameters
    ----------
    vec : ndarray of shape (3, 1) 
        Cartesian vector whose direction in Miller indices is to be found.
    lat : ndarray of shape (3, 3)
        Column vectors representing the lattice unit cell
    lat_params : list of lenght 6
        Lattice parameters. The fist three represent the magnitude of each of 
        the lattice vectors.
    tol : float
        Snapping angular tolerance in degrees.
    vec_type : str ("direction" | "plane-normal")
        If "plane", `vec` is interpreted as a normal vector to the plane to be 
        found in Miller indices.
    max_mill: int, optional
        Maximum Miller index. By default, set to 20.
    degrees_in : bool, optional
        If True, the angular tolerance and the angular lattice parameters will be interpreted as degrees, otherwise radians. False by default.
    degrees_out : bool, optional
        If True, the angular differences will be returned in degrees, otherwise
        in radians. False by default.

    Returns 
    -------
    tuple of (ndarray of shape (3, N), ndarray of shape (N,))
        The first array is an array of column vectors representing Miller 
        indices within the specified tolerance. The second array is the
        corresponding angular difference between a given found Miller vector
        and the specified vector. Both arrays are sorted by smallest angle.
    NEW:
     - Added lattice parameters as input.

    """

    if tol >= 90:
        raise ValueError('Specify an angular tolerance lower than 90 degrees.')

    elif tol >= 20:
        warnings.warn('Large angular tolerance specified: {}'.format(tol))
        pass

    if vec_type not in ['direction', 'plane-normal']:
        raise ValueError('`vec_type` must be one of "direction", "plane".')

    if degrees_in:
        tol = np.deg2rad(tol)

    tol_dist = 2 * np.sin(tol / 2)

    vec_unit = vec / np.linalg.norm(vec, axis=0)
    print(vec_unit)

    # Transform vector to a lattice basis and check if in search space.
    # Search space is the lattice half-space with positive a-component.
    vec_lat = np.dot(np.linalg.inv(lat), vec_unit)
    flip_vec_to_half_space(vec_lat)
    vec_unit = np.dot(lat, vec_lat)
    print(vec_unit)

    trials_lat = coordgeometry.find_non_parallel_int_vecs(
        max_mill, tile=True).T
    print('trials_lat: ', trials_lat[:,2])
    
    if vec_type == 'direction':
        trials_cart = lat @ trials_lat
        trials_cart_unit = trials_cart / np.linalg.norm(trials_cart, axis=0)

    elif vec_type == 'plane-normal':
        trials_cart_unit = plane_normal(
            trials_lat, latt_vecs=lat.T, normed=True)
        
        # lattice.plane_normal(np.array([[0,0,1]]).T, M_mon, 'monoclinic', latt_params_mzro2, degrees=True)
        
        print('trials_cart_unit: \n', trials_cart_unit)
    diff = trials_cart_unit - vec_unit
    diff_mag = np.linalg.norm(diff, axis=0)
    print('diff: ', diff)

    # Get indices of trial lattice vectors which are within tolerance
    good_diff_idx = np.where(diff_mag < tol_dist)[0]

    # Get lattice vectors within tolerance
    good_mill = trials_lat[:, good_diff_idx]

    if good_mill.shape[1] == 0:
        raise ValueError(
            'Could not find Miller indices within specified angular tolerance.')

    # Find the angles of lattice vectors within tolerance
    good_angs = 2 * np.arcsin(diff_mag[good_diff_idx] / 2)
    if degrees_out:
        good_angs = np.rad2deg(good_angs)

    # Order good lattice vetors by difference magnitude
    srt_idx = np.argsort(good_angs)
    good_mill_srt = good_mill[:, srt_idx]
    good_angs_srt = good_angs[srt_idx]

    return (good_mill_srt, good_angs_srt)




    
