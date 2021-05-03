import numpy as np

from crystex import coordgeometry
from crystex import lattice
from crystex import projections
from crystex import rotations
from crystex.symmetry import SYM_OPS


def fibre_misorientation(fibre, euler_data, lattice_system, latt_params,
                         axes='xyz', mask=None, user_rot=None, ret_poles=True):
    """
    Find the misorientation away from a fibre texture.

    Parameters
    ----------
    fibre : ndarray of shape (3, 1)
        Fibre pole specified as a column vector.
    euler_data : ndarray of shape (N, 3)
        Euler angles data for N number of pixels.
    lattice_system : string
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal,
        orthorhombic, monoclinic, triclinic.
    latt_params :  list of lenght 6
        Lattice parameters. The fist three represent the magnitude of each of 
        the lattice vectors and the second three the angles between these vectors
        in degrees.
    axes : string
        Set alignment of sample axes with projection sphere axes. Options:
        'xyz' (default); 'yzx'; 'zxy'; 'yxz'; 'zyx'; 'xzy'.
    mask : ndarray of bool, optional
        Mask to be applied to misorientation angles.
    user_rot : list 
        A rotation axis and angle in degrees defined by the user. 
        Example: [[0,1,0], 90]

    Returns
    -------
    mis_ang : ndarray
        Misorientation angles away from the given fibre in radians.
    weights : ndarray
        Weights for each misorientation angle calculated as 1 / sin(mis_ang).

    TODO:
    - Find why dimensions of cart_3dpoles have changed to (1, N, 3) instead of (N, 3)
    """

    cart_3dpoles = projections.project_crystal_poles(
        fibre, crys='poly', eulers=euler_data, proj_type='equal_area', lattice_sys=lattice_system,latt_params=latt_params, pole_type='plane-normal', degrees=True, axes=axes, ret_poles=True, user_rot=user_rot
    )[1][0]

    # Dimensions of cart_3dpoles are (1, N, 3)
    cart_3dpoles = cart_3dpoles.squeeze()
    sp_3dpoles = coordgeometry.cart2spherical(cart_3dpoles)

    # Calculate fibre misorientation angle [0°, 90°]
    mis_ang = abs(sp_3dpoles[2])
    mis_ang[np.where(mis_ang > np.pi / 2)] = np.pi - \
        mis_ang[np.where(mis_ang > np.pi / 2)]
    mis_ang = np.ma.array(mis_ang, mask=mask)
    weights = 1 / (np.sin(mis_ang.compressed()))

    if ret_poles:
        return mis_ang, weights, cart_3dpoles
    else:
        return mis_ang, weights


def misorientation_pair(eulers1, eulers2, degrees=True, ax_ang=True):
    """
    Calculate the misorientation between a pair of orientations given in Euler 
    angles in Bunge notation. Direction for rotation is 1 to 2.

    """

    gr1_rot = rotations.euler2rot_mat_n(eulers1)
    gr2_rot = rotations.euler2rot_mat_n(eulers2)

    gr1_inv_rot = np.linalg.inv(gr1_rot)
    gr2_inv_rot = np.linalg.inv(gr2_rot)

    rot_12 = gr2_rot @ gr1_inv_rot
    ax_ang_12 = rotations.rotmat2ax_ang(rot_12, degrees=True)

    if ax_ang:
        return ax_ang_12


def euler_pair_disorientation(eulers_A, eulers_B, point_group,
                              degrees_in=False, degrees_out=False):
    """
    Calculate disorientation axis and angle between a pair of orientations
    given in Euler angles in Bunge notation. Direction for rotation is A to B.

    Parameters
    ----------
    eulers_A : ndarray of shape (3,)
        See notes.
    eulers_B : ndarray of shape (3,)
        See notes.
    point_group : str
        One of "6/mmm", "cubic", "triclinic"
    degrees_in : bool, optional
        If True `eulers_A` and `eulers_B` values are interpreted as degrees, 
        otherwise as radians. False by default.
    degrees_out : bool, optional
        If True, disorientation angle is returned in degrees, otherwise in
        radians. False by default.

    Returns
    -------
    tuple of ndarray (axis_crystal, axis_sample, angle)
        axis_crystal : ndarray of shape (3, 1)
            Column vector representing disorientation axis within a local
            crystal Cartesian basis.
        axis_sample : ndarray of shape (3, 1)
            Column vector representing disorientation axes within a sample
            Cartesian basis.
        angle : float
            Disorientation angle in radians or degrees.

    Notes
    -----
    Euler angles are expected in Bunge notation (intrinsic, ZXZ passive 
    rotations, i.e. from sample to crystal coordinate system).

    """

    ax_crys, ax_samp, ang = euler_pair_disorientation_all(
        eulers_A, eulers_B, point_group, degrees_in, degrees_out)

    return (ax_crys[:, 0:1], ax_samp[:, 0:1], ang)


def euler_pair_disorientation_all(eulers_A, eulers_B, point_group,
                                  degrees_in=False, degrees_out=False):
    """
    Calculate all equivalent disorientations between a pair of orientations
    given in Euler angles in Bunge notation. Direction for rotation is A to B.

    Symmetrically equivalent axes may be returned for one disorientation
    angle.

    Parameters
    ----------
    eulers_A : ndarray of shape (3,)
        See notes.
    eulers_B : ndarray of shape (3,)
        See notes.
    point_group : str
        One of "6/mmm", "cubic", "triclinic"
    degrees_in : bool, optional
        If True `eulers_A` and `eulers_B` values are interpreted as degrees, 
        otherwise as radians. False by default.
    degrees_out : bool, optional
        If True, disorientation angle is returned in degrees, otherwise in
        radians. False by default.

    Returns
    -------
    tuple of ndarray (axis_crystal, axis_sample, angle)
        axis_crystal : ndarray of shape (3, N)
            Column vectors representing symmetrically-equivalent disorientation
            axes within a local crystal Cartesian basis.
        axis_sample : ndarray of shape (3, N)
            Column vectors representing symmetrically-equivalent disorientation
            axes within a sample Cartesian basis.
        angle : float
            Disorientation angle in radians or degrees.

    Notes
    -----
    Euler angles are expected in Bunge notation (intrinsic, ZXZ passive 
    rotations, i.e. from sample to crystal coordinate system).

    """
    sym_ops = SYM_OPS[point_group]

    rot_A = rotations.euler2rot_mat_n(eulers_A, degrees=degrees_in)[0]
    rot_B = rotations.euler2rot_mat_n(eulers_B, degrees=degrees_in)[0]

    # print(rotations.rotmat2axang(rot_A, degrees=True))
    # print(rotations.rotmat2axang(rot_B, degrees=True))

    num_ops = len(sym_ops)**2
    mis_rot = np.ones((num_ops, 3, 3)) * np.nan
    mis_ax_crys = np.ones((3, num_ops)) * np.nan
    mis_ax_samp = np.ones((3, num_ops)) * np.nan
    mis_ang = np.ones((num_ops,)) * np.nan

    all_sidx = [(i, j) for i in range(len(sym_ops))
                for j in range(len(sym_ops))]
    s_idx = 0

    for sym_A in sym_ops:

        inv_rot_A_sym = np.linalg.inv(sym_A @ rot_A)

        for sym_B in sym_ops:

            rot_B_sym = sym_B @ rot_B
            r_i = rot_B_sym @ inv_rot_A_sym
            ax_i, ang_i = rotations.rotmat2axang(r_i)

            mis_rot[s_idx] = r_i
            mis_ax_crys[:, s_idx] = ax_i
            mis_ax_samp[:, s_idx] = inv_rot_A_sym @ ax_i
            mis_ang[s_idx] = ang_i

            s_idx += 1

    if degrees_out:
        mis_ang = np.rad2deg(mis_ang)

    dis_idx = np.where(np.isclose(mis_ang, np.min(mis_ang)))[0]

    dis_rot = mis_rot[dis_idx]
    dis_ax_crys = mis_ax_crys[:, dis_idx]
    dis_ax_samp = mis_ax_samp[:, dis_idx]
    dis_ang = mis_ang[dis_idx]
    dis_all_sym_idx = [i for i_idx, i in enumerate(
        all_sidx) if i_idx in dis_idx]

    return (dis_ax_crys, dis_ax_samp, dis_ang[0], dis_all_sym_idx)


def gb_disorientation(eulers_A, eulers_B, point_group, lat, ang_tol,
                      max_mill=20, degrees_in=False, degrees_out=False):
    """
    Find the disorientation axis and angle of a grain boundary from the Euler
    angles of two grains.

    Parameters
    ----------
    eulers_A : ndarray of shape (3,)
        See notes.
    eulers_B : ndarray of shape (3,)
        See notes.
    point_group : str
        One of "6/mmm", "cubic", "triclinic"
    lat : ndarray of shape (3, 3)
        Column vectors representing the lattice unit cell 
    ang_tol : float
        Snapping angular tolerance of the disorientation axis.
    max_mill: int, optional
        Maximum Miller index. By default, set to 20.        
    degrees_in : bool, optional
        If True `eulers_A`, `eulers_B` and `ang_tol` values are interpreted as
        degrees, otherwise as radians. False by default.
    degrees_out : bool, optional
        If True, disorientation angle and axis deviation from returned Miller
        indices are returned in degrees, otherwise in radians. False by 
        default.

    Returns
    -------
    tuple of (ax_mill, ax_diff, ang)
        ax_mill : ndarray of int shape (3, 1)
            Integer column vector representing the disorientation axis in 
            Miller indices.
        ax_diff : float
            Angular deviation of the found lattice disorientation axis from the
            actual axis. In radians or degrees.
        ang : float
            Disorientation angle. In radians or degrees.

    Notes
    -----
    Euler angles are expected in Bunge notation (intrinsic, ZXZ passive 
    rotations, i.e. from sample to crystal coordinate system).

    """

    # Find the angle and disorientation axis in Cartesian coordinates
    ax_cart_crys, ax_cart_samp, ang, sym_idx = euler_pair_disorientation_all(
        eulers_A, eulers_B, point_group, degrees_in, degrees_out)

    # print('ax_cart: \n{}\n'.format(ax_cart))

    ax_mill_all = []
    ax_diff_all = []
    ax_cart_crys_all = []
    ax_cart_samp_all = []
    sym_idx_all = []

    for ax_idx in range(ax_cart_crys.shape[1]):

        # "Snap" the axis to the nearest lattice site within some angular tolerance
        i = ax_cart_crys[:, ax_idx:ax_idx + 1]
        j = ax_cart_samp[:, ax_idx:ax_idx + 1]
        k = sym_idx[ax_idx]

        ax_mill, ax_diff = lattice.cart2miller_all(
            i, lat, ang_tol, 'direction', max_mill, degrees_in, degrees_out)

        ax_cart_crys_all.append(np.tile(i, (1, ax_mill.shape[1])))
        ax_cart_samp_all.append(np.tile(j, (1, ax_mill.shape[1])))
        sym_idx_all.append(np.tile(k, (ax_mill.shape[1], 1)))

        ax_mill_all.append(ax_mill)
        ax_diff_all.append(ax_diff)

    ax_mill_all = np.hstack(ax_mill_all)
    ax_diff_all = np.hstack(ax_diff_all)
    ax_cart_crys_all = np.hstack(ax_cart_crys_all)
    ax_cart_samp_all = np.hstack(ax_cart_samp_all)
    sym_idx_all = np.vstack(sym_idx_all)

    sort_idx = np.argsort(ax_diff_all)
    ax_mill_all = ax_mill_all[:, sort_idx]
    ax_diff_all = ax_diff_all[sort_idx]
    ax_cart_crys_all = ax_cart_crys_all[:, sort_idx]
    ax_cart_samp_all = ax_cart_samp_all[:, sort_idx]
    sym_idx_all = sym_idx_all[sort_idx]

    sym_ops = np.array(SYM_OPS[point_group])[sym_idx_all[0]]

    ret = (
        ax_cart_crys_all[:, 0],
        ax_cart_samp_all[:, 0],
        ax_mill_all[:, 0],
        ax_diff_all[0],
        sym_ops,
        ang,
    )

    return ret
