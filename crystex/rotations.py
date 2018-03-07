import numpy as np

from crystex import coordgeometry


def ax_ang2rot_mat(axes, angles, degrees=False):
    """
    Generates pre-multiplication rotation matrices for given axes and angles.

    Parameters
    ----------
    axes : ndarray
        Array of shape (N, 3), which if N is 1, will be tiled to the size
        (M, 3). Otherwise, N must be equal to M (for M, see `angles`).
    angles : ndarray
        Array of shape (M).
    degrees : bool (optional)
        If True, `angles` interpreted as degrees.

    Returns
    -------
    ndarray of shape (N or M, 3, 3).

    Notes
    -----
    Computed using the Rodrigues' rotation formula.

    Examples
    --------

    Find the rotation matrix for a single axis and angle:

    >>> ax_ang2rot_mat(np.array([[0,0,1]]), np.array([np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]]])

    Find the rotation matrices for different angles about the same axis:

    >>> ax_ang2rot_mat(np.array([[0,0,1]]), np.array([np.pi/4, -np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.70710678,  0.        ],
            [-0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]]])

    Find the rotation matrices about different axes by the same angle:

    >>> ax_ang2rot_mat(np.array([[0,0,1], [0,1,0]]), np.array([np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.        ,  0.70710678],
            [ 0.        ,  1.        ,  0.        ],
            [-0.70710678,  0.        ,  0.70710678]]])

    Find the rotation matrices about different axes and angles:

    >>> ax_ang2rot_mat(
        np.array([[0,0,1], [0,1,0]]), np.array([np.pi/4, -np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.        , -0.70710678],
            [ 0.        ,  1.        ,  0.        ],
            [ 0.70710678,  0.        ,  0.70710678]]])

    """

    # Check dimensions

    if axes.ndim == 1:
        axes = axes[np.newaxis]

    angles_err_msg = '`angles` must be a number or array of shape (M,).'

    if isinstance(angles, np.ndarray):
        if angles.ndim != 1:
            raise ValueError(angles_err_msg)

    else:
        try:
            angles = np.array([angles])

        except ValueError:
            print(angles_err_msg)

    if axes.shape[0] == angles.shape[0]:
        n = axes.shape[0]
    else:
        if axes.shape[0] == 1:
            n = angles.shape[0]
        elif angles.shape[0] == 1:
            n = axes.shape[0]
        else:
            raise ValueError(
                'Incompatible dimensions: the first dimension of `axes` or'
                '`angles` must be one otherwise the first dimensions of `axes`'
                'and `angles` must be equal.')

    # Convert to radians if necessary
    if degrees:
        angles = np.deg2rad(angles)

    # Normalise axes to unit vectors:
    axes = axes / np.linalg.norm(axes, axis=1)[:, np.newaxis]

    cross_prod_mat = np.zeros((n, 3, 3))
    cross_prod_mat[:, 0, 1] = -axes[:, 2]
    cross_prod_mat[:, 0, 2] = axes[:, 1]
    cross_prod_mat[:, 1, 0] = axes[:, 2]
    cross_prod_mat[:, 1, 2] = -axes[:, 0]
    cross_prod_mat[:, 2, 0] = -axes[:, 1]
    cross_prod_mat[:, 2, 1] = axes[:, 0]

    rot_mats = np.tile(np.eye(3), (n, 1, 1)) + (
        np.sin(angles)[:, np.newaxis, np.newaxis] * cross_prod_mat) + (
            (1 - np.cos(angles)[:, np.newaxis, np.newaxis]) * (
                cross_prod_mat @ cross_prod_mat))

    return rot_mats


def rotmat2axang(rot_mat, degrees=False):
    """
    Convert a rotation matrix into axis-angle representation.

    Parameters
    ----------
    rot_mat : ndarray of shape (3, 3)
        Rotation matrix which pre-multiplies column vectors.
    degrees : bool, optional
        If True, returns angle in degrees, otherwise in radians.

    Returns
    -------
    tuple of (axis, angle)
        axis : ndarray of shape (3,)
        angle : float

    Notes
    -----
    Following similar function in Matlab from here:
    https://github.com/marcdegraef/3Drotations/blob/master/src/MatLab/om2ax.m 

    TODO:
    - Understand `P` factor in  http://doi.org/10.1088/0965-0393/23/8/083501
      and apply here if necessary.
    - Vectorise
    - Add unit tests

    """

    tol = 1e-7

    # Check dimensions
    if rot_mat.ndim == 3:
        rot_mat = rot_mat.squeeze()

    trc = np.trace(rot_mat)
    ang = np.arccos(np.clip(0.5 * (trc - 1), -1, 1))

    if np.isclose(ang, 0.0):

        # Set axis to [001] if angle is 0.
        ax = np.array([0, 0, 1])

    else:

        # Find eigenvalues, eigenvectors for `rot_mat`
        eigval, eigvec = np.linalg.eig(rot_mat)

        # Get index of eigenvalue which is 1 + 0i
        eig_cond = np.logical_and(
            abs(np.real(eigval) - 1) < tol,
            abs(np.imag(eigval) < tol)
        )

        if np.sum(eig_cond) != 1:
            raise ValueError(
                'Not exactly one eigenvector with eigenvalue of 1 found. '
                'Eigenvalues are: {}'.format(eigval))

        # Set the axes to eigenvector with eigenvalue = 1
        # Determine the sign of each component using the conditions below.
        ax = np.real(eigvec[:, np.where(eig_cond)[0]]).squeeze()

        if (rot_mat[2, 1] - rot_mat[1, 2]) != 0:
            s = np.sign(rot_mat[2, 1] - rot_mat[1, 2])
            ax[0] = s * abs(ax[0])

        if (rot_mat[0, 2] - rot_mat[2, 0]) != 0:
            s = np.sign(rot_mat[0, 2] - rot_mat[2, 0])
            ax[1] = s * abs(ax[1])

        if (rot_mat[1, 0] - rot_mat[0, 1]) != 0:
            s = np.sign(rot_mat[1, 0] - rot_mat[0, 1])
            ax[2] = s * abs(ax[2])

        if degrees:
            ang = np.degrees(ang)

    return ax, ang


def euler2rot_mat_n(angles, degrees=False):
    """
    Converts sets of Euler angles in Bunge convention to rotation matrices.

    Parameters
    ----------
    angles : ndarray
        An array of shape (N, 3) of Euler angles using Bunge (zx'z") 
        convention (φ1, Φ, φ2).
    degrees : bool
        Specify whether units of angles are radians (default) or degrees. 

    Returns
    -------
    ndarray
        An array of shape (N, 3, 3).

    Notes
    -----
    Angular ranges are φ1: [0, 2π], Φ: [0, π], φ2: [0, 2π].
    By definition the Euler angles in Bunge convention represent a rotation of
    the reference frame, i.e. a passive transformation. ax_ang2rot_mat() is
    constructed for an active rotations and therefore we apply the rotations 
    of opposite sign and in the opposite order (-φ2, -Φ, -φ1) [1].

    [1] Rowenhorst et al. (2015) 23(8), 83501.
        doi.org/10.1088/0965-0393/23/8/083501

    """

    # Add dimension for 1D
    if angles.ndim == 1:
        angles = angles[np.newaxis]

    # Degrees option
    if degrees:
        angles = np.radians(angles)

    #  Euler angles:
    φ1 = angles[:, 0]
    Φ = angles[:, 1]
    φ2 = angles[:, 2]

    Rz_phi1 = ax_ang2rot_mat(np.array([[0, 0, 1]]), -angles[:, 0])
    Rx_Phi = ax_ang2rot_mat(np.array([[1, 0, 0]]), -angles[:, 1])
    Rz_phi2 = ax_ang2rot_mat(np.array([[0, 0, 1]]), -angles[:, 2])

    return Rz_phi2 @ Rx_Phi @ Rz_phi1


def rot_mat2euler(R):
    """
    Converts a rotation matrix to a set of three Euler angles using Bunge
    (zx'z") convention (φ1, Φ, φ2).

    Parameters
    ----------
    R : ndarray
        An array of size (3,3) repesenting a rotation matrix.

    Returns
    -------
    ndarray
        An array of size (1,3) with the Euler angles according to Bunge
        convention (φ1, Φ, φ2).

    Notes
    -----
    Angular ranges are φ1: [0, 2π], Φ: [0, π], φ2: [0, 2π]
    Not vectorised yet - only works for a single rotation matrix.

    If cos φ1 = c1, cos Φ = c2, cos φ2 = c3, 
    sin φ1 = s1, sin Φ = s2, sin φ2 = s3
    The angles are derived from the rotation matrix:
    R = [[ c1c3 - s1c2s3,   s1c3 + c1c2s3,   s2s3 ]
         [ - c1s3 - s1c2c3, - s1s3 + c1c2c3, s2c3 ]
         [ s1s2,            - c1s2,          c2   ]]

    """

    φ1 = np.arctan2(R[2, 0], -R[2, 1])
    Φ = np.arccos(R[2, 2])
    φ2 = np.arctan2(R[0, 2], R[1, 2])

    return np.array([φ1, Φ, φ2])


def rotate_axes(axes):
    """
    Notes:
    Convention for an active rotation used: looking down the axis of rotation,
    counter-clockwise rotation is > 0, and clockwise < 0.

    TODO:
    - Add all options: ['xyz', 'yzx', 'zxy', 'yxz', 'zyx', 'xzy']
    """
    if axes == 'xyz':
        Rtot = np.eye(3, 3)[np.newaxis]

    elif axes == 'yzx':
        R1 = ax_ang2rot_mat(np.array([0, 1, 0]), -90.0, degrees=True)
        R2 = ax_ang2rot_mat(np.array([0, 0, 1]), -90.0, degrees=True)
        Rtot = R2 @ R1

    elif axes == 'zxy':
        R1 = ax_ang2rot_mat(np.array([1, 0, 0]), 90.0, degrees=True)
        R2 = ax_ang2rot_mat(np.array([0, 0, 1]), 90.0, degrees=True)
        Rtot = R2 @ R1

    elif axes == 'yxz':
        R1 = ax_ang2rot_mat(
            np.array([0, 1, 0]), -180.0, degrees=True)
        R2 = ax_ang2rot_mat(np.array([0, 0, 1]), 90.0, degrees=True)
        Rtot = R2 @ R1

    elif axes == 'zyx':
        R1 = ax_ang2rot_mat(np.array([0, 1, 0]), 90.0, degrees=True)
        Rtot = R1

    elif axes == 'xzy':
        R1 = ax_ang2rot_mat(np.array([1, 0, 0]), -90.0, degrees=True)
        Rtot = R1

    return Rtot


def rotate_eulers(val, ang, data_set, degrees=True):
    """
    Add or substract a rotation of a given value to a given Euler angle 
    in a data set.

    Parameters
    ----------
    val : float
        Value by which to change `ang`. Positive value will be added and negative
        subtracted.
    ang : string 
        Euler angle to be changed. Allowed values based on Bunge convention:
        'phi1', 'Phi', 'phi2', 'φ1', 'Φ', 'φ2'.
    data_set : ndarray of shape (N, 3)
         An array of Euler angles using Bunge (zx'z") convention (φ1, Φ, φ2).
    degrees : bool, optional (default True)
        Units of angles. Note `ang` and `data_set` must have the same units.
    Returns
    -------


    """
    if ang == 'phi1' or ang == 'φ1':
        ang_idx = 0
    elif ang == 'Phi' or ang == 'Φ':
        ang_idx = 1
    elif ang == 'phi2' or ang == 'φ2':
        ang_idx = 2

    data_set[:, ang_idx] = data_set[:, ang_idx] + val

    return data_set
