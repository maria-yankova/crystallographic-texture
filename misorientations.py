import coordgeometry
import projections
import numpy as np


def fibre_misorientation(fibre, euler_data, lattice_system, latt_params,
                         axes='xyz', mask=None, user_rot=None):
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
    """

    cart_3dpoles = projections.ploject_crystal_poles(fibre, crys='poly', eulers=euler_data,
                                                     proj_type='equal_area', lattice_sys=lattice_system,
                                                     latt_params=latt_params, pole_type='plane-normal',
                                                     degrees=True, axes=axes, ret_poles=True,
                                                     user_rot=user_rot)[1]

    sp_3dpoles = coordgeometry.cart2spherical(cart_3dpoles)

    # Calculate fibre misorientation angle [0°, 90°]
    mis_ang = abs(sp_3dpoles[2])
    mis_ang[np.where(mis_ang > np.pi / 2)] = np.pi - \
        mis_ang[np.where(mis_ang > np.pi / 2)]
    mis_ang = np.ma.array(mis_ang, mask=mask)
    weights = 1 / (np.sin(mis_ang.compressed()))

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
    ax_ang_12 = rotations.rotmat2ax_ang(rot_12.squeeze(), degrees=True)

    if ax_ang:
        return ax_ang_12
