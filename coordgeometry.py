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
