import numpy as np
import coordgeometry
import rotations
import lattice


def equal_area_proj(xyz):
    """
    Returns the normalised equal-area projection of `xyz`, which is an array of column vectors. 

    Parameters
    ----------
    xyz : ndarray
        An array of column vectors in Cartesian coordinates.

    Returns
    -------
    Tuple : angles, radii

    Notes
    -----


    """

    # old version:
    # spc = cart2spherical(xyz)
    # # Rotate poles in South hemisphere to North
    # for i,phi in enumerate(spc[2]):
    #     if spc[2][i] > np.pi/2:
    #         spc[2][i] = np.abs(spc[2][i] - np.pi)
    #         spc[1][i] = spc[1][i] + np.pi
    # return spc[1],2*np.sin(spc[2]/2)/np.sqrt(2)

    xyz = xyz / np.linalg.norm(xyz, axis=0)

    ρ, θ, φ = coordgeometry.cart2spherical(xyz)  # radius, azimuthal, polar

    # Rotate poles in South hemisphere to North
    for i, phi in enumerate(φ):
        if φ[i] > np.pi / 2:
            φ[i] = np.abs(φ[i] - np.pi)
            θ[i] = θ[i] + np.pi

    np.putmask(θ, θ >= np.pi, θ - 2 * np.pi)  # wrap angles in (-π, π)
    R = 2 * np.sin(φ / 2) / np.sqrt(2)

    return θ, R


def stereographic_proj(xyz):
    """
    Returns the normalised stereographic projection [theta,r] of `xyz`, which is an array of column vectors.

    Parameters
    ----------
    xyz : ndarray
        An array of column vectors in Cartesian coordinates.

    Returns
    -------
    Tuple : angles, radii
    """

    xyz = xyz / np.linalg.norm(xyz, axis=0)

    ρ, θ, φ = coordgeometry.cart2spherical(xyz)  # radius, azimuthal, polar

    # Rotate poles in South hemisphere to North
    for i, phi in enumerate(φ):
        if φ[i] > np.pi / 2:
            φ[i] = np.abs(φ[i] - np.pi)
            θ[i] = θ[i] + np.pi

    np.putmask(θ, θ >= np.pi, θ - 2 * np.pi)  # wrap angles in (-π, π)
    R = np.tan(φ / 2)

    return θ, R


def ploject_crystal_poles(poles, proj_type=None, lattice_sys=None, latt_params=None,
                         pole_type=None, degrees=False, align='cz'):
    """
    Project a set of ctystal poles.
    
    Parameters
    ----------
    poles : ndarray of shape (3 or 4, n)
        Array of poles given in Miller (Miller-Bravais for 'hexagonal' 
        lattice system) indices as column vectors.
    proj_type: string
        Projection type is 'stereographic' or 'equal_area'.
    lattice_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal, 
        orthorhombic, monoclinic, triclinic.
    latt_params : list of lenght 6
        Lattice parameters. The fist three represent the magnitude of each of the lattice vectors.
        If all three are None, a = b = c = 1. 
        If all three angles are None, example angles sets are used as described in Notes.
    degrees : bool, optional
        Units of `α`, `β`, `γ`. Radians by default.a
    align : string, optional
        Alignment option between crystal and orthonormal reference frames. 
        Three options implemented (as described in [1]): 
        - 'ax': a-axis || x-axis and c*-axis || z*-axis
        - 'by': b-axis || y-axis and a*-axis || x*-axis
        - 'cz': c-axis || z-axis and a*-axis || x*-axis [Default]
        where * corresponds to reciprocal lattice vectors.
    
    Returns
    -------
    proj_poles : tuple of ndarrays of shape (n,)
        Arrays of `n` polar angles and radii as projections of poles.
        
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
    """
    proj_opt = { 'stereographic' : stereographic_proj,
                   'equal_area' : equal_area_proj }
    if proj_type in proj_opt:
        project = proj_opt[proj_type]
    else:
        raise ValueError('"{}" is not a valid projection type. '
                             '`proj_type` must be one of: {}.'.format(
                                 proj_type, proj_opt.keys()))

    if latt_params:
        params_dict = { 'a' : latt_params[0],
                    'b' : latt_params[1], 
                    'c' : latt_params[2], 
                    'α' : latt_params[3], 
                    'β' : latt_params[4], 
                    'γ' : latt_params[5] }
        M = lattice.crystal2ortho(lattice_sys, **params_dict, normed=True,
                                degrees=degrees, align=align)
    else:
        M = lattice.crystal2ortho(lattice_sys, normed=True,
                                degrees=degrees, align=align)

    cell_e = np.array([[1,0,0], [0,1,0], [0,0,1]]).T

    # Crystal vectors in orthonormal basis as column vectors
    cell_ortho = np.dot(M.T, cell_e)

    pole_types = ['direction', 'plane-normal']
    if pole_type not in pole_types:
        raise ValueError('"{}" is not a valid `pole_type` option. '
                             '`pole_type` must be one of: {}.'.format(
                                 pole_type, pole_types))
    
    elif pole_type == 'plane-normal':

        # Convert Miller-Bravais to Miller indices
        if lattice_sys == 'hexagonal' and poles.shape[0] == 4:
            poles = lattice.miller_brav2miller(poles, idx_type='plane')

        # Reciprocal lattice vectors in orthonormal basis
        cell_rec = lattice.reciprocal_lattice_vecs(cell_ortho)
        M_rec = cell_rec        # Column vectors of reciprocal a,b,c
        # Reciprocal lattice vectors for poles (column vectors)
        g_poles = np.dot(cell_rec, poles)

        proj_poles = project(g_poles)
    
    elif pole_type == 'direction':

        # Convert Miller-Bravais to Miller indices
        if lattice_sys == 'hexagonal' and poles.shape[0] == 4:
            poles = lattice.miller_brav2miller(poles, idx_type='direction')

        d_poles = np.dot(M.T, poles)

        proj_poles = project(d_poles)

    return proj_poles
