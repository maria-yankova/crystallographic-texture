import numpy as np

from crystex import coordgeometry
from crystex import rotations
from crystex import lattice
from crystex import symmetry
from crystex import numutils

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
    # print('ρ, θ, φ: ', ρ, θ, φ)
    # Rotate poles in South hemisphere to North
    for i, phi in enumerate(φ):
        if φ[i] > np.pi / 2:
            φ[i] = np.abs(φ[i] - np.pi)
            θ[i] = θ[i] + np.pi

    np.putmask(θ, (θ-np.pi) >= -1e-5, θ - 2 * np.pi)  # wrap angles in (-π, π)
    R = 2 * np.sin(φ / 2) / np.sqrt(2)

    # if np.isclose((R - 1), 0.0) and (θ - 0.0) < -1e-8:
    #     θ *= -1
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
    # print('ρ, θ, φ: ', ρ, θ, φ)
    # Rotate poles in South hemisphere to North
    for i, phi in enumerate(φ):
        if φ[i] > np.pi / 2:
            φ[i] = np.abs(φ[i] - np.pi)
            θ[i] = θ[i] + np.pi

    np.putmask(θ, θ >= np.pi, θ - 2 * np.pi)  # wrap angles in (-π, π)
    R = np.tan(φ / 2)

    return θ, R


def project_crystal_poles(poles,  eulers=None, rot_mat=None, proj_type=None, 
                            lattice_sys=None, latt_params=None,
                            pole_type=None, degrees=False, align='cz', crys=None,
                            axes='xyz', ret_poles=False,
                            user_rot=None, apply_sym=False, ret_sym_sep=False):
    """
    Project a set of crystal poles specified using Miller(-Bravais) indices.

    Parameters
    ----------
    poles : ndarray of shape (3 or 4, n)
        Array of poles given in Miller (Miller-Bravais for 'hexagonal'
        lattice system) indices as column vectors.
    rot_mat : ndarray of shape (n,3,3)
        Array of `n` rotation matrices for conversion from crystal to sample
        coordinate system. See notes. Note, specify either `rot_mat` or `eulers`.
    eulers : ndarray of shape (N, 3), optional
        An array of Euler angles using Bunge (zx'z") convention (φ1, Φ, φ2) in degrees.
        Note, specify either `rot_mat` or `eulers`.
    proj_type: string
        Projection type is 'stereographic' or 'equal_area'.
    lattice_sys : string, optional
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
        where * refers to reciprocal lattice vectors.
    crys : string
        Specifies type of crystal: 'single' or 'poly'. For 'single' crystal,
        `proj_poles` expects the projections of pole(s) in a single crystal
        aligned with the sample coordinate system. For a 'poly' crystal,
        `proj_poles` expects the projections of a given pole in a set of crystals.
    axes  : string
        Set alignment of sample axes with projection sphere axes. Options:
        'xyz' (default); 'yzx'; 'zxy'; 'yxz'; 'zyx'; 'xzy'.
    ret_poles : bool, optional (default False)
        Optionally, return the 3d pole vectors before they are projected.
    user_rot : list 
        A rotation axis and angle in degrees defined by the user. 
        Example: [[0,1,0], 90]
    apply_sym : bool
        Apply symmetry (optional). Works for cubic, hexagonal and monoclinic only.

    Returns
    -------
    proj_poles : list of tuples of ndarrays of shape (n,)
        Arrays of polar angles and radii as projections of `n` poles.

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
    If rotation matrices are obtained from euler angles in Bunge convention,
    the inverse rotation matrices are needed to rotate from crystal to sample
    reference frame.

    References
    ----------
    [1] Giacovazzo et al.(2002) Fundamentals of Crystallography. Oxf Univ Press. p. 75-76.

    TODO:
    - Add 'direction' pole option for polycrystal.
    """
    # Check valid entries for crystal type and rot_mat
    all_crys = ['single', 'poly']
    if crys not in all_crys:
        raise ValueError('"{}" is not a valid crystal type. '
                         '`crys` must be one of: {}.'.format(
                             crys, all_crys))

    # Check valid entry for rot_mat and eulers
    if crys == 'poly':
        if rot_mat is not None and eulers is not None:
            raise ValueError(
                'Specify either `rot_mat` or `eulers` but not both.')
        elif eulers is not None:
            rot_mat = np.linalg.inv(
                rotations.euler2rot_mat_n(eulers, degrees=True))
        elif rot_mat is None and eulers is None:
            raise ValueError('Please specify `rot_mat` or `eulers` corresponding to '
                             'orientation of individual poles in polycrystal.')
    elif crys == 'single' and rot_mat:
        raise ValueError('"{}" and "{}" is not a valid set of options for `crys`'
                         ' and `rot_mat`. Specify either `crys`=\'single\' or '
                         '`crys`=\'poly\' and `rot_mat`'.format(crys, rot_mat))
    
    sym_ops = [np.eye(3)]
    
    # Apply symmetry 
    if apply_sym:
        if lattice_sys=='hexagonal':
            sym_ops = symmetry.SYM_OPS['6/mmm'] # 12 symmetry operators
            # vals_true = np.array([False,False,False,True,True,True,True,True,True,False,False,False,])
            # sym_ops = np.array(sym_ops)[vals_true]
        elif lattice_sys=='monoclinic':
            sym_ops = symmetry.SYM_OPS['2/m'] # 2 symmetry operators
        elif lattice_sys=='cubic':
            sym_ops = symmetry.SYM_OPS[lattice_sys]
        elif lattice_sys=='tetragonal':
            sym_ops = symmetry.SYM_OPS[lattice_sys]
        elif lattice_sys=='triclinic':
            sym_ops = symmetry.SYM_OPS[lattice_sys]
        else:
            raise ValueError('Symmetry operators only implemented for cubic, tetragonal, hexagonal and monoclinic crystals')
    # print('len(sym_ops)', len(sym_ops))
    # Check valid entry for projection type
    proj_opt = {'stereographic': stereographic_proj,
                'equal_area': equal_area_proj}
    if proj_type in proj_opt:
        project = proj_opt[proj_type]
    else:
        raise ValueError('"{}" is not a valid projection type. '
                         '`proj_type` must be one of: {}.'.format(
                             proj_type, proj_opt.keys()))

    # Check valid entry for axes alignment
    all_axes = ['xyz', 'yzx', 'zxy', 'yxz', 'zyx', 'xzy']
    if axes not in all_axes:
        raise ValueError('"{}" is not a valid axes option. '
                         '`axes` must be one of: {}.'.format(
                             axes, all_axes))
    else:
        R_ax = rotations.rotate_axes(axes)
    # Find user defined rotation matrix
    if user_rot:
        R_usr = rotations.ax_ang2rot_mat(
            np.array(user_rot[0]), user_rot[1], degrees=True)

    if latt_params:
        params_dict = {'a': latt_params[0],
                       'b': latt_params[1],
                       'c': latt_params[2],
                       'α': latt_params[3],
                       'β': latt_params[4],
                       'γ': latt_params[5]}
        M = lattice.crystal2ortho(**params_dict, normed=False,
                                  degrees=degrees, align=align)
    else:
        M = lattice.crystal2ortho(lattice_sys, normed=True,
                                  degrees=degrees, align=align)

    cell_e = np.eye(3)
    
    # Crystal vectors in orthonormal basis as column vectors
    cell_ortho = np.dot(M.T, cell_e)
    # print('cell_ortho: ', cell_ortho)

    pole_types = ['direction', 'plane-normal']
    proj_poles = []
    all_ppp = []
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
        
        # Reciprocal lattice vectors for poles (column vectors)
        g_poles_sym = []
        for p in poles.T:
            g_poles = np.dot(cell_rec, p)
            g_poles_sym.append((sym_ops @ g_poles).T)
            # print('p: ', p)
            # print('g_poles shape: ', g_poles.shape)
            # print('symm gpoles: ', (sym_ops @ g_poles).T)
        if crys == 'poly':
            for g_i in range(len(g_poles_sym)):
                pp = (rot_mat @ g_poles_sym[g_i]).T
                # print('g_i: ', g_i)
                ppp = np.squeeze(R_ax, axis=0) @ pp
                if user_rot:
                    ppp = np.squeeze(R_usr, axis=0) @ ppp
                proj_p = []
                for ps in range(ppp.shape[0]):
                    proj_p.append(project(ppp[ps]))
                if ret_sym_sep:
                    proj_poles.append(proj_p)
                else:
                    proj_poles.append(np.concatenate(proj_p, axis=1))
                all_ppp.append(ppp)
                
        else:
            for g_i in range(len(g_poles_sym)):
                proj_poles.append(project(g_poles_sym[g_i]))
        # g_poles = np.dot(cell_rec, poles)
        # if crys == 'poly':
        #     for g_i in range(g_poles.shape[1]):
        #         pp = np.dot(rot_mat, g_poles[:, g_i]).T
        #         print(pp.shape)
        #         ppp = np.dot(np.squeeze(R_ax, axis=0), pp)
        #         if user_rot:
        #             ppp = np.dot(np.squeeze(R_usr, axis=0), ppp)
        #         proj_poles.append(project(ppp))
        # else:
        #     proj_poles.append(project(g_poles))

    elif pole_type == 'direction':
        # rot_mat_sym = []
        # for sym in sym_ops:
        #     rot_mat_sym.append(sym @ rot_mat)
        # rot_mat = np.reshape(np.array(rot_mat_sym), (len(sym_ops)*rot_mat.shape[0], 3, 3))
        # Convert Miller-Bravais to Miller indices
        if lattice_sys == 'hexagonal' and poles.shape[0] == 4:
            poles = lattice.miller_brav2miller(poles, idx_type='direction')
        
        d_poles_sym = []
        for p in poles.T:
            poles_sym = np.dot(sym_ops, p).T
            d_poles_sym.append(np.dot(M.T, poles_sym))
        
        # print('d_poles_sym: ', d_poles_sym, '\n\n')    
        # for p in poles.T:
        #     poles_sym = np.dot(sym_ops, p).T
        # d_poles = np.dot(M.T, poles)
        # print(rot_mat)
        if crys == 'poly':
            
            for d_i in range(len(d_poles_sym)):
                pp = (rot_mat @ d_poles_sym[d_i]).T
                ppp = np.squeeze(R_ax, axis=0) @ pp
                if user_rot:
                    ppp = np.squeeze(R_usr, axis=0) @ ppp
                proj_p = []
                for ps in range(ppp.shape[0]):
                    proj_p.append(project(ppp[ps]))
                proj_poles.append(np.concatenate(proj_p, axis=1))
                all_ppp.append(ppp)
            # for d_i in range(d_poles.shape[1]):
            #     pp = np.dot(rot_mat, d_poles[:, d_i]).T
            #     ppp = np.dot(np.squeeze(R_ax, axis=0), pp)
            #     if user_rot:
            #         ppp = np.dot(np.squeeze(R_usr, axis=0), ppp)
            #     proj_poles.append(project(ppp))
        else:
            proj_poles.append(project(d_poles))

    # if np.isclose(proj_poles[2][1],0.2, rtol=2e-03):

    if ret_poles:
        return proj_poles, all_ppp
    else:
        return proj_poles


def bin_proj_poles(proj_poles, bins=50, normed=True):
    """
    Compute the histograms of projected poles data points.

    Parameters
    ----------
    proj_poles : list of tuples of ndarrays of shape (n,)
        Arrays of polar angles and radii as projections of `n` poles.
    bins : int, optional
        Number of bins. Default is 50.
    normed : bool, optional
        If False, returns the number of samples in each bin. If True, returns 
        the bin density bin_count / sample_count / bin_area

    Returns
    -------
    Hs : list of ndarrays, shape (bins, bins)
        List of `n` histograms calculated for a grid in Cartesian coordinates.
    xedges : ndarray
        The bin edges along the first dimension.
    yedges : ndarray
        The bin edges along the second dimension.

    """
    Hs = []

    for p_i, p in enumerate(proj_poles):

        # Convert polar to Cartesian coordinates
        x, y = coordgeometry.polar2cart_2D(p[1], p[0] + np.pi)

        # Calculate bidimensional histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, normed=normed)

        Hs.append(np.flipud(np.fliplr(H)))

    return Hs, xedges, yedges
