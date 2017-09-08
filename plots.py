import numpy as np
import lattice
import projections
from matplotlib import pyplot as plt
from matplotlib import cm


def plot_pole_fig(proj_poles, poles, crys=None,  lattice_sys=None, axes='xyz',
                grid=False):
    """
    Return a figure object for a pole figure.

    Parameters
    ----------
    proj_poles : tuple of ndarrays of shape (n,)
        Arrays of `n` polar angles and radii as projections of poles.
    crys : string
        Specifies type of crystal: 'single' or 'poly'. For 'single' crystal, 
        `proj_poles` expects the projections of pole(s) in a single crystal 
        aligned with the sample coordinate system. For a 'poly' crystal, 
        `proj_poles` expects the projections of a given pole in a set of crystals.
    poles : ndarray of shape (3, n)
        Specify poles to be plotted as column vectors.
    lattice_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal, 
        orthorhombic, monoclinic, triclinic.
    axes  : string
        Set alignment of sample axes with projection sphere axes. Options:
        'xyz' (default); 'yzx'; 'zxy'; 'yxz'; 'zyx'; 'xzy'.
    grid : bool
        Turn grid lines on plot on or off (default).

    Returns
    -------
    f : matplotlib figure

    Notes
    -----
    We can either plot multiple poles for a single crystal or 

    TODO:
    - Sort out plot labelling: based on lattice system for single crystal, and
        based on plotted pole for polycrystal.
    - Think about whether to have more than one pole figure for a single crystal 
    as well.
    - Add check the lenght of proj_poles = number of poles given.
    - Add label with plot details: phase, projection type, upper hemisphere, 
    data plotted.

    """

    all_crys = ['single', 'poly']
    if crys not in all_crys:
        raise ValueError('"{}" is not a valid crystal type. '
                         '`crys` must be one of: {}.'.format(
                             crys, all_crys))

    # Check valid entry for axes alignment
    all_axes = ['xyz', 'yzx', 'zxy', 'yxz', 'zyx', 'xzy']
    if axes not in all_axes:
        raise ValueError('"{}" is not a valid axes option. '
                         '`axes` must be one of: {}.'.format(
                             axes, all_axes))

    if crys == 'single':
        # proj_poles = proj_poles[0]
        f, ax = plt.subplots(1, 1, subplot_kw={'polar': True}, figsize=(5, 5))
        ax.scatter(proj_poles[0][0], proj_poles[0][1])

        ax.set_title("Stereographic projection", va='bottom')
        ax.set_rmax(1)
        if lattice_sys != 'hexagonal':
            ax.set_xticklabels(['', '', '[010]', '', '', '', '', ''])
            ax.annotate('[001]', xy=(0, 0), xytext=(10.3, 0.15))
        else:
            ax.set_xticklabels([])
            ax.annotate('[0001]', xy=(0, 0), xytext=(10.3, 0.15))

        ax.set_yticklabels([])
        if not grid:
                ax.yaxis.grid(False)
                ax.xaxis.grid(False)

    elif crys == 'poly':
        n_figs = len(proj_poles)

        # get poles labels
        poles_lbl = []
        for i in range(n_figs):
            poles_lbl.append(''.join([str(x) for x in poles[:,i]]))

        f = plt.figure(1, figsize=(10, 10))
        for n in range(n_figs):
            ax = f.add_subplot(1, n_figs, n+1, projection='polar')
            cax = ax.scatter(proj_poles[n][0], proj_poles[n][1], cmap=cm.hsv,s=0.005)
            ax.set_rmax(1)
            ax.set_xticklabels([axes[0].upper(), '', axes[1].upper(), '', '', '', '', ''])
            ax.set_yticklabels([])
            ax.set_title("{" + poles_lbl[n] + "}", va='bottom')
            if not grid:
                ax.yaxis.grid(False)
                ax.xaxis.grid(False)

    return f
