import numpy as np
import lattice
import projections
from matplotlib import pyplot as plt

def plot_pole_fig(proj_poles, crys=None, pole=None, lattice_sys=None):
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
    pole : string
        If `crys`='poly', specify pole to be plotted as string of Miller(-Bravais)
        indices (for example '001').  
    lattice_sys : string, optional
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal, 
        orthorhombic, monoclinic, triclinic.

    
    Returns
    -------
    f : matplotlib figure

    Notes
    -----
    We can either plot multiple poles for a single crystal or 

    TODO:
    - Label plot based on lattice system.
    - Add option for proj_poles to be ndarray.

    """

    all_crys = ['single', 'poly']
    if crys not in all_crys:
        raise ValueError('"{}" is not a valid crystal type. '
                             '`crys` must be one of: {}.'.format(
                                 crys, all_crys))
    
    if crys == 'poly' and  not pole:
        raise ValueError('Please specify which pole is to be plotted using' 
        'Miller(-Bravais) indices (for example \'001\').')


    f,ax = plt.subplots(1, 1, subplot_kw={'polar':True},figsize=(5,5))
    ax.scatter(proj_poles[0],proj_poles[1])

    ax.set_title("Stereographic projection", va='bottom')
    ax.set_rmax(1)
    if lattice_sys != 'hexagonal':
        ax.set_xticklabels(['', '', '[010]', '', '', '', '', ''])
        ax.annotate('[001]', xy=(0, 0), xytext=(10.3, 0.15))
    else:
        ax.set_xticklabels([])
        ax.annotate('[0001]', xy=(0, 0), xytext=(10.3, 0.15))
    
    ax.set_yticklabels([])
    
    return f
