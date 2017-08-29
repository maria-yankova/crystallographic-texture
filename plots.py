import numpy as np
import lattice
import projections
from matplotlib import pyplot as plt

def plot_pole_fig(proj_poles, lattice_sys):
    """
    Return a figure object for a pole figure.abs

    Parameters
    ----------
    proj_poles : tuple of ndarrays of shape (n,)
        Arrays of `n` polar angles and radii as projections of poles.
    lattice_sys : string
        Lattice system is one of cubic, hexagonal, rhombohedral, tetragonal, 
        orthorhombic, monoclinic, triclinic.
    
    Returns
    -------
    f : matplotlib figure

    TODO:
    - Label plot based on lattice system.
    - Add option for proj_poles to be ndarray.

    """

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
