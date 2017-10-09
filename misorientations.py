import coordgeometry
import projections
import numpy as np


def fibre_misorientation(fibre, pole_data, lattice_system, latt_params, axes, mask):
    """
    Find the misorientation away from a fibre texture.

    """


    cart_3dpoles = projections.ploject_crystal_poles(fibre, crys='poly', eulers=pole_data,
                                                proj_type='equal_area', lattice_sys=lattice_system, 
                                                latt_params=latt_params, pole_type='plane-normal', 
                                                degrees=True, axes=axes, ret_poles=True)[1]
    
    sp_3dpoles = coordgeometry.cart2spherical(cart_3dpoles)

    # Calculate fibre misorientation angle [0°, 90°]
    mis_ang = abs(sp_3dpoles[2])
    mis_ang[np.where(mis_ang > np.pi/2)] = np.pi - mis_ang[np.where(mis_ang > np.pi/2)]
    mis_ang = np.ma.array(mis_ang, mask=mask)
    weights = 1/(np.sin(mis_ang.compressed()))

    return mis_ang, weights
