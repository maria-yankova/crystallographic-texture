import numpy as np
from numpy import linalg as la


def zero_prec(A, tol=1e-12):
    """
    Sets elements of an array within some tolerance `tol` to 0.0.

    """
    A[abs(A) < tol] = 0.0
    return A


def plane(x, y, a, b, c):
    z = a * x + b * y + c
    return z


def fit_plane(points):
    """
    Fit points to an equation of a plane using singular value decompositon.

    We aim to find the parameters (a, b, c) in the equation of a plane: 
    `ax + by + c = z`. We can do this by setting up a system of linear
    equations: `AX = B`, where `A` is an (N, 3) array whose first two columns
    are the `x` and `y` data, `X` is a (3, 1) array containing the parameters
    to fit (a, b, c), and `B` is an (N, 1) array containing the `z` data. We
    solve by taking the generalised inverse: 

        `X = ginv(A) @ B = inv(A.T @ A) @ A.T @ B`.

    Parameters
    ----------
    points : ndarray of shape (3, N)

    Returns
    -------
    tupe of (fit, mean)
        fit : ndarray of shape (3,)
            Contains the fitted plane parameters (a, b, c)
        mean : ndarray of shape (3, 1)
            The position mean of the points.

    """

    # First subtract the mean of the points:
    points_mean = np.mean(points, axis=1)[:, np.newaxis]
    points = points - points_mean

    # Fit a plane:
    A = np.hstack([
        points[0:2].T,
        np.ones((points.shape[1], 1))
    ])
    B = points[2:3].T
    X = np.linalg.inv(A.T @ A) @ A.T @ B
    X = X.squeeze()

    return X, points_mean


def get_from_dict(d, address=None):
    """
    Return the value in a nested dict given by a list of keys.

    Parameters
    ----------
    d : dict
        Nested dict to search for a given value, which will be returned.
    address : list of str, optional
        Address of dict value to search for. If not assigned, will return np.array(`d`)

    Returns
    -------
    ndarray

    """

    if address is None:
        address = []

    out = d
    for i in address:
        out = out.get(i)
        if out is None:
            break

    return out


def index_lst(lst, idx):
    """Return indexed elements of a list."""
    return [lst[i] for i in idx]


def col_wise_dot(a, b):
    """ Compute the dot product between columns of a and columns of b."""

    return np.einsum('ij,ij->j', a, b)


def col_wise_angles(a, b, degrees=False):
    """ a, b are 3 x N arrays whose columns represented vectors to find the angles betweeen."""

    A = la.norm(np.cross(a, b, axis=0), axis=0)
    B = col_wise_dot(a, b)
    angles = np.arctan2(A, B)

    if degrees:
        angles = np.rad2deg(angles)

    return angles
