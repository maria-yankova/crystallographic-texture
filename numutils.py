def zero_prec(A, tol=1e-12):
    """
    Sets elements of an array within some tolerance `tol` to 0.0.

    """
    A[abs(A) < tol] = 0.0
    return A
