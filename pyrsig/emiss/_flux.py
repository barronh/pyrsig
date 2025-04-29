def divergence(c, dy, dx, order=4, withdiag=True):
    """
    calculate gradient of c in w-e and s-n directions
    and optionally in the ne-sw and se-nw diagonals

    Arguments
    ---------
    c : array
        2-dimensional array of column densities
    dy : float or array
        pixel edge length in the y direction
    dx : float or array
        pixel edge length in the x direction

    Returns
    -------
    dcdx, dcdy[, dcdr, dcds]: tuple
        tuple of 2 or 4 divergence

    Notes
    -----
    Adapted from https://github.com/Kang-Sun-CfA/Oversampling_matlab/blob
                        /v0.2/popy.py#L1711C1-L1732C1
    """
    import numpy as np
    c = np.asarray(c)
    if np.isscalar(dx):
        dx = np.broadcast_to(dx, c.shape)
    if np.isscalar(dy):
        dy = np.broadcast_to(dy, c.shape)
    dcdx = np.full_like(c, np.nan)
    dcdy = np.full_like(c, np.nan)
    if withdiag:
        dd = (dx**2 + dy**2)**.5
        dcdr = np.full_like(c, np.nan)
        dcds = np.full_like(c, np.nan)
    if order == 2:
        dcdx[:, 1:-1] = (c[:, 2:] - c[:, 0:-2]) / (2 * dx[:, 1:-1])
        dcdy[1:-1, :] = (c[2:, :] - c[0:-2, :]) / (2 * dy[1:-1, :])
        if withdiag:
            dcdr[1:-1, 1:-1] = (
                (c[2:, 2:] - c[0:-2, 0:-2]) / (2 * dd[1:-1, 1:-1])
            )
            dcds[1:-1, 1:-1] = (
                (c[2:, 0:-2] - c[0:-2, 2:]) / (2 * dd[1:-1, 1:-1])
            )
    elif order == 4:
        dcdx[:, 2:-2] = (
            (-c[:, 4:] + 8 * c[:, 3:-1] - 8 * c[:, 1:-3] + c[:, 0:-4])
            / (12 * dx[:, 2:-2])
        )
        dcdy[2:-2, :] = (
            (-c[4:, ] + 8 * c[3:-1, ] - 8 * c[1:-3, ] + c[0:-4, ])
            / (12 * dy[2:-2, :])
        )
        if withdiag:
            dcdr[2:-2, 2:-2] = (
                (
                    -c[4:, 4:] + 8 * c[3:-1, 3:-1]
                    - 8 * c[1:-3, 1:-3] + c[0:-4, 0:-4]
                ) / (12 * dd[2:-2, 2:-2])
            )
            dcds[2:-2, 2:-2] = (
                (
                    -c[4:, 0:-4] + 8 * c[3:-1, 1:-3]
                    - 8 * c[1:-3, 3:-1] + c[0:-4, 4:]
                ) / (12 * dd[2:-2, 2:-2])
            )
    else:
        raise KeyError('order must be 2 or 4')

    if withdiag:
        return dcdx, dcdy, dcdr, dcds
    else:
        return dcdx, dcdy
