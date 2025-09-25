__all__ = ['emg', 'fitemg1d', 'fitemg']
__doc__ = """
emg Satellite-derived Emissions
===============================

Exponentially Modified Gaussian (emg) Satellite-derived Emissions is designed
to estimate emissions from satellite data following many previous studies based
on Bierle et al.

Functions:
- to_polar(x, y, degrees=False)
- to_cartesian(r, alpha, theta=0, degrees=False)
- rotate_array(a, theta, degrees=False)
- emg(x, alpha, x0, sigma, mu, beta, return_parts=False)
- fitemg1d(y, dx, verbose=0, addvldhat=True)
- fitemg(vcds, us, vs, degrees=False, dx=1, nxplume...)
"""


def to_polar(x, y, degrees=False):
    """
    Convert cartesian (x, y) to polar (r, alpha)
    https://en.wikipedia.org/wiki/List_of_common_coordinate_transformations
    #From_Cartesian_coordinates

    Arguments
    ---------
    x, y : float or np.ndarray
        x and y coordinates
    degrees : bool
        If True, then alpha returned is provided in degrees.
        If False, then alpha returned is provided in radians.

    Returns
    -------
    r, alpha : float or np.ndarray
        radius and angle in polar coordinates

    Example
    -------
    import numpy as np
    from pyrsig.emiss._emg import to_polar
    us = np.array([1, 0, -1, 0, 1, -1])
    vs = np.array([0, 1, 0, -1, 1, -1])
    ws, wd = to_polar(us, vs, degrees=True)
    print(ws)
    #[   1.    1.    1.    1.   1.41421356 1.41421356]
    print(wd)
    # [   0.   90.  180.  -90.   45. -135.]
    """
    import numpy as np
    x = np.asarray(x)
    y = np.asarray(y)
    alpha = np.arctan2(y.astype('f'), x.astype('f'))
    r = (x**2 + y**2)**.5
    if degrees:
        alpha = np.degrees(alpha)

    return r, alpha


def to_cartesian(r, alpha, theta=0, degrees=False):
    """
    Calculate the (x, y) with an optional rotation defined by theta.
    https://en.wikipedia.org/wiki/Rotation_of_axes

    Arguments
    ---------
    r : float or np.ndarray
        radius of polar coordinate
    alpha : float or np.ndarray
        angle (in radians) of polar coordinate
    theta : float
        angle to rotate for the cartesian plane
    degrees : bool
        If True, then alpha and theta are provided in degrees.
        If False, then alpha and theta are provided in radians.

    Returns
    -------
    x, y : np.ndarray
        cartesian coordinates with optional rotation

    Example
    -------
    import numpy as np
    from pyrsig.emiss._emg import to_cartesian
    x, y = to_cartesian(np.sqrt(2), np.radians(45))
    x, y
    # (1.0000000000000002, 1.0)
    x, y = to_cartesian(1, 45, -45, degrees=True)
    x, y
    # (2.220446049250313e-16, 1.0)
    """
    import numpy as np
    alpha = np.asarray(alpha)
    theta = np.asarray(theta)
    if degrees:
        alpha = np.radians(alpha)
        theta = np.radians(theta)
    x = r * np.cos(alpha)
    y = r * np.sin(alpha)
    xp = x * np.cos(theta) + y * np.sin(theta)
    yp = -x * np.sin(theta) + y * np.cos(theta)
    return xp, yp


def rotate_array(a, theta, degrees=False):
    """
    Arguments
    ---------
    a : np.ndarray
        symmetric array (n, n)
    theta : float
        Direction (in degrees) to rotate symmetric array
    degrees : bool
        If True, then theta is in degrees

    Returns
    -------
    b : np.ndarray
        Rotated symmetric grid (n, n)

    Example
    -------
    import numpy as np
    from pyrsig.emiss._emg import rotate_array
    tmpl = np.eye(5)
    tmpl[:2] = 0
    rots = np.zeros(tmpl.shape)
    vcds = np.zeros(tmpl.shape)
    for v in [1, -1]:
      for u in [1, -1]:
        a = tmpl[::v, ::u]
        d = np.degrees(np.arctan2(v, u)) # v, u or y, x
        b = rotate_array(a, -d, degrees=d)
        vcds += a
        rots += b
        print(f'# u={u:+.0f};v={v:+.0f};d={d}')

    print('# Sum Originals:')
    print('#' + np.array2string(np.flipud(vcds)).replace('\n', '\n#'))
    print('# Sum Rotated:')
    print('#' + np.array2string(np.flipud(rots)).replace('\n', '\n#'))
    # u=+1;v=+1;d=45.0
    # u=-1;v=+1;d=135.0
    # u=+1;v=-1;d=-45.0
    # u=-1;v=-1;d=-135.0
    # Sum Originals:
    #[[1. 0. 0. 0. 1.]
    # [0. 1. 0. 1. 0.]
    # [0. 0. 4. 0. 0.]
    # [0. 1. 0. 1. 0.]
    # [1. 0. 0. 0. 1.]]
    # Sum Rotated:
    #[[0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]
    # [0. 0. 4. 4. 4.]
    # [0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]]
    """
    import numpy as np
    ny, nx = a.shape
    assert (ny % 2 == 1)
    assert (nx % 2 == 1)

    n = a.shape[0] // 2
    if degrees:
        theta = np.radians(theta)
    j, i = np.indices(a.shape)

    y = j - n
    x = i - n

    r, alpha = to_polar(x, y)
    # Calculate the (x', y') in rotation
    xp, yp = to_cartesian(r, alpha, theta)
    # xp[n, n] = 0
    # yp[n, n] = 0
    # Sample at the closest grid cell
    ip = np.minimum(n, np.maximum(
        -n, np.ma.masked_invalid(np.round(xp))
    )).astype('i').filled(0) + n
    jp = np.minimum(n, np.maximum(
        -n, np.ma.masked_invalid(np.round(yp))
    )).astype('i').filled(0) + n
    return a[jp, ip]


def emg(x, alpha, x0, sigma, mu, beta, return_parts=False):
    """
    Eq 1 of Goldberg et al.[1] as described

    e = 1/x0 exp(mu/x0 + sigma^2 / (2 x0^2) - x / x0)
    G = F((x - mu)/sigma - sigma/x_o)
    f = e * G
    y = alpha f + beta

    Note: F is the Gaussian cumulative distribution function"

    [1] Goldberg et al. doi: 10.5194/acp-22-10875-2022

    Arguments
    ---------
    x : array
        x0 is the e-folding distance downwind, representing the length scale
        of the NO2 decay
    alpha : float
        Mass distributed across distance
    sigma : float
        sigma is the standard deviation of the Gaussian function, representing
        the Gaussian smoothing length scale
    mu : float
        mu is the location of the apparent source relative to the assumed
        pollution source center
    beta : float
        Background line density
    return_parts : bool
        If False (default), return final output.
        If True, return parts for further analysis

    Returns
    -------
    out : array or tuple
        If False (default), return y (alpha * e * G + beta)
        If True, return alpha, e, G, beta and alpha * e * G + beta

    Example
    -------
    import numpy as np
    from pyrsig.emiss._emg import emg
    # Approximation of data in Goldberg[1] Figure 8a inset
    # g/y y/h * h molNOx/gNOx NO2/NOx [=] mol
    approx_alpha = 62e9 / 8760 * 1.7  / 46 / 1.32
    approx_x0 = 20e3 # iteratively identified
    exparams = {
        'alpha': approx_alpha, 'x0': approx_x0,
        'sigma': 28e3, 'mu': -8000.,
        'beta': 2.2,
    }
    x = np.arange(-75e3, 105e3, 5000)
    y = emg(x, **exparams)
    print(np.array2string(np.round(x / 1000, 1).astype('i')))
    print(np.array2string(np.round(y, 1)))
    # [-75 -70 -65 -60 -55 -50 -45 -40 -35 -30 -25 -20 -15 -10  -5   0   5  10
    #   15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90  95 100]
    # [2.3 2.3 2.3 2.4 2.5 2.6 2.7 2.9 3.1 3.3 3.6 3.8 4.1 4.3 4.4 4.5 4.6 4.6
    #  4.6 4.4 4.3 4.1 3.9 3.7 3.5 3.3 3.1 2.9 2.8 2.7 2.6 2.5 2.4 2.4 2.4 2.3]
    """
    import numpy as np
    from scipy.stats import norm
    e = 1 / x0 * np.exp((mu / x0) + (sigma**2 / (2 * x0**2)) - x / x0)
    # e = np.where(x < mu, 0, e)
    G = norm.cdf((x - mu) / sigma - sigma / x0)
    pdf = e * G
    # precisely equal to scipy equivalent
    # from scipy.stats import exponnorm
    # pdf = exponnorm(K=x0 / sigma, loc=mu, scale=sigma).pdf(x)
    out = alpha * pdf + beta
    if return_parts:
        return alpha, e, G, beta, out
    else:
        return out


def fitemg1d(vld, dx, verbose=0, addvldhat=True):
    """
    Arguments
    ---------
    vld : np.array
        shape (n,) is the line density assuming focal point at n // 2
    dx : float
        Width in length units (consistent with units of b, eg, b [=] mol/m2
        then dx [=] m)
    addvldhat : bool
        If True (default), add the fit line (vldhat) to the output

    Returns
    -------
    out : dict
        x, vld, params (from fitemg1d), and (optionally) vldhat

    Example
    -------
    from pyrsig.emiss import fitemg1d
    from scipy.stats import exponnorm
    # Approximation of data in Goldberg[1] Figure 8a inset
    # g/y y/h * h molNOx/gNOx NO2/NOx [=] mol
    approx_alpha = 62e9 / 8760 * 1.7  / 46 / 1.32
    approx_x0 = 20e3 # iteratively identified
    alpha = 198155
    x0 = 20e3
    sigma = 28e3
    mu = -8000.
    beta = 2.2
    x = np.arange(-75e3, 156e3, 5e3)
    f = exponnorm(K=approx_x0 / sigma, loc=mu, scale=sigma).pdf(x)
    y = approx_alpha * f + beta
    yerr = np.random.normal(size=y.size) * 0.1
    yfit = fitemg1d(y, dx=1e3)
    print(np.array2string(np.round(x / 1000, 1).astype('i')))
    print(np.array2string(np.round(y, 1)))
    print(np.array2string(np.round(yfit['vldhat'], 1)))
    # [-75 -70 -65 -60 -55 -50 -45 -40 -35 -30 -25 -20 -15 -10  -5   0   5  10
    #   15  20  25  30  35  40  45  50  55  60  65  70  75  80  85  90  95 100
    #  105 110 115 120 125 130 135 140 145 150 155]
    # [2.3 2.3 2.3 2.4 2.5 2.6 2.7 2.9 3.1 3.3 3.6 3.8 4.1 4.3 4.4 4.5 4.6 4.6
    #  4.6 4.4 4.3 4.1 3.9 3.7 3.5 3.3 3.1 2.9 2.8 2.7 2.6 2.5 2.4 2.4 2.4 2.3
    #  2.3 2.3 2.3 2.2 2.2 2.2 2.2 2.2 2.2 2.2 2.2]
    # [2.3 2.3 2.3 2.4 2.5 2.6 2.7 2.9 3.1 3.3 3.6 3.8 4.1 4.3 4.4 4.5 4.6 4.6
    #  4.6 4.4 4.3 4.1 3.9 3.7 3.5 3.3 3.1 2.9 2.8 2.7 2.6 2.5 2.4 2.4 2.4 2.3
    #  2.3 2.3 2.3 2.2 2.2 2.2 2.2 2.2 2.2 2.2 2.2]
    """
    from scipy.optimize import curve_fit
    import numpy as np
    ny = vld.size
    assert (ny % 2 == 1)
    n = ny // 2
    x = (np.arange(ny)[:] - n) * dx
    # fitting requires a reasonable starting guess
    dist = np.abs(x[-1] - x[0])
    sigma0 = dist / 3
    x00 = sigma0 / 2
    mu0 = -2 * np.float32(dx)
    pdf_max = emg(x, alpha=1, x0=x00, sigma=sigma0, mu=mu0, beta=0).max()
    # should this be divided by (pdf_max * dx)?
    alpha0 = (vld.max() - vld.min()) / pdf_max
    beta0 = vld.min()
    guess = np.array([alpha0, x00, sigma0, mu0, beta0], dtype='f')
    if verbose > 0:
        print('p0', dict(zip('alpha x0 sigma mu beta'.split(), guess)))
    lb = np.array([1e-1, 1e-1, 1e-1, x.min(), 0])
    ub = np.array([1e10, x.max() * 4, x.max() * 4, x.max(), vld.max()])
    pfit, pcov = curve_fit(
        emg, x, vld, p0=guess, bounds=(lb, ub), check_finite=False
    )
    pfit = dict(zip('alpha x0 sigma mu beta'.split(), pfit))
    out = {'x': x, 'params': pfit}
    if addvldhat:
        out['vldhat'] = emg(x, **pfit)
    return out


def fitemg(
    vcds, us=None, vs=None, degrees=False, dx=1, nxplume=None, verbose=0
):
    """
    Arguments
    ---------
    vcds : xr.DataArray
        Vertical column densities 3d (t, y, x) where (x, y) must be square
        (x == y). Units seem to work best when mol/m2. The area unit must
        match the length unit of dx.
    us : array-like
        1d (t,) U component of wind in the x direction at center x/y
    vs : array-like
        1d (t,) V component of wind in the y direction at center x/y
    dx : float
        edge length in units commensurate with vcds (typically m)
    nxplume : int
        Number of cells to sum in the cross-plume direction after rotation when
        calculating the line density (mol/m) along the plume. Defaults to the
        half width minus 1.
    degrees : bool
        Perform calculations of wind direction in degrees (default=True)

    Returns
    -------
    fitresult : dict
        vldhat: best fit function
        ws: wind speeds
        wd: wind directions
        wsmean: mean of wind speeds
        rots: rotated rasters
    """
    import numpy as np
    from warnings import warn
    import copy
    import xarray as xr

    dims = ('time', 'y', 'x')
    attrs = copy.deepcopy({k: v for k, v in vcds.attrs.items()})
    vcds = np.ma.masked_invalid(vcds.data)
    if vcds.ndim == 2:
        # add a third dimension
        vcds = vcds[np.newaxis, :, :]
    elif vcds.ndim > 3:
        # reduces higher dimensions to a single third dimension
        vcds = vcds.reshape(-1, *vcds.shape[-2:])
    if vcds.ndim != 3:
        raise ValueError('dimension must be 2 or higher')

    if us is None or vs is None:
        rots = vcds
        if us is not None or vs is not None:
            warn('if us or vs is None, the other is ignored')
    else:
        us = np.asarray(us)
        vs = np.asarray(vs)
        rots = np.full_like(vcds, np.nan)
        wss, wds = to_polar(us, vs, degrees=degrees)
        for ti, (ws, wd, a) in enumerate(zip(wss, wds, vcds)):
            rots[ti] = rotate_array(a, -wd, degrees=degrees)

    rot = rots.mean(0)
    ny, nx = rot.shape
    if nx != ny:
        msg = 'nx and ny must be equal for rotation.'
        raise ValueError(msg)
    n = ny // 2
    if nxplume is None:
        nxplume = n - 1

    sliceacross = slice(
        max(0, n - nxplume), min(ny, n + nxplume + 1)
    )
    vld = rot[sliceacross, :].sum(0) * dx
    out = fitemg1d(vld, dx=dx, verbose=verbose)
    outf = xr.Dataset()
    outf.coords['x'] = out['x']
    outf.coords['y'] = out['x']
    outf['ROTATED_VCD'] = dims, rots, attrs
    outf['vld'] = ('x',), vld
    outf['vldhat'] = ('x',), emg(out['x'], **out['params']), out['params']
    outf['ws'] = ('time',), wss, dict(units='m/s')
    wdunits = {True: 'degrees', False: 'radians'}[degrees]
    outf['wd'] = ('time',), wds, dict(units=wdunits)
    tattrs = dict(
        units='s', description='vldhat.x0 / ws.mean()', long_name='tau'
    )
    outf['tau'] = (), out['params']['x0'] / wss.mean(), tattrs
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    emis_ps = out['params']['alpha'] / outf['tau'].data
    eattrs = dict(
        long_name='emis_no2', units='molNO2/s',
        description=(
            'emis_no2 = vldhat.alpha / tau = vldhat.alpha * ws.mean() '
            '/ vldhat.x0'
        ),
        comment='emis_nox = emis_no2 * NOx/NO2 (e.g., 1.32)',

    )
    outf['emis_no2'] = (), emis_ps, eattrs
    return outf
