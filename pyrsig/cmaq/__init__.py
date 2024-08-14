__all__ = ['open_mfioapi', 'pair_rsigcmaq']

from .pair import pair_rsigcmaq
import pandas as pd


def save_ioapi(ds, path, format='NETCDF3_CLASSIC', **kwds):
    """
    Providing a function to clean-up meta-data for IOAPI.

    Arguments
    ---------
    ds : xr.Dataset
        Dataset that should be saved as IOAPI. Dimensions and coordinates
        must support the conversion.
    path : str
        Path to save ioapi file
    format : str
        'NETCDF3_CLASSIC' or 'NETCDF4_CLASSIC'
    kwds :
        Passed to xr.Dataset.to_netcdf

    Returns
    -------
    ds.to_netcdf
        Saved file
    """
    from .. import __version__
    import pandas as pd
    import xarray as xr
    import numpy as np

    ods = ds[[]].copy(deep=True)
    props = ods.attrs
    props.update(ds.attrs)
    outkeys = [
        vk for vk, vv in ds.data_vars.items()
        if 'PERIM' in vv.dims or 'ROW' in vv.dims
    ]
    nv = len(outkeys)
    if 'ROW' in ds[outkeys[0]].dims:
        ods.attrs['FTYPE'] = 1
    elif 'PERIM' in ds[outkeys[0]].dims:
        ods.attrs['FTYPE'] = 2

    assert len(set([k[:16] for k in outkeys])) == nv
    varlist = ''.join([k[:16].ljust(16) for k in outkeys])
    dates = pd.to_datetime(ds.TSTEP.values)
    if len(dates) == 1:
        if 'TSTEP' in ds.attrs:
            # A bit circular here, but parsing TSTEP as definitive when there
            # is no time coordinate.
            tstep = ds.attrs['TSTEP']
            tstepstr = f'{tstep:06d}'
            dts = int(tstepstr[-2:])
            dtm = int(tstepstr[-4:-2])
            dth = int(tstepstr[:-4])
            dt = dth * 3600 + dtm * 60 + dts
        else:
            # Assume 24h, which is like typical rsig
            dt = 24 * 3600
    else:
        dt = np.diff(dates).astype('d').max() / 1e9

    dth = dt // 3600
    dtm = (dt % 3600) // 60
    dts = (dt % 60) // 1
    tstepstr = f'{dth:.0f}{dtm:02.0f}{dts:02.0f}'
    timec = pd.to_datetime(
        ds.TSTEP.min().values
        + np.arange(len(dates)) * pd.to_timedelta(dt, unit='s')
    )
    jdays = timec.strftime('%Y%j').astype('i')
    hms = timec.strftime('%H%M%S').astype('i')
    ods['TFLAG'] = xr.DataArray(
        np.array([jdays, hms]).T, name='TFLAG', dims=('TSTEP', 'DATE-TIME'),
        attrs=dict(
            long_name='TFLAG'.ljust(16), units='<YYYYJJJ,HHMMSS>',
            var_desc='Time flag'.ljust(80)
        )
    ).expand_dims(VAR=nv).transpose('TSTEP', 'VAR', 'DATE-TIME')
    ods.attrs['SDATE'] = int(ods['TFLAG'][0, 0, 0])
    ods.attrs['STIME'] = int(ods['TFLAG'][0, 0, 1])
    ods.attrs['TSTEP'] = int(tstepstr)

    for dk in outkeys:
        ok = dk[:16]
        ods[ok] = ds[dk].astype('f')
        vprops = ods[ok].attrs
        vprops['long_name'] = vprops.get('long_name', ok)[:16].ljust(16)
        vprops['var_desc'] = vprops.get('var_desc', ok)[:80].ljust(80)
        vprops['units'] = vprops.get('units', 'unknown')[:16].ljust(16)
        ods[ok].encoding.update(ds[dk].encoding)

    now = pd.to_datetime('now', utc=True)
    props['CDATE'] = props['WDATE'] = int(now.strftime('%Y%j'))
    props['CTIME'] = props['WTIME'] = int(now.strftime('%H%M%S'))
    props['NCOLS'], props['NROWS'] = ds.sizes['COL'], ds.sizes['ROW']
    props['NLAYS'], props['NVARS'] = ds.sizes['LAY'], ods.sizes['VAR']
    props['XORIG'] = float(props['XORIG'] + ds.COL.min() - 0.5)
    props['YORIG'] = float(props['YORIG'] + ds.COL.min() - 0.5)
    s = [1.]
    for i, sm in enumerate(ods.LAY):
        s.append(2 * sm - s[-1])

    props['VAR-LIST'] = varlist
    props['VGLVLS'] = np.array(s, dtype='f')
    props['UPNAM'] = f'pyrsig {__version__}'.ljust(16)
    defprops = {
        'EXECUTION_ID': '????'.ljust(80),
        'IOAPI_VERSION': 'not applicable'.ljust(16), 'EXEC_ID': '?'.ljust(80),
        'FTYPE': 1, 'NTHIK': 1, 'GDTYP': 2, 'P_ALP': 33.0, 'P_BET': 45.0,
        'P_GAM': -97.0, 'XCENT': -97.0, 'YCENT': 40.0,
        'VGTYP': -9999, 'VGTOP': np.float32(5000.0),
        'GDNAM': f'{"UNKNOWN":16s}'.ljust(16), 'metadata': '',
        'FILEDESC': ''.ljust(80 * 60), 'UPNAM': 'pyrsig'.ljust(16),
        'HISTORY': 'pyrsig.cmaq.save_ioapi'.ljust(80 * 60)
    }
    ods.attrs.update(props)
    for pk, pdef in defprops.items():
        ods.attrs.setdefault(pk, pdef)

    coords = list(ods.coords)
    odds = ods.drop_indexes(coords).reset_coords(coords, drop=True)
    return odds.to_netcdf(path, format=format, **kwds)


def open_ioapi(path, metapath=None, earth_radius=6370000., **kwds):
    """
    Open an IOAPI file, add coordinate data, and optionally add RSIG metadata.

    Arguments
    ---------
    path : str
        Path to IOAPI formatted files.
    metapath : str
        Path to metadata associated with the RSIG query. The metadata will be
        added as metadata global property.
    earth_radius : float
        Assumed radius of the earth. 6370000 is the WRF default.
    kwds : mappable
        Passed to xr.open_dataset

    Returns
    -------
    ds : xarray.Dataset
        Dataset with IOAPI metadata
    """
    import xarray as xr
    kwds.setdefault('engine', 'netcdf4')
    f = xr.open_dataset(path, **kwds)
    f = add_ioapi_meta(
        f, path=path, metapath=metapath, earth_radius=earth_radius
    )
    return f


def add_ioapi_meta(ds, metapath=None, earth_radius=6370000., path=''):
    """
    Open an IOAPI file, add coordinate data, and optionally add RSIG metadata.

    Arguments
    ---------
    ds : xr.Dataset
        IOAPI dataset Path to IOAPI formatted files.
    metapath : str
        Path to metadata associated with the RSIG query. The metadata will be
        added as metadata global property.
    earth_radius : float
        Assumed radius of the earth. 6370000 is the WRF default.

    Returns
    -------
    outds : xarray.Dataset
        Dataset with IOAPI metadata
    """
    from ..utils import get_proj4
    import numpy as np
    import warnings
    f = ds
    try:
        tflag = f['TFLAG'].astype('i').values[:, 0, :]
        yyyyjjj = tflag[:, 0]
        yyyyjjj = np.where(yyyyjjj < 1, 1970001, yyyyjjj)
        HHMMSS = tflag[:, 1]
        tstrs = []
        for j, t in zip(yyyyjjj, HHMMSS):
            tstrs.append(f'{j:07d}T{t:06d}')

        time = pd.to_datetime(tstrs, format='%Y%jT%H%M%S')
        f.coords['TSTEP'] = time
    except Exception:
        pass

    if 'VGLVLS' in f.attrs:
        lvls = f.attrs['VGLVLS']
        if len(lvls) > 1:
            f.coords['LAY'] = (lvls[:-1] + lvls[1:]) / 2.
        else:
            f.coords['LAY'] = [np.mean(lvls)]

    nrs = [f.sizes.get('ROW', None), f.attrs.get('NROWS', None)]
    for nr in nrs:
        if nr is not None:
            f.coords['ROW'] = np.arange(f.attrs['NROWS']) + 0.5
            break

    ncs = [f.sizes.get('COL', None), f.attrs.get('NCOLS', None)]
    for nc in ncs:
        if nc is not None:
            f.coords['COL'] = np.arange(f.attrs['NCOLS']) + 0.5
            break

    try:
        proj4str = get_proj4(f.attrs, earth_radius=earth_radius)
        f.attrs['crs_proj4'] = proj4str
    except ValueError as e:
        warnings.warn(str(e))

    if metapath is None:
        import os
        if os.path.exists(path + '.txt'):
            metapath = path + '.txt'

    if metapath is False:
        metapath = None

    if metapath is not None:
        with open(metapath, 'r') as metaf:
            metatxt = metaf.read()
        f.attrs['metadata'] = metatxt

    return f


def open_mfioapi(
    paths, metapaths=None, earth_radius=6370000., **kwargs
):
    """
    Minimal version of open_mfdataset that is compatible with open_ioapi.
    preprocess :  keyword defaults to add_ioapi_meta
    concat_dim :  keyword defaults to 'TSTEP'

    Arguments
    ---------
    paths : iterable
        Paths to ioapi files to be opened.
    metapaths : iterable
        Paths to be added as a string metadata
    earth_radius : float
        Radius of the earth for projection.
    kwargs :
        See xr.open_mfdataset

    Returns
    -------

    """
    import xarray as xr
    import functools

    addio = functools.partial(add_ioapi_meta, earth_radius=earth_radius)
    kwargs.setdefault('concat_dim', 'TSTEP')
    kwargs.setdefault('preprocess', addio)
    outf = xr.open_mfdataset(paths, **kwargs)
    if metapaths is None:
        metapaths = []
    elif isinstance(metapaths, str):
        metapaths = [metapaths]

    metastr = ''
    for metapath in metapaths:
        with open(metapath, 'r') as mf:
            metastr += metapath + ':\n' + mf.read()

    outf.attrs['metadata'] = metastr

    return outf
