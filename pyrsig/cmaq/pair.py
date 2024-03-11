__all__ = ['pair_rsigcmaq']
__doc__ = """
pair_rsigcmaq is designed to pair RSIG data with CMAQ files.

Example:

    from pair_rsigcmaq import pair_rsigcmaq
    import pandas a pd

    outpath = pair_rsigcmaq('CMAQ_ACONC_20160101', 'O3', 'airnow.ozone')
    df = pd.read_csv(outpath)
    # Print descriptive statistics
    print(df[['ozone', 'CMAQ_O3']].describe().to_csv())
    # Print correlation statistics
    print(df[['ozone', 'CMAQ_O3']].corr().to_csv())

More details:

    from pair_rsigcmaq import pair_rsigcmaq
    help(pair_rsigcmaq)

Or use as a script:

    python -m pyrsig.cmaq.pair --help
"""

import xarray as xr
import numpy as np
import pyrsig
import pandas as pd
import pyproj
import os
from os.path import basename
from argparse import RawTextHelpFormatter, ArgumentParser

_prsr_description = (
    """
Pair CMAQ with data in RSIG using time and horizontal position.
Seven examples are provided below:

Pair AirNow ozone, co, and no2:

%(prog)s airnow.ozone O3  CMAQv54_ACONC.202317*
%(prog)s airnow.co    CO  CMAQv54_ACONC.202317*
%(prog)s airnow.no2   NO2  CMAQv54_ACONC.202317*

Pair AirNow pm25 with an ELMO output:

%(prog)s airnow.pm25  PM25 CMAQv54_AELMO.202317*

Pair Pandora NO2, TropOMI NO2, or VIIRS AOD with a PHOTDIAG1 file:

%(prog)s pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount"""
    + """ NO2_COLUMN CMAQ_PHOTDIAG1.202317*
%(prog)s tropomi.nrti.no2.nitrogendioxide_tropospheric_column"""
    + """ NO2_COLUMN CMAQ_PHOTDIAG1.202317*
%(prog)s viirsnoaa.jrraod.AOD550 AOD_W550_ANGST CMAQ_PHOTDIAG1.202317*

For a full list of species that RSIG can pair, use the --help-rsig

%(prog)s --help-rsig
"""
)


def pair_rsigcmaq(
    qds, qkey, datakey, bdate=None, edate=None, bbox=None, prefix='CMAQ_',
    persist=False, verbose=0
):
    """
    Arguments
    ---------
    qds : xarray.Dataset or str
        If a str, it must be the path to CMAQ file.
        If an xarray.Dataset, it must have CMAQ structure and:
        - encoding['source'] string describing the file
        - crs_proj4 added by pyrsig.open_ioapi
    qkey : str
        CMAQ variable key (must be in the CMAQ file)
    datakey : str
        RSIG variable key
    bdate : datetime
        Starting date time. If None, infer from file.
    edate : datetime
        Ending date time. If None, infer from file.
    bbox : tuple
        ll_lon, ll_lat, ur_lon, ur_lat in decimal degrees East and North
        If bbox is None, it will be inferred from the file.
    persist : bool
        If True (default: False), save to disk and return output path
    verbose : int
        Level of verbosity. Negative omits warning about cached paired files.

    Returns
    -------
    outpath : str
        path to output
    """
    workdir = '.'
    p1s = pd.to_timedelta('1s')
    if isinstance(qds, str):
        qpath = qds
        pyrsig.open_ioapi(qpath)
    else:
        qpath = qds.encoding.get('source', 'unknown_cmaq')

    if verbose > 0:
        print(qpath)
    # Open CMAQ ACONC file
    qds = pyrsig.open_ioapi(qpath)
    Y, X = xr.broadcast(qds.ROW, qds.COL)
    proj = pyproj.Proj(qds.crs_proj4)
    if bbox is None:
        qlon, qlat = proj(X.values, Y.values, inverse=True)
        lonb = np.quantile(qlon, [0, 1]) + np.array([-1, 1])
        latb = np.quantile(qlat, [0, 1]) + np.array([-1, 1])
        bbox = lonb[0], latb[0], lonb[1], latb[1]
        if verbose > 0:
            print(bbox)

    api = pyrsig.RsigApi(bbox=bbox, workdir=workdir)

    if bdate is None or edate is None:
        dates = pd.to_datetime(qds.TSTEP.values)
    if bdate is None:
        bdate = dates[0]
    if edate is None:
        edate = dates[-1] + (dates[-1] - dates[-2]) - p1s
    if verbose > 0:
        print(f'{bdate} to {edate}')
    opath = bdate.strftime(f'{workdir}/{datakey}_and_{basename(qpath)}.csv')
    if os.path.exists(opath) and verbose >= 0:
        print(f'Keeping {opath}')
        return opath

    # Acquire (or use cached) pandora data
    df = api.to_dataframe(
        datakey, bdate=bdate, edate=edate,
        unit_keys=False, parse_dates=True
    )

    # Get a translate lat/lon to x/y
    df['x'], df['y'] = proj(df['LONGITUDE'], df['LATITUDE'])

    # Extract model at observed times and locations
    qvar = qds[qkey][:, 0].sel(
        TSTEP=df['time'].dt.tz_convert(None).to_xarray(),
        COL=df['x'].to_xarray(),
        ROW=df['y'].to_xarray(),
        method='nearest'
    )
    df[f'{prefix}{qkey}'] = qvar.values

    if persist:
        df.to_csv(opath, index=False)
        return opath
    else:
        return df


if __name__ == '__main__':
    prsr = ArgumentParser(
        description=_prsr_description, formatter_class=RawTextHelpFormatter
    )
    prsr.add_argument('--help-rsig', default=False, action='store_true')
    args1, othr = prsr.parse_known_args()
    if args1.help_rsig:
        api = pyrsig.RsigApi()
        print('RSIG Supports:')
        for key in api.keys():
            print(f' - {key}')

    prsr.add_argument('--bbox', default=None)
    prsr.add_argument('rsigkey', help='RSIG id (see pyrsig.RsigApi().keys())')
    prsr.add_argument('cmaqkey', help='e.g., O3, NO2, NO2_COLUMN')
    helpstr = (
        'Paths to CMAQ outputs like ACONC or PHOTDIAG1 or AELMO'
        + '(e.g., ./CMAQ_cb6r5.12US1.ACONC.20160101'
    )
    prsr.add_argument('cmaqpaths', nargs='+', help=helpstr)

    args = prsr.parse_args()

    bbox = args.bbox
    datakey = args.rsigkey
    qkey = args.cmaqkey

    for qpath in args.cmaqpaths:
        pair_rsigcmaq(qpath, qkey=qkey, datakey=datakey, bbox=args.bbox)
