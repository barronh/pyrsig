__all__ = ['RsigApi', 'open_ioapi']
__version__ = '0.1.0'

import pandas as pd

_def_grid_kw = {
    '12US1': dict(
        GDNAM='12US1', GDTYP=2, NCOLS=459, NROWS=299,
        XORIG=-2556000.0, YORIG=-1728000.0, XCELL=12000., YCELL=12000.,
        P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97., YCENT=40.
    ),
    '4US1': dict(
        GDNAM='4US1', GDTYP=2, NCOLS=459 * 3, NROWS=299 * 3,
        XORIG=-2556000.0, YORIG=-1728000.0, XCELL=4000., YCELL=4000.,
        P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97., YCENT=40.
    ),
    '1US1': dict(
        GDNAM='1US1', GDTYP=2, NCOLS=459 * 12, NROWS=299 * 12,
        XORIG=-2556000.0, YORIG=-1728000.0, XCELL=1000., YCELL=1000.,
        P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97., YCENT=40.
    ),
    '12US2': dict(
        GDNAM='12US2', GDTYP=2, NCOLS=396, NROWS=246,
        XORIG=-2412000.0, YORIG=-1620000.0, XCELL=12000., YCELL=12000.,
        P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97., YCENT=40.
    ),
    '4US2': dict(
        GDNAM='4US2', GDTYP=2, NCOLS=396 * 3, NROWS=246 * 3,
        XORIG=-2412000.0, YORIG=-1620000.0, XCELL=4000., YCELL=4000.,
        P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97., YCENT=40.
    ),
    '1US2': dict(
        GDNAM='1US2', GDTYP=2, NCOLS=396 * 12, NROWS=246 * 12,
        XORIG=-2412000.0, YORIG=-1620000.0, XCELL=1000., YCELL=1000.,
        P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97., YCENT=40.
    ),
    '36US3': dict(
        GDNAM='36US3', GDTYP=2, NCOLS=172, NROWS=148,
        XORIG=-2952000.0, YORIG=-2772000.0, XCELL=36000., YCELL=36000.,
        P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97., YCENT=40.
    ),
    '108NHEMI2': dict(
        GDNAM='108NHEMI2', GDTYP=6, NCOLS=187, NROWS=187,
        XORIG=-10098000.0, YORIG=-10098000.0, XCELL=108000., YCELL=108000.,
        P_ALP=1., P_BET=45., P_GAM=-98., XCENT=-98., YCENT=90.
    ),
    '36NHEMI2': dict(
        GDNAM='36NHEMI2', GDTYP=6, NCOLS=187 * 3, NROWS=187 * 3,
        XORIG=-10098000.0, YORIG=-10098000.0, XCELL=36000., YCELL=36000.,
        P_ALP=1., P_BET=45., P_GAM=-98., XCENT=-98., YCENT=90.
    ),
}

_shared_grid_kw = dict(
    VGTYP=7, VGTOP=5000., NLAYS=35, earth_radius=6370000., g=9.81, R=287.04,
    A=50., T0=290, P0=1000e2, REGRID_AGGREGATE='None'
)

for key in _def_grid_kw:
    for pk, pv in _shared_grid_kw.items():
        _def_grid_kw[key].setdefault(pk, pv)


_keys = (
    'airnow.pm25', 'airnow.pm10', 'airnow.ozone', 'airnow.no', 'airnow.no2',
    'airnow.nox', 'airnow.so2', 'airnow.co', 'airnow.temperature',
    'airnow.pressure', 'airnow.rh', 'airnow2.pm25', 'airnow2.ozone',
    'airnow2.no2', 'airnow2.so2', 'airnow2.co',
    'aqs.pm25', 'aqs.pm25_daily_average', 'aqs.pm25_daily_filter', 'aqs.pm10',
    'aqs.ozone', 'aqs.ozone_8hour_average', 'aqs.ozone_daily_8hour_maximum',
    'aqs.co', 'aqs.so2', 'aqs.no2', 'aqs.nox', 'aqs.noy', 'aqs.rh',
    'aqs.temperature', 'aqs.pressure',
    'ceilometer.aerosol_layer_heights',
    'cmaq.equates.conus.aconc.O3', 'cmaq.equates.conus.aconc.NO2',
    'cmaq.equates.conus.aconc.PM25',
    'hms.fire_ecosys', 'hms.fire_power', 'hms.smoke',
    'metar.elevation', 'metar.visibility', 'metar.seaLevelPress',
    'metar.temperature', 'metar.dewpoint', 'metar.relativeHumidity',
    'metar.windDir', 'metar.windSpeed', 'metar.windGustSpeed', 'metar.wind',
    'metar.altimeter', 'metar.minTemp24Hour', 'metar.maxTemp24Hour',
    'metar.precip1Hour', 'metar.precip3Hour', 'metar.precip6Hour',
    'metar.precip24Hour', 'metar.pressChange3Hour', 'metar.snowCover'
    'nesdis.pm25', 'nesdis.co', 'nesdis.co2', 'nesdis.ch4', 'nesdis.n2o',
    'nesdis.nh3', 'nesdis.nox', 'nesdis.so2', 'nesdis.tnmhc',
    'pandora.ozone', 'purpleair.pm25_corrected',
    'purpleair.pm25_corrected_hourly', 'purpleair.pm25_corrected_daily',
    'purpleair.pm25_corrected_monthly', 'purpleair.pm25_corrected_yearly',
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
    'tropomi.offl.no2.air_mass_factor_troposphere',
    'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column',
    'tropomi.offl.co.carbonmonoxide_total_column',
    'tropomi.offl.ch4.methane_mixing_ratio',
    'tropomi.offl.ch4.methane_mixing_ratio',
    'tropomi.offl.ch4.methane_mixing_ratio_bias_corrected',
    'viirsnoaa.jrraod.AOD550', 'viirsnoaa.vaooo.AerosolOpticalDepth_at_550nm',
)

_point_prefixes = ('airnow', 'aqs', 'purpleair', 'pandora')


def _progress(blocknum, readsize, totalsize):
    """
    Arguments
    ---------
    blocknum : int
        block number of blocks to be read
    readsize : int
        chunksize read
    totalsize : int
        -1 unknown or size of file
    """
    totalblocks = (totalsize // readsize) + 1
    pblocks = totalblocks // 10
    if pblocks <= 0:
        pblocks = 100
    if totalsize > 0:
        print(
            '\r' + 'Retrieving {:.0f}'.format(readsize/totalsize*100), end='',
            flush=True
        )
    else:
        if blocknum == 0:
            print('Retrieving .', end='', flush=True)
        if (blocknum % pblocks) == 0:
            print('.', end='', flush=True)


def _getfile(url, outpath, maxtries=5, verbose=1):
    """
    Arguments
    ---------
    url : str
        path to retrieve
    outpath : str
        path to save file to
    maxtries : int
        try this many times before quitting
    """
    import time
    from urllib.request import urlretrieve
    import ssl
    import os

    ssl._create_default_https_context = ssl._create_unverified_context

    # If the file exists, get the current size
    if os.path.exists(outpath):
        stat = os.stat(outpath)
        dlsize = stat.st_size
    else:
        dlsize = 0

    # if the size is non-zero, assume it is good
    if dlsize > 0:
        print('Using cached:', outpath)
        return

    # Try to download the file maxtries times
    tries = 0
    if verbose > 0:
        reporthook = _progress
    else:
        reporthook = None
    while dlsize <= 0 and tries < maxtries:
        # Remove 0-sized files.
        outdir = os.path.dirname(outpath)
        if os.path.exists(outpath):
            os.remove(outpath)
        os.makedirs(outdir, exist_ok=True)
        if verbose:
            print('Calling RSIG', outpath, '')
        t0 = time.time()
        urlretrieve(
            url=url,
            filename=outpath,
            reporthook=reporthook,
        )
        # Check timing
        t1 = time.time()
        stat = os.stat(outpath)
        dlsize = stat.st_size

        if dlsize == 0:
            print('Failed', url, t1 - t0)
        tries += 1

        if verbose > 0:
            print('')


def open_ioapi(path, earth_radius=6370000.):
    import xarray as xr
    import numpy as np
    f = xr.open_dataset(path, engine='netcdf4')
    lvls = f.attrs['VGLVLS']
    tflag = f['TFLAG'][:, 0, :].astype('i').values
    yyyyjjj = tflag[:, 0]
    yyyyjjj = np.where(yyyyjjj < 1, 1970001, yyyyjjj)
    HHMMSS = tflag[:, 1]
    tstrs = []
    for j, t in zip(yyyyjjj, HHMMSS):
        tstrs.append(f'{j:07d}T{t:06d}')

    try:
        time = pd.to_datetime(tstrs, format='%Y%jT%H%M%S')
        f.coords['TSTEP'] = time
    except Exception:
        pass

    f.coords['LAY'] = (lvls[:-1] + lvls[1:]) / 2.
    f.coords['ROW'] = np.arange(f.attrs['NROWS']) + 0.5
    f.coords['COL'] = np.arange(f.attrs['NCOLS']) + 0.5
    props = {k: v for k, v in f.attrs.items()}
    props['x_0'] = -props['XORIG']
    props['y_0'] = -props['YORIG']
    props.setdefault('earth_radius', earth_radius)

    if f.attrs['GDTYP'] == 2:
        f.attrs['crs_proj4'] = (
            '+proj=lcc +lat_1={P_ALP} +lat_2={P_BET} +lat_0={YCENT}'
            ' +lon_0={XCENT} +R={earth_radius} +x_0={x_0} +y_0={y_0}'
            ' +to_meter={XCELL} +no_defs'
        ).format(**props)

    return f


class RsigApi:
    """
    RsigApi is a python-based interface to RSIG's web-based API

    Properties
    ----------
    grid_kw : dict
      Dictionary of regridding IOAPI properties. Defaults to 12US1

    viirsnoaa_kw : dict
      Dictionary of filter properties

    tropomi_kw : dict
      Dictionary of filter properties

    purpleair_kw : dict
      Dictionary of filter properties and api_key. Unlike other options,
      purpleair_kw will not work with the defaults. The user *must* update the
      api_key property to their own key. Contact PurpleAir for more details.

    """
    def describe(self, key):
        import requests
        if key not in self._description:
            r = requests.get(
                f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION='
                f'1.0.0&REQUEST=DescribeCoverage&COVERAGE={key}&compress=1'
            )
            self._description[key] = r.text
        return self._description[key]

    def capabilities(self):
        """
        At this time, the capabilities does not list cmaq.*

        """
        import requests
        if self._capabilities is None:
            self._capabilities = requests.get(
                f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION='
                '1.0.0&REQUEST=GetCapabilities&compress=1'
            )
        return self._capabilities

    def keys(self, offline=True):
        """
        Arguments
        ---------
        offline : bool
            If True, uses small cached set of coverages.
            If False, finds all coverages from capabilities.

        """
        if offline:
            keys = tuple(_keys)
        else:
            keys = []
            for line in self.capabilities().text.split('\n'):
                if line.startswith('            <name>'):
                    keys.append(line.split('name')[1][1:-2])

        return keys

    def __init__(
        self, key=None, bdate=None, edate=None, bbox=None, grid_kw=None,
        tropomi_kw=None, purpleair_kw=None, viirsnoaa_kw=None,
        server='ofmpub.epa.gov', compress=1, corners=1, encoding=None,
        workdir='.'
    ):
        """
        Arguments
        ---------
        key : str
          Default key for query (e.g., 'aqs.o3', 'purpleair.pm25_corrected',
          or 'tropomi.offl.no2.nitrogendioxide_tropospheric_column')
        bdate : str or pd.Datetime
          beginning date (inclusive) defaults to yesterday at 0Z
        edate : str or pd.Datetime
          ending date (inclusive) defaults to bdate + 23:59:59
        bbox : tuple
          wlon, slat, elon, nlat in decimal degrees (-180 to 180)
        grid_kw : str or dict
          If str, must be 12US1, 1US1, 12US2, 1US2, 36US3, 108NHEMI2, 36NHEMI2
            and will be used to set parameters based on EPA domains.
          If dict, IOAPI mapping parameters see default for details.
        viirsnoaa_kw : dict
          Dictionary of VIIRS NOAA filter parameters default
          {'minimum_quality': 'high'} options include 'high' or 'medium')
        tropomi_kw : dict
          Dictionary of TropOMI filter parameters default
          {'minimum_quality': 75, 'maximum_cloud_fraction': 1.0} options
          are 0-100 and 0-1.
        purpleair_kw : dict
          Dictionary of purpleair filter parameters and api_key.
          {
            'out_in_flag': 0, # options 0, 2, ''
            'freq': 'hourly', # options hourly, daily, monthly, yearly
            'maximum_difference': 5, # integer
            'maximum_ratio': 0.70, # float
            'agg_pct': 75, # 0-100
            'api_key': '<your key here>'
          }
        server : str
          'ofmpub.epa.gov' for external  users
          'maple.hesc.epa.gov' for on EPA VPN users
        compress : int
          1 to transfer files with gzip compression
          0 to transfer uncompressed files (slow)
        encoding : dict
          IF encoding is provided, netCDF files will be stored as NetCDF4
          with encoding for all variables. If _FillValue is provided, it will
          not be applied to TFLAG and COUNT.
        workdir : str
          Working directory (must exist) defaults to '.'

        """
        self._description = {}
        self._keys = None
        self._capabilities = None
        self.server = server
        self.key = key
        self.compress = compress
        self.workdir = workdir
        self.encoding = encoding
        if bbox is None:
            self.bbox = (-126, 24, -66, 50)
        else:
            self.bbox = bbox
        if bdate is None:
            bdate = (
                pd.to_datetime('now') - pd.to_timedelta('1day')
            ).replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)

        self.bdate = pd.to_datetime(bdate)
        if edate is None:
            self.edate = edate
        else:
            self.edate = pd.to_datetime(edate)

        self.corners = corners
        if grid_kw is None:
            grid_kw = '12US1'
        if isinstance(grid_kw, str):
            if grid_kw not in _def_grid_kw:
                raise KeyError('unknown grid, you must specify properites')
            grid_kw = _def_grid_kw[grid_kw].copy()

        self.grid_kw = grid_kw

        if tropomi_kw is None:
            tropomi_kw = {'minimum_quality': 75, 'maximum_cloud_fraction': 1.0}

        self.tropomi_kw = tropomi_kw

        if viirsnoaa_kw is None:
            viirsnoaa_kw = {'minimum_quality': 'high'}

        self.viirsnoaa_kw = viirsnoaa_kw

        if purpleair_kw is None:
            purpleair_kw = {
                'out_in_flag': 0, 'freq': 'hourly',
                'maximum_difference': 5, 'maximum_ratio': 0.70,
                'agg_pct': 75, 'api_key': '<your key here>'
            }

        self.purpleair_kw = purpleair_kw

    def get_file(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, request='GetCoverage', compress=0, verbose=0
    ):
        """
        Build url, outpath, and download the file. Returns outpath

        """
        url, outpath = self._build_url(
            formatstr, key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=grid, request=request, compress=compress
        )
        if verbose > 0:
            print(url)
        _getfile(url, outpath, verbose=verbose)
        return outpath

    def _build_url(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, request='GetCoverage',
        compress=1
    ):
        """
        Arguments
        ---------
        formatstr : str
          'xdr', 'ascii', 'netcdf-ioapi', 'netcdf-coards'
        request : str
            'GetCoverage' or 'GetMetadata'
        all other keywords see __init__

        """
        if key is None:
            key = self.key

        if bdate is None:
            bdate = self.bdate
        else:
            bdate = pd.to_datetime(bdate)

        if edate is None:
            if self.edate is None:
                edate = (
                    bdate + pd.to_timedelta('+1day') + pd.to_timedelta('-1s')
                )
            else:
                edate = self.edate
        else:
            edate = pd.to_datetime(edate)

        if bbox is None:
            bbox = self.bbox

        if edate < bdate:
            raise ValueError('edate cannot be before bdate')

        if bbox[2] < bbox[0]:
            raise ValueError('elon cannot be less than wlon')

        if bbox[3] < bbox[1]:
            raise ValueError('nlat cannot be less than slat')

        corners = self.corners
        grid_kw = self.grid_kw
        purpleair_kw = self.purpleair_kw
        tropomi_kw = self.tropomi_kw
        viirsnoaa_kw = self.viirsnoaa_kw
        if compress is None:
            compress = self.compress
        wlon, slat, elon, nlat = bbox
        if grid and request == 'GetCoverage':
            gridstr = self._build_grid(grid_kw)
        else:
            gridstr = ''

        if key.startswith('viirsnoaa'):
            viirsnoaastr = '&MINIMUM_QUALITY={minimum_quality}'.format(
                **viirsnoaa_kw
            )
        else:
            viirsnoaastr = ''

        if key.startswith('tropomi'):
            tropomistr = (
                '&MINIMUM_QUALITY={minimum_quality}'
                '&MAXIMUM_CLOUD_FRACTION={maximum_cloud_fraction}'
            ).format(**tropomi_kw)
        else:
            tropomistr = ''

        if key.startswith('purpleair'):
            purpleairstr = (
                '&OUT_IN_FLAG={out_in_flag}&MAXIMUM_DIFFERENCE='
                '{maximum_difference}&MAXIMUM_RATIO={maximum_ratio}'
                '&AGGREGATE={freq}&MINIMUM_AGGREGATION_COUNT_PERCENTAGE='
                '{agg_pct}&KEY={api_key}'
            ).format(**purpleair_kw)
        else:
            purpleairstr = ''

        if any([key.startswith(pre) for pre in _point_prefixes]):
            cornerstr = ''
        else:
            cornerstr = f'&CORNERS={corners}'

        url = (
            f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION=1.0.0'
            f'&REQUEST={request}&FORMAT={formatstr}'
            f'&TIME={bdate:%Y-%m-%dT%H:%M:%SZ}/{edate:%Y-%m-%dT%H:%M:%SZ}'
            f'&BBOX={wlon},{slat},{elon},{nlat}'
            f'&COVERAGE={key}'
            f'&COMPRESS={compress}'
        ) + purpleairstr + viirsnoaastr + tropomistr + gridstr + cornerstr

        outpath = (
            f'{self.workdir}/{key}_{bdate:%Y-%m-%dT%H:%M:%SZ}'
            f'_{edate:%Y-%m-%dT%H:%M:%SZ}'
        )

        if formatstr.lower() == 'ascii':
            outpath += '.csv'
        elif formatstr.lower() == 'netcdf-ioapi':
            outpath += '.nc'
        elif formatstr.lower() == 'netcdf-coards':
            outpath += '.nc'
        elif formatstr.lower() == 'xdr':
            outpath += '.xdr'
        if request == 'GetMetadata':
            outpath += '.txt'
        elif compress:
            outpath += '.gz'

        return url, outpath

    def _build_grid(self, grid_kw):
        """
        Build the regrid portion of the URL

        """
        grid_kw.setdefault('earth_radius', 6370000)
        if grid_kw.get('GDTYP', 2) == 2:
            gridstr = (
                '&REGRID=weighted'
                '&LAMBERT={P_ALP},{P_BET},{XCENT},{YCENT}'
                '&ELLIPSOID={earth_radius},{earth_radius}'
                '&GRID={NCOLS},{NROWS},{XORIG},{YORIG},{XCELL},{YCELL}'
            ).format(**grid_kw)
            if grid_kw.get('REGRID_AGGREGATE', 'None').strip() != 'None':
                gridstr += (
                    "&REGRID_AGGREGATE={REGRID_AGGREGATE}".format(**grid_kw)
                )

            return gridstr
        else:
            raise KeyError('GDTYP only implemented for ')

    def to_dataframe(
        self, key=None, bdate=None, edate=None, bbox=None, unit_keys=True,
        parse_dates=False, verbose=0
    ):
        """
        All arguments default to those provided during initialization.

        Arguments
        ---------
        key : str
          Default key for query (e.g., 'aqs.o3', 'purpleair.pm25_corrected',
          or 'tropomi.offl.no2.nitrogendioxide_tropospheric_column')
        bdate : str or pd.Datetime
          beginning date (inclusive) defaults to yesterday at 0Z
        edate : str or pd.Datetime
          ending date (inclusive) defaults to bdate + 23:59:59
        bbox : tuple
          wlon, slat, elon, nlat in decimal degrees (-180 to 180)
        unit_keys : bool
          If True, keep unit in column name.
          If False, move last parenthetical part of key to attrs of Series.
        parse_dates : bool
          If True, parse Timestamp(UTC)
        verbose : int
          level of verbosity

        Returns
        -------
        df : pandas.DataFrame
            Results from download

        """
        outpath = self.get_file(
            'ascii', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=False, verbose=verbose,
            compress=1
        )
        df = pd.read_csv(outpath, delimiter='\t', na_values=[-9999., -999])
        if not unit_keys:
            columns = [k for k in df.columns]
            newcolumns = []
            unit_dict = {}
            for k in columns:
                if '(' not in k:
                    newk = k
                    unit = 'unknown'
                else:
                    idx = k.rfind('(')
                    newk = k[:idx]
                    unit = k[idx+1:-1]
                unit_dict[newk] = unit
                newcolumns.append(newk)
            df.columns = newcolumns
            for k in newcolumns:
                if hasattr(df[k], 'attrs'):
                    df[k].attrs.update(dict(units=unit_dict.get(k, 'unknown')))
        if parse_dates:
            if 'Timestamp(UTC)' in df:
                df['time'] = pd.to_datetime(df['Timestamp(UTC)'])
            if 'Timestamp' in df:
                df['time'] = pd.to_datetime(df['Timestamp'])

        return df

    def to_ioapi(
        self, key=None, bdate=None, edate=None, bbox=None, removegz=False,
        verbose=0
    ):
        """
        All arguments default to those provided during initialization.

        Arguments
        ---------
        key : str
          Default key for query (e.g., 'aqs.o3', 'purpleair.pm25_corrected',
          or 'tropomi.offl.no2.nitrogendioxide_tropospheric_column')
        bdate : str or pd.Datetime
          beginning date (inclusive) defaults to yesterday at 0Z
        edate : str or pd.Datetime
          ending date (inclusive) defaults to bdate + 23:59:59
        bbox : tuple
          wlon, slat, elon, nlat in decimal degrees (-180 to 180)
        removegz : bool
          If True, then remove the downloaded gz file. Bad for caching.

        Returns
        -------
        ds : xarray.Dataset
            Results from download

        """
        import gzip
        import shutil
        import os

        # always use compression for network speed.
        outpath = self.get_file(
            'netcdf-ioapi', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=True, compress=1, verbose=verbose
        )
        # Uncompress the netcdf file. If encoding is available, apply it
        if os.path.exists(outpath[:-3]):
            print('Using cached:', outpath[:-3])
        else:
            with gzip.open(outpath, 'rb') as f_in:
                with open(outpath[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    f_out.flush()
            if self.encoding is not None:
                import xarray as xr

                with xr.open_dataset(outpath[:-3]) as tmpf:
                    tmpf.load()
                for key in tmpf.data_vars:
                    tvar = tmpf[key]
                    tvar.encoding.update(self.encoding)
                    if key in ('TFLAG', 'COUNT'):
                        tvar.encoding.pop('_FillValue', '')

                tmpf.to_netcdf(outpath[:-3], format='NETCDF4_CLASSIC')

        f = open_ioapi(outpath[:-3])
        if removegz:
            os.remove(outpath)

        return f
