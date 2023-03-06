__all__ = ['RsigApi']
__version__ = '0.1.0'

import pandas as pd


_keys = (
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
    'tropomi.offl.no2.air_mass_factor_troposphere',
    'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column',
    'tropomi.offl.co.carbonmonoxide_total_column',
    'tropomi.offl.ch4.methane_mixing_ratio',
    'tropomi.offl.ch4.methane_mixing_ratio',
    'tropomi.offl.ch4.methane_mixing_ratio_bias_corrected',
    'viirsnoaa.jrraod.AOD550', 'viirsnoaa.vaooo.AerosolOpticalDepth_at_550nm',
    'airnow.pm25', 'airnow.pm10',
    'airnow.ozone', 'airnow.no', 'airnow.no2', 'airnow.nox', 'airnow.so2',
    'airnow.co', 'airnow.temperature', 'airnow.pressure', 'airnow.rh',
    'airnow2.pm25', 'airnow2.ozone', 'airnow2.no2', 'airnow2.so2',
    'airnow2.co',
    'aqs.pm25', 'aqs.pm25_daily_average', 'aqs.pm25_daily_filter', 'aqs.pm10',
    'aqs.ozone', 'aqs.ozone_8hour_average', 'aqs.ozone_daily_8hour_maximum',
    'aqs.co', 'aqs.so2', 'aqs.no2', 'aqs.nox', 'aqs.noy', 'aqs.rh',
    'aqs.temperature', 'aqs.pressure',
    'ceilometer.aerosol_layer_heights',
    'cmaq.equates.conus.aconc.O3', 'cmaq.equates.conus.aconc.NO2',
    'cmaq.equates.conus.aconc.PM25'
    'metar.elevation', 'metar.visibility', 'metar.seaLevelPress',
    'metar.temperature', 'metar.dewpoint', 'metar.relativeHumidity',
    'metar.windDir', 'metar.windSpeed', 'metar.windGustSpeed', 'metar.wind',
    'metar.altimeter', 'metar.minTemp24Hour', 'metar.maxTemp24Hour',
    'metar.precip1Hour', 'metar.precip3Hour', 'metar.precip6Hour',
    'metar.precip24Hour', 'metar.pressChange3Hour', 'metar.snowCover'
    'nesdis.pm25', 'nesdis.co', 'nesdis.co2', 'nesdis.ch4', 'nesdis.n2o',
    'nesdis.nh3', 'nesdis.nox', 'nesdis.so2', 'nesdis.tnmhc'
    'pandora.ozone',
    'purpleair.pm25_corrected', 'purpleair.pm25_corrected_hourly',
    'purpleair.pm25_corrected_daily', 'purpleair.pm25_corrected_monthly',
    'purpleair.pm25_corrected_yearly',
)


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

        print('')


class RsigApi:
    """
    RsigApi is a python-based interface to RSIG's web-based API

    Properties
    ----------
    grid_kw
      Dictionary of regridding IOAPI properties. Defaults to 12US1

    tropomi_kw
      Dictionary of filter properties

    purpleair_kw
      Dictionary of filter properties and api_key

    Methods
    -------
    capabilities
      Full xml text describing all RSIG capabilities. Refreshed every time,
      so slow because this service is slow.

    keys
      List of keys that RSIG can process. Using offline=False is slow because
      it depends on capabilities method.

    to_dataframe
        Access data from RSIG as a pandas.DataFrame

    to_ioapi
        Access data from RSIG as a xarray.Dataset

    """
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
        if self._keys is None:
            if offline:
                self._keys = tuple(_keys)
            else:
                self._keys = []
                for line in self.capabilities().text.split('\n'):
                    if line.startswith('            <name>'):
                        self._keys.append(line.split('name')[1][1:-2])

        return self._keys

    def __init__(
        self, key=None, bdate=None, edate=None, bbox=None, grid_kw=None,
        tropomi_kw=None, purpleair_kw=None, server='ofmpub.epa.gov',
        compress=1, workdir='.'
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
        grid_kw : dict
          Dictionary of IOAPI mapping parameters
        tropomi_kw : dict
          Dictionary of TropOMI filter parameters
        purpleair_kw : dict
          Dictionary of purpleair filter parameters and api_key
        server : str
          'ofmpub.epa.gov', 'maple.hesc.epa.gov'
        compress : int
            1 to compress; 0 to not
        workdir : str
            Working directory (must exist) defaults to .
        """
        self._keys = None
        self._capabilities = None
        self.server = server
        self.key = key
        self.compress = compress
        self.workdir = workdir
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
            edate = (
                self.bdate + pd.to_timedelta('+1day') + pd.to_timedelta('-1s')
            )

        self.edate = pd.to_datetime(edate)
        if grid_kw is None:
            grid_kw = dict(
                GDTYP=2, VGTYP=7, NCOLS=459, NROWS=299, NLAYS=35,
                XORIG=-2556000.0, YORIG=-1728000.0, XCELL=12000., YCELL=12000.,
                VGTOP=5000., P_ALP=33., P_BET=45., P_GAM=-97., XCENT=-97.,
                YCENT=40., earth_radius=6370000., g=9.81, R=287.04, A=50.,
                T0=290, P0=1000e2, corners=1
            )

        self.grid_kw = grid_kw

        if tropomi_kw is None:
            tropomi_kw = dict(minimum_quality=75, maximum_cloud_fraction=1.0)

        self.tropomi_kw = tropomi_kw

        if purpleair_kw is None:
            purpleair_kw = dict(
                out_in_flag=0, freq='hourly',
                maximum_difference=5, maximum_ratio=0.70,
                agg_pct=75, api_key='<your key here'
            )

        self.purpleair_kw = purpleair_kw

    def _get_file(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, grid_kw=None, purpleair_kw=None, request='GetCoverage',
        compress=0, verbose=0
    ):
        """
        Build url, outpath, and download the file. Returns outpath
        """
        url, outpath = self._build_url(
            formatstr, key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=grid, grid_kw=grid_kw, purpleair_kw=purpleair_kw,
            request=request, compress=compress
        )
        if verbose > 0:
            print(url)
        _getfile(url, outpath, verbose=verbose)
        return outpath

    def _build_url(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, grid_kw=None, purpleair_kw=None, request='GetCoverage',
        compress=1
    ):
        """
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
            edate = self.edate
        else:
            edate = pd.to_datetime(edate)
        if bbox is None:
            bbox = self.bbox
        if grid_kw is None:
            grid_kw = self.grid_kw
        if purpleair_kw is None:
            purpleair_kw = self.purpleair_kw
        if compress is None:
            compress = self.compress
        wlon, slat, elon, nlat = bbox
        if grid and request == 'GetCoverage':
            gridstr = self._build_grid(grid_kw)
        else:
            gridstr = ''
        if key.startswith('tropomi'):
            tropomistr = '&MINIMUM_QUALITY=75&MAXIMUM_CLOUD_FRACTION=1.000000'
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

        url = (
            f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION=1.0.0'
            f'&REQUEST={request}&FORMAT={formatstr}'
            f'&TIME={bdate:%Y-%m-%dT%H:%M:%SZ}/{edate:%Y-%m-%dT%H:%M:%SZ}'
            f'&BBOX={wlon},{slat},{elon},{nlat}'
            f'&COVERAGE={key}'
            f'&COMPRESS={compress}'
        ) + purpleairstr + tropomistr + gridstr
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
            return (
                '&REGRID=weighted&CORNERS={corners}'
                '&LAMBERT={P_ALP},{P_BET},{XCENT},{YCENT}'
                '&ELLIPSOID={earth_radius},{earth_radius}'
                '&GRID={NCOLS},{NROWS},{XORIG},{YORIG},{XCELL},{YCELL}'
            ).format(**grid_kw)
        else:
            raise KeyError('GDTYP only implemented for ')

    def to_dataframe(
        self, key=None, bdate=None, edate=None, bbox=None, tropomi_kw=None,
        purpleair_kw=None, verbose=0
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
        grid_kw : dict
          Dictionary of IOAPI mapping parameters
        tropomi_kw : dict
          Dictionary of TropOMI filter parameters
        purpleair_kw : dict
          Dictionary of purpleair filter parameters and api_key

        Returns
        -------
        df : pandas.DataFrame
            Results from download
        """
        outpath = self._get_file(
            'ascii', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=False, purpleair_kw=purpleair_kw, verbose=verbose,
            compress=1
        )
        return pd.read_csv(outpath, delimiter='\t', na_values=[-9999., -999])

    def to_ioapi(
        self, key=None, bdate=None, edate=None, bbox=None,
        grid_kw=None, tropomi_kw=None, purpleair_kw=None, verbose=0
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
        grid_kw : dict
          Dictionary of IOAPI mapping parameters
        tropomi_kw : dict
          Dictionary of TropOMI filter parameters
        purpleair_kw : dict
          Dictionary of purpleair filter parameters and api_key

        Returns
        -------
        ds : xarray.Dataset
            Results from download
        """
        import gzip
        import xarray as xr
        import shutil
        import os
        import numpy as np

        outpath = self._get_file(
            'netcdf-ioapi', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=True, grid_kw=grid_kw, purpleair_kw=purpleair_kw, compress=1,
            verbose=verbose
        )
        if os.path.exists(outpath[:-3]):
            print('Using cached:', outpath[:-3])
        else:
            with gzip.open(outpath, 'rb') as f_in:
                with open(outpath[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    f_out.flush()

        f = xr.open_dataset(outpath[:-3], engine='netcdf4')
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
        except:
            pass
        f.coords['LAY'] = (lvls[:-1] + lvls[1:]) / 2.
        f.coords['ROW'] = np.arange(f.attrs['NROWS']) + 0.5
        f.coords['COL'] = np.arange(f.attrs['NCOLS']) + 0.5
        props = {k: v for k, v in f.attrs.items()}
        props['x_0'] = -props['XORIG']
        props['y_0'] = -props['YORIG']
        props.setdefault(
            'earth_radius', self.grid_kw.get('earth_radius', 6370000.)
        )

        if f.attrs['GDTYP'] == 2:
            f.attrs['proj4str'] = (
                '+proj=lcc +lat_1={P_ALP} +lat_2={P_BET} +lat_0={YCENT}'
                ' +lon_0={XCENT} +R={earth_radius} +x_0={x_0} +y_0={y_0}'
                ' +to_meter={XCELL} +no_defs'
            ).format(**props)

        return f
