__all__ = ['RsigApi']
__version__ = '0.1.0'

import pandas as pd


def progress(blocknum, readsize, totalsize):
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


def getfile(url, outpath, maxtries=5, verbose=1):
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
        reporthook = progress
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
    capabilities
      Full xml text describing all RSIG capabilities. Refreshed every time,
      so slow because this service is slow.

    keys
      List of all keys that RSIG can process. Slow because it depends on
      capabilities

    grid_kw
      Dictionary of regridding IOAPI properties. Defaults to 12US1

    tropomi_kw
      Dictionary of filter properties

    purpleair_kw
      Dictionary of filter properties and api_key

    Methods
    -------
    to_dataframe
    Access data from RSIG as a pandas.DataFrame

    to_ioapi
    Access data from RSIG as a xarray.Dataset

    """
    @property
    def capabilities(self):
        import requests
        if self._capabilities is None:
            self._capabilities = requests.get(
                f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION='
                '1.0.0&REQUEST=GetCapabilities&compress=1'
            )
        return self._capabilities

    @property
    def keys(self):
        if self._keys is None:
            self._keys = []
            for line in self.capabilities.text.split('\n'):
                if line.startswith('            <name>'):
                    self._keys.append(line.split('name')[1][1:-2])

        return self._keys

    def __init__(
        self, key=None, bdate=None, edate=None, bbox=None, grid_kw=None,
        tropomi_kw=None, purpleair_kw=None, server='ofmpub.epa.gov', compress=1
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
        """
        self._keys = None
        self._capabilities = None
        self.server = server
        self.key = key
        self.compress = compress
        if bbox is None:
            self.bbox = (-126, -24, 50, -66)
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
            compress=0
        ):
            """
            Build url, outpath, and download the file. Returns outpath
            """
            url, outpath = self._build_url(
                formatstr, key=key, bdate=bdate, edate=edate, bbox=bbox,
                grid=grid, grid_kw=grid_kw, purpleair_kw=purpleair_kw,
                request=request, compress=compress
            )
            getfile(url, outpath)
            return outpath

    def _build_url(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, grid_kw=None, purpleair_kw=None, request='GetCoverage',
        compress=0
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
            f'./{key}_{bdate:%Y-%m-%dT%H:%M:%SZ}'
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
        self, key=None, bdate=None, edate=None, bbox=None, purpleair_kw=None
    ):
        """
        Arguments
        ---------

        """
        outpath = self._get_file(
            'ascii', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=False, purpleair_kw=purpleair_kw, compress=1
        )
        return pd.read_csv(outpath, delimiter='\t')

    def to_ioapi(
        self, key=None, bdate=None, edate=None, bbox=None,
        grid_kw=None, purpleair_kw=None
    ):
        import gzip
        import xarray as xr
        import shutil
        import os
        outpath = self._get_file(
            'netcdf-ioapi', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=True, grid_kw=grid_kw, purpleair_kw=purpleair_kw, compress=1
        )
        if os.path.exists(outpath[:-3]):
            print('Using cached:', outpath[:-3])
        else:
            with gzip.open(outpath, 'rb') as f_in:
                with open(outpath[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    f_out.flush()
        f = xr.open_dataset(outpath[:-3], engine='netcdf4')

        return f
