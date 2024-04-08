__all__ = ['RsigApi', 'RsigGui', 'open_ioapi', 'open_mfioapi', 'cmaq']
__version__ = '0.8.2'

from . import cmaq
from .cmaq import open_ioapi, open_mfioapi
import pandas as pd
import requests
from .utils import customize_grid, def_grid_kw as _def_grid_kw

# Used to shorten pandora names for 80 character PEP
_trvca = 'tropospheric_vertical_column_amount'

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
    'pandora.ozone'
    f'pandora.L2_rfuh5p1_8.formaldehyde_{_trvca}',
    f'pandora.L2_rfuh5p1_8.formaldehyde_{_trvca}_uncertainty',
    'pandora.L2_rfus5p1_8.direct_formaldehyde_air_mass_factor',
    'pandora.L2_rfus5p1_8.direct_formaldehyde_air_mass_factor_uncertainty',
    'pandora.L2_rfus5p1_8.formaldehyde_total_vertical_column_amount',
    'pandora.L2_rfus5p1_8.formaldehyde_vertical_column_amount_uncertainty',
    f'pandora.L2_rnvh3p1_8.water_vapor_{_trvca}',
    f'pandora.L2_rnvh3p1_8.water_vapor_{_trvca}_uncertainty',
    'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount',
    'pandora.L2_rnvh3p1_8.tropospheric_nitrogen_dioxide',
    'pandora.L2_rnvh3p1_8.tropospheric_nitrogen_dioxide_uncertainty',
    'pandora.L2_rnvs3p1_8.direct_nitrogen_dioxide_air_mass_factor',
    'pandora.L2_rnvs3p1_8.direct_nitrogen_dioxide_air_mass_factor_uncertainty',
    'pandora.L2_rout2p1_8.ozone_vertical_column_amount',
    'pandora.L2_rout2p1_8.direct_ozone_air_mass_factor',
    'pandora.L2_rout2p1_8.ozone_air_mass_factor_uncertainty',
    'pandora.L2_rsus1p1_8.sulfur_dioxide_vertical_column_amount',
    'pandora.L2_rsus1p1_8.direct_sulfur_dioxide_air_mass_factor',
    'pandora.L2_rsus1p1_8.sulfur_dioxide_air_mass_factor_uncertainty',
    'pandora.L2_rnvssp1_8.nitrogen_dioxide_vertical_column_amount',
    'pandora.L2_rnvssp1_8.direct_nitrogen_dioxide_air_mass_factor',
    'pandora.L2_rnvssp1_8.direct_nitrogen_dioxide_air_mass_factor_uncertainty',
    'purpleair.pm25_corrected',
    'purpleair.pm25_corrected_hourly', 'purpleair.pm25_corrected_daily',
    'purpleair.pm25_corrected_monthly', 'purpleair.pm25_corrected_yearly',
    'regridded.conus.monthly.tropomi.offl.no2',
    'regridded.conus.monthly.tropomi.offl.hcho',
    'regridded.conus.monthly.tropomi.offl.ch4',
    'regridded.conus.monthly.tropomi.offl.co',
    'regridded.conus.monthly.tropomi.rpro.no2',
    'regridded.conus.seasonal.tropomi.offl.no2',
    'regridded.conus.seasonal.tropomi.offl.hcho',
    'regridded.conus.seasonal.tropomi.offl.ch4',
    'regridded.conus.seasonal.tropomi.offl.co',
    'regridded.conus.seasonal.tropomi.rpro.no2',
    'tempo.proxy_l2.no2.vertical_column_total',
    'tempo.proxy_l2.no2.vertical_column_total_uncertainty',
    'tempo.proxy_l2.no2.vertical_column_troposphere',
    'tempo.proxy_l2.no2.vertical_column_stratosphere',
    'tempo.proxy_l2.no2.amf_total',
    'tempo.proxy_l2.no2.amf_total_uncertainty',
    'tempo.proxy_l2.no2.amf_troposphere',
    'tempo.proxy_l2.no2.amf_stratosphere',
    'tempo.proxy_l2.no2.ground_pixel_quality_flag',
    'tempo.proxy_l2.hcho.vertical_column',
    'tempo.proxy_l2.hcho.vertical_column_uncertainty',
    'tempo.proxy_l2.hcho.amf',
    'tempo.proxy_l2.hcho.amf_uncertainty',
    'tempo.proxy_l2.o3p.total_ozone_column',
    'tempo.proxy_l2.o3p.troposphere_ozone_column',
    'tempo.proxy_l2.o3p.stratosphere_ozone_column',
    'tempo.proxy_l2.o3p.ozone_information_content',
    'tempo.proxy_l2.o3p.ground_pixel_quality_flag',
    'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
    'tropomi.offl.no2.air_mass_factor_troposphere',
    'tropomi.offl.hcho.formaldehyde_tropospheric_vertical_column',
    'tropomi.offl.co.carbonmonoxide_total_column',
    'tropomi.offl.ch4.methane_mixing_ratio',
    'tropomi.offl.ch4.methane_mixing_ratio_bias_corrected',
    'viirsnoaa.jrraod.AOD550', 'viirsnoaa.vaooo.AerosolOpticalDepth_at_550nm',
)

_nocorner_prefixes = (
    'airnow', 'aqs', 'purpleair', 'pandora', 'cmaq', 'regridded',
    'calipso'
)
_nolonlats_prefixes = ('cmaq', 'regridded')
_noregrid_prefixes = ('cmaq', 'regridded')


def _actionf(msg, action, ErrorTyp=None):
    """
    Convenience function for warning or raising an error.

    Arguments
    ---------
    msg : str
        Message to raise or warn.
    action : str
        If 'error', raise ErrorTyp(msg)
        If 'warn', warnings.warn using msg
        Else do nothing.
    ErrorTyp : Exception
        Defaults to ErrorTyp

    Returns
    -------
    None
    """
    import warnings

    if ErrorTyp is None:
        ErrorTyp = ValueError
    if action == 'error':
        raise ErrorTyp(msg)
    elif action == 'warn':
        warnings.warn(msg)


def parsexml(root):
    """Recursive xml parsing:
    Given a root, return dictionaries for each element and its children.
    Each element has children, attributes (attr), tag, and text.
    If any of these has no elements, it will be removed.
    """
    out = {}
    out['tag'] = root.tag.split('}')[-1]
    out['attr'] = root.attrib
    out['text'] = root.text
    out['children'] = []

    for child in root:
        childd = parsexml(child)
        out['children'].append(childd)

    if len(out['children']) == 0:
        del out['children']
    if out['text'] is None:
        out['text'] = ''

    out['text'] = out['text'].strip()
    if len(out['text']) == 0:
        del out['text']
    if len(out['attr']) == 0:
        del out['attr']

    return out


def coverages_from_xml(txt):
    """Based on xml text, create coverage data"""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(txt)

    xmlout = parsexml(root)
    out = []
    for c in xmlout['children']:
        record = {k: v for k, v in c.items() if k != 'children'}
        kids = c['children']
        for e in kids:
            if 'attr' not in e and len(e.get('children', [])) == 0:
                record[e['tag']] = e.get('text', '')

            if e['tag'] == 'lonLatEnvelope':
                envtxt = ''
                for p in e['children']:
                    envtxt += ' ' + p['text']
                record['bbox_str'] = envtxt.strip()

            if e['tag'] == 'domainSet':
                for s in e['children']:
                    if s['tag'] == 'temporalDomain':
                        for tp in s['children']:
                            for te in tp['children']:
                                record[te['tag']] = te['text']

        out.append(record)

    return out


def _progress(blocknum, readsize, totalsize):
    """
    Display progress using dots or % indicator.

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


def _create_unverified_tls_context(*args, **kwds):
    """
    Thin wrapper around ssl._create_unverified_context that adds the option to
    use TLS negotiation, which is currently used by RSIG servers.
    """
    import ssl
    # Set up SSL context to allow legacy TLS versions
    ctx = ssl._create_unverified_context(*args, **kwds)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    return ctx


class LegacyAdapter(requests.adapters.HTTPAdapter):
    # "Transport adapter" that allows us to use custom ssl_context.

    def __init__(self, **kwargs):
        import ssl
        def_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = _create_unverified_tls_context
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        self.ssl_context = ctx
        ssl._create_default_https_context = def_ctx
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        import urllib3
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)


def legacy_get(url, *args, **kwds):
    import copy
    session = requests.session()
    la = LegacyAdapter()

    # Maple has a bad certificate, so here if you are using maple
    # I disable the certificate, but no the warning
    if 'maple' in url:
        la.ssl_context.check_hostname = False
        kwds = copy.copy(kwds)
        kwds.setdefault('verify', False)

    session.mount('https://', la)

    return session.get(url, *args, **kwds)


def _getfile(url, outpath, maxtries=5, verbose=1, overwrite=False):
    """
    Download file from RSIG using fault tolerance and optional caching
    when overwrite is False.

    Arguments
    ---------
    url : str
        path to retrieve
    outpath : str
        path to save file to
    maxtries : int
        try this many times before quitting
    verbose : int
        Level of verbosity
    overwrite : bool
        If True, overwrite existing files.
        If False, reuse existing files.

    Returns
    -------
    None
    """
    import time
    from urllib.request import urlretrieve
    import ssl
    import os

    # If the file exists, get the current size
    if not overwrite and os.path.exists(outpath):
        stat = os.stat(outpath)
        dlsize = stat.st_size
    else:
        dlsize = 0

    # if the size is non-zero, assume it is good
    if dlsize > 0:
        print('Using cached:', outpath)
        return

    _def_https_context = ssl._create_default_https_context
    ssl._create_default_https_context = _create_unverified_tls_context

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

    ssl._create_default_https_context = _def_https_context


class RsigApi:
    def __init__(
        self, key=None, bdate=None, edate=None, bbox=None, grid_kw=None,
        tropomi_kw=None, purpleair_kw=None, viirsnoaa_kw=None, tempo_kw=None,
        pandora_kw=None, calipso_kw=None,
        server='ofmpub.epa.gov', compress=1, corners=1, encoding=None,
        overwrite=False, workdir='.', gridfit=False
    ):
        """
        RsigApi is a python-based interface to RSIG's web-based API

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
          and will be used to set parameters based on EPA domains. If dict,
          IOAPI mapping parameters see default for details.
        viirsnoaa_kw : dict
          Dictionary of VIIRS NOAA filter parameters default
          {'minimum_quality': 'high'} other options 'medium' or 'low'
        tropomi_kw : dict
          Dictionary of TropOMI filter parameters default
          {'minimum_quality': 75, 'maximum_cloud_fraction': 1.0} options
          are 0-100 and 0-1.
        purpleair_kw : dict
          Dictionary of purpleair filter parameters and api_key.
            'out_in_flag': 0, # options 0, 2, ''
            'freq': 'hourly', # options hourly, daily, monthly, yearly, none
            'maximum_difference': 5, # integer
            'maximum_ratio': 0.70, # float
            'agg_pct': 75, # 0-100
            'default_humidity': 50,
            'api_key': 'your_key_here'
        tempo_kw : dict
          Dictionary of TEMPO filter parameters default
            'api_key': 'your_key_here' # 'password'
        pandora_kw : dict
          Dictionary of Pandora filter parameters default
          {'minimum_quality': 'high'} other options 'medium' or 'low'
        calipso_kw : dict
          Dictionary of Calipso filter parameters default
          {'MINIMUM_CAD': 20, 'MAXIMUM_UNCERTAINTY': 99}
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
        overwrite : bool
          If True, overwrite downloaded files in workdir.
          If False, reuse downloaded files in workdir.
        workdir : str
          Working directory (must exist) defaults to '.'
        gridfit : bool
          Default (False) keep grid as supplied.
          If True, redefine grid to remove cells outside the bbox.

        Properties
        ----------
        grid_kw : dict
          Dictionary of regridding IOAPI properties. Defaults to 12US1

        viirsnoaa_kw : dict
          Dictionary of filter properties

        tropomi_kw : dict
          Dictionary of filter properties

        tempo_kw : dict
          Dictionary of filter properties

        purpleair_kw : dict
          Dictionary of filter properties and api_key. Unlike other options,
          purpleair_kw will not work with the defaults. The user *must* update
          teh api_key property to their own key. Contact PurpleAir for more
          details.

        """
        self._description = {}
        self._keys = None
        self._capabilities = None
        self._describecoverages = None
        self._coveragesdf = None
        self._capabilitiesdf = None
        self.server = server
        self.key = key
        self.compress = compress
        self.workdir = workdir
        self.encoding = encoding
        self.overwrite = overwrite

        if bbox is None:
            self.bbox = (-126, 24, -66, 50)
        else:
            self.bbox = bbox
        if bdate is None:
            bdate = (
                pd.to_datetime('now', utc=True) - pd.to_timedelta('1day')
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

        if gridfit:
            grid_kw = customize_grid(grid_kw, self.bbox)

        self.grid_kw = grid_kw

        if tropomi_kw is None:
            tropomi_kw = {}

        tropomi_kw.setdefault('minimum_quality', 75)
        tropomi_kw.setdefault('maximum_cloud_fraction', 1.0)

        self.tropomi_kw = tropomi_kw

        if tempo_kw is None:
            tempo_kw = {}

        tempo_kw.setdefault('minimum_quality', 'normal')
        tempo_kw.setdefault('maximum_cloud_fraction', 1.0)
        tempo_kw.setdefault('api_key', 'your_key_here')

        self.tempo_kw = tempo_kw

        if viirsnoaa_kw is None:
            viirsnoaa_kw = {}

        viirsnoaa_kw.setdefault('minimum_quality', 'high')

        self.viirsnoaa_kw = viirsnoaa_kw

        if pandora_kw is None:
            pandora_kw = {}

        pandora_kw.setdefault('minimum_quality', 'high')

        self.pandora_kw = pandora_kw

        if calipso_kw is None:
            calipso_kw = {}

        calipso_kw.setdefault('MINIMUM_CAD', 20)
        calipso_kw.setdefault('MAXIMUM_UNCERTAINTY', 99)

        self.calipso_kw = calipso_kw

        if purpleair_kw is None:
            purpleair_kw = {}

        defpurp_kw = {
            'out_in_flag': 0, 'freq': 'hourly',
            'maximum_difference': 5, 'maximum_ratio': 0.70,
            'agg_pct': 75, 'api_key': 'your_key_here',
            'default_humidity': 50.000000
        }
        for k, v in defpurp_kw.items():
            purpleair_kw.setdefault(k, v)

        self.purpleair_kw = purpleair_kw

    def set_grid_kw(self, grid_kw):
        if isinstance(grid_kw, str):
            if grid_kw not in _def_grid_kw:
                raise KeyError('unknown grid, you must specify properites')
            grid_kw = _def_grid_kw[grid_kw].copy()

        self.grid_kw = grid_kw

    def resize_grid(self, clip=True):
        """
        Update grid_kw property so that it only covers the bbox by adjusting
        the XORIG, YORIG, NCOLS and NROWS. If clip is True, this has the affect
        of reducing the number of rows and columns. This is useful when the
        area of interest is much smaller than the grid defined in grid_kw.

        Arguments
        ---------
        clip : bool

        Returns
        -------
        None
        """
        self.grid_kw = customize_grid(self.grid_kw, self.bbox, clip=clip)

    def describe(self, key, as_dataframe=True, raw=False):
        """
        describe returns details about the coverage specified by key. Details
        include spatial bounding box, time coverage, time resolution, variable
        label, and a short description.

        DescribeCoverage with a COVERAGE should be faster than descriptions
        because it only returns a small xml chunk. Currently, DescribeCoverage
        with a COVERAGE specified is unreliable because of malformed xml. If
        this fails, describe will instead request all coverages and query the
        specific coverage.

        Arguments
        ---------
        as_dataframe : bool
            Defaults to True and descriptions are returned as a dataframe.
            If False, returns a list of elements.
        raw : bool
            Return raw xml instead of parsing. Useful for debugging.

        Returns
        -------
        coverages : pandas.DataFrame or list
            dataframe or list of parsed descriptions

        Example
        -------
            df = rsigapi.describe('airnow.no2')
            print(df.to_csv())
            # ,name,label,description,bbox_str,beginPosition,timeResolution
            # 0,no2,no2(ppb),UTC hourly mean surface measured nitrogen ...,
            # ... -157 21 -51 64,2003-01-02T00:00:00Z,PT1H
        """
        import warnings

        if key not in self._description:
            r = legacy_get(
                f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION='
                f'1.0.0&REQUEST=DescribeCoverage&COVERAGE={key}&compress=1'
            )
            self._description[key] = r.text

        if raw:
            return self._description[key]

        try:
            coverages = coverages_from_xml(self._description[key])
        except Exception as e:
            warnings.warn(str(e) + '; using descriptions')
            return self.descriptions().query(f'name == "{key}"')

        if as_dataframe:
            coverages = pd.DataFrame.from_records(coverages)
            coverages['prefix'] = coverages['name'].apply(
                lambda x: x.split('.')[0]
            )
            coverages = coverages.drop('tag', axis=1)

        return coverages

    def descriptions(self, as_dataframe=True, refresh=False, verbose=0):
        """
        Experimental and may change.

        descriptions returns details about all coverages. Details include
        spatial bounding box, time coverage, time resolution, variable label,
        and a short description.

        Currently, parses capabilities using xml.etree.ElementTree and returns
        coverages from details available in CoverageOffering elements from
        DescribeCoverage.

        Currently cleaning up data xml elements that are bad and doing a
        per-coverage parsing to increase fault tolerance in the xml.

        Arguments
        ---------
        as_dataframe : bool
            Defaults to True and descriptions are returned as a dataframe.
            If False, returns a list of elements.
        refresh : bool
            If True, get new copy and save to ~/.pyrsig/descriptons.xml
            If False (default), reload from saved if available.
        verbose : int
            If verbose is greater than 0, show warnings from parsing.

        Returns
        -------
        coverages : pandas.DataFrame or list
            dataframe or list of parsed descriptions

        Example
        -------

            rsigapi = pyrsig.RsigApi()
            desc = rsigapi.descriptions()
            print(desc.query('prefix == "tropomi"').name.unique())
            # ['tropomi.nrti.no2.nitrogendioxide_tropospheric_column'
            #  ... 43 other name here
            #  'tropomi.rpro.ch4.methane_mixing_ratio_bias_corrected']
        """
        import re
        import pandas as pd
        import warnings
        import os

        descpath = os.path.expanduser('~/.pyrsig/DescribeCoverage.csv')
        if not refresh and as_dataframe:
            if self._coveragesdf is not None:
                return self._coveragesdf
            elif os.path.exists(descpath):
                self._coveragesdf = pd.read_csv(descpath)
                return self._coveragesdf

        print('Refreshing descriptions...')
        # Start Cleaning Section
        # BHH 2023-05-10
        # This section provides "cleaning" to the xml content provided by
        # DescribeCoverage. This should not have to happen and should be
        # removable at some point in the future.
        # Working with TP to fix xml

        descmidre = re.compile(
            r'\</CoverageDescription\>.+?\<CoverageDescription.+?\>',
            flags=re.MULTILINE + re.DOTALL
        )
        mismatchtempre = re.compile(
            r'\</lonLatEnvelope\>\s+\</spatialDomain\>',
            flags=re.MULTILINE + re.DOTALL
        )

        # Regex, replacement
        resubsdesc = [
            (descmidre, ''),  # concated coverages have extra open/close tags
            (re.compile('<='), '&lt;='),  # associated with <= 32 in Modis
            (re.compile('qa_value <'), 'qa_value &lt;'),  # w/ tropomi.ntri
            (
                mismatchtempre,
                '</lonLatEnvelope><domainSet><spatialDomain></spatialDomain>',
            ),  # Missing open block for spatialDomain in goes (eg imager.calb)
            (
                re.compile(r'</CoverageOffering>\s+</CoverageOfferingBrief>'),
                '</CoverageOffering>',
            ),  # Ceiliometers have wrong opening tags and extra close tag
            (
                re.compile('CoverageOfferingBrief'), 'CoverageOffering'
            ),  # Ceiliometers have wrong opening tags and extra close tag
            (
                re.compile(
                    r'<rangeSet>\s+<RangeSet>\s+<supportedCRSs>',
                    flags=re.MULTILINE + re.DOTALL
                ),
                '<rangeSet><RangeSet></RangeSet></rangeSet><supportedCRSs>'
            ),  # Ceiliometers have missing rangeset content and closing tags
        ]

        if self._describecoverages is None or refresh:
            if verbose > 1:
                print('Requesting...', flush=True)
            self._describecoverages = legacy_get(
                f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION='
                '1.0.0&REQUEST=DescribeCoverage'
            ).text

            ctext = self._describecoverages

            for reg, sub in resubsdesc:
                ctext = reg.sub(sub, ctext)

            # End Cleaning Section
            self._describecoverages = ctext

        ctext = self._describecoverages

        # Selecting coverages and removing garbage when necessary.
        cleanre = re.compile(
            r'\</name\>.+?\</CoverageOffering\>',
            flags=re.MULTILINE + re.DOTALL
        )
        # <CoverageOffering>.+?</CoverageOffering>
        coverre = re.compile(
            r'\<CoverageOffering\>.+?\</CoverageOffering\>',
            flags=re.MULTILINE + re.DOTALL
        )

        coverages = []
        limited_details = []
        for rex in coverre.finditer(ctext):
            secttxt = ctext[rex.start():rex.end()]
            secttxt = (
                '<CoverageDescription version="1.0.0"'
                + ' xmlns="http://www.opengeospatial.org/standards/wcs"'
                + ' xmlns:gml="http://www.opengis.net/gml"'
                + ' xmlns:xlink="http://www.w3.org/1999/xlink">'
                + secttxt + '</CoverageDescription>'
            )
            try:
                coverage = coverages_from_xml(secttxt)
                coverages.extend(coverage)
            except Exception as e:
                try:
                    secttxt = cleanre.sub(
                        '</name></CoverageOffering>', secttxt
                    )
                    coverage = coverages_from_xml(secttxt)
                    coverages.extend(coverage)
                    limited_details.append(coverage[0]["name"])
                except Exception as e2:
                    # If a secondary error was raised, print it... but raise
                    # the original error
                    print(e)
                    raise e2

        nlimited = len(limited_details)
        if nlimited > 0 and verbose > 0:
            limitedstr = ', '.join(limited_details)
            warnings.warn(
                f'Limited details for {nlimited} coverages: {limitedstr}'
            )

        if as_dataframe:
            coverages = pd.DataFrame.from_records(coverages)
            coverages['bbox_str'] = coverages['bbox_str'].fillna(
                '-180 -90 180 90'
            )
            coverages['endPosition'] = coverages['endPosition'].fillna('now')
            coverages['prefix'] = coverages['name'].apply(
                lambda x: x.split('.')[0]
            )
            coverages = coverages.drop('tag', axis=1)
            self._coveragesdf = coverages
            # If you have arrived here, it means the file did not exist
            # or was intended to be refreshed. So, make it.
            os.makedirs(os.path.dirname(descpath), exist_ok=True)
            self._coveragesdf.to_csv(descpath, index=False)

        return coverages

    def capabilities(self, as_dataframe=True, refresh=False, verbose=0):
        """
        At this time, the capabilities does not list cmaq.*
        """
        import re
        import pandas as pd
        import os
        import io

        cappath = os.path.expanduser('~/.pyrsig/GetCapabilities.csv')
        if not refresh and as_dataframe:
            if self._capabilitiesdf is not None:
                return self._capabilitiesdf
            elif os.path.exists(cappath):
                self._capabilitiesdf = pd.read_csv(cappath)
                return self._capabilitiesdf

        if refresh or self._capabilities is None:
            self._capabilities = legacy_get(
                f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION='
                '1.0.0&REQUEST=GetCapabilities&compress=1'
            )

        if as_dataframe:
            os.makedirs(os.path.dirname(cappath), exist_ok=True)
            cre = re.compile(
                '<CoverageOfferingBrief>.+?</CoverageOfferingBrief>',
                re.DOTALL + re.M
            )
            gre = re.compile(
                r'<lonLatEnvelope srsName="WGS84\(DD\)">\s*<gml:pos>(.+?)'
                + r'</gml:pos>\s*<gml:pos>(.+?)</gml:pos>\s*</lonLatEnvelope>',
                re.M
            )
            tre = re.compile(r'>\s+<', re.M)
            ctext = self._capabilities.text
            ctext = '\n'.join(cre.findall(ctext))
            ctext = gre.sub(r'<bbox_str>\1 \2</bbox_str>', ctext)
            ctext = tre.sub(r'><', ctext)
            # Cleanup... for known issues
            ctext = ctext.replace('>yyy', '>')
            ctext = ctext.replace('<=', 'less than or equal to ')
            ctext = ctext.replace('qa_value < 0', 'qa_value less than 0')
            ctext = ctext.replace('>0=', 'greater than 0 =')
            ctext = ctext.replace('<0=', 'less than 0 = ')
            # version 1.5
            if hasattr(pd, 'read_xml'):
                ctext = f"""<?xml version="1.0" encoding="UTF-8" ?>
                <WCS_Capabilities>
                {ctext}
                </WCS_Capabilities>"""
                capabilitiesdf = pd.read_xml(io.StringIO(ctext))
            else:
                ccsv = ctext.replace('"', '\'')
                ccsv = ccsv.replace('</name><label>', '","')
                ccsv = ccsv.replace('</label><description>', '","')
                ccsv = ccsv.replace('</description><bbox_str>', '","')
                ccsv = ccsv.replace(
                    '</bbox_str></CoverageOfferingBrief>', '"\n'
                )
                ccsv = ccsv.replace('<CoverageOfferingBrief><name>', '"')
                ccsv = 'name,label,description,bbox_str\n' + ccsv
                capabilitiesdf = pd.read_csv(io.StringIO(ccsv))

            capabilitiesdf['prefix'] = capabilitiesdf['name'].apply(
                lambda x: x.split('.')[0]
            )
            capabilitiesdf.to_csv(cappath, index=False)
            self._capabilitiesdf = capabilitiesdf
            return self._capabilitiesdf

        return self._capabilities

    def keys(self, offline=True):
        """
        Arguments
        ---------
        offline : bool
            If True, uses small cached set of tested coverages.
            If False, finds all coverages from capabilities service.

        """
        if offline:
            keys = tuple(_keys)
        else:
            keys = sorted(self.capabilities(refresh=True).name.unique())

        return keys

    def get_file(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, request='GetCoverage', compress=0, overwrite=None,
        verbose=0
    ):
        """
        Build url, outpath, and download the file. Returns outpath

        """
        if overwrite is None:
            overwrite = self.overwrite
        url, outpath = self._build_url(
            formatstr, key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=grid, request=request, compress=compress
        )
        if verbose > 0:
            print(url)

        _getfile(url, outpath, verbose=verbose, overwrite=overwrite)

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

        if key is None:
            raise ValueError('key must be specified')

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
        tempo_kw = self.tempo_kw
        viirsnoaa_kw = self.viirsnoaa_kw
        pandora_kw = self.pandora_kw
        calipso_kw = self.calipso_kw
        if compress is None:
            compress = self.compress

        wlon, slat, elon, nlat = bbox

        # If already gridded, do not use grid keywords
        nogridkw = any([key.startswith(pre) for pre in _noregrid_prefixes])

        if (grid and not nogridkw) and request == 'GetCoverage':
            gridstr = self._build_grid(grid_kw)
        else:
            gridstr = ''

        if key.startswith('viirsnoaa'):
            viirsnoaastr = '&MINIMUM_QUALITY={minimum_quality}'.format(
                **viirsnoaa_kw
            )
        else:
            viirsnoaastr = ''

        if key.startswith('pandora'):
            pandorastr = '&MINIMUM_QUALITY={minimum_quality}'.format(
                **pandora_kw
            )
        else:
            pandorastr = ''

        if key.startswith('calipso'):
            calipsostr = (
                '&MINIMUM_CAD={MINIMUM_CAD}'
                + '&MAXIMUM_UNCERTAINTY={MAXIMUM_UNCERTAINTY}'
            ).format(**calipso_kw)
        else:
            calipsostr = ''

        if key.startswith('tropomi'):
            tropomistr = (
                '&MINIMUM_QUALITY={minimum_quality}'
                '&MAXIMUM_CLOUD_FRACTION={maximum_cloud_fraction}'
            ).format(**tropomi_kw)
        else:
            tropomistr = ''

        if key.startswith('tempo.l2'):
            if tempo_kw['api_key'] == 'your_key_here':
                raise ValueError('''You must set the tempo_kw api_key
(e.g., api.tempo_kw["api_key"] = "...") before submitting a query.''')
            tempostr = (
                '&MAXIMUM_CLOUD_FRACTION={maximum_cloud_fraction}'
                '&MINIMUM_QUALITY={minimum_quality}&KEY={api_key}'
            ).format(**tempo_kw)
        else:
            tempostr = ''

        if key.startswith('purpleair'):
            if purpleair_kw['api_key'] == 'your_key_here':
                raise ValueError('''You must set the purpleair_kw api_key
(e.g., api.purpleair_kw["api_key"] = "9...") before submitting a query.''')
            purpleairstr = (
                '&OUT_IN_FLAG={out_in_flag}&MAXIMUM_DIFFERENCE='
                '{maximum_difference}&MAXIMUM_RATIO={maximum_ratio}'
                '&AGGREGATE={freq}&MINIMUM_AGGREGATION_COUNT_PERCENTAGE='
                '{agg_pct}&DEFAULT_HUMIDITY={default_humidity}&KEY={api_key}'
            ).format(**purpleair_kw)
        else:
            purpleairstr = ''

        if any([key.startswith(pre) for pre in _nocorner_prefixes]):
            cornerstr = ''
        else:
            cornerstr = f'&CORNERS={corners}'

        if any([key.startswith(pre) for pre in _nolonlats_prefixes]):
            nolonlatsstr = '&NOLONLATS=1'
        else:
            nolonlatsstr = ''

        url = (
            f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION=1.0.0'
            f'&REQUEST={request}&FORMAT={formatstr}'
            f'&TIME={bdate:%Y-%m-%dT%H:%M:%SZ}/{edate:%Y-%m-%dT%H:%M:%SZ}'
            f'&BBOX={wlon},{slat},{elon},{nlat}'
            f'&COVERAGE={key}'
            f'&COMPRESS={compress}'
        ) + (
            purpleairstr + viirsnoaastr + tropomistr + tempostr + pandorastr
            + calipsostr + gridstr + cornerstr + nolonlatsstr
        )

        outpath = (
            f'{self.workdir}/{key}_{bdate:%Y-%m-%dT%H%M%SZ}'
            f'_{edate:%Y-%m-%dT%H%M%SZ}'
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
        GDTYP = grid_kw.get('GDTYP', 2)
        if GDTYP == 1:
            projstr = '&LONLAT=1'
        elif GDTYP == 2:
            projstr = '&LAMBERT={P_ALP},{P_BET},{XCENT},{YCENT}'
        elif GDTYP == 6:
            projstr = '&STEREOGRAPHIC={XCENT},{YCENT},{P_BET}'
        elif GDTYP == 7:
            projstr = '&MERCATOR={P_GAM}'
        else:
            raise KeyError('GDTYP only implemented for ')

        gridstr = (
            '&REGRID=weighted'
            + projstr
            + '&ELLIPSOID={earth_radius},{earth_radius}'
            + '&GRID={NCOLS},{NROWS},{XORIG},{YORIG},{XCELL},{YCELL}'
        )
        if grid_kw.get('REGRID_AGGREGATE', 'None').strip() != 'None':
            gridstr += "&REGRID_AGGREGATE={REGRID_AGGREGATE}"

        return gridstr.format(**grid_kw)

    def to_dataframe(
        self, key=None, bdate=None, edate=None, bbox=None, unit_keys=True,
        parse_dates=False, withmeta=False, verbose=0, backend='ascii'
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
        withmeta: bool
          If True, add 'GetMetadata' results as a "metadata" attribute of the
          dataframe. This is useful for understanding the underlying datasets
          used to create the result.
        verbose : int
          level of verbosity

        Returns
        -------
        df : pandas.DataFrame
            Results from download

        """
        from . import xdr
        assert backend in {'ascii', 'xdr'}
        outpath = self.get_file(
            backend, key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=False, verbose=verbose,
            compress=1
        )
        if backend == 'ascii':
            df = pd.read_csv(outpath, delimiter='\t', na_values=[-9999., -999])
        else:
            df = xdr.from_xdrfile(outpath, na_values=[-9999., -999])

        if withmeta:
            metapath = self.get_file(
                'ascii', key=key, bdate=bdate, edate=edate, bbox=bbox,
                grid=False, verbose=verbose, request='GetMetadata',
                compress=1
            )
            metatxt = open(metapath, 'r').read()
            df.attrs['metadata'] = metatxt

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
        self, key=None, bdate=None, edate=None, bbox=None, withmeta=False,
        removegz=False, verbose=0
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
        withmeta : bool
          If True, add 'GetMetadata' results at an attribute "metadata" to the
          netcdf file. This is useful for understanding the underlying datasets
          used to create the result.
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
        if not self.overwrite and os.path.exists(outpath[:-3]):
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

        if withmeta:
            metapath = self.get_file(
                'netcdf-ioapi', key=key, bdate=bdate, edate=edate, bbox=bbox,
                grid=True, compress=1, request='GetMetadata', verbose=verbose
            )
        else:
            metapath = None

        f = open_ioapi(outpath[:-3], metapath=metapath)
        if removegz:
            os.remove(outpath)

        return f

    def to_netcdf(
        self, key=None, bdate=None, edate=None, bbox=None, grid=False,
        withmeta=False, removegz=False, verbose=0
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
        grid : bool
          Add column and row variables with grid assignments.
        withmeta : bool
          If True, add 'GetMetadata' results at an attribute "metadata" to the
          netcdf file.
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
        import xarray as xr

        # always use compression for network speed.
        outpath = self.get_file(
            'netcdf-coards', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=grid, compress=1, verbose=verbose
        )
        # Uncompress the netcdf file. If encoding is available, apply it
        if not self.overwrite and os.path.exists(outpath[:-3]):
            print('Using cached:', outpath[:-3])
        else:
            with gzip.open(outpath, 'rb') as f_in:
                with open(outpath[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    f_out.flush()

        f = xr.open_dataset(outpath[:-3])

        if withmeta:
            metapath = self.get_file(
                'netcdf-coards', key=key, bdate=bdate, edate=edate, bbox=bbox,
                grid=grid, compress=1, request='GetMetadata', verbose=verbose
            )
            with open(metapath, 'r') as metaf:
                metatxt = metaf.read()
            f.attrs['metadata'] = metatxt

        if removegz:
            os.remove(outpath)

        return f


class RsigGui:
    @classmethod
    def from_api(cls, api):
        gui = cls()
        gui._bbwe.value = api.bbox[::2]
        gui._bbsn.value = api.bbox[1::2]
        if api.bdate is not None:
            bdate = pd.to_datetime(api.bdate)
            gui._dates.value = bdate.floor('1D')
            gui._hours.value = (
                bdate - bdate.floor('1D')
            ).total_seconds() // 3600
        if api.edate is not None:
            edate = pd.to_datetime(api.edate)
            gui._datee.value = edate.floor('1D')
            gui._houre.value = (
                edate - edate.floor('1D')
            ).total_seconds() // 3600
        if api.key is not None:
            gui._prodd.value = api.key
        if api.grid_kw is not None:
            gui._gridd.value = api.grid_kw['GDNAM']
        if api.workdir is not None:
            gui._workd.value = api.workdir

        return gui

    def __init__(self):
        """
        RsigGui Object designed for IPython with ipywidgets in Jupyter

        Example:
        gui = RsigGui()
        gui.form  # As last line in cell, displays controls for user
        gui.plotopts()  # Plots current options
        gui.check()  # Check bounding box and date options make sense
        rsigapi = gui.get_api() # Convert gui to standard api
        # proceed with normal RsigApi usage
        """
        from datetime import date
        from ipywidgets import Layout, Box, Dropdown, Label, FloatRangeSlider
        from ipywidgets import DatePicker, Textarea, BoundedIntText, Output

        api = RsigApi()
        descdf = api.descriptions().copy()
        descdf['begin'] = descdf['beginPosition']
        descdf['end'] = descdf['endPosition']
        descdf['bbox'] = descdf['bbox_str']
        descdf['opt_txt'] = descdf.apply(
            lambda x: '{name}\t({begin}-{end})\t({bbox})'.format(**x), axis=1
        )
        descdf['sort'] = ~descdf.name.isin(api.keys())
        prodopts = descdf.sort_values(by=['sort', 'name'], ascending=True)[
            ['opt_txt', 'name']
        ].values.tolist()
        l100 = Layout(width='95%')
        l50 = Layout(width='30em')
        self._prodd = prodd = Dropdown(
            options=prodopts, description='Product', layout=l100,
            value='tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        self._gridd = gridd = Dropdown(
            options=list(_def_grid_kw), value='12US1', description='grid',
            layout=l50
        )
        self._dates = datesa = DatePicker(
            description='Start Date', disabled=False, layout=l50,
            value=(
                date.today()
                - pd.to_timedelta('7d')
            )
        )
        self._datee = dateea = DatePicker(
            description='End Date', disabled=False, value=datesa.value,
            layout=l50
        )
        self._hours = hours = BoundedIntText(
            min=0, max=23, value=0, description='Start HR', layout=l50
        )
        self._houre = houre = BoundedIntText(
            min=0, max=23, value=23, description='End HR', layout=l50
        )
        self._bbsn = FloatRangeSlider(
            min=-90, max=90, value=(24, 50), description='South-North',
            layout=l100
        )
        self._bbwe = FloatRangeSlider(
            min=-180, max=180, value=(-126, -66), description='West-East',
            layout=l100
        )
        self._workd = workd = Textarea(
            value='.', description='Work Dir', layout=l100
        )
        self._out = Output(layout=l100)
        form_items = [
            Label(value='RSIG Options'),
            prodd, self._bbsn, self._bbwe,
            Box([datesa, hours]), Box([dateea, houre]),
            gridd, workd, self._out
        ]
        [
            fi.observe(self._update_out, names='value')
            for fi in form_items + [datesa, hours, dateea, houre]
        ]
        self._form = Box(form_items, layout=Layout(
            display='flex', flex_flow='column', border='solid 2px',
            align_items='stretch', width='100%'
        ))

    def _update_out(self, *args):
        from IPython.display import clear_output, display
        fig = self.plotopts()
        with self._out:
            clear_output(wait=True)
            display(fig)

    def date_range(self):
        import pandas as pd
        return pd.date_range(self.bdate, self.edate)

    @property
    def form(self):
        return self._form

    @property
    def key(self):
        return self._prodd.value

    @property
    def bdate(self):
        import pandas as pd
        return (
            pd.to_datetime(self._dates.value)
            + pd.to_timedelta(self._hours.value, unit='H')
        )

    @property
    def edate(self):
        import pandas as pd
        hms = self._houre.value * 3600 + 3599
        return pd.to_datetime(
            self._datee.value
        ) + pd.to_timedelta(hms, unit='s')

    @property
    def grid_kw(self):
        return self._gridd.value

    @property
    def bbox(self):
        w, e = self._bbwe.value
        s, n = self._bbsn.value
        return (w, s, e, n)

    @property
    def workdir(self):
        return self._workd.value

    def plotopts(self):
        import pycno
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        bbw, bbs, bbe, bbn = self.bbox
        c = {True: 'g', False: 'r'}.get(self.check(), 'r')
        ax.plot(
            [bbw, bbe, bbe, bbw, bbw],
            [bbs, bbs, bbn, bbn, bbs],
            color=c
        )
        if c == 'r':
            ax.text(
                .5, .5, 'Invalid Options', horizontalalignment='center',
                transform=ax.transAxes, color='r', fontsize=30,
                bbox={'edgecolor': c, 'facecolor': 'white'}
            )
        fig.suptitle(f'Query Options: {self.key}, {self.grid_kw}')
        ax.set(title=f'{self.bdate:%FT%H:%M:%S} {self.edate:%FT%H:%M:%S}')
        pycno.cno().drawstates(ax=ax)
        return fig

    def get_api(self):
        rsigapi = RsigApi(
            key=self.key, bdate=self.bdate, edate=self.edate,
            bbox=self.bbox, grid_kw=self.grid_kw, workdir=self.workdir
        )
        return rsigapi

    def check(self, action='return'):
        bbw, bbs, bbe, bbn = self.bbox
        iswe = bbw < bbe
        issn = bbs < bbn
        isbe = self.bdate <= self.edate

        if not iswe:
            _actionf('West is East of East', action)
        if not issn:
            _actionf('South is North of North', action)
        if not isbe:
            _actionf('bdate is later than edate', action)

        return iswe & issn & isbe
