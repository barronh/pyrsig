__all__ = ['RsigApi', 'RsigGui', 'open_ioapi']
__version__ = '0.4.4'

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
    'NORTHSOUTHAM': dict(
        GDNAM='NORTHSOUTHAM', GDTYP=7, NCOLS=179, NROWS=154,
        XORIG=251759.25, YORIG=-1578187., XCELL=27000., YCELL=27000.,
        P_ALP=0., P_BET=0., P_GAM=-98., XCENT=-98., YCENT=0.
    ),
}

_shared_grid_kw = dict(
    VGTYP=7, VGTOP=5000., NLAYS=35, earth_radius=6370000., g=9.81, R=287.04,
    A=50., T0=290, P0=1000e2, REGRID_AGGREGATE='None'
)

for key in _def_grid_kw:
    for pk, pv in _shared_grid_kw.items():
        _def_grid_kw[key].setdefault(pk, pv)

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
    'pandora.L2_rfus5p1_8.formaldehyde_vertical_column_amount_uncertainty'
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
    'tempo.proxy_l2.no2.vertical_column_total',
    'tempo.proxy_l2.no2.vertical_column_total_uncertainty',
    'tempo.proxy_l2.no2.vertical_column_troposphere',
    'tempo.proxy_l2.no2.vertical_column_stratosphere',
    'tempo.proxy_l2.no2.amf_total',
    'tempo.proxy_l2.no2.amf_total_uncertainty',
    'tempo.proxy_l2.no2.amf_troposphere',
    'tempo.proxy_l2.no2.amf_stratosphere',
    'tempo.proxy_l2.no2.ground_pixel_quality_flag'
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

_nocorner_prefixes = ('airnow', 'aqs', 'purpleair', 'pandora', 'cmaq')
_nolonlats_prefixes = ('cmaq',)
_noregrid_prefixes = ('cmaq',)


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
                record['bbox_str'] = envtxt

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

    ssl._create_default_https_context = ssl._create_unverified_context

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


def get_proj4(attrs, earth_radius=6370000.):
    """
    Create a proj4 formatted grid definition using IOAPI attrs and earth_radius

    Arguments
    ---------
    attrs : dict-like
        Mappable of IOAPI properties that supports the items method
    earth_radius : float
        Assumed radius of the earth. 6370000 is the WRF default.

    Returns
    -------
    projstr : str
        proj4 formatted string such that the domain southwest corner starts at
        (0, 0) and ends at (NCOLS, NROWS)
    """
    props = {k: v for k, v in attrs.items()}
    props['x_0'] = -props['XORIG']
    props['y_0'] = -props['YORIG']
    props.setdefault('earth_radius', earth_radius)

    if props['GDTYP'] == 2:
        projstr = (
            '+proj=lcc +lat_1={P_ALP} +lat_2={P_BET} +lat_0={YCENT}'
            ' +lon_0={XCENT} +R={earth_radius} +x_0={x_0} +y_0={y_0}'
            ' +to_meter={XCELL} +no_defs'
        ).format(**props)
    elif props['GDTYP'] == 6:
        projstr = (
            '+proj=stere +lat_0={lat_0} +lat_ts={P_BET} +lon_0={XCENT}'
            + ' +x_0={x_0} +y_0={y_0} +R={earth_radius} +to_meter={XCELL}'
            + ' +no_defs'
        ).format(lat_0=props['P_ALP'] * 90, **props)
    elif props['GDTYP'] == 7:
        projstr = (
            '+proj=merc +R={earth_radius} +lat_ts=0 +lon_0={XCENT}'
            + ' +x_0={x_0} +y_0={y_0} +to_meter={XCELL}'
            + ' +no_defs'
        ).format(**props)
    else:
        raise ValueError('GDTYPE {GDTYP} not implemented'.format(**props))

    return projstr


def customize_grid(grid_kw, bbox, clip=True):
    """
    Redefine grid_kw to cover bbox by removing extra rows and columns and
    redefining XORIG, YORIG, NCOLS and NROWS.

    Arguments
    ---------
    grid_kw : dict or str
        If str, must be a known grid in default grids.
        If dict, must include all IOAPI grid metadata properties
    bbox : tuple
        wlon, slat, elon, nlat in decimal degrees (-180 to 180)
    clip : bool
        If True, limit grid to original grid bounds

    Returns
    -------
    ogrid_kw : dict
        IOAPI grid metadata properties with XORIG/YORIG and NCOLS/NROWS
        adjusted such that it only covers bbox or (if clip) only covers
        the portion of bbox covered by the original grid_kw.
    """
    import pyproj
    import numpy as np

    if isinstance(grid_kw, str):
        grid_kw = _def_grid_kw[grid_kw]

    ogrid_kw = {k: v for k, v in grid_kw.items()}
    proj4str = get_proj4(grid_kw)
    proj = pyproj.Proj(proj4str)
    llx, lly = proj(*bbox[:2])
    urx, ury = proj(*bbox[2:])
    midx, midy = proj((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    maxy = np.max([lly, ury, midy])
    miny = np.min([lly, ury, midy])
    maxx = np.max([llx, urx, midx])
    minx = np.min([llx, urx, midx])

    lli, llj = np.floor([minx, miny]).astype('i')
    uri, urj = np.ceil([maxx, maxy]).astype('i')
    if clip:
        lli, llj = np.maximum(0, [lli, llj])
        uri = np.minimum(grid_kw['NCOLS'], uri)
        urj = np.minimum(grid_kw['NROWS'], urj)
    ogrid_kw['XORIG'] = grid_kw['XORIG'] + lli * grid_kw['XCELL']
    ogrid_kw['YORIG'] = grid_kw['YORIG'] + llj * grid_kw['YCELL']
    ogrid_kw['NCOLS'] = uri - lli
    ogrid_kw['NROWS'] = urj - llj
    return ogrid_kw


def open_ioapi(path, metapath=None, earth_radius=6370000.):
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

    Returns
    -------
    ds : xarray.Dataset
        Dataset with IOAPI metadata
    """
    import xarray as xr
    import numpy as np
    import warnings

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


class RsigApi:
    def __init__(
        self, key=None, bdate=None, edate=None, bbox=None, grid_kw=None,
        tropomi_kw=None, purpleair_kw=None, viirsnoaa_kw=None,
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
          {'minimum_quality': 'high'} options include 'high' or 'medium')
        tropomi_kw : dict
          Dictionary of TropOMI filter parameters default
          {'minimum_quality': 75, 'maximum_cloud_fraction': 1.0} options
          are 0-100 and 0-1.
        purpleair_kw : dict
          Dictionary of purpleair filter parameters and api_key.
            'out_in_flag': 0, # options 0, 2, ''
            'freq': 'hourly', # options hourly, daily, monthly, yearly
            'maximum_difference': 5, # integer
            'maximum_ratio': 0.70, # float
            'agg_pct': 75, # 0-100
            'api_key': '<your key here>'
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
        because it only returns a smalll xml chunk. Currently, DescribeCoverage
        with a COVERAGE specified is unreliable because of malformed xml. If
        this fails, describe will instead request all coverages and query it
        for the specific coverage. This is much slower and is being addressed.

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
        import requests
        import warnings

        if key not in self._description:
            r = requests.get(
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

    def descriptions(self, as_dataframe=True, verbose=0):
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
        import requests

        if as_dataframe and self._coveragesdf is not None:
            return self._coveragesdf

        if self._describecoverages is None:
            if verbose > 1:
                print('Requesting...', flush=True)
            self._describecoverages = requests.get(
                f'https://{self.server}/rsig/rsigserver?SERVICE=wcs&VERSION='
                '1.0.0&REQUEST=DescribeCoverage&compress=1'
            ).text

        ctext = self._describecoverages

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
            (descmidre, ''),
            (re.compile('<='), '&lt;='),  # associated with <= 32 in Modis
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
        for reg, sub in resubsdesc:
            ctext = reg.sub(sub, ctext)

        # End Cleaning Section

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
            coverages['prefix'] = coverages['name'].apply(
                lambda x: x.split('.')[0]
            )
            coverages = coverages.drop('tag', axis=1)
            self._coveragesdf = coverages

        return coverages

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
            If True, uses small cached set of tested coverages.
            If False, finds all coverages from capabilities service.

        """
        if offline:
            keys = tuple(_keys)
        else:
            keys = []
            for line in self.capabilities().text.split('\n'):
                if line.startswith('            <name>'):
                    keys.append(line.split('name')[1][1:-2])

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
        viirsnoaa_kw = self.viirsnoaa_kw
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
            purpleairstr + viirsnoaastr + tropomistr + gridstr + cornerstr
            + nolonlatsstr
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
        if GDTYP == 2:
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
        parse_dates=False, withmeta=False, verbose=0
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
        outpath = self.get_file(
            'ascii', key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=False, verbose=verbose,
            compress=1
        )
        df = pd.read_csv(outpath, delimiter='\t', na_values=[-9999., -999])
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
        (
            gui._bbw.value,
            gui._bbs.value,
            gui._bbe.value,
            gui._bbn.value,
        ) = api.bbox
        if api.bdate is not None:
            gui._dates.value = api.bdate
        if api.edate is not None:
            gui._datee.value = api.edate
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
        from ipywidgets import Layout, Box, Dropdown, Label, FloatSlider
        from ipywidgets import DatePicker, Textarea
        form_item_layout = Layout(
            display='flex', flex_flow='row', justify_content='space-between'
        )

        prodopts = RsigApi().keys()
        self._prodd = prodd = Dropdown(options=prodopts)
        self._gridd = gridd = Dropdown(
            options=list(_def_grid_kw), value='12US1'
        )
        self._dates = datesa = DatePicker(
            description='Start Date', disabled=False,
            value=(
                date.today()
                - pd.to_timedelta('7d')
            )
        )
        self._datee = dateea = DatePicker(
            description='End Date', disabled=False, value=datesa.value
        )
        self._bbw = bbw = FloatSlider(min=-180, max=180, value=-126)
        self._bbe = bbe = FloatSlider(min=-180, max=180, value=-66)
        self._bbs = bbs = FloatSlider(min=-90, max=90, value=24)
        self._bbn = bbn = FloatSlider(min=-90, max=90, value=50)
        self._workd = workd = Textarea(value='.')
        form_items = [
            Box([Label(value='RSIG Options')], layout=form_item_layout),
            Box([
                Label(value='Data Product'), prodd
            ], layout=form_item_layout),
            Box([Label(value='Southest'), bbs, Label(value='Northest'), bbn]),
            Box([Label(value='Westest'), bbw, Label(value='Eastest'), bbe]),
            Box([
                Label(value='Date Start'), datesa,
                Label(value='End'), dateea
            ], layout=form_item_layout),
            Box([Label(value='Grid Option'), gridd], layout=form_item_layout),
            Box([
                Label(value='Working Directory'), workd
            ], layout=form_item_layout),
        ]

        self._form = Box(form_items, layout=Layout(
            display='flex', flex_flow='column', border='solid 2px',
            align_items='stretch', width='50%'
        ))

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
        return self._dates.value

    @property
    def edate(self):
        return self._datee.value

    @property
    def grid_kw(self):
        return self._gridd.value

    @property
    def bbox(self):
        return tuple([
            v.value
            for v in [self._bbw, self._bbs, self._bbe, self._bbn]
        ])

    @property
    def workdir(self):
        return self._workd.value

    def plotopts(self):
        import pycno
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        bbw, bbs, bbe, bbn = self.bbox
        ax.plot(
            [bbw, bbe, bbe, bbw, bbw],
            [bbs, bbs, bbn, bbn, bbs],
            color='r'
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
