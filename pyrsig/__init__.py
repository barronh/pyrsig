__all__ = ['RsigApi', 'RsigGui', 'open_ioapi', 'open_mfioapi', 'cmaq']
__version__ = '0.10.0'

from . import cmaq
from .cmaq import open_ioapi, open_mfioapi
import pandas as pd
from .utils import customize_grid, def_grid_kw as _def_grid_kw
from .utils import coverages_from_xml, legacy_get


_corner_prefixes = (
    'gasp', 'goes', 'modis', 'omibehr', 'tempo', 'tropomi', 'viirs'
)
_nolonlats_prefixes = ('cmaq', 'regridded')
_noregrid_prefixes = ('cmaq', 'regridded')
_shpxdrprefixes = ['hms.']
_shpbinprefixes = [
    'landuse.atlantic.population_iclus',
    'landuse.gulf.population_iclus',
    'landuse.pacific.population_iclus',
]


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
          IOAPI mapping parameters. For details, look at the defaults:
            import pyrsig; print(pyrsig.RsigApi().grid_kw)
          The REGRID_AGGREGATE defines how the regridded values are aggregated
          in time. Options are None (default), daily, or all.
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
            'minimum_quality': 'normal'
            'maximum_cloud_fraction': 1.0
            'maximum_solar_zenith_angle': 70.
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
        self._capabilities = None
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
        tempo_kw.setdefault('maximum_solar_zenith_angle', 70.)

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

    def descriptions(self, refresh=False, verbose=0):
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
        from .data import get_descriptions
        if self._coveragesdf is None:
            self._coveragesdf = get_descriptions(
                server=self.server, refresh=refresh
            )
        return self._coveragesdf

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
        descdf = self.descriptions(refresh=not offline)
        keys = tuple(sorted(descdf['name'].unique()))
        return keys

    def get_file(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, corners=None, request='GetCoverage', compress=0,
        overwrite=None, verbose=0
    ):
        """
        Build url, outpath, and download the file. Returns outpath

        """
        from .utils import get_file
        if overwrite is None:
            overwrite = self.overwrite
        url, outpath = self._build_url(
            formatstr, key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=grid, request=request, compress=compress, corners=corners
        )
        if verbose > 0:
            print(url)

        get_file(url, outpath, verbose=verbose, overwrite=overwrite)

        return outpath

    def _build_url(
        self, formatstr, key=None, bdate=None, edate=None, bbox=None,
        grid=False, corners=None, request='GetCoverage',
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

        if corners is None:
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
                '&MAXIMUM_SOLAR_ZENITH_ANGLE={maximum_solar_zenith_angle}'
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

        if corners == 1:
            if any([key.startswith(pre) for pre in _corner_prefixes]):
                cornerstr = f'&CORNERS={corners}'
            else:
                cornerstr = ''
        else:
            cornerstr = ''

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
        parse_dates=False, corners=None, withmeta=False, verbose=0,
        backend='ascii', grid=False
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
        from . import bin
        assert backend in {'ascii', 'xdr', 'bin'}
        if any([key.startswith(pfx) for pfx in _shpxdrprefixes]):
            backend = 'xdr'
        elif any([key.startswith(pfx) for pfx in _shpbinprefixes]):
            backend = 'bin'

        outpath = self.get_file(
            backend, key=key, bdate=bdate, edate=edate, bbox=bbox,
            grid=grid, verbose=verbose, corners=corners,
            compress=1
        )
        if backend == 'ascii':
            df = pd.read_csv(outpath, delimiter='\t', na_values=[-9999., -999])
        elif backend == 'xdr':
            df = xdr.from_xdrfile(outpath, na_values=[-9999., -999])
        elif backend == 'bin':
            df = bin.from_binfile(outpath)
        else:
            raise KeyError(f'format {backend} unknown; use xdr, bin or ascii')

        if withmeta:
            metapath = self.get_file(
                'ascii', key=key, bdate=bdate, edate=edate, bbox=bbox,
                grid=grid, verbose=verbose, request='GetMetadata',
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


# Add easy access defaults
_defapi = RsigApi()
descriptions = _defapi.descriptions
to_dataframe = _defapi.to_dataframe
to_ioapi = _defapi.to_ioapi
to_netcdf = _defapi.to_netcdf
