__all__ = ['get_descriptions', 'make_descriptions']

from os.path import dirname, join, expanduser, exists

_userdescpath = expanduser('~/.pyrsig/DescribeCoverage.csv')
_pkgdescpath = join(dirname(__file__), 'DescribeCoverage.csv')
_describecoverages = None


def get_descriptions(server='ofmpub.epa.gov', refresh=False, verbose=0):
    """
    Make a refreshed user copy or find the installed package version of
    descriptions.

    Arguments
    ---------
    server : str
        Path to server (ofmpub.epa.gov or maple.hesc.epa.gov)
    refresh : bool
        If True, refresh a user copy of descriptions from DescribeCoverage
    verbose : int
        Level of verbosity

    Returns
    -------
    descdf : pandas.DataFrame
        Description of known coverages.
    """
    import pandas as pd
    if refresh:
        make_descriptions(_userdescpath, server=server, verbose=verbose)

    if exists(_userdescpath):
        descpath = _userdescpath
    elif exists(_pkgdescpath):
        descpath = _pkgdescpath
    elif not refresh:
        # None found, force refresh
        return get_descriptions(server=server, refresh=True, verbose=verbose)
    df = pd.read_csv(descpath)
    return df


def make_descriptions(descpath, server='ofmpub.epa.gov', verbose=0):
    """
    Arguments
    ---------
    descpath : str
        Path to make a copy of descriptions from DescribeCoverage
    server : str
        Path to server (ofmpub.epa.gov or maple.hesc.epa.gov)
    verbose : int
        Level of verbosity

    Returns
    -------
    None
    """
    global _describecoverages
    import re
    import pandas as pd
    import warnings
    import os
    from ..utils import coverages_from_xml, legacy_get

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

    if _describecoverages is None:
        if verbose > 1:
            print('Requesting...', flush=True)
        _describecoverages = legacy_get(
            f'https://{server}/rsig/rsigserver?SERVICE=wcs&VERSION='
            '1.0.0&REQUEST=DescribeCoverage'
        ).text

        ctext = _describecoverages

        for reg, sub in resubsdesc:
            ctext = reg.sub(sub, ctext)

        # End Cleaning Section
        _describecoverages = ctext

    ctext = _describecoverages

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

    coverages = pd.DataFrame.from_records(coverages)
    coverages['bbox_str'] = coverages['bbox_str'].fillna(
        '-180 -90 180 90'
    )
    coverages['endPosition'] = coverages['endPosition'].fillna('now')
    coverages['prefix'] = coverages['name'].apply(
        lambda x: x.split('.')[0]
    )
    coverages = coverages.drop('tag', axis=1)

    # If you have arrived here, it means the file did not exist
    # or was intended to be refreshed. So, make it.
    os.makedirs(os.path.dirname(descpath), exist_ok=True)
    coverages.to_csv(descpath, index=False)
