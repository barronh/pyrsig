__all__ = ['get_proj4', 'customize_grid', 'def_grid_kw', 'shared_grid_kw']

def_grid_kw = {
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
    'global_1pt0': dict(
        GDNAM='GLOBAL', GDTYP=1, NCOLS=360, NROWS=180,
        XORIG=-180, YORIG=-90, XCELL=1., YCELL=1.,
        P_ALP=0., P_BET=0., P_GAM=0., XCENT=0., YCENT=0.
    ),
    'global_0pt1': dict(
        GDNAM='GLOBAL', GDTYP=1, NCOLS=3600, NROWS=1800,
        XORIG=-180, YORIG=-90, XCELL=0.1, YCELL=0.1,
        P_ALP=0., P_BET=0., P_GAM=0., XCENT=0., YCENT=0.
    ),
}

shared_grid_kw = dict(
    VGTYP=7, VGTOP=5000., NLAYS=35, earth_radius=6370000., g=9.81, R=287.04,
    A=50., T0=290, P0=1000e2, REGRID_AGGREGATE='None'
)

for key in def_grid_kw:
    for pk, pv in shared_grid_kw.items():
        def_grid_kw[key].setdefault(pk, pv)


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

    if props['GDTYP'] == 1:
        projstr = '+proj=lonlat +R={earth_radius}'.format(**props)
    elif props['GDTYP'] == 2:
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
        grid_kw = def_grid_kw[grid_kw]

    ogrid_kw = {k: v for k, v in grid_kw.items()}
    # Lonlat box must be treated separately
    if ogrid_kw['GDTYP'] == 1:
        llx, lly = bbox[:2]
        urx, ury = bbox[2:]
        ncols = int(np.ceil((urx - llx) / ogrid_kw['XCELL']) + 4)
        nrows = int(np.ceil((ury - lly) / ogrid_kw['YCELL']) + 4)
        xorig = (int(llx / ogrid_kw['XCELL']) - 1) * ogrid_kw['XCELL']
        yorig = (int(lly / ogrid_kw['YCELL']) - 1) * ogrid_kw['YCELL']
        ogrid_kw['NCOLS'] = ncols
        ogrid_kw['NROWS'] = nrows
        ogrid_kw['XORIG'] = xorig
        ogrid_kw['YORIG'] = yorig
        return ogrid_kw

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


def quickstats(df, refkey='obs'):
    """
    Simple utility to calculate quantiles, mean, correlation, bias statistics,
    and Index of Agreement. All statistics have the input data units except
    for nmb, fmb, correlation, and ioa -- all those are unitless.

    Arguments
    ---------
    df : pandas.DataFrame
        Must contain at least refkey as a column. All other columns will be
        treated as estimates.
    refkey : str
        Key to be used as the reference (or truth or observation) value.

    Returns
    -------
    statdf : pandas.DataFrame
        Contains results where each statistic is a row and columns are refkey
        and all the estimates

    Note:
    For nmb or fmb in percent, multiply by 100.
    """
    statsdf = df.describe()
    statsdf.loc['r'] = df.corr().loc[refkey]
    meands = statsdf.loc['mean']
    # Mean Bias [ppb] # same as ozone and CMAQ_O3
    statsdf.loc['mb'] = meands - meands[refkey]
    # Normalized Mean Bias [1]
    statsdf.loc['nmb'] = statsdf.loc['mb'] / meands[refkey]
    # Fractional Mean Bias [1]
    statsdf.loc['fmb'] = 2 * statsdf.loc['mb'] / meands.sum()
    # IOA [1]
    bias = df.subtract(df[refkey], axis=0)
    sqerr = bias**2
    apdev = df.subtract(meands[refkey]).abs()
    statsdf.loc['ioa'] = (
        1 - sqerr.sum() / (apdev.add(apdev[refkey], axis=0)**2).sum()
    )
    return statsdf
