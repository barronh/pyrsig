__all__ = ['def_grid_kw']
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
    'global_0pt01': dict(
        GDNAM='GLOBAL', GDTYP=1, NCOLS=36000, NROWS=18000,
        XORIG=-180, YORIG=-90, XCELL=0.01, YCELL=0.01,
        P_ALP=0., P_BET=0., P_GAM=0., XCENT=0., YCENT=0.
    ),
    'global_0pt02': dict(
        GDNAM='GLOBAL', GDTYP=1, NCOLS=18000, NROWS=9000,
        XORIG=-180, YORIG=-90, XCELL=0.02, YCELL=0.02,
        P_ALP=0., P_BET=0., P_GAM=0., XCENT=0., YCENT=0.
    ),
    'TEMPOL3_0pt02': dict(
        GDNAM='GLOBAL', GDTYP=1, NCOLS=7750, NROWS=2950,
        XORIG=-170, YORIG=14, XCELL=0.02, YCELL=0.02,
        P_ALP=0., P_BET=0., P_GAM=0., XCENT=0., YCENT=0.
    ),
    'HRRR3K': dict(
        GDNAM='HRRR3K', GDTYP=2, NCOLS=1799, NROWS=1059,
        XORIG=-2699020.14252193, YORIG=-1588806.15255666,
        XCELL=3000., YCELL=3000., earth_radius=6371229.,
        P_ALP=38.5, P_BET=38.5, P_GAM=-97.5, XCENT=-97.5, YCENT=38.5,
    ),
}

shared_grid_kw = dict(
    VGTYP=7, VGTOP=5000., NLAYS=35, earth_radius=6370000., g=9.81, R=287.04,
    A=50., T0=290, P0=1000e2, REGRID_AGGREGATE='None'
)

for key in def_grid_kw:
    for pk, pv in shared_grid_kw.items():
        def_grid_kw[key].setdefault(pk, pv)
