def test_polygon():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bbox = (-80, 30, -60, 50)
        rsigapi = RsigApi(bdate='2023-03-01', bbox=bbox, workdir=td)
        ds = rsigapi.to_dataframe('hms.smoke')
        print(ds.shape)
    return ds


def test_swath():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bbox = (-80, 30, -60, 50)
        rsigapi = RsigApi(bdate='2023-07-01', bbox=bbox, workdir=td)
        dkey = 'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        ds = rsigapi.to_dataframe(dkey, backend='xdr')
        print(ds.shape)

    return


def test_site():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bbox = (-80, 30, -60, 50)
        rsigapi = RsigApi(bdate='2023-07-01', bbox=bbox, workdir=td)
        dkey = 'airnow.ozone'
        ds = rsigapi.to_dataframe(dkey, backend='xdr')
        print(ds.shape)

    return


def test_profile():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bbox = (-80, 30, -60, 50)
        rsigapi = RsigApi(bdate='2023-07-01', bbox=bbox, workdir=td)
        dkey = 'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount'
        ds = rsigapi.to_dataframe(dkey, backend='xdr')
        print(ds.shape)

    return


def test_point():
    from .. import RsigApi
    import tempfile
    return
    with tempfile.TemporaryDirectory() as td:
        bbox = (-80, 30, -60, 50)
        rsigapi = RsigApi(bdate='2023-07-01', bbox=bbox, workdir=td)
        # rsigapi.purpleair_kw['api_key'] = 'cannot test without api'
        dkey = 'purpleair.pm25_corrected'
        ds = rsigapi.to_dataframe(dkey, backend='xdr')
        print(ds.shape)
    return


def test_calipso():
    from .. import RsigApi
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bbox = (-150, 20, -60, 55)
        rsigapi = RsigApi(bdate='2022-07-01', bbox=bbox, workdir=td)
        # rsigapi.purpleair_kw['api_key'] = 'cannot test without api'
        dkey = 'calipso.l2_05kmapro.Total_Backscatter_Coefficient_532'
        ds = rsigapi.to_dataframe(dkey, backend='xdr')
        print(ds.shape)
    return


def test_grid():
    from .. import RsigApi
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bbox = (-180, 0, 180, 90)
        rsigapi = RsigApi(
            bdate='2016-07-01T17', edate='2016-07-01T17:59:59', bbox=bbox,
            workdir=td
        )
        # rsigapi.purpleair_kw['api_key'] = 'cannot test without api'
        dkey = 'cmaq.equates.conus.aconc.SFC_TMP'
        ds = rsigapi.to_dataframe(dkey, backend='xdr')
        print(ds.shape)
    return


def test_subsetgrid():
    from .. import RsigApi
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        bbox = (-150, 20, -60, 55)
        rsigapi = RsigApi(
            bdate='2016-07-01T17', edate='2016-07-01T17:59:59', bbox=bbox,
            workdir=td
        )
        # rsigapi.purpleair_kw['api_key'] = 'cannot test without api'
        dkey = 'cmaq.equates.hemi.aconc.SFC_TMP'
        ds = rsigapi.to_dataframe(dkey, backend='xdr')
        print(ds.shape)
    return


def test_regriddedswath():
    import warnings
    warnings.warn('skipped test_regriddedswath')
