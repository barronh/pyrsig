def _checkdf(df, hasmeta=False):
    assert len(df.shape) == 2  # require two-dimensional return
    assert df.shape[0] > 0  # require a row
    assert df.shape[1] > 0  # require a column
    if hasmeta:
        assert 'metadata' in df.attrs


def test_pandora():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2023-03-01', bbox=(-80, 30, -60, 50), workdir=td
        )
        df = rsigapi.to_dataframe(
            'pandora.L2_rnvh3p1_8.tropospheric_nitrogen_dioxide'
        )
        _checkdf(df)


def test_hmssmoke():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bbox = (-80, 30, -60, 50)
        rsigapi = RsigApi(bdate='2023-03-01', bbox=bbox, workdir=td)
        df = rsigapi.to_dataframe('hms.smoke')
        _checkdf(df)


def test_viirsnoaa():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', bbox=(-85, 30, -70, 45), workdir=td
        )
        df = rsigapi.to_dataframe('viirsnoaa.jrraod.AOD550')
        _checkdf(df)


def test_aqs_ozone_cache():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        # get fresh data
        df = rsigapi.to_dataframe('aqs.ozone')
        _checkdf(df)

        # get cached data
        df = rsigapi.to_dataframe('aqs.ozone')
        _checkdf(df)

        # force overwrite of data
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td,
            overwrite=True
        )
        df = rsigapi.to_dataframe('aqs.ozone')
        _checkdf(df)


def test_aqs_no2_verbose():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        df = rsigapi.to_dataframe('aqs.no2', verbose=1)
        _checkdf(df)


def test_aqs_no2_withmeta():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        df = rsigapi.to_dataframe('aqs.no2', withmeta=True, verbose=1)
        _checkdf(df, hasmeta=True)


def test_aqs_no2_unittime():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        df = rsigapi.to_dataframe(
            'aqs.no2', unit_keys=False, parse_dates=True, verbose=1
        )
        _checkdf(df)


def test_aqs_no2_bdate():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(workdir=td, edate='2022-03-01T01')
        df = rsigapi.to_dataframe(
            'aqs.no2', bdate='2022-03-01T00'
        )
        print(df.shape)
        rsigapi = RsigApi(workdir=td, overwrite=True)
        df = rsigapi.to_dataframe(
            'aqs.no2', bdate='2022-03-01T00', edate='2022-03-01T01'
        )
        _checkdf(df)


def test_cmaqequates():
    """currently not operational. I don't yet know why"""
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2016-03-01', bbox=(-85, 30, -70, 45), workdir=td
        )
        df = rsigapi.to_dataframe('cmaq.equates.conus.aconc.O3')
        _checkdf(df)
