def test_pandora():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2023-03-01', bbox=(-80, 30, -60, 50), workdir=td
        )
        ds = rsigapi.to_dataframe(
            'pandora.L2_rnvh3p1_8.tropospheric_nitrogen_dioxide'
        )
        print(ds.shape)


def test_viirsnoaa():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', bbox=(-85, 30, -70, 45), workdir=td
        )
        ds = rsigapi.to_dataframe('viirsnoaa.jrraod.AOD550')
        print(ds.shape)


def test_aqs_ozone_cache():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        df = rsigapi.to_dataframe('aqs.ozone')
        print(df.shape)
        df = rsigapi.to_dataframe('aqs.ozone')
        print(df.shape)
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td,
            overwrite=True
        )
        df = rsigapi.to_dataframe('aqs.ozone')
        print(df.shape)


def test_aqs_no2_verbose():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        df = rsigapi.to_dataframe('aqs.no2', verbose=1)
        print(df.shape)


def test_aqs_no2_withmeta():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        df = rsigapi.to_dataframe('aqs.no2', withmeta=True, verbose=1)
        print(df.shape, len(df.attrs['metadata']))


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
        print(df.shape)


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
        print(df.shape)


def test_cmaqequates():
    """currently not operational. I don't yet know why"""
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2016-03-01', bbox=(-85, 30, -70, 45), workdir=td
        )
        ds = rsigapi.to_dataframe('cmaq.equates.conus.aconc.O3')
        print(ds.shape)
