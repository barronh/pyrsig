def test_initdef():
    from .. import RsigApi

    RsigApi()


def test_init():
    from .. import RsigApi

    RsigApi(
        bdate='2022-03-01T00:00:00', edate='2022-03-01T23:59:59',
        bbox=(-126, 20, -110, 50)
    )


def test_tropomi():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td,
            bbox=(-97, 20, -65, 50)
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)


def test_tropomi_encoding_remove():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-02', workdir=td,
            bbox=(-97, 20, -65, 50), encoding={'zlib': True, 'complevel': 1}
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
            removegz=True
        )
        print(ds.dims)


def _test_viirsnoaa():
    """currently not operational. I don't yet know why"""
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(bdate='2022-03-01', workdir=td)
        ds = rsigapi.to_ioapi(
            'viirsnoaa.jrraod.AOD550'
        )
        print(ds.dims)


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


def test_aqs_no2_verbose():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01T00', edate='2022-03-01T01', workdir=td
        )
        df = rsigapi.to_dataframe('aqs.no2', verbose=1)
        print(df.shape)


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


def test_capabilities():
    from .. import RsigApi

    rsigapi = RsigApi()
    keys = rsigapi.keys()
    print(len(keys))
    keys = rsigapi.keys(offline=False)
    print(len(keys))
