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


def test_tropomi_stereo():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td,
            bbox=(-97, 20, -65, 50), grid_kw='108NHEMI2'
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)


def test_tropomi_merc():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td,
            bbox=(-97, 20, -65, 50), grid_kw='NORTHSOUTHAM'
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)


def test_tropomi_cache():
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
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td, overwrite=True,
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


def test_tropomi_withmeta():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-02', workdir=td,
            bbox=(-97, 20, -65, 50), encoding={'zlib': True, 'complevel': 1}
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
            withmeta=True, removegz=True
        )
        print(ds.dims, len(ds.attrs['metadata']))


def test_equates():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2016-03-01', workdir=td,
            bbox=(-97, 20, -65, 50)
        )
        ds = rsigapi.to_ioapi(
            'cmaq.equates.conus.aconc.O3'
        )
        print(ds.dims)
