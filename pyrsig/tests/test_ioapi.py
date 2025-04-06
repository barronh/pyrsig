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
        ds.close()


def test_tropomi_save():
    from .. import RsigApi
    from ..cmaq import save_ioapi, open_ioapi
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td,
            bbox=(-97, 20, -65, 50)
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        save_ioapi(ds, f'{td}/test.nc')
        ds2 = open_ioapi(f'{td}/test.nc')
        for pk in ds.attrs:
            p1 = ds.attrs[pk]
            p2 = ds2.attrs[pk]
            if pk in ('WTIME', 'CTIME', 'UPNAM'):
                continue
            if isinstance(p1, str):
                p1 = p1.strip()
                p2 = p2.strip()
            # print(pk, p1, p2)
            assert (pk == pk and np.all(p1 == p2))

        for k in ['TFLAG', 'LATITUDE', 'LONGITUDE', 'NO2', 'COUNT']:
            # print(k)
            v1 = ds[k]
            v2 = ds2[k]
            assert (k == k and np.allclose(v1, v2))
            for pk in v1.attrs:
                p1 = v1.attrs[pk]
                p2 = v2.attrs[pk]
                if isinstance(p1, str):
                    p1 = p1.strip()
                    p2 = p2.strip()
                # print(pk, p1, p2)
                assert (k == k and pk == pk and np.all(p1 == p2))
        ds.close()
        ds2.close()


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
        ds.close()


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
        ds.close()


def test_tropomi_lonlat():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td,
            bbox=(-97, 20, -65, 50), grid_kw='global_1pt0'
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)
        ds.close()


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
        ds.close()
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)
        ds.close()
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td, overwrite=True,
            bbox=(-97, 20, -65, 50)
        )
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)
        ds.close()


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
        ds.close()


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
        ds.close()


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
        ds.close()
