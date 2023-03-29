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


