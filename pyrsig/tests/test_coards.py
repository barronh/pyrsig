def test_tropomi_coards():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td,
            bbox=(-97, 20, -65, 50)
        )
        ds = rsigapi.to_netcdf(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)


def test_tropomi_coards_withmeta():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(
            bdate='2022-03-01', workdir=td,
            bbox=(-97, 20, -65, 50)
        )
        ds = rsigapi.to_netcdf(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column',
            withmeta=True
        )
        print(ds.dims, len(ds.attrs['metadata']))
