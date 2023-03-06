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
        rsigapi = RsigApi(bdate='2022-03-01', workdir=td)
        ds = rsigapi.to_ioapi(
            'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
        )
        print(ds.dims)


def test_aqs():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rsigapi = RsigApi(bdate='2022-03-01', workdir=td)
        df = rsigapi.to_dataframe('aqs.ozone')
        print(df.shape)
