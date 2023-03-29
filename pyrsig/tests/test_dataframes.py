def test_viirsnoaa():
    """currently not operational. I don't yet know why"""
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
