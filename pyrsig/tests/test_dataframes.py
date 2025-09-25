class _apictx:
    """
    Context manager to create an api in a temporary working space.
    """
    def __init__(self, **kwds):
        import tempfile
        self.tdir = tempfile.TemporaryDirectory()
        kwds.setdefault('bdate', '2023-03-01')
        kwds.setdefault('bbox', (-80, 30, -60, 50))
        kwds['workdir'] = self.tdir.name
        self.kwds = kwds

    def __enter__(self):
        from .. import RsigApi
        return RsigApi(**self.kwds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tdir.cleanup()


def _checkdf(df, hasmeta=False):
    assert len(df.shape) == 2  # require two-dimensional return
    assert df.shape[0] > 0  # require a row
    assert df.shape[1] > 0  # require a column
    if hasmeta:
        assert 'metadata' in df.attrs


def test_pandora():
    with _apictx() as api:
        key = 'pandora.L2_rnvh3p1_8.tropospheric_nitrogen_dioxide'
        df = api.to_dataframe(key)
        _checkdf(df)


def test_hmssmoke():
    with _apictx() as api:
        df = api.to_dataframe('hms.smoke')
        _checkdf(df)


def test_viirsnoaa():
    bbox = (-85, 30, -70, 45)
    opts = dict(bbox=bbox, bdate='2019-01-01T16', edate='2019-01-01T18:59:59')
    with _apictx(**opts) as api:
        df = api.to_dataframe('viirsnoaa.jrraod.AOD550')
        _checkdf(df)


def test_airnow_ozone_cache():
    with _apictx(edate='2023-03-01T00:59:59') as api:
        # get fresh data
        df = api.to_dataframe('airnow.ozone')
        _checkdf(df)

        # get cached data
        df = api.to_dataframe('airnow.ozone')
        _checkdf(df)

        # force overwrite of data
        api.overwrite = True
        sbox = [v for v in api.bbox]
        sbox[3] = sbox[1] + (sbox[3] - sbox[1]) / 2
        sbox[2] = sbox[0] + (sbox[2] - sbox[0]) / 2
        odf = api.to_dataframe('airnow.ozone', bbox=sbox)
        _checkdf(odf)
        assert odf.shape[0] < df.shape[0]


def test_aqs_ozone():
    with _apictx(edate='2023-03-01T00:59:59') as api:
        df = api.to_dataframe('aqs.ozone', verbose=1)
        _checkdf(df)


def test_airnow_ozone_geo():
    with _apictx(edate='2023-03-01T00:59:59') as api:
        df = api.to_geodataframe('airnow.ozone', verbose=1)
        _checkdf(df)


def test_airnow_no2_verbose():
    with _apictx(edate='2023-03-01T00:59:59') as api:
        df = api.to_dataframe('airnow.no2', verbose=1)
        _checkdf(df)


def test_aqs_no2_verbose():
    with _apictx(edate='2023-03-01T00:59:59') as api:
        df = api.to_dataframe('aqs.no2', verbose=1)
        _checkdf(df)


def test_airnow_no2_withmeta():
    with _apictx(edate='2023-03-01T00:59:59') as api:
        df = api.to_dataframe('airnow.no2', withmeta=True, verbose=1)
        _checkdf(df, hasmeta=True)


def test_airnow_no2_unittime():
    with _apictx(edate='2023-03-01T00:59:59') as api:
        df = api.to_dataframe(
            'airnow.no2', unit_keys=False, parse_dates=True, verbose=1
        )
        _checkdf(df)


def test_airnow_no2_bdate():
    with _apictx(bdate=None, edate='2023-03-01T00:59:59') as api:
        df1 = api.to_dataframe('airnow.no2', bdate='2023-03-01T00')
        _checkdf(df1)
        api.overwrite = True
        df2 = api.to_dataframe(
            'airnow.no2', bdate='2023-03-01T00', edate='2023-03-01T00:59:59'
        )
        _checkdf(df2)
        assert (df1 == df2).all().all()


def test_cmaqequates():
    with _apictx(bdate='2017-03-01T00', edate='2017-03-01T00:59:59') as api:
        df = api.to_dataframe('cmaq.equates.conus.aconc.O3')
        _checkdf(df)
