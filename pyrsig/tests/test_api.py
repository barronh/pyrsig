def test_initdef():
    from .. import RsigApi

    RsigApi()


def test_init():
    from .. import RsigApi

    RsigApi(
        bdate='2022-03-01T00:00:00', edate='2022-03-01T23:59:59',
        bbox=(-126, 20, -110, 50)
    )


def test_capabilities():
    from .. import RsigApi

    rsigapi = RsigApi()
    keys = rsigapi.keys()
    print(len(keys))
    keys = rsigapi.keys(offline=False)
    print(len(keys))


def test_descriptions():
    from .. import RsigApi
    import warnings

    rsigapi = RsigApi()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        desc = rsigapi.descriptions()
    print(desc.columns, desc.shape)


def test_describe():
    from .. import RsigApi

    rsigapi = RsigApi()
    desc = rsigapi.describe('aqs.ozone')
    print(len(desc))


def test_customize_grid():
    """
    Test updated to reflect new customize_grid functionality
    """
    from .. import customize_grid, RsigApi
    bbox = (-90, 15, -60, 30)
    nofit = RsigApi(bbox=bbox, grid_kw='12US1', gridfit=False)
    fit_grid_kw = customize_grid(nofit.grid_kw, bbox=bbox)
    fit = RsigApi(bbox=bbox, grid_kw='12US1', gridfit=True)
    propkeys = list(nofit.grid_kw)
    check_kw = dict(XORIG=672000., YORIG=-1728000.0, NCOLS=190, NROWS=94)
    for pk in propkeys:
        # Unless specifically changed, should match original
        pref = check_kw.get(pk, nofit.grid_kw[pk])
        assert pref == fit.grid_kw[pk]
        assert pref == fit_grid_kw[pk]


def test_findkeys():
    from .. import findkeys
    df1 = findkeys()
    df2 = findkeys(name='tempo.l2.no2')
    df3 = findkeys(name='tempo.l2.no2', label='.*molec')
    df4 = findkeys(bbox=(90, 30, 95, 50))
    df5 = findkeys(temporal='2018-01-01/now')
    assert df2.shape[0] < df1.shape[0]
    assert df3.shape[0] < df2.shape[0]
    assert df4.shape[0] < df1.shape[0]
    assert df5.shape[0] < df1.shape[0]
