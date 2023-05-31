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
    from .. import customize_grid, RsigApi
    bbox = (-90, 15, -60, 20)
    nofit = RsigApi(bbox=bbox, grid_kw='12US1', gridfit=False)
    fit = RsigApi(bbox=bbox, grid_kw='12US1', gridfit=True)
    fitnoclip = customize_grid(nofit.grid_kw, bbox, clip=False)
    propkeys = list(nofit.grid_kw)
    for pk in propkeys:
        pref = nofit.grid_kw[pk]
        pclip = fit.grid_kw[pk]
        pnoclip = fitnoclip[pk]
        if pk not in ('XORIG', 'YORIG', 'NCOLS', 'NROWS'):
            assert pref == pclip
            assert pref == pnoclip
    cx0, cy0 = fit.grid_kw['XORIG'], fit.grid_kw['YORIG']
    cnc, cnr = fit.grid_kw['NCOLS'], fit.grid_kw['NROWS']
    assert (cx0, cy0, cnc, cnr) == (804000.0, -1728000.0, 179, 25)
    ncx0, ncy0 = fitnoclip['XORIG'], fitnoclip['YORIG']
    ncnc, ncnr = fitnoclip['NCOLS'], fitnoclip['NROWS']
    assert (ncx0, ncy0, ncnc, ncnr) == (804000.0, -2820000.0, 262, 116)
