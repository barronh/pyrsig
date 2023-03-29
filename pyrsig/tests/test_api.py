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
