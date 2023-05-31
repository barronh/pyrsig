def test_initdef():
    import warnings
    from sys import version_info as vi
    from .. import RsigGui

    vstr = f'{vi.major}.{vi.minor}.{vi.micro}'
    try:
        gui = RsigGui()
        gui.check()
    except ImportError as e:
        if vstr <= '3.6.1':
            warnings.warn('Cannot test gui with version 3.6.1 or lower')
        else:
            raise e


def test_roundtrip():
    import warnings
    from sys import version_info as vi
    from .. import RsigGui

    vstr = f'{vi.major}.{vi.minor}.{vi.micro}'
    try:
        g1 = RsigGui()
    except ImportError as e:
        if vstr <= '3.6.1':
            warnings.warn('Cannot test gui with version 3.6.1 or lower')
            return
        else:
            raise e

    a1 = g1.get_api()
    g2 = RsigGui.from_api(a1)
    propkeys = ['key', 'bdate', 'edate', 'grid_kw', 'workdir']
    for pk in propkeys:
        p1 = getattr(g1, pk)
        p2 = getattr(g2, pk)
        assert p1 == p2
