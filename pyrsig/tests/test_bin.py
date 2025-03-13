def test_polygon():
    from .. import RsigApi
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        bbox = (-75, 43, -60, 45)
        rsigapi = RsigApi(bdate='2023-03-01', bbox=bbox, workdir=td)
        df = rsigapi.to_dataframe('landuse.atlantic.population_iclus1')

    print(df.shape)
    return
