__all__ = ['from_binfile']


def from_binfile(path):
    """
    Arguments
    ---------
    path : str
        Path to binary file that has a first line
        prefix shxn shpn dbfn
        Where shxn is the length of the shx file
        Where shpn is the length of the shp file
        Where dbfn is the length of the shp file

    Returns
    -------
    df : geopandas.GeoDataFrame
    """
    import tempfile
    import gzip
    import geopandas as gpd

    if path.endswith('.gz'):
        bf = gzip.open(path)
    else:
        bf = open(path, 'rb')
    _l = bf.readline().decode().strip()
    prefix, shxn, shpn, dbfn = _l.split()
    shxn = int(shxn)
    shpn = int(shpn)
    dbfn = int(dbfn)
    shxs = bf.tell()
    shxe = shxs + shxn
    shps = shxe
    shpe = shps + shpn
    dbfs = shpe
    dbfe = dbfs + dbfn
    with tempfile.TemporaryDirectory() as td:
        filestem = f'{td}/{prefix}'
        bf.seek(shxs, 0)
        with open(filestem + '.shx', 'wb') as shxf:
            shxf.write(bf.read(shxe - shxs))
        with open(filestem + '.shp', 'wb') as shpf:
            shpf.write(bf.read(shpe - shps))
        with open(filestem + '.dbf', 'wb') as dbff:
            dbff.write(bf.read(dbfe - dbfs))
        outdf = gpd.read_file(filestem + '.shp')

    outdf.crs = 4326
    return outdf
