__all__ = ['from_xdrfile', 'from_xdr']


def getgridprops(gdnam, crshdr, crsparts, projparts, gridparts, vgparts):
    import numpy as np
    vgprops = {}
    vgprops['VGTYP'] = np.int32(vgparts[0])
    vgprops['VGTOP'] = np.float32(vgparts[1])
    vgprops['VGLVLS'] = np.array(vgparts[2:], dtype='f')
    gridkeys = ['NCOLS', 'NROWS', 'XORIG', 'YORIG', 'XCELL', 'YCELL']
    gridprops = dict(zip(gridkeys, gridparts))
    projattrs = {'GDNAM': gdnam}
    projattrs.update(gridprops)
    crskeys = crshdr.split(': ')[-1].split()
    crsprops = dict(zip(crskeys, crsparts))
    if 'stereographic' in crshdr:
        projattrs['GDTYP'] = 6
        projattrs['P_ALP'] = crsprops['lat_0'] / 90
        projattrs['XCENT'] = projattrs['P_BET'] = crsprops['lon_0']
        projattrs['YCENT'] = crsprops['lat_0']
        projattrs['P_GAM'] = crsprops['lat_sec']
    elif 'lcc' in crshdr:
        projattrs['GDTYP'] = 2
        projattrs['P_ALP'] = crsprops['lat_1']
        projattrs['P_BET'] = crsprops['lat_2']
        projattrs['P_GAM'] = crsprops['lon_0']
        projattrs['XCENT'] = crsprops['lon_0']
        projattrs['YCENT'] = crsprops['lat_0']
    elif 'lonlat' in crshdr:
        projattrs['GDTYP'] = 1
    else:
        raise KeyError(f'Need implement {crshdr}')
        # projattrs['GDTYP'] = 7 # merc
        # projattrs['GDTYP'] = 1 # lonlat
    return {'vertical': vgprops, 'projection': projattrs, 'crs': crsprops}


def from_xdrfile(
    path, na_values=None, decompress=None, as_dataframe=True,
    decompress_inline=True
):
    """
    Currently supports profile, site and swath (v2.0). Each is in XDR format
    with a custom set of header rows in text format. The text header rows also
    describe the binary portion of the file.

    Arguments
    ---------
    path : str
        Path to file in XDR format with RSIG headers
    decompress : bool
        If None, use decompress if path ends in .gz
        If True, decompress to temporary file.
        If False, do not decompress to temporary file (was never compressed)
    decompress_inline : bool
        if True (default), use gzip.open to decompress and read file
        if False, decompress file on disk
    as_dataframe : bool
        If True (default), return data as a pandas.Dataframe.
        If False, return a xarray.Dataset. Only subset and grid support
        as_dataframe.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content
    """
    import gzip
    import os
    import pandas as pd

    if decompress is None:
        decompress = path.endswith('.gz')
    fsize = os.stat(path).st_size
    if fsize == 20:
        return pd.DataFrame()
    elif fsize < 20:
        raise IOError(
            'File size less than 20-bytes; likely failed to download.'
        )
    if decompress:
        if decompress_inline:
            inf = gzip.open(path)
        else:
            temppath = path.replace('.gz', '')
            assert temppath != path
            if not os.path.exists(temppath):
                with gzip.open(path, 'rb') as inf:
                    with open(temppath, 'wb') as outf:
                        outf.write(inf.read())
            inf = open(temppath, 'rb')
    else:
        inf = open(path, 'rb')
    outf = from_xdr(
        inf, decompress=decompress, na_values=na_values,
        as_dataframe=as_dataframe
    )
    inf.close()
    return outf


def from_xdr(inf, na_values=None, decompress=False, as_dataframe=True):
    """
    Currently supports profile, site and swath (v2.0). Each is in XDR format
    with a custom set of header rows in text format. The text header rows also
    describe the binary portion of the file.

    Infers RSIG format using first 40 characters.

    * Site 2.0: from_site
    * Profile 2.0: from_profile
    * Swath 2.0: from_swath
    * Point 1.0: from_point
    * CALIPSO 1.0: from_calipso
    * Polygon 1.0: from_polygon
    * Grid 1.0: from_grid
    * Subset 9.0: from_subset

    Arguments
    ---------
    inf : file
        Data file in XDR format with RSIG headers
    na_values : scalar
        Used to remove known missing values.
    decompress : bool
        If True, decompress to temporary file.
        If False, do not decompress to temporary file (was never compressed)
    as_dataframe : bool
        If True (default), return data as a pandas.Dataframe.
        If False, return a xarray.Dataset. Only subset and grid support
        as_dataframe.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content
    """
    import numpy as np

    inf.seek(0, 0)
    defspec = inf.read(40).decode().split('\n')[0].lower().strip()
    inf.seek(0, 0)
    if defspec.startswith('profile'):
        df = from_profile(inf)
    elif defspec.startswith('site'):
        df = from_site(inf)
    elif defspec.startswith('point'):
        df = from_point(inf)
    elif defspec.startswith('swath'):
        df = from_swath(inf)
    elif defspec.startswith('calipso'):
        df = from_calipso(inf)
    elif defspec.startswith('polygon'):
        df = from_polygon(inf)
    elif defspec.startswith('grid'):
        df = from_grid(inf, as_dataframe=as_dataframe)
    elif defspec.startswith('subset'):
        df = from_subset(inf, as_dataframe=as_dataframe)
    elif defspec.startswith('regridded-swath'):
        df = from_regriddedswath(inf)
    else:
        raise IOError(
            f'{defspec} not in supported formats (profile, site, swath,'
            ' calipso, polygon, grid, subset, or regridded-swath)'
        )

    if na_values is not None:
        df.replace(na_values, np.nan, inplace=True)

    return df


def from_polygon(bf):
    """
    Currently supports Polygon (v1.0) which has 10 header rows in text format.
    The text header rows also describe the binary portion of the file, which
    includes three segments of data corresponding to shx, shp, and dbf files
    embeded within the file.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
    xdrdump master_hms_smoke_2018-07-01_to_2018-07-02.xdr|m
    Polygon 1.0
    https://www.ospo.noaa.gov/Products/land/hms.html,hmsserver
    2018-07-01T00:00:00-0000 2018-07-02T23:59:59-0000
    # Variable names:
    smoke
    # Variable units:
    ug/m3
    # name shxdatabytes shpdatabytes dbfdatabytes and
    # shxfiledata shpfiledata dbffiledata:
    hms_smoke 828 30812 2468
    """
    import tempfile
    import geopandas as gpd

    for i in range(10):
        _l = bf.readline().decode().strip()
        if i == 0:
            assert (_l.lower() == 'polygon 1.0')
        if i == 2:
            dates, datee = _l.replace(':', '').replace('-0000', 'Z').split()
        elif i == 9:
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
        filestem = f'{td}/{prefix}_{dates}_{datee}'
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


def from_profile(bf):
    """
    Currently supports Profile (v2.0) which has 14 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
    # Line 1: definition spec
    # Line 2: Info
    # Line 3: Start/End date
    # Line 4-5: bbox
    # Line 6-7: Dimensions
    # Line 8-9: Variable names
    # Line 10-11: Units
    # Line 12-14: definition of data chunks lines
    #   Next nprof * 80 characters are notes for each profile
    #   Next nprof * 8 are N_p number of points for each profile as long int
    #   Next N * sum(N_p for p in nprof) * 8 are data as 64-bit reals

    Example Header
    --------------
    Profile 2.0
    http://www.esrl.noaa.gov/gmd/grad/neubrew/,NEUBrewSubset
    2010-07-10T00:00:00-0000 2010-07-10T23:59:59-0000
    # Subset domain: <min_lon> <min_lat> <max_lon> <max_lat>:
    -126 24 -66 50
    # Dimensions: variables profiles:
    6 5
    # Variable names:
    timestamp id longitude latitude elevation ozone
    # Variable units:
    yyyymmddhhmmss - deg deg m molecules/cm3
    # char notes[profiles][80] and
    # MSB 64-bit integers points[profiles] and
    # IEEE-754 64-bit reals data_1[variables][points_1]
       ... data_P[variables][points_P]:
    """
    import numpy as np
    import pandas as pd

    for i in range(14):
        _l = bf.readline()
        if i == 0:
            assert (_l.decode().strip().lower() == 'profile 2.0')
        elif i == 6:
            nvar, nprof = np.array(_l.decode().strip().split(), dtype='i')
        elif i == 8:
            varkeys = _l.decode().strip().split()
        elif i == 10:
            units = _l.decode().strip().split()

    notes = [bf.read(80).decode().strip() for i in range(nprof)]
    longnp = np.frombuffer(bf.read(nprof * 8), dtype='>i8')
    # Read all values at once
    dfs = []
    varwunits = [varkey + f'({units[i]})' for i, varkey in enumerate(varkeys)]
    for i, npoints in enumerate(longnp):
        nfloats = npoints * nvar
        vals = np.frombuffer(bf.read(nfloats * 8), dtype='>d').reshape(
            nvar, npoints
        )
        vals = vals.byteswap().view(vals.dtype.newbyteorder())
        df = pd.DataFrame(dict(zip(varwunits, vals)))
        df['NOTE'] = notes[i]
        dfs.append(df)

    df = pd.concat(dfs)
    infmt = '%Y%m%d%H%M%S'
    outfmt = '%Y-%m-%dT%H:%M:%S%z'
    ntimestamps = df['timestamp(yyyymmddhhmmss)']
    timestamps = pd.to_datetime(
        ntimestamps, format=infmt, utc=True
    ).dt.strftime(outfmt)
    df.drop('timestamp(yyyymmddhhmmss)', axis=1, inplace=True)
    df['Timestamp(UTC)'] = timestamps
    renames = {
        'id(-)': 'STATION(-)',
        'longitude(deg)': 'LONGITUDE(deg)',
        'latitude(deg)': 'LATITUDE(deg)',
        'elevation(m)': 'ELEVATION(m)'
    }
    renames = {k: v for k, v in renames.items() if k in df.columns}
    df.rename(columns=renames, inplace=True)
    frontkeys = [
        'Timestamp(UTC)', 'LONGITUDE(deg)', 'LATITUDE(deg)', 'ELEVATION(m)',
        'STATION(-)'
    ]
    lastkeys = ['NOTE']
    outkeys = frontkeys + [
        k for k in df.columns if k not in (frontkeys + lastkeys)
    ] + lastkeys
    df = df[outkeys]

    return df


def from_swath(bf):
    """
    Currently supports Swath (v2.0) which has 14 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
    # 14 header rows to be read using \n
    # Line 1: definition spec
    # Line 2: Info
    # Line 3: Start/End date
    # Line 4-5: Dimensions
    # Line 6-7: Variable names
    # Line 8-9: Units
    # Line 10-11: bbox
    # Line 12-14: definition of data chunks lines
    #   Next nsite * 4 are number of times for each site (N_s) as 32-bit int
    #   Next nvar * sum(2 for s in nsite) * 4 are data as 64-bit reals
    #   Next nvar * sum(N_s for s in nsite) * 4 are data as 64-bit reals

    Swath 2.0
    http://www.star.nesdis.noaa.gov/smcd/emb/viirs_aerosol/,VIIRSSubset
    2013-06-15T00:00:00-0000
    # Dimensions: variables timesteps scans:
    12 24 4
    # Variable names:
    Longitude Latitude Scan_Start_Time faot550 Longitude_SW Longitude_SE
       ... Longitude_NW Longitude_NE Latitude_SW Latitude_SE Latitude_NW
       ... Latitude_NE
    # Variable units:
    deg deg YYYYDDDHHMM - deg deg deg deg deg deg deg deg
    # Domain: <min_lon> <min_lat> <max_lon> <max_lat>
    -113 26 -111 26.5
    # MSB 64-bit integers (yyyydddhhmm) timestamps[scans] and
    # MSB 64-bit integers points[scans] and
    # IEEE-754 64-bit reals data_1[variables][points_1]
       ... data_S[variables][points_S]:
    """

    import numpy as np
    import pandas as pd

    for i in range(14):
        _l = bf.readline()
        if i == 0:
            defspec = _l.decode().strip().lower()
        elif i == 2:
            pass  # stime = _l.decode().strip()
        elif i == 4:
            nvar, nt, nscan = np.array(_l.decode().strip().split(), dtype='i')
        elif i == 6:
            varkeys = _l.decode().strip().split()
        elif i == 8:
            units = _l.decode().strip().split()

    assert (defspec == defspec)

    # MSB 64-bit integers (yyyydddhhmm) timestamps[scans]
    nts = np.frombuffer(bf.read(nscan * 8), dtype='>i8')
    # MSB 64-bit integers points[scans]
    nps = np.frombuffer(bf.read(nscan * 8), dtype='>i8')
    infmt = '%Y%j%H%M'
    outfmt = '%Y-%m-%dT%H:%M:%S%z'
    timestamps = pd.to_datetime(nts, format=infmt, utc=True).strftime(outfmt)

    # Read all values at once
    # IEEE-754 64-bit reals data_1[variables][points_1] ...
    #           data_S[variables][points_S]
    dfs = []
    varwunits = [varkey + f'({units[i]})' for i, varkey in enumerate(varkeys)]
    # To-do
    # Switch from iterative unpacking to numpy.frombuffer, which is much faster
    # this requires some fancy indexing and complex repeats.
    for i, npoints in enumerate(nps):
        nfloats = npoints * nvar
        vals = np.frombuffer(bf.read(nfloats * 8), dtype='>d').reshape(
            nvar, npoints
        )
        vals = vals.byteswap().view(vals.dtype.newbyteorder())
        df = pd.DataFrame(dict(zip(varwunits, vals)))
        df['Timestamp(UTC)'] = timestamps[i]
        dfs.append(df)

    df = pd.concat(dfs)
    renames = {
        'Longitude(deg)': 'LONGITUDE(deg)',
        'Latitude(deg)': 'LATITUDE(deg)',
    }
    renames = {k: v for k, v in renames.items() if k in df.columns}
    df.rename(columns=renames, inplace=True)
    outkeys = ['Timestamp(UTC)', 'LONGITUDE(deg)', 'LATITUDE(deg)']
    outkeys = outkeys + [k for k in df if k not in outkeys]
    df = df[outkeys]
    return df


def from_site(bf):
    """
    Currently supports Site (v2.0) which has 14 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
    # 13 header rows to be read using \n
    # Line 1: definition spec
    # Line 2: Info
    # Line 3: Start/End date
    # Line 4-5: Dimensions
    # Line 6-7: Variable names
    # Line 8-9: Units
    # Line 10-13: definition of data chunks lines
    #   Next nsite * 80 characters are notes for each profile
    #   Next nsite * 4 are IDs number of times for each site as 32-bit integers
    #     or 64-bit integers.
    #   Next sum(2 for s in nsite) * 4 are data as 32-bit reals
    #     or 64-bit integers.
    #   Next sum(N_s for s in nsite) * 4 are data as 32-bit reals
    #     or 64-bit integers.

    Example Header
    --------------
    SITE 2.0
    http://airnow.gov/,reformat_airnow_obs,SiteSubset
    2005-08-26T00:00:00-0000
    # data dimensions: timesteps stations
    3 9
    # Variable names:
    PM25
    # Variable units:
    ug/m3
    # char notes[stations][80] and
    # MSB 64-bit integers ids[stations] and
    # IEEE-754 64-bit reals sites[stations][2=<longitude,latitude>] and
    # IEEE-754 64-bit reals data[timesteps][stations]:

    """
    import numpy as np
    import pandas as pd

    for i in range(13):
        _l = bf.readline()
        if i == 0:
            assert (_l.decode().strip().lower() == 'site 2.0')
        elif i == 2:
            stime = _l.decode().strip()
        elif i == 4:
            nt, nsite = np.array(_l.decode().strip().split(), dtype='i')
        elif i == 6:
            varkeys = _l.decode().strip().split()
        elif i == 8:
            units = _l.decode().strip().split()
        elif i == 10:
            idtype = _l.decode().strip()
        elif i == 11:
            xytype = _l.decode().strip()
        elif i == 12:
            valtype = _l.decode().strip()

    idtype = np.dtype({True: '>i8', False: '>i4'}['64' in idtype])
    xytype = np.dtype({True: '>f8', False: '>f4'}['64' in xytype])
    valtype = np.dtype({True: '>f8', False: '>f4'}['64' in valtype])

    time = pd.date_range(stime, periods=nt, freq='h')
    notes = [bf.read(80).decode().strip() for i in range(nsite)]

    buff = bf.read(nsite * idtype.itemsize)
    nis = np.frombuffer(buff, dtype=idtype)
    buff = bf.read(nsite * xytype.itemsize * 2)
    xys = np.frombuffer(buff, dtype=xytype).reshape(nsite, 2)

    # Read all values at once
    varwunits = [varkey + f'({units[i]})' for i, varkey in enumerate(varkeys)]

    # Unpacking data using frombuffer, which is much faster than iteratively
    # calling xdr.unpack_farray
    vals = np.frombuffer(bf.read(), dtype=valtype).reshape(1, nsite, nt)
    atimes = np.array(time, ndmin=1)[None, :].repeat(nsite, 0).ravel()
    anotes = np.array(notes)[:, None].repeat(nt, 1).T.ravel()
    astation = nis[:, None].repeat(nt, 1).T.ravel()
    axs = xys[:, 0, None].repeat(nt, 1).T.ravel()
    ays = xys[:, 1, None].repeat(nt, 1).T.ravel()
    data = {
        'Timestamp(UTC)': atimes,
        'SITE_NAME': anotes,
        'STATION(-)': astation.astype(f'={astation.dtype.char}'),
        'LONGITUDE(deg)': axs.astype(f'={axs.dtype.char}'),
        'LATITUDE(deg)': ays.astype(f'={ays.dtype.char}'),
    }
    for vi, varwunit in enumerate(varwunits):
        data[varwunit] = vals[vi].ravel().astype(f'={vals.dtype.char}')
    df = pd.DataFrame(data)

    # Serves as a comment on how this was done iteratively.
    """
    dfs = []
    for i, id in enumerate(nis):
        lon, lat = xys[i]
        vals = np.array(up.unpack_farray(nt, up.unpack_float)).reshape(1, nt)
        #assert np.allclose(valchk[0, i], vals)
        df = pd.DataFrame(dict(zip(varwunits, vals)))
        df['Timestamp(UTC)'] = time
        df['SITE_NAME'] = notes[i]
        df['STATION(-)'] = id
        df['LONGITUDE(deg)'] = lon
        df['LATITUDE(deg)'] = lat
        dfs.append(df)

    df = pd.concat(dfs)
    """
    frontkeys = [
        'Timestamp(UTC)', 'LONGITUDE(deg)', 'LATITUDE(deg)', 'STATION(-)'
    ]
    lastkeys = ['SITE_NAME']
    outkeys = frontkeys + [
        k for k in df.columns if k not in (frontkeys + lastkeys)
    ] + lastkeys
    df = df[outkeys]
    return df


def from_point(bf):
    """
    Currently supports Point (v1.0) which has 12 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
    # 12 header rows to be read using \n
    # Line 1: definition spec
    # Line 2: Info
    # Line 3: Start/End date
    # Line 4-5: Dimensions
    # Line 6-7: Variable names
    # Line 8-9: Units
    # Line 10-12: definition of data chunks lines
    #   Next npoint * 80 characters are notes for each point
    #   Next nvariable * npoint * 8 bytes are data[variables][]

    Example Header
    --------------
    Point 1.0
    https://satepsanone.nesdis.noaa.gov/pub/FIRE/BBEP-geo,GOESBBSubset
    2018-11-09T00:00:00-00002018-11-10T23:00:00-0000
    # Dimensions: variables points:
    4 1876
    # Variable names:
    Timestamp Longitude Latitude PM25_emission
    # Variable units:
    yyyymmddhhmmss deg deg kg
    # IEEE-754 64-bit reals data[variables][points]:
    """
    import numpy as np
    import pandas as pd

    for i in range(11):
        _l = bf.readline()
        if i == 0:
            assert (_l.decode().strip().lower() == 'point 1.0')
        elif i == 2:
            # stime = _l.decode().strip()
            pass  # not loading time because it is duplicative
        elif i == 4:
            nvar, npoint = np.array(_l.decode().strip().split(), dtype='i')
        elif i == 6:
            varkeys = _l.decode().strip().split()
        elif i == 8:
            units = _l.decode().strip().split()

    notes = [bf.read(80).decode().strip() for i in range(npoint)]

    # Use numpy to unpack frombuffer becuase it is faster than unpack
    vals = np.frombuffer(bf.read(), dtype='>d').reshape(nvar, npoint)
    vals = vals.byteswap().view(vals.dtype.newbyteorder())
    varkeys = [
        {'id': 'STATION'}.get(varkey, varkey).upper()
        for varkey in varkeys
    ]
    varwunits = [varkey + f'({units[i]})' for i, varkey in enumerate(varkeys)]
    data = {
        'NOTE': notes,
    }
    for vi, varwunit in enumerate(varwunits):
        data[varwunit] = vals[vi].ravel().astype(f'={vals.dtype.char}')

    df = pd.DataFrame(data)
    infmt = '%Y%m%d%H%M%S'
    outfmt = '%Y-%m-%dT%H:%M:%S%z'
    ntimestamps = df['TIMESTAMP(yyyymmddhhmmss)']
    timestamps = pd.to_datetime(
        ntimestamps, format=infmt, utc=True
    ).dt.strftime(outfmt)
    df.drop('TIMESTAMP(yyyymmddhhmmss)', axis=1, inplace=True)
    ti = varwunits.index('TIMESTAMP(yyyymmddhhmmss)')
    varwunits[ti] = 'Timestamp(UTC)'

    df['Timestamp(UTC)'] = timestamps
    outkeys = varwunits + ['NOTE']
    df = df[outkeys].rename(
        columns={'PM25_ATM_HOURLY(ug/m3)': 'pm25_atm_hourly(ug/m3)'}
    )
    return df


def from_calipso(bf):
    """
    Currently supports Calipso (v1.0) which has 15 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Example Header
    --------------
    CALIPSO 1.0
    https://eosweb.larc.nasa.gov/project/calipso/calipso_table,CALIPSOSubset
    2016-05-04T00:00:00-0000
    # Dimensions: variables timesteps profiles:
    5 24 8
    # Variable names: VariableName is Total_Backscatter_Coefficient_532
    Profile_UTC_Time Longitude Latitude Elevation VariableName
    # Variable units:
    yyyymmdd.f deg deg m per_kilometer_per_steradian
    # Domain: <min_lon> <min_lat> <max_lon> <max_lat>
    -126 24 -66 50
    # MSB 64-bit integers (yyyydddhhmm) profile_timestamps[profiles] and
    # IEEE-754 64-bit reals profile_bounds[profiles][2=<lon,lat>][2=<min,max>]
    # MSB 64-bit integers profile_dimensions[profiles][2=<points,levels>] and
    # IEEE-754 64-bit reals profile_data_1[variables][points_1][levels]
    #                       ... profile_data_S[variables][points_S][levels]:
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    headerlines = []
    for i in range(15):
        _l = bf.readline().decode()
        headerlines.append(_l)
        if i == 0:
            assert (_l.strip().lower() == 'calipso 1.0')
        elif i == 2:
            # stime = _l.strip()
            pass  # not loading time because it is duplicative
        elif i == 4:
            nvar, ntime, nprof = np.array(_l.strip().split(), dtype='i')
        elif i == 6:
            varkeys = _l.strip().split()
        elif i == 8:
            units = _l.strip().split()
        elif i == 10:
            # bbox = [float(d) for d in _l.strip().split()]
            pass  # not loading time because it is duplicative

    stime = bf.tell()
    etime = stime + nprof * 8
    sbnd = etime
    ebnd = sbnd + (nprof * 4) * 8
    sdims = ebnd
    edims = sdims + (nprof * 2) * 8
    sdata = edims
    # time = np.frombuffer(bf.read(etime - stime), dtype='>i8')
    # bshape = [nprof, 2, 2]
    # tbnds = np.frombuffer(bf.read(ebnd - sbnd), dtype='>d').reshape(*bshape)
    bf.seek(sdims, 0)
    dims = np.frombuffer(bf.read(edims - sdims), dtype='>i8').reshape(nprof, 2)
    if (
        not (dims[:, 1] == dims[0, 1]).all()
    ):
        raise IOError(
            'CALIPSO 1.0 xdr reader needs to be updated; levels vary. Use'
            + ' ASCII backend instead of xdr until resolved. Please report to'
            + ' https://github.com/barronh/pyrsig/issues'
        )
    times = []
    lons = []
    lats = []
    elev = []
    data = []
    for iprof, (npoint, nlev) in enumerate(dims):
        edata = sdata + (3 * npoint + 2 * npoint * nlev) * 8
        vals = np.frombuffer(bf.read(edata - sdata), dtype='>d')
        vals = vals.byteswap().view(vals.dtype.newbyteorder())
        sd = 0
        ed = sd + npoint * 3
        ptimes, plons, plats = vals[sd:ed].reshape(3, npoint)
        sd = ed
        ed = sd + npoint * nlev * 2
        pelev, pdata = vals[sd:ed].reshape(2, npoint, nlev)
        times.append(ptimes)
        lons.append(plons)
        lats.append(plats)
        elev.append(pelev)
        data.append(pdata)
        sdata = edata
    ds = xr.Dataset()
    alldata = [times, lons, lats, elev, data]
    for key, vals, unit in zip(varkeys, alldata, units):
        attrs = dict(long_name=key, units=unit)
        dims = {1: ('points',), 2: ('points', 'levels')}[vals[0].ndim]
        ds[key] = (dims, np.concatenate(vals, axis=0), attrs)
    ds.attrs['description'] = '\n'.join(headerlines)
    df = ds.to_dataframe().reset_index(drop=True)
    renamer = {key: f'{key}({unit})' for key, unit in zip(varkeys, units)}
    renamer['Latitude'] = 'LATITUDE(deg)'
    renamer['Longitude'] = 'LONGITUDE(deg)'
    renamer['Elevation'] = 'ELEVATION(m)'
    df['Timestamp(UTC)'] = (
        pd.to_datetime(df['Profile_UTC_Time'] // 1, utc=True)
        + pd.to_timedelta(df['Profile_UTC_Time'] % 1, unit='d')
    ).dt.round('1s')
    df = df[['Timestamp(UTC)'] + varkeys[1:-1] + varkeys[:1] + varkeys[-1:]]
    df.rename(columns=renamer, inplace=True)
    return df


def from_grid(bf, as_dataframe=True):
    """
    Currently supports Grid (v1.0) which has 12 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers
    as_dataframe : bool
        If True (default), return data as a pandas.Dataframe.
        If False, return a xarray.Dataset.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Example Header
    --------------
    xdrdump master_hrrr_wind_2020-02-17.xdr
    Grid 1.0
    http://home.chpc.utah.edu/~u0553130/Brian_Blaylock/,HRRRSubset
    2020-02-17T00:00:00-0000
    # Dimensions: timesteps variables rows columns:
    24 2 85 73
    # Variable names:
    wind_u wind_v
    # Variable units:
    m/s m/s
    # IEEE-754 64-bit reals longitudes[rows][columns] and
    # IEEE-754 64-bit reals latitudes[rows][columns] and
    # IEEE-754 64-bit reals data[timesteps][variables][rows][columns]:
    -7.8518169999999998e+01
    -7.8484610000000004e+01
    -7.8451070000000001e+01
    ...
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    headerlines = []
    for i in range(12):
        _l = bf.readline().decode().strip()
        headerlines.append(_l)
        if i == 0:
            assert (_l.lower() == 'grid 1.0')
        elif i == 1:
            rsig_program = _l
        elif i == 2:
            stime = pd.to_datetime(_l)
        elif i == 4:
            ntime, nvar, nrow, ncol = np.array(_l.split(), dtype='i')
        elif i == 6:
            varkeys = _l.split()
        elif i == 8:
            units = _l.split()
        elif i == 10:
            # bbox = [float(d) for d in _l.strip().split()]
            pass  # not loading time because it is duplicative

    nbytes = nrow * ncol * 8
    lon = np.frombuffer(bf.read(nbytes), dtype='>d').reshape(nrow, ncol)
    lon = lon.byteswap().view(lon.dtype.newbyteorder())
    lat = np.frombuffer(bf.read(nbytes), dtype='>d').reshape(nrow, ncol)
    lat = lat.byteswap().view(lat.dtype.newbyteorder())
    dshape = ntime, nvar, nrow, ncol
    nbytes = np.prod(dshape) * 8
    data = np.frombuffer(bf.read(nbytes), dtype='>d').reshape(*dshape)
    data = data.byteswap().view(data.dtype.newbyteorder())
    ds = xr.Dataset()
    ds.attrs['rsig_program'] = rsig_program
    dt = pd.to_timedelta('3600s')
    reftime = '1970-01-01T00:00:00+0000'
    attrs = dict(units=f'seconds since {reftime}', long_name='time')
    ds.coords['time'] = ('time',), stime + np.arange(ntime) * dt, attrs
    attrs = dict(units='index', long_name='x_center')
    ds.coords['x'] = ('x',), np.arange(ncol, dtype='d') + 0.5
    attrs = dict(units='index', long_name='y_center')
    ds.coords['y'] = ('y',), np.arange(nrow, dtype='d') + 0.5
    attrs = dict(units='degrees_north', long_name='latitude')
    ds['lat'] = ('y', 'x'), lat, attrs
    attrs = dict(units='degrees_east', long_name='longitude')
    ds['lon'] = ('y', 'x'), lon, attrs
    for vi, varkey in enumerate(varkeys):
        attrs = dict(long_name=varkey, units=units[vi])
        ds[varkey] = ('time', 'y', 'x'), data[:, vi, :, :], attrs

    if as_dataframe:
        df = ds.to_dataframe().astype('d')
        time = df.index.get_level_values('time')
        df['Timestamp(UTC)'] = time.strftime('%Y-%m-%dT%H:%M:%S+0000')
        keepcols = ['Timestamp(UTC)', 'lon', 'lat'] + varkeys
        renamer = {vk: f'{vk}({vu})' for vk, vu in zip(varkeys, units)}
        renamer['lon'] = 'LONGITUDE(deg)'
        renamer['lat'] = 'LATITUDE(deg)'
        df = df[keepcols].rename(columns=renamer)
        return df
    else:
        return ds


def from_subset(bf, as_dataframe=True):
    """
    Currently supports Subset (v9.0) which has 17 header rows in text format.
    The text header rows also describe the binary portion of the file. This
    data can be very large for high resolution continental data. For example,
    CMAQ EQUATES CONUS for one 4D variable requires 4GB of memory as a
    dataframe whereas the same data from CMAQ EQUATES HEMI is just 0.065GB.
    Both are much smaller as a Dataset where coordinate data is not repeated.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers
    as_dataframe : bool
        If True (default), return data as a pandas.Dataframe.
        If False, return a xarray.Dataset.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
    /xdrdump master_cmaq_amad_hemi_ext_2006-08-28.xdr |m
    SUBSET 9.0 CMAQ
    108NHEMI1
    http://www.epa.gov/ttn/scram/,CMAQSubset
    2006-08-28T00:00:00-0000
    # data dimensions: timesteps variables layers rows columns:
    1 4 35 187 187
    # subset indices (0-based time, 1-based layer/row/column): first-timestep
      ... last-timestep first-layer last-layer first-row last-row first-column
      ... last-column:
     0 0 1 35 1 187 1 187
    # Variable names:
    LONGITUDE LATITUDE ELEVATION EXT_Recon
    # Variable units:
    deg deg m 1/km
    # stereographic projection: lat_0 lon_0 lat_sec major_semiaxis
      ... minor_semiaxis
    90 -98 45 6.37e+06 6.37e+06
    # Grid: ncols nrows xorig yorig xcell ycell vgtyp vgtop vglvls[36]:
    187 187 -1.0098e+07 -1.0098e+07 108000 108000 7 5000 1 0.9975 0.995 ...
    # IEEE-754 32-bit reals data[variables][timesteps][layers][rows][columns]:
    -1.4300000000000000e+02
    -1.4269029235839844e+02
    ...
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    from .utils import get_proj4
    import pyproj

    headerlines = []
    for i in range(17):
        _l = bf.readline().decode().strip()
        headerlines.append(_l)
        if i == 0:
            assert (_l.lower() == 'subset 9.0 cmaq')
        elif i == 1:
            gdnam = _l
        elif i == 2:
            rsig_program = _l
        elif i == 3:
            stime = pd.to_datetime(_l)
        elif i == 5:
            ntime, nvar, nlay, nrow, ncol = np.array(_l.split(), dtype='i')
        elif i == 7:
            ftime, ltime, flay, llay, frow, lrow, fcol, lcol = np.array(
                _l.split(), dtype='i'
            )
        elif i == 9:
            varkeys = _l.split()
        elif i == 11:
            units = _l.split()
        elif i == 12:
            crshdr = _l.lower()
        elif i == 13:
            crsparts = np.array(_l.split(), dtype='d')
        elif i == 15:
            projparts = np.array(_l.split(), dtype='d')
            vgparts = np.array(projparts[-nlay - 3:], dtype='f')
            gridparts = projparts[:-len(vgparts)]
        elif i == 16:
            is64 = '64-bit' in _l
            if is64:
                dtval = '>d'
            else:
                dtval = '>f'

    gridprops = getgridprops(
        gdnam, crshdr, crsparts, projparts, gridparts, vgparts
    )
    vgprops = gridprops['vertical']
    vglvls = vgprops['VGLVLS']
    projattrs = gridprops['projection']
    earth_radius = gridprops['crs']['major_semiaxis']

    proj4 = get_proj4(projattrs, earth_radius=earth_radius)
    proj = pyproj.Proj(proj4)
    data = np.frombuffer(bf.read(), dtype=dtval).reshape(
        nvar, ntime, nlay, nrow, ncol
    )
    ds = xr.Dataset()
    ds.attrs.update(**projattrs)
    ds.attrs.update(**vgprops)
    ds.attrs['UPNAM'] = rsig_program.ljust(16)
    dt = pd.to_timedelta('3600s')
    t = stime + np.arange(ntime) * dt
    reftime = '1970-01-01T00:00:00+0000'
    attrs = dict(long_name='TSTEP', units=f'seconds since {reftime}')
    ds.coords['TSTEP'] = t
    z = ((vglvls[1:] + vglvls[:-1]) / 2)[flay - 1:llay]
    attrs = dict(long_name='LAY', units='layer_center')
    ds.coords['LAY'] = ('LAY',), z, attrs
    y = np.arange(nrow) + frow - 0.5
    attrs = dict(long_name='ROW', units='cell_center')
    ds.coords['ROW'] = ('ROW',), y, attrs
    x = np.arange(ncol) + fcol - 0.5
    attrs = dict(long_name='COL', units='cell_center')
    ds.coords['COL'] = ('COL',), x, attrs
    Y, X = xr.broadcast(ds['ROW'], ds['COL'])
    LON, LAT = proj(X.values, Y.values, inverse=True)
    attrs = dict(long_name='LONGITUDE'.ljust(16), units='degrees_east    ')
    attrs['var_desc'] = attrs['long_name'].ljust(80)
    ds['LONGITUDE'] = ('ROW', 'COL'), LON.astype('f'), attrs
    attrs = dict(long_name='LATITUDE'.ljust(16), units='degrees_north   ')
    attrs['var_desc'] = attrs['long_name'].ljust(80)
    ds['LATITUDE'] = ('ROW', 'COL'), LAT.astype('f'), attrs
    dims = ('TSTEP', 'LAY', 'ROW', 'COL')
    for vi, vk in enumerate(varkeys):
        attrs = dict(
            units=units[vi].ljust(16), long_name=vk.ljust(16),
            var_desc=vk.ljust(80)
        )
        vals = data[vi]
        ds[vk] = dims, vals, attrs

    if as_dataframe:
        df = ds.astype(dtval[1:]).to_dataframe()
        time = df.index.get_level_values('TSTEP')
        df['Timestamp(UTC)'] = time.strftime('%Y-%m-%dT%H:%M:%S+0000')
        keepcols = ['Timestamp(UTC)', 'LONGITUDE', 'LATITUDE'] + varkeys
        renamer = {vk: f'{vk}({vu})' for vk, vu in zip(varkeys, units)}
        renamer['LONGITUDE'] = 'LONGITUDE(deg)'
        renamer['LATITUDE'] = 'LATITUDE(deg)'
        df = df[keepcols].rename(columns=renamer)
        return df
    else:
        return ds


def from_regriddedswath(bf):
    """
    Currently supports regridded-swath (v2.0) which has 19 header rows in text
    format. The text header rows also describe the binary portion of the file.
    This data can be very large for high resolution continental data. For
    example, CMAQ EQUATES CONUS for one 4D variable requires 4GB of memory as
    a dataframe whereas the same data from CMAQ EQUATES HEMI is just 0.065GB.
    Both are much smaller as a Dataset where coordinate data is not repeated.

    Arguments
    ---------
    bf : file
        File opened in byte read in XDR format with RSIG headers
    as_dataframe : bool
        If True (default), return data as a pandas.Dataframe.
        If False, return a xarray.Dataset.

    Notes
    -----
    In progress.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
    xdrdump master_regridded_viirs_ivaot_faot550_2013-06-15.xdr|m
    REGRIDDED-Swath 2.0
    http://....nesdis.noaa.gov/smcd/emb/viirs_aerosol/,VIIRSSubset,XDRConvert
    2013-06-15T20:00:00-0000
    # timesteps
    2
    # Variable name:
    faot550
    # Variable units:
    -
    # stereographic proj: lat_0 lon_0 lat_sec major_semiaxis minor_semiaxis
    90 -98 45 6370000.000000 6370000.000000
    # Grid: ncols nrows xorig yorig xcell ycell vgtyp vgtop vglvls[2]:
    137 137 -7398000.0 -7398000.0 108000.0 108000.0 2 10000.0 1 0.995
    # MSB 64-bit integers points[timesteps] and
    # IEEE-754 64-bit reals longitudes[timesteps][points] and
    # IEEE-754 64-bit reals latitudes[timesteps][points] and
    # MSB 64-bit integers columns[timesteps][points] and
    # MSB 64-bit integers rows[timesteps][points] and
    # IEEE-754 64-bit reals data[timesteps][points]:
    154
    719
    -1.2970142966950573e+02
    -1.2908750532347997e+02
    -1.2783553910026154e+02
    ...
    """
    import numpy as np
    import pandas as pd
    # from .utils import get_proj4

    headerlines = []
    for i in range(19):
        _l = bf.readline().decode().strip()
        headerlines.append(_l)
        if i == 0:
            assert (_l.lower() == 'regridded-swath 2.0')
        elif i == 1:
            rsig_program = _l
        elif i == 2:
            stime = pd.to_datetime(_l)
        elif i == 4:
            ntime, = np.array(_l.split(), dtype='i')
        elif i == 6:
            varkeys = _l.split()
        elif i == 8:
            units = _l.split()
        elif i == 9:
            crshdr = _l.lower()
        elif i == 10:
            crsparts = np.array(_l.split(), dtype='d')
        elif i == 12:
            projparts = np.array(_l.split(), dtype='d')
            nlay = 1
            vgparts = np.array(projparts[-nlay - 3:], dtype='f')
            gridparts = projparts[:-len(vgparts)]

    dt = pd.to_timedelta('1h')
    times = []
    points = np.frombuffer(bf.read(ntime * 8), dtype='>i8')
    outfmt = '%Y-%m-%dT%H:%M:%S%z'
    dts = pd.Series(np.concatenate([
        np.ones(points[ti]) * ti * dt
        for ti in range(ntime)
    ]))
    times = (stime + dts).dt.strftime(outfmt).values
    npoints = points.sum()
    lons = np.frombuffer(bf.read(npoints * 8), dtype='>d')
    lats = np.frombuffer(bf.read(npoints * 8), dtype='>d')
    cols = np.frombuffer(bf.read(npoints * 8), dtype='>i8')
    rows = np.frombuffer(bf.read(npoints * 8), dtype='>i8')
    vals = np.frombuffer(bf.read(npoints * 8), dtype='>d')
    valkey = f'{varkeys[0]}({units[0]})'
    gdn = 'unknown'  # gdnam
    gattrs = getgridprops(gdn, crshdr, crsparts, projparts, gridparts, vgparts)
    df = pd.DataFrame({
        'Timestep(UTC)': times.astype(f'={times.dtype.char}'),
        'LONGITUDE(deg)': lons.astype(f'={lons.dtype.char}'),
        'LATITUDE(deg)': lats.astype(f'={lats.dtype.char}'),
        'COLUMN(-)': cols.astype(f'={cols.dtype.char}'),
        'ROW(-)': rows.astype(f'={rows.dtype.char}'),
        valkey: vals.astype(f'={vals.dtype.char}'),
    })
    df.attrs.update(gattrs)
    df.attrs['rsig_program'] = rsig_program
    return df


if __name__ == '__main__':
    import pyrsig

    baseurl = (
        'https://ofmpub.epa.gov/rsig/rsigserver?SERVICE=wcs&VERSION=1.0.0'
        + '&REQUEST=GetCoverage&FORMAT=xdr'
    )
    baseurl += '&BBOX=-130.0,20.,-65.,60&COMPRESS=1'
    ckey = 'pandora.L2_rnvs3p1_8.nitrogen_dioxide_vertical_column_amount'
    url = (
        f'{baseurl}&TIME=2023-10-17T13:00:00Z/2023-10-17T13:59:59Z'
        + f'&COVERAGE={ckey}'
    )
    r = pyrsig.legacy_get(url)
    print('Profile test')
    dfprof = from_xdr(r.content, decompress=True)

    url = (
        f'{baseurl}&TIME=2023-10-17T13:00:00Z/2023-10-17T14:59:59Z'
        + '&COVERAGE=airnow.ozone'
    )
    # Get it and decompress it
    r = pyrsig.legacy_get(url)
    print('Site test')
    dfsite = from_xdr(r.content, decompress=True)

    tempokey = open('/home/bhenders/.tempokey', 'r').read().strip()
    ckey = 'tempo.l2.no2.vertical_column_troposphere'
    url = (
        f'{baseurl}&TIME=2023-11-17T13:00:00Z/2023-11-17T13:59:59Z'
        + '&COVERAGE={ckey}&KEY={tempokey}'
    )
    # Get it and decompress it
    r = pyrsig.legacy_get(url)
    print('Swath test')
    dfswath = from_xdr(r.content, decompress=True)
