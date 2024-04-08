__all__ = ['from_xdrfile', 'from_xdr']


def from_xdrfile(path, na_values=None, decompress=None):
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
        If True, decompress buffer.
        If False, buffer is already decompressed (or was never compressed)

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content
    """
    if decompress is None:
        decompress = path.endswith('.gz')
    with open(path, 'rb') as inf:
        buf = inf.read()
        return from_xdr(buf, decompress=decompress, na_values=na_values)


def from_xdr(buffer, na_values=None, decompress=False):
    """
    Currently supports profile, site and swath (v2.0). Each is in XDR format
    with a custom set of header rows in text format. The text header rows also
    describe the binary portion of the file.

    Infers RSIG format using first 40 characters.

    * Site 2.0: from_site
    * Profile 2.0: from_profile
    * Swath 2.0: from_swath

    Arguments
    ---------
    buffer : bytes
        Data buffer in XDR format with RSIG headers
    decompress : bool
        If True, decompress buffer.
        If False, buffer is already decompressed (or was never compressed)

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content
    """
    import gzip
    import numpy as np
    if decompress:
        buffer = gzip.decompress(buffer)

    defspec = buffer[:40].decode().split('\n')[0].lower().strip()
    if defspec.startswith('profile'):
        df = from_profile(buffer)
    elif defspec.startswith('site'):
        df = from_site(buffer)
    elif defspec.startswith('point'):
        df = from_point(buffer)
    elif defspec.startswith('swath'):
        df = from_swath(buffer)
    elif defspec.startswith('calipso'):
        df = from_calipso(buffer)
    else:
        raise IOError('{defspec} not in profile, site, swath')

    if na_values is not None:
        df = df.replace(na_values, np.nan)

    return df


def from_profile(buffer):
    """
    Currently supports Profile (v2.0) which has 14 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    buffer : bytes
        Data buffer in XDR format with RSIG headers
    decompress : bool
        If True, decompress buffer.
        If False, buffer is already decompressed (or was never compressed)

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
    """
    import numpy as np
    import pandas as pd
    import xdrlib
    import io

    bf = io.BytesIO(buffer)
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

    n = bf.tell()
    up = xdrlib.Unpacker(buffer)
    up.set_position(n)

    notes = [up.unpack_fstring(80).decode().strip() for i in range(nprof)]
    longnpb = up.unpack_fstring(nprof * 8)
    longnp = np.frombuffer(longnpb, dtype='>l')
    sdn = up.get_position()
    # Read all values at once
    # vals = np.frombuffer(buffer[sdn:], dtype='>d')
    dfs = []
    up.set_position(sdn)
    varwunits = [varkey + f'({units[i]})' for i, varkey in enumerate(varkeys)]
    for i, npoints in enumerate(longnp):
        nfloats = npoints * nvar
        vals = np.array(up.unpack_farray(nfloats, up.unpack_double)).reshape(
            nvar, npoints
        )
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


def from_swath(buffer):
    """
    Currently supports Swath (v2.0) which has 14 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    buffer : bytes
        Data buffer in XDR format with RSIG headers
    decompress : bool
        If True, decompress buffer.
        If False, buffer is already decompressed (or was never compressed)

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
    """

    import numpy as np
    import pandas as pd
    import xdrlib
    import io

    bf = io.BytesIO(buffer)
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

    n = bf.tell()
    up = xdrlib.Unpacker(buffer)
    up.set_position(n)

    # MSB 64-bit integers (yyyydddhhmm) timestamps[scans]
    nts = np.frombuffer(up.unpack_fstring(nscan * 8), dtype='>l')
    # MSB 64-bit integers points[scans]
    nps = np.frombuffer(up.unpack_fstring(nscan * 8), dtype='>l')

    infmt = '%Y%j%H%M'
    outfmt = '%Y-%m-%dT%H:%M:%S%z'
    timestamps = pd.to_datetime(nts, format=infmt, utc=True).strftime(outfmt)

    sdn = up.get_position()
    # Read all values at once
    # IEEE-754 64-bit reals data_1[variables][points_1] ...
    #           data_S[variables][points_S]
    dfs = []
    up.set_position(sdn)
    varwunits = [varkey + f'({units[i]})' for i, varkey in enumerate(varkeys)]
    # To-do
    # Switch from iterative unpacking to numpy.frombuffer, which is much faster
    # this requires some fancy indexing and complex repeats.
    for i, npoints in enumerate(nps):
        nfloats = npoints * nvar
        vals = np.array(up.unpack_farray(nfloats, up.unpack_double)).reshape(
            nvar, npoints
        )
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


def from_site(buffer):
    """
    Currently supports Site (v2.0) which has 14 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    buffer : bytes
        Data buffer in XDR format with RSIG headers
    decompress : bool
        If True, decompress buffer.
        If False, buffer is already decompressed (or was never compressed)

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
    #   Next sum(2 for s in nsite) * 4 are data as 64-bit reals
    #   Next sum(N_s for s in nsite) * 4 are data as 64-bit reals
    """
    import numpy as np
    import pandas as pd
    import xdrlib
    import io

    bf = io.BytesIO(buffer)
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

    time = pd.date_range(stime, periods=nt, freq='H')
    n = bf.tell()
    up = xdrlib.Unpacker(buffer)
    up.set_position(n)

    notes = [up.unpack_fstring(80).decode().strip() for i in range(nsite)]

    nis = np.frombuffer(up.unpack_fstring(nsite * 4), dtype='>i')
    xys = np.frombuffer(up.unpack_fstring(nsite * 8), dtype='>f').reshape(
        nsite, 2
    )

    sdn = up.get_position()
    # Read all values at once
    # vals = np.fromstring(buffer[sdn:], dtype='>d')
    up.set_position(sdn)
    varwunits = [varkey + f'({units[i]})' for i, varkey in enumerate(varkeys)]

    # Unpacking data using from buffer, which is much faster than iteratively
    # calling xdr.unpack_farray
    vals = np.frombuffer(buffer[sdn:], dtype='>f').reshape(1, nsite, nt)
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


def from_point(buffer):
    """
    Currently supports Point (v1.0) which has 12 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    buffer : bytes
        Data buffer in XDR format with RSIG headers
    decompress : bool
        If True, decompress buffer.
        If False, buffer is already decompressed (or was never compressed)

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
    """
    import numpy as np
    import pandas as pd
    import xdrlib
    import io

    bf = io.BytesIO(buffer)
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

    n = bf.tell()
    up = xdrlib.Unpacker(buffer)
    up.set_position(n)

    notes = [up.unpack_fstring(80).decode().strip() for i in range(npoint)]

    sdn = up.get_position()
    # Use numpy to unpack from buffer becuase it is faster than unpack
    vals = np.frombuffer(buffer[sdn:], dtype='>d').reshape(nvar, npoint)
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


def from_calipso(buffer):
    """
    Currently supports Calipso (v1.0) which has 15 header rows in text format.
    The text header rows also describe the binary portion of the file.

    Arguments
    ---------
    buffer : bytes
        Data buffer in XDR format with RSIG headers
    decompress : bool
        If True, decompress buffer.
        If False, buffer is already decompressed (or was never compressed)

    Returns
    -------
    df : pd.DataFrame
        Dataframe with XDR content

    Notes
    -----
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
    import xdrlib
    import io
    import xarray as xr

    bf = io.BytesIO(buffer)
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

    n = bf.tell()
    up = xdrlib.Unpacker(buffer)
    up.set_position(n)
    sdn = up.get_position()
    stime = sdn
    etime = stime + nprof * 8
    sbnd = etime
    ebnd = sbnd + (nprof * 4) * 8
    sdims = ebnd
    edims = sdims + (nprof * 2) * 8
    sdata = edims
    # time = np.frombuffer(buffer[stime:etime], dtype='>l')
    # bounds = np.frombuffer(
    #     buffer[sbnd:ebnd], dtype='>d'
    # ).reshape(nprof, 2, 2)
    dims = np.frombuffer(buffer[sdims:edims], dtype='>l').reshape(nprof, 2)
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
        vals = np.frombuffer(buffer[sdata:edata], dtype='>d')
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
