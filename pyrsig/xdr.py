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
    elif defspec.startswith('swath'):
        df = from_swath(buffer)
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
