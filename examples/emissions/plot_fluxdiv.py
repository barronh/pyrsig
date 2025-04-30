"""
Calculate Emissions by Flux Divergence
======================================

This example uses Phoenix Arizona as an example application of the technique.
This example uses just 4 days, but can easily be extended to the whole year or
the ozone season to get a more robust estimate.

Steps:
1. Create a 3km custom L3 NO2 product on the HRRR grid,
2. Retrieve HRRR winds at cell centers for the same domain,
4. Fit Eq 1 of Goldberg et al. (10.5194/acp-22-10875-2022; equivalent to scipy.stats.exponnorm),
5. Derive tau parameter and estimate emissions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyrsig
import pandas as pd
import xarray as xr

# %%
# Define Location and Date Range
# ------------------------------
# 
# - Location is a single point
# - Date range can be a few days at a time
# - bbox is the box to retrieve VCDs

workdir = 'phoenix'
loc = (-112.0777, 33.4482)  # Phoenix Arizona - fit is not good due to multiple loci
# loc = (-95.1, 29.720621)  # Houston Texas
dates = pd.date_range('2023-05-01T00Z', '2023-05-04T00Z')
bbox = np.array(loc + loc) + np.array([-1, -1, 1, 1]) * 1.5


# %%
# Get TropOMI NO2 Over bbox
# --------------------------
#

rsig = pyrsig.RsigApi(
    bbox=bbox, workdir=workdir, grid_kw='HRRR3K', gridfit=True
)
vkey = 'tropomi.offl.no2.nitrogendioxide_tropospheric_column'
ds = rsig.to_ioapi(bdate=dates, key=vkey)

# removing missing values
ds = ds.where(lambda x: x > -9e30)

# %%
# Get Winds At Domain Center
# --------------------------
# - Find the domain center
# - Create a 0.01 degree box around it
# - retrieve the HRRR 80m wind components (requries xdr)

wkey = 'hrrr.wind_80m'
rsig.grid_kw
wds = rsig.to_ioapi(bdate=dates, key=wkey).where(lambda x: x > -9e30)

# Add wind components to the VCD dataset
ds['WIND_80M_U'] = wds['WIND_80M_U']
ds['WIND_80M_V'] = wds['WIND_80M_V']

# %%
# Calculate the Column Divergence
# -------------------------------
#

ds['DCDX'] = ds['NO2'] * np.nan
ds['DCDY'] = ds['NO2'] * np.nan
for ti, t in enumerate(ds.TSTEP.dt.strftime('%FT%HZ').values):
    print(t)
    v = ds['NO2'][ti, 0]
    if not v.isnull().all():
        dcdx, dcdy = pyrsig.emiss.divergence(v.data, 3000., 3000, withdiag=False)
        ds['DCDX'][ti, 0] = dcdx
        ds['DCDY'][ti, 0] = dcdy

# %%
# Multiply by Orthogonal Wind Components
# --------------------------------------
#

ds['FDV'] = ds['DCDX'] * ds['WIND_80M_U'] + ds['DCDY'] * ds['WIND_80M_V']
ds['FDV'].attrs.update(
    long_name='column divergence',
    units='molecules/cm2/s'
)

# %%
# Add Chemical Lifetime Correction for Emissions
# ----------------------------------------------
# - tau is a simple assumption of 2h, but should be improved
#   for specific application.
# - Units are converted to moles/m2/s for more common representation
#

tau = 7200 # s
Z = (
        ds['FDV'].mean(('TSTEP', 'LAY'))
        + ds['NO2'].mean(('TSTEP', 'LAY')) / tau
) / 6.022e23 * 1e4
Z.attrs.update(
    long_name='div(V u) + V / tau',
    units='moles/m2/s'
)
ds['emiss_nox'] = Z
outds = ds[['FDV', 'emiss_nox']]
outds.attrs.update(ds.attrs)
outds.to_netcdf('flux.nc')

# %%
# Plot Emission Results
# ---------------------
# - Create a simple plot of emissions
# - Then intersect cells with Maricopa county and sum for county total
#

qm = Z.plot()
figpath = f'{workdir}/flux.png'
qm.figure.savefig(figpath)

# Requires geopandas to add county and primary roads
try:
    import geopandas as gpd
    cntyname = 'Maricopa'
    cntypath = 'https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip'
    cnty = gpd.read_file(cntypath, bbox=tuple(bbox)).to_crs(ds.crs_proj4)
    roadpath = 'https://www2.census.gov/geo/tiger/TIGER2023/PRIMARYROADS/tl_2023_us_primaryroads.zip'
    road = gpd.read_file(roadpath, bbox=tuple(bbox)).to_crs(ds.crs_proj4)
    cnty.plot(facecolor='none', edgecolor='gray', linewidth=.6, ax=qm.axes)
    road.plot(facecolor='none', edgecolor='gray', linewidth=.3, ax=qm.axes)
    qm.figure.savefig(figpath)
except Exception as e:
    print('failed to add map', str(e))

# Requires geopandas to add county total
try:
    import shapely
    r, c = xr.broadcast(Z.ROW, Z.COL)
    cntypoly = cnty.query(f'NAME == "{cntyname}"').unary_union
    incnty = cntypoly.contains(shapely.points(np.stack([r, c], axis=-1)))  # Is each cell in Maricopa?
    massrate_nox = (Z.data[incnty].sum() * ds.XCELL * ds.YCELL) * 1.32 * 46.  #  gNO2/s
    mass = massrate_nox / 1e12 * 365 * 24 * 3600 # Tg/yr
    label = f'{cntyname} NOx as NO2 = {mass:.4f} [Tg/yr]'
    print(label)
    qm.axes.text(0.05, 0.975, label, transform=qm.axes.transAxes, va='top')
    qm.figure.savefig(figpath)
except Exception as e2:
    print('failed to calculate total', str(e2))
    