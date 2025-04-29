"""
Calculate Emissions by Flux Divergence
======================================

This example uses Houston Texas as an example of where the technique provides
a reasonable emission estimate. This example uses just 4 days, but could be
extended to the ozone season to get a more robust estimate.

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
import geopandas as gpd

# %%
# Define Location and Date Range
# ------------------------------
# 
# - Location is a single point
# - Date range can be a few days at a time
# - bbox is the box to retrieve VCDs

# loc = (-112.0777, 33.4482)  # Phoenix Arizona - fit is not good due to multiple loci
loc = (-95.1, 29.720621)  # Houston Texas
dates = pd.date_range('2023-05-01T00Z', '2023-05-04T00Z')
bbox = np.array(loc + loc) + np.array([-1, -1, 1, 1]) * 1.5


# %%
# Get TropOMI NO2 Over bbox
# --------------------------
#

rsig = pyrsig.RsigApi(
    bbox=bbox, workdir='flux', grid_kw='HRRR3K', gridfit=True
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
wds = rsig.to_ioapi(bdate=dates, key=wkey).where(lambda x: x > -9e30)

# Add wind components to the VCD dataset
ds['WIND_80M_U'] = wds['WIND_80M_U']
ds['WIND_80M_V'] = wds['WIND_80M_V']
ds['DCDX'] = ds['NO2'] * np.nan
ds['DCDY'] = ds['NO2'] * np.nan
for ti, t in enumerate(ds.TSTEP.dt.strftime('%FT%HZ')):
    print(t)
    v = ds['NO2'][ti, 0]
    if not v.isnull().all():
        dcdx, dcdy = pyrsig.emiss.divergence(v.data, 3000., 3000, withdiag=False)
        ds['DCDX'][ti, 0] = dcdx
        ds['DCDY'][ti, 0] = dcdy

ds['FDV'] = ds['DCDX'] * ds['WIND_80M_U'] + ds['DCDY'] * ds['WIND_80M_V']
Z = FDV.mean(('TSTEP', 'LAY'))
qm = Z.plot()
qm.figure.savefig('flux.png')