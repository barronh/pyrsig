"""
Get IOAPI formatted NetCDF TropOMI NO2
=======================================

Shows how to get TropOMI as an xarray Dataset. This example downloads a NetCDF
file with IOAPI metadata, which is opened and returned.
"""

# sphinx_gallery_thumbnail_path = '_static/tropominc.png'

import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
ds = rsigapi.to_ioapi('tropomi.offl.no2.nitrogendioxide_tropospheric_column')
print(ds.data_vars)
# Data variables:
#     TFLAG      (TSTEP, VAR, DATE-TIME) int32 ...
#     LONGITUDE  (TSTEP, LAY, ROW, COL) float32 ...
#     LATITUDE   (TSTEP, LAY, ROW, COL) float32 ...
#     COUNT      (TSTEP, LAY, ROW, COL) int32 ...
#     NO2        (TSTEP, LAY, ROW, COL) float32 ...
