"""
Get DataFrame for TropOMI NO2
=============================

Shows how to get TropOMI NO2 tropospheric column densities as a pandas
DataFrame.
"""

# sphinx_gallery_thumbnail_path = '_static/tropomidf.png'

import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
df = rsigapi.to_dataframe('tropomi.offl.no2.nitrogendioxide_tropospheric_column')
print(df.shape, *df.columns)
# (303444, 4) Timestamp(UTC) LONGITUDE(deg) LATITUDE(deg) nitrogendioxide_tropospheric_column(molecules/cm2)
