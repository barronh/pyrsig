"""
Get DataFrame for TropOMI NO2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
df = rsigapi.to_dataframe('tropomi.offl.no2.nitrogendioxide_tropospheric_column')
print(df.shape, *df.columns)
# (303444, 4) Timestamp(UTC) LONGITUDE(deg) LATITUDE(deg) nitrogendioxide_tropospheric_column(molecules/cm2)
