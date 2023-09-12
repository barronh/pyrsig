"""
Get DataFrame for AQS ozone
^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import pyrsig

# sphinx_gallery_thumbnail_path = '_static/aqsozone.png'

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
print([k for k in rsigapi.keys() if 'ozone' in k])
# ['airnow.ozone', 'airnow2.ozone', 'aqs.ozone', 'aqs.ozone_8hour_average', 'aqs.ozone_daily_8hour_maximum', 'pandora.ozone']
df = rsigapi.to_dataframe('aqs.ozone')
print(df.shape, *df.columns)
# (26760, 6) Timestamp(UTC) LONGITUDE(deg) LATITUDE(deg) STATION(-) ozone(ppb) SITE_NAME
