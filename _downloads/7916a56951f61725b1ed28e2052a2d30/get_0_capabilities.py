"""
Get List of Possible Coverages
==============================

Shows how to get a list of RSIG coverage that are available. The offline
list is a quick reference to commonly used coverages, use offlien=False for
a full list.
"""

# sphinx_gallery_thumbnail_path = '_static/getcapabilities.png'

import pyrsig

rsigapi = pyrsig.RsigApi()
keys = rsigapi.keys()
print(len(keys), keys)
# 80 ('airnow.pm25', ... 'aqs.ozone', ... 'cmaq.equates.conus.aconc.O3', ... 'hms.smoke',
#     'metar.wind', ... 'pandora.ozone', 'purpleair.pm25_corrected', ...
#     'tropomi.offl.no2.nitrogendioxide_tropospheric_column', ...
#     'viirsnoaa.jrraod.AOD550', ...)
keys = rsigapi.keys(offline=False)  # slow and likely to many options
print(len(keys))
# 3875
