"""
Get DataFrame for PurpleAir PM25
================================

Shows how to get PurpleAir measurements as a DataFrame. PurpleAir, unlike other
coverages, requires an api_key that you must get from PurpleAir.
"""

# sphinx_gallery_thumbnail_path = '_static/purpleairpm.png'

import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
rsigapi.purpleair_kw['api_key'] = '<put your api key here>'
df = rsigapi.to_dataframe('purpleair.pm25_corrected')
