import pyrsig


cmaqkey = 'cmaq.equates.conus.aconc.O3'
datakey = 'airnow.ozone'

api = pyrsig.RsigApi()

# Return CMAQ for default bbox (-126, 24, -66, -50) for a single day
ds = api.to_ioapi(cmaqkey, bdate='2018-07-01')

# pair_rsigcmaq will match the bbox, bdate, and edate from the CMAQ dataset.
df = pyrsig.cmaq.pair_rsigcmaq(ds, 'O3', datakey)
# Or, use persist=True to save pairing to disk and return output path instead.
# import pandas as pd
# outpath = pyrsig.cmaq.pair_rsigcmaq(ds, 'O3', datakey, persist=True)
# df = pd.read_csv(outpath)

# Calculate stats table with common quantile, correlation, and bias metrics
statsdf = pyrsig.utils.quickstats(df[['ozone', 'CMAQ_O3']], 'ozone')
# Print them for the user to review.
print(statsdf.to_csv())
# count,21607.0,24192.0
# mean,25.355614523071228,25.866846084594727
# std,11.526748941988622,7.350318908691406
# min,0.0,0.00026858781347982585
# 25%,19.0,22.68592643737793
# 50%,28.0,27.47078800201416
# 75%,33.0,30.673885345458984
# max,64.0,48.06443405151367
# r,1.0,0.6378620099737814
# mb,0.0,0.5112315615234984
# nmb,0.0,0.02016245991822938
# fmb,0.0,0.019961226206574992
# ioa,1.0,0.7575190422148421
