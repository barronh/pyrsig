# pyrsig

Python interface to RSIG Web API

## Install

From pypi.org (most stable):

```bash
pip install pyrsig
```

From github (latest):

```bash
pip install git+https://github.com/barronh/pyrsig.git
```

## User Guide

Examples and more information are available at 
https://barronh.github.io/pyrsig

## Example

## Get DataFrame for AQS ozone

```python
import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
print([k for k in rsigapi.keys() if 'ozone' in k])
# ['airnow.ozone', 'airnow2.ozone', 'aqs.ozone', 'aqs.ozone_8hour_average', 'aqs.ozone_daily_8hour_maximum', 'pandora.ozone']
df = rsigapi.to_dataframe('aqs.ozone')
print(df.shape, *df.columns)
# (26760, 6) Timestamp(UTC) LONGITUDE(deg) LATITUDE(deg) STATION(-) ozone(ppb) SITE_NAME
```

## Get DataFrame for PurpleAir PM25

```python
import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
rsigapi.purpleair_kw['api_key'] = '<put your api key here>'
df = rsigapi.to_dataframe('purpleair.pm25_corrected')
```

## Get DataFrame for TropOMI NO2

```python
import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
df = rsigapi.to_dataframe('tropomi.offl.no2.nitrogendioxide_tropospheric_column')
print(df.shape, *df.columns)
# (303444, 4) Timestamp(UTC) LONGITUDE(deg) LATITUDE(deg) nitrogendioxide_tropospheric_column(molecules/cm2)
```

## Get IOAPI formatted NetCDF TropOMI NO2

```python
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
```

## Get List of Possible

```python
import pyrsig

rsigapi = pyrsig.RsigApi()
keys = rsigapi.keys()
print(len(keys), keys)
# 80 ('airnow.pm25', ... 'aqs.ozone', ... 'cmaq.equates.conus.aconc.O3', ... 'hms.smoke',
#     'metar.wind', ... 'pandora.ozone', 'purpleair.pm25_corrected', ...
#     'tropomi.offl.no2.nitrogendioxide_tropospheric_column', ...
#     'viirsnoaa.jrraod.AOD550', ...)
keys = rsigapi.keys(offline=False) # slow and likely to many options
print(len(keys))
# 3875
```

## Change Log

Not all changes are listed, but notable changes are itemized for ease of review.

* v0.8.5: Added simple control over cache warning.
* v0.8.4: Added support for Subset 9.0 CMAQ and Grid 1.0 xdr formats.
          Updated keys to rely on descriptions (via DescribeCoverage).
          Added utilites for basic polygon/cmaq intersections for HMS.
* v0.8.3: Added xdr Polygon 1.0 format capability, added package
          DescribeCoverage in data module and restructured utilities.
* v0.8.2: Added xdr CALIPSO 1.0 format capability.
* v0.8.1: Added xdr Point 1.0 format capability.
* v0.8.0: Restructuring module code and adding CMAQ pairing.
* v0.7.0: Added offline descriptions for review of space/time coverage.
* v0.7.0: Added TEMPO options for screening
* v0.6.0: Added latitude longitude grid pass thru support.
* v0.5.1: Added convenience function for opening many IOAPI files at once.
* v0.5.1: Updated TEMPO proxy naming.
* v0.4.6: Added support for legacy TLS servers (e.g, ofmpub and maple)
* v0.4.5: Updated TEMPO proxy naming
* v0.4.4: Adding pandora explicit support
* v0.4.3: updated to work with CMAQ EQUATES data (must exclude grid=False)
* v0.4.3: updated to support GDTYP=7 (equatorial mercator)
* v0.4.2: updated to support GDTYP=2 (equatorial mercator)
