# pyrsig

Python interface to RSIG Web API

## Install

```bash
pip install git+https://github.com/barronh/pyrsig.git
```

## Example

## Get DataFrame for AQS ozone

```python
import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
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
