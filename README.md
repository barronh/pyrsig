# pyrsig

Python interface to RSIG Web API

## Install

```bash
pip install git+https://github.com/barronh/pyrsig.git
```

## Example

## Get IOAPI formatted NetCDF TropOMI NO2

```python
import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
ds = rsigapi.to_ioapi('tropomi.offl.no2.nitrogendioxide_tropospheric_column')
```

## Get DataFrame for AQS ozone

```python
import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
df = rsigapi.to_dataframe('aqs.ozone')
```

## Get DataFrame for PurpleAir PM25

```python
import pyrsig

rsigapi = pyrsig.RsigApi(bdate='2022-03-01')
rsigapi.purpleair_kw['api_key'] = '<put your api key here>'
df = rsigapi.to_dataframe('purpleair.pm25_corrected')
```