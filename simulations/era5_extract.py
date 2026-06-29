#!/usr/bin/env python
"""Extract ERA5 surface (1000 hPa) u/v wind for a lat/lon bbox at one time index,
returning the native ERA5 grid points as JSON for the Julia driver's F_src buffers.

This is the data-access shim for the live-data path (item 3): the Julia adapter env
is MTK/NetCDF-free, so the driver shells out to a NetCDF-capable Python (anaconda's
netCDF4) to read the real Camp Fire ERA5 file. It is NOT the formal EarthSciData
loader — it gets real ERA5 values into F_src so the front is driven by, and refreshes
from, real data; the EarthSciData/EarthSciIO loader binding is the remaining step.

Usage: era5_extract.py <ncfile> <lon_min> <lon_max> <lat_min> <lat_max> <time_index>
Output (stdout): {"lons":[...], "lats":[...], "u":[...], "v":[...], "valid_time":"..."}
  one entry per native ERA5 grid point inside the bbox (row-major, lat-major).
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import netCDF4 as nc

ncfile = sys.argv[1]
lon0, lon1, lat0, lat1 = (float(sys.argv[i]) for i in (2, 3, 4, 5))
tidx = int(sys.argv[6])

d = nc.Dataset(ncfile)
lat = np.array(d["latitude"][:])
lon = np.array(d["longitude"][:])
lev = np.array(d["pressure_level"][:])
li = int(np.argmax(lev))                 # 1000 hPa = lowest level = surface proxy
u = d["u"][tidx, li, :, :]
v = d["v"][tidx, li, :, :]
ila = np.where((lat >= lat0) & (lat <= lat1))[0]
ilo = np.where((lon >= lon0) & (lon <= lon1))[0]

lons, lats, uu, vv = [], [], [], []
for a in ila:
    for o in ilo:
        lons.append(float(lon[o]))
        lats.append(float(lat[a]))
        uu.append(float(u[a, o]))
        vv.append(float(v[a, o]))

vt = nc.num2date(d["valid_time"][tidx], d["valid_time"].units)
print(json.dumps({"lons": lons, "lats": lats, "u": uu, "v": vv, "valid_time": str(vt)}))
