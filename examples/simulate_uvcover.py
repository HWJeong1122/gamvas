
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as au
from astropy.coordinates import Angle, SkyCoord

import gamvas as gv

# Source name & sky coordinates
source = "1749+096"
ra  = Angle("04h23m15.801s").to(au.deg)
dec = Angle("-01d20m33.066s").to(au.deg)
source_coord = SkyCoord(ra, dec)

# observation setup
frequency = np.array([21.5e9, 22.0e9, 22.5e9])    # frequency [Hz]
bandwidth = np.array([256e6, 256e6, 256e6])       # bandwidth [Hz]
array = ["KC", "KY", "KU", "KT"]                  # antenna
tarr = None                     # gamvas.load.open_fits.tarr (optional)
date = "2000-01-01"             # observing date
tstart = 0                      # observation start time [hour]
duration = 24                   # observation duration   [hour]
scanlen = 3                     # scan length            [minute]
gaptime = 0.1                   # time gap between scans [hour]
tint = 10                       # integration time       [second]
ellim = 15                      # elevation limit        [deg]

# =====================================================================
# Simulate (u,v)-coverage
# =====================================================================
# Create VLBI array and simulate (u,v)-coverage
carr = gv.simulation.create_array.CreateArray(
    source=source, source_coord=source_coord,
    frequency=frequency + bandwidth / 2, array=array, tarr=tarr, date=date,
    tstart=tstart, duration=duration, scanlen=scanlen, gaptime=gaptime,
    tint=tint, ellim=ellim
)

uvcov = carr.carr(returned=True)
tarr = carr.tarr

# check data fields
print("# Data field (tarr):\n", tarr.dtype.names)
print("# Data field (uvcov):\n", uvcov.dtype.names)

# check uv-coverage
uvmax = np.max(
    np.append(np.abs(uvcov["u"]), np.abs(uvcov["v"]))
) * 1.1

plt.gca().set_aspect("equal")
plt.gca().grid(True)
plt.xlabel(r"$U~(\lambda)$")
plt.ylabel(r"$V~(\lambda)$")
plt.scatter(+uvcov["u"], +uvcov["v"], c="k", s=5, marker="o")
plt.scatter(-uvcov["u"], -uvcov["v"], c="k", s=5, marker="o")
plt.xlim(-uvmax, uvmax)
plt.ylim(-uvmax, uvmax)
plt.gca().invert_xaxis()
plt.show()
