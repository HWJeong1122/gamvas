
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import gamvas as gv
from imports import *

# NOTE:
#   This script is for reproducing resultant plots
#   and for printing statistics
#   using obtained model parameters
#   (for single-frequency)

# TODO: basic setting
source = "1807+698"
date = "2025-02-10"
uvw = "u"           # (str): uv-weighting (for drawing images)
select = "ll"       # (str): polarization type
snrflag = 3         # (float): snr-flagging value
scanlen = 300       # (float) [sec]: scan length
gaptime = 60        # (float) [sec]: gap time between scans
uvave = 10          # (float) [sec]: uv-averaging

npix = 256          # (int): number of image pixels
mrng = 22

# TODO: spectrum setting (for multi-freq.)
spectrum = "single"
fitset = "sf"
ifsingle = True
set_spectrum = False

band = "22"
ferr = 0.20     # applied fractional errors to visibility

path = f"{os.getcwd()}/"

# TODO: set to your resultant 'model_params.xlsx'
file_xlsx =\
    f"/path/to/your/model_params.xlsx"

rd_prms =\
    gv.utils.rd_mprms(
        file=file_xlsx,
        cid=1
        # core component ID = 1 (i.e., core := model #1)
        # NOTE: core component is fixed to (0,0) on the map
    )
mprms = rd_prms[0]  # measured median values
eprms = rd_prms[1]  # estimated error ranges (mean of upper and lower limits)
k = rd_prms[2]      # the number of model parameters

uvf =\
    gv.load.open_fits(
        path=f"/path/to/your/uvf/files/",
        file=f"file.name.uvf",
        mrng=mrng * gv.mas
    )

uvf.load_uvf(
    select=select, uvw=uvw,
    scanlen=scanlen, gaptime=gaptime, uvave=uvave
)

if band == "129":
    nant = 3
    ftype = ["amp", "clphs"]
else:
    nant = 4
    ftype = ["amp", "clamp", "clphs"]

uvf.flag_uvvis(type="snr", value=3)
uvf.flag_uvvis(type="nant", value=nant)

### flag below if you don't want to consider systematics
### estimated from median absolute deviation (MAD)
uvf.apply_systematics()

### activate below if you want to consider fractional errors
### in chi-square calculations
# uvf.add_error_fraction(ferr)

uvf.ploter.bprms = uvf.bprms
uvf.ploter.prms = mprms
uvf.ploter.spectrum = spectrum
uvf.append_visibility_model(
    freq_ref=uvf.freq, freq=uvf.freq,
    theta=mprms, fitset=fitset, model="gaussian",
    spectrum=spectrum, set_spectrum=set_spectrum
)
uvf.selfcal(type="phs")
uvf.selfcal(type="a&p")

# print statistics at each frequency
clamp_uvcomb, clphs_uvcomb =\
    gv.utils.set_uvcombination(
        uvf.data,
        uvf.tmpl_clamp,
        uvf.tmpl_clphs
    )

uvcomb =\
    (
        copy.deepcopy(uvf.clamp["clamp"]),
        copy.deepcopy(uvf.clphs["clphs"]),
        copy.deepcopy(uvf.clamp["sigma_clamp"]),
        copy.deepcopy(uvf.clphs["sigma_clphs"]),
        clamp_uvcomb,
        clphs_uvcomb
    )

res =\
    gv.utils.print_stats(
        uvf=uvf,
        uvcomb=uvcomb,
        k=k,            # the number of model parameters
        logz=np.nan,    # estimated objective function (not important here)
        dlogz=np.nan,   # uncertainty of objective function
        ftype=ftype
    )
# res[0]: input ftypes
# res[1]: out chi2 values (including vis term first)
# res[2]: out AIC values (including vis term first)
# res[3]: out BIC values (including vis term first)

uvf.ploter.clq_mod =\
    gv.utils.set_closure(
        uvf.data["u"],
        uvf.data["v"],
        uvf.data["vism"],
        np.zeros(uvf.data["vism"].shape[0]),
        uvf.data["ant_name1"],
        uvf.data["ant_name2"],
        clamp_uvcomb,
        clphs_uvcomb
    )

uvf.ploter.draw_radplot(uvf, plotimg=True, plotvism=True)
uvf.ploter.draw_closure(type="clamp", plotimg=True, model=True)
uvf.ploter.draw_closure(type="clphs", plotimg=True, model=True)
images =\
    uvf.ploter.draw_image(
        uvf=uvf, plotimg=True,
        npix=npix, mindr=3, plot_resi=True, addnoise=True,
        freq_ref=uvf.freq, freq=uvf.freq, model="gaussian",
        ifsingle=ifsingle, set_spectrum=set_spectrum,
        returned=True
    )
# images[0]: resultant image
# images[1]: corresponding residual map
