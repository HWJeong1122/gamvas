
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import gamvas as gv

# NOTE:
#   This script is for reproducing resultant plots
#   and for printing statistics
#   using obtained model parameters
#   (for multi-frequency)

# TODO: basic setting
source = "1928+738"
date = "2024-12-13"
uvw = "u"           # (str): uv-weighting (for drawing images)
select = "ll"       # (str): polarization type
snrflag = 3         # (float): snr-flagging value
scanlen = 300       # (float) [sec]: scan length
gaptime = 60        # (float) [sec]: gap time between scans
uvave = 10          # (float) [sec]: uv-averaging

npix = 256          # (int): number of image pixels
mrng = 22

# TODO: spectrum setting (for multi-freq.)
spectrum = "ssa"
fitset = "mf"
ifsingle = False
set_spectrum = True

bands = ["22", "43", "86", "129"]
ferrs = [0.20, 0.20, 0.20, 0.30]    # applied fractional errors to visibility

path = f"{os.getcwd()}/"

# TODO: set to your resultant 'model_params.xlsx'
file_xlsx =\
    f"{path}/1928+738/{date}/Pol_{select.upper()}/mf/model_params.xlsx"

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

uvfs = [None for i in range(len(bands))]

for i, band in enumerate(bands):
    uvfs[i] =\
        gv.load.open_fits(
            path=f"{path}/uvfs/",
            file=f"c.{band}.{source}.{date}.uvf",
            mrng=mrng * gv.mas
        )

    uvfs[i].load_uvf(
        select=select, uvw=uvw,
        scanlen=scanlen, gaptime=gaptime, uvave=uvave
    )

    if band == "129":
        nant = 3
        ftype = ["amp", "clphs"]
    else:
        nant = 4
        ftype = ["amp", "clamp", "clphs"]

    uvfs[i].flag_uvvis(type="snr", value=3)
    uvfs[i].flag_uvvis(type="nant", value=nant)

    ### flag below if you don't want to consider systematics
    ### estimated from median absolute deviation (MAD)
    uvfs[i].apply_systematics()

    ### activate below if you want to consider fractional errors
    ### in chi-square calculations
    # uvfs[i].add_error_fraction(ferrs[i])

    uvfs[i].ploter.bprms = uvfs[i].bprms
    uvfs[i].ploter.prms = mprms
    uvfs[i].append_visibility_model(
        freq_ref=uvfs[0].freq, freq=uvfs[i].freq,
        theta=mprms, fitset=fitset, model="gaussian",
        spectrum=spectrum, set_spectrum=set_spectrum
    )
    uvfs[i].selfcal(type="phs")
    uvfs[i].selfcal(type="a&p")

    # print statistics at each frequency
    clamp_uvcomb, clphs_uvcomb =\
        gv.utils.set_uvcombination(
            uvfs[i].data,
            uvfs[i].tmpl_clamp,
            uvfs[i].tmpl_clphs
        )

    uvcomb =\
        (
            copy.deepcopy(uvfs[i].clamp["clamp"]),
            copy.deepcopy(uvfs[i].clphs["clphs"]),
            copy.deepcopy(uvfs[i].clamp["sigma_clamp"]),
            copy.deepcopy(uvfs[i].clphs["sigma_clphs"]),
            clamp_uvcomb,
            clphs_uvcomb
        )

    res =\
        gv.utils.print_stats(
            uvf=uvfs[i],
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

uvall = gv.utils.set_uvf(uvfs, type="mf")
uvall.ploter.spectrum = spectrum

# print statistics in multi-freqency dataset
clamp_uvcomb_mf, clphs_uvcomb_mf =\
    gv.utils.set_uvcombination(
        uvall.data,
        uvall.tmpl_clamp,
        uvall.tmpl_clphs
    )

uvcomb_mf =\
    (
        copy.deepcopy(uvall.clamp["clamp"]),
        copy.deepcopy(uvall.clphs["clphs"]),
        copy.deepcopy(uvall.clamp["sigma_clamp"]),
        copy.deepcopy(uvall.clphs["sigma_clphs"]),
        clamp_uvcomb_mf,
        clphs_uvcomb_mf
    )

print("\n# Statistics from multi-frequency dataset")
res_mf =\
    gv.utils.print_stats(
        uvf=uvall,
        uvcomb=uvcomb_mf,
        k=k,
        logz=np.nan,
        dlogz=np.nan,
        ftype=["amp", "clamp", "clphs"]
    )

uvall.ploter.clq_mod =\
    gv.utils.set_closure(
        uvall.data["u"],
        uvall.data["v"],
        uvall.data["vism"],
        np.zeros(uvall.data["vism"].shape[0]),
        uvall.data["ant_name1"],
        uvall.data["ant_name2"],
        clamp_uvcomb_mf,
        clphs_uvcomb_mf
    )

uvall.ploter.draw_radplot(uvall, plotimg=True, plotvism=True)
uvall.ploter.draw_closure(type="clamp", plotimg=True, model=True)
uvall.ploter.draw_closure(type="clphs", plotimg=True, model=True)
for i, band in enumerate(bands):
    images =\
        uvall.ploter.draw_image(
            uvf=uvall, plotimg=True,
            npix=npix, mindr=3, plot_resi=True, addnoise=True,
            freq_ref=uvfs[0].freq, freq=uvfs[i].freq, model="gaussian",
            ifsingle=ifsingle, set_spectrum=set_spectrum,
            returned=True
        )
    # images[0]: resultant image
    # images[1]: corresponding residual map
