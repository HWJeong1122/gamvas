
import os
import re
import numpy as np

import gamvas as gv

import sys
import matplotlib.pyplot as plt

# =================================================================
# Set base directory path & frequency
# =================================================================
path = f"{os.getcwd()}/"
freqs = [21.8, 43.3, 86.3, 107.8]
bands = ["k", "q", "wl", "wh"]
uvfs = [None for _ in range(len(freqs))]

# =================================================================
# Check single-frequency results
# =================================================================
for nf, freq in enumerate(freqs):
    # read model parameters in '.xlsx' ('model_params.xlsx')
    theta_sf, error_sf, k_sf = gv.utils.rd_theta(
        path=f"{path}/1749+096/2025-01-02/Pol_RR/gaussian.{freq}/",
        file="model_params.xlsx",
        # id_core=1   # ID no. of the core component, if needed
    )

    # open & load calibrated UVFITS file
    uvfs[nf] = gv.load.open_fits(
        path=f"{path}/data/",
        file=f"flag.f24hj01a.1749+096.{bands[nf]}.uvf"
    )
    uvfs[nf].load_uvf(
        select_pol="rr", select_if="all",
        gaptime=180, scanlen=180
    )

    # set data: complex visibility & closure relations
    uvfs[nf].set_data()
    uvfs[nf].set_closure()

    # debias visibility amplitude
    uvfs[nf].debias_amplitude()

    # compute systematics through median absolute deviation (MAD)
    syscal_type = ["vis", "logclamp", "clphs"]
    uvfs[nf].systematics_cal(dotype=syscal_type)

    # (optional) average data (IF channels)
    uvfs[nf].average(dotype="ifchan")

    # (optional) average data (time)
    uvfs[nf].average(
        dotype="time", value=10, mode="scan"
    )

    # flag data
    uvfs[nf].flag_data(dotype="snr", value=5)
    uvfs[nf].flag_data(dotype="nant", value=3)

    # set data (averaged): complex visibility & closure relations
    uvfs[nf].set_data()
    uvfs[nf].set_closure()

    # apply estimated systematics
    uvfs[nf].systematics_apply(dotype=syscal_type)

    # load reconstructed image and model parameters from FITS file
    model_component = gv.utils.get_fits(
        path=f"{path}/1749+096/2025-01-02/Pol_RR/imgfits/",
        file=f"1749+096.2025-01-02.img.sf.{freq:.0f}.fits",
        dotype="component"
    )
    print(model_component)
    print(model_component.dtype.names)

    image_res = gv.utils.get_fits(
        path=f"{path}/1749+096/2025-01-02/Pol_RR/imgfits/",
        file=f"1749+096.2025-01-02.img.sf.{freq:.0f}.fits",
        dotype="image"
    )
    image = image_res[0]
    mapfov = image_res[1]
    extent = [+mapfov/2, -mapfov/2, -mapfov/2, mapfov/2]
    plt.imshow(image, extent=extent, cmap="gist_heat")
    plt.show()

    # set model visibility from model parameters in the FITS file
    uvcoverage = uvfs[nf].uvcov
    visibility_model = gv.utils.dft_fits(
        path=f"{path}/1749+096/2025-01-02/Pol_RR/imgfits/",
        file=f"1749+096.2025-01-02.img.sf.{freq:.0f}.fits",
        uvcov=uvcoverage,
        dotype="component"
    )

    # transfer model visibility to UVFITS file
    uvfs[nf].model_visibility_append(vism=visibility_model)
    # NOTE:
    #     if any, you may want to establish model visibilities
    #     from the model parameters in '.xlsx' ('model_params.xlsx').
    #     Then,
    # uvfs[nf].model_visibility_append(theta=theta_sf)

    # self-calibration
    uvfs[nf].selfcal(dotype="phs")
    uvfs[nf].selfcal(dotype="a&p")

    # statistics
    clamp_uvcomb, clphs_uvcomb = gv.utils.set_uvcombination(uvfs[nf])
    uvcomb = (
        np.ma.getdata(uvfs[nf].clamp["clamp"]),
        np.ma.getdata(uvfs[nf].clphs["clphs"]),
        np.ma.getdata(uvfs[nf].clamp["sig_logclamp"]),
        np.ma.getdata(uvfs[nf].clphs["sig_clphs"]),
        clamp_uvcomb,
        clphs_uvcomb
    )

    print(f"\nStatistics @ {freq:.0f} GHz")
    stats = gv.utils.print_stats(
        uvf=uvfs[nf], uvcomb=uvcomb, k=k_sf,
        dotype=["vis", "amp", "clamp", "clphs"]
    )

    # basic plots with model
    uvfs[nf].ploter.draw_radplot(uvfs[nf], plotimg=True, plotmodel=True)
    uvfs[nf].ploter.draw_closure(dotype="clamp", plotimg=True, plotmodel=True)
    uvfs[nf].ploter.draw_closure(dotype="clphs", plotimg=True, plotmodel=True)

# =================================================================
# Check multi-frequency results
# =================================================================

# read multi-frequency model parameters in '.xlsx' ('model_params.xlsx')
theta_mf, error_mf, k_mf = gv.utils.rd_theta(
    path=f"{path}/1749+096/2025-01-02/Pol_RR/mf.gaussian.ssa/",
    file="model_params.xlsx",
    # id_core=1   # ID no. of the core component, if needed
)

# replace single-frequency model visibilities with multi-frequency ones
for _uvf in uvfs:
    _uvf.model_visibility_drop()
    _uvf.model_visibility_append(
        freq_ref=uvfs[0].freq0, theta=theta_mf,
        model="gaussian", spectrum="ssa"
    )
    _uvf.selfcal(dotype="phs")
    _uvf.selfcal(dotype="a&p")

# set multi-frequency UVFITS file
uvf = gv.utils.set_uvf(uvfs, dotype="mf")

# basic plots with model
uvf.ploter.draw_radplot(uvf, plotimg=True, plotmodel=True)
uvf.ploter.draw_closure(
    dotype="clamp", plotimg=True, plotmodel=True, plotalif=False)
uvf.ploter.draw_closure(
    dotype="clphs", plotimg=True, plotmodel=True, plotalif=False)
