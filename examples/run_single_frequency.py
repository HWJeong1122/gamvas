
import os
import re
import numpy as np

import gamvas as gv

if __name__ == "__main__":

    # For MacOS only:
    #   use fork start method to avoid multiprocessing issues
    import multiprocessing as mpr
    mpr.set_start_method("fork")

    # define number of CPUs to use
    # ncpu = 1                # use a single CPU core
    ncpu = os.cpu_count()   # use all available CPU cores

    # =================================================================
    # TODO: Define paths and files
    # =================================================================
    cwd = f"{os.getcwd()}/"
    path_uvf = f"{cwd}/data/"
    file = "flag.f24hj01a.1749+096.k.uvf"
    # file = "flag.f24hj01a.1749+096.q.uvf"
    # file = "flag.f24hj01a.1749+096.wl.uvf"
    # file = "flag.f24hj01a.1749+096.wh.uvf"

    # =================================================================
    # TODO: Define polarization, IF channels, and model-fit parameters
    # =================================================================
    # Polarization & IF channels
    select_pol = "rr"
    select_if = "all"

    # model-fit parameters
    maxn = 4                # maximum allowed number of components
    fixnmod = False         # if True, fix number of components to 'maxn'
    model = "gaussian"      # fit Gaussian models
    spectrum = "flat"       # use flat spectrum
    snrflag = 5             # SNR threshold
    gaptime = 180           # gap time between scans (seconds)
    scanlen = gaptime       # scan length (seconds)
    avgtime = 10            # average time (seconds, None = scan length)
    avgmode = "scan"        # average mode (scan or day)
    avgweight = True        # use weighted average
    rscsbl = False          # amplitude scaling method
    # NOTE: if rscsbl == True and 'amp' term is not included in 'ftype',
    #     amplitude scaling is applied to the short baseline;
    #     otherwise, the scaling factor is calculated using all visibilities

    # =================================================================
    # TODO: Define field of view & boundary conditions
    # =================================================================
    mapfov = 30         # field of view (mas)
    bnd_a = 5           # global maximum angular size (mas)
    bnd_l = [-1, +6]    # global boundary condition for RA (mas)
    bnd_m = [-1, +6]    # global boundary condition for DEC (mas)
    boundset = None     # set of boundary conditions

    # =================================================================
    # Open & load UVFITS file
    # =================================================================
    # open UVFITS file
    uvf = gv.load.open_fits(
        path=path_uvf, file=file, mapfov=mapfov, mapunit="mas"
    )

    # load UVFITS file
    uvf.load_uvf(
        select_pol=select_pol, select_if=select_if,
        gaptime=gaptime, scanlen=scanlen
    )

    # set data: complex visibility & closure relations
    uvf.set_data()
    uvf.set_closure()

    # debias visibility amplitude
    uvf.debias_amplitude()

    # compute systematics through median absolute deviation (MAD)
    syscal_type = ["vis", "logclamp", "clphs"]
    uvf.systematics_cal(dotype=syscal_type)

    # (optional) average data (IF channels)
    uvf.average(dotype="ifchan", weighted=avgweight)

    # (optional) average data (time)
    uvf.average(
        dotype="time", weighted=avgweight, value=avgtime, mode=avgmode
    )

    # flag data
    uvf.flag_data(dotype="snr", value=snrflag)
    uvf.flag_data(dotype="nant", value=3)

    # set data (averaged): complex visibility & closure relations
    uvf.set_data()
    uvf.set_closure()

    # apply estimated systematics
    uvf.systematics_apply(dotype=syscal_type)

    # # (optional) increase uncertainty of complex visibility: 0.1 = 10%
    # uvf.increase_sigma_fraction(value=0.1)
    # uvf.increase_sigma_factor(value=1)

    # NOTE: manually define boundary conditions
    #     if any, you may want to define boundary conditions separately
    #     for each model components, for example,
    out_sbl = uvf.get_sblf()
    sblf = out_sbl[0]
    fmin, fmax = uvf.ufreq.min(), uvf.ufreq.max()
    maxn = 4
    bnd_s = [[0, sblf], [0, sblf], [0, sblf], [0, sblf]] # (flux density, Jy)
    bnd_a = [[0, 1], [0, 3], [0, 5], [0, 5]] # (angular size, mas)
    bnd_l = [[-1, +1], [-1, +3], [-1, +6], [-1, +6]] # (RA, mas)
    bnd_m = [[-1, +1], [-1, +4], [-1, +6], [-1, +6]] # (Dec, mas)
    bnd_f = [[fmin, fmax], [fmin, fmax],
             [fmin, fmax], [fmin, fmax]] # (turnover freq, Hz)
    bnd_i = [[-3, 0], [-3, 0], [-3, 0], [-3, 0]] # (spectral index)
    boundset = (bnd_s, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i)

    # basic information
    source = uvf.source
    date = uvf.date
    ufreq = uvf.ufreq
    freq = round(uvf.freq_mean)
    bands = [f"{freq}"]

    # base path to save modeling results
    path_fig_ = f"{cwd}/{source}/{date}/Pol_{select_pol.upper()}/"
    gv.utils.mkdir(path_fig_)

    # check plot (radial plot)
    uvf.ploter.draw_radplot(
        uvf, plotimg=True, dotype=["amp", "phs", "sigma", "snr"]
    )

    # check plot (time plot)
    uvf.ploter.draw_tplot(uvf, plotimg=True)

    # check plot (logarithmic closure amplitude)
    uvf.ploter.draw_closure(dotype="clamp", plotimg=True)

    # check plot (closure phase)
    uvf.ploter.draw_closure(dotype="clphs", plotimg=True)

    # check plot (uv-coverage)
    uvf.ploter.draw_uvcover(uvf, plotimg=True, dotype="a&p")

    # check plot (uv-coverage)
    uvf.ploter.draw_dirtymap(uvf, plotimg=True)

    # define fit type & weight
    ftype = ["clamp", "clphs"]
    fwght = [1, 1]

    mfu = gv.modeling.modeling(
        uvfs=uvf, select_pol=select_pol, ftype=ftype, fwght=fwght, ufreq=ufreq,
        bands=bands, spectrum=spectrum, maxn=maxn, fixnmod=fixnmod,
        mapfov=mapfov, bnd_a=bnd_a, bnd_l=bnd_l, bnd_m=bnd_m,
        path_fig=path_fig_, source=source, date=date, ncpu=ncpu, model=model,
        boundset=boundset, rscsbl=rscsbl, save_uvfits=True, save_imgfits=True
    )
    mfu.run()

    # Re-run the model-fit with the inclusion of 'amp' term
    mfu.ftype = ["amp", "clamp", "clphs"]
    mfu.fwght = [1, 1, 1]
    mfu.run()
