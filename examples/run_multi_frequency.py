
import os
import re
import numpy as np

import gamvas as gv

if __name__ == "__main__":

    # NOTE (for MacOS only):
    #   use fork start method to avoid multiprocessing issues
    import multiprocessing as mpr
    mpr.set_start_method("fork")

    # =================================================================
    # TODO: Define the number of CPU cores to use
    # =================================================================
    # ncpu = 1                # use a single CPU core
    ncpu = os.cpu_count()   # use all available CPU cores

    # =================================================================
    # TODO: Define paths and files
    # =================================================================
    # in this example, we will use UVFITS file calibrated from
    # single-frequency modeling
    cwd = f"{os.getcwd()}/"
    path_uvf = f"{cwd}/1749+096/2025-01-02/Pol_RR/uvfits/"
    files = [
        "gv.sf.gaussian.flat.22.1749+096.2025-01-02.uvf",
        "gv.sf.gaussian.flat.43.1749+096.2025-01-02.uvf",
        "gv.sf.gaussian.flat.86.1749+096.2025-01-02.uvf",
        "gv.sf.gaussian.flat.108.1749+096.2025-01-02.uvf"
    ]

    # =================================================================
    # TODO: Define polarization, IF channels, and model-fit parameters
    # =================================================================
    # polarization & IF channels
    select_pol = "rr"
    select_if = "all"

    # model-fit parameters
    maxn = 5                # maximum allowed number of components
    fixnmod = False         # if True, fix number of components to 'maxn'
    model = "gaussian"      # fit Gaussian models
    spectrum = "ssa"        # use synchrotron self-absorption spectrum
    snrflag = 5             # SNR threshold
    gaptime = 180           # gap time between scans (seconds)
    scanlen = gaptime       # scan length (seconds)
    avgtime = None          # average time (seconds, None = scan length)
    avgmode = "scan"        # average mode (scan or day)
    avgweight = True        # use weighted average

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
    uvfs = [None for i in range(len(files))]
    bands = []

    for i, file in enumerate(files):
        # open UVFITS file
        uvfs[i] = gv.load.open_fits(
            path=path_uvf, file=file, mapfov=mapfov, mapunit="mas"
        )

        # load UVFITS file
        uvfs[i].load_uvf(
            select_pol=select_pol, select_if=select_if,
            gaptime=gaptime, scanlen=scanlen
        )

        # set data: complex visibility & closure relations
        uvfs[i].set_data()
        uvfs[i].set_closure()

        # compute systematics through median absolute deviation (MAD)
        syscal_type = ["vis", "logclamp", "clphs"]
        uvfs[i].systematics_cal(dotype=syscal_type)

        # # (optional) average data (IF channels)
        # uvfs[i].average(dotype="ifchan", weighted=avgweight)

        # (optional) average data (time)
        uvfs[i].average(
            dotype="time", weighted=avgweight, value=avgtime, mode=avgmode
        )

        # debias visibility amplitude
        uvfs[i].debias_amplitude()

        # flag data
        uvfs[i].flag_data(dotype="snr", value=snrflag)
        uvfs[i].flag_data(dotype="nant", value=3)

        # set data (averaged): complex visibility & closure relations
        uvfs[i].set_data()
        uvfs[i].set_closure()

        # apply estimated systematics
        uvfs[i].systematics_apply(dotype=syscal_type)

        # (optional) increase uncertainty of complex visibility: 0.1 = 10%
        uvfs[i].add_fractional_error(value=0.1)
        # uvfs[i].increase_sigma_factor(value=1)

        bands.append(f"{round(uvfs[i].freq_mean)}")

    uvf = gv.utils.set_uvf(uvfs, dotype="mf")

    # NOTE: manually define boundary conditions
    #     if any, you may want to define boundary conditions separately
    #     for each model components, for example,
    # _vis = uvf.get_data("vis")
    # sblf = np.nanmax(np.abs(_vis))
    # fmin, fmax = uvf.ufreq.min(), uvf.ufreq.max()
    # maxn = 4
    # bnd_s = [[0, sblf], [0, sblf], [0, sblf], [0, sblf]] # (flux density, Jy)
    # bnd_a = [[0, 3], [0, 3], [0, 5], [0, 5]] # (angular size, mas)
    # bnd_l = [[-1, +1], [-1, +3], [-1, +6], [-1, +6]] # (RA, mas)
    # bnd_m = [[-1, +1], [-1, +4], [-1, +6], [-1, +6]] # (Dec, mas)
    # bnd_f = [[fmin, fmax], [fmin, fmax],
    #          [fmin, fmax], [fmin, fmax]] # (turnover frequency, GHz)
    # bnd_i = [[-3, 0], [-3, 0], [-3, 0], [-3, 0]] # (spectral index)
    # boundset = (bnd_s, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i)

    # basic information
    source = uvf.source
    date = uvf.date
    ufreq = uvf.ufreq

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
    uvf.ploter.draw_closure(dotype="clamp", plotimg=True, plotalif=False)

    # check plot (closure phase)
    uvf.ploter.draw_closure(dotype="clphs", plotimg=True, plotalif=False)

    # check plot (uv-coverage)
    uvf.ploter.draw_uvcover(uvf, plotimg=True, dotype="a&p")

    # check plot (uv-coverage)
    uvf.ploter.draw_dirtymap(uvf, plotimg=True)

    # define fit type & weight
    # NOTE: in multi-frequency framework, amplitude should be included
    ftype = ["amp", "clamp", "clphs"]
    fwght = [0.1, 1, 1]

    mfu = gv.modeling.modeling(
        uvfs=uvfs, select_pol=select_pol, sampler="slice", factor_sblf=1.5,
        ftype=ftype, fwght=fwght, ufreq=ufreq, bands=bands, spectrum=spectrum,
        maxn=maxn, fixnmod=fixnmod, mapfov=mapfov, bnd_a=bnd_a, bnd_l=bnd_l,
        bnd_m=bnd_m, path_fig=path_fig_, source=source, date=date, ncpu=ncpu,
        model=model, boundset=boundset
    )
    mfu.run()

    # re-run the model with different fit type & weight
    mfu.ftype = ["amp", "clamp", "clphs"]
    mfu.fwght = [1, 1, 1]
    mfu.run()
