
import os
import sys
import gc
import copy
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from scipy import optimize
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling.models import Gaussian2D
from astropy.time import Time as Ati
from astropy import units as u

import gamvas

nan = np.nan
r2m = u.rad.to(u.mas)
d2m = u.deg.to(u.mas)
d2r = u.deg.to(u.rad)

def close_figure(fig):
    """
    Close all figures and free memory.
    """
    plt.close(fig)
    plt.close("all")
    gc.collect()


def mkdir(path):
    """
    Make a directory if it does not exist.
    """
    if not os.path.isdir(path):
        os.system('mkdir %s'%(path))


def cal_rms(image):
    """
    Compute the root-mean-square (RMS) value of the input image
    Arguments:
        image (2D-array): input image
    Returns:
        float: statistical RMS value of the input image
    """
    # cent = image.size
    # unit = int(cent / 10)
    # roi_1 = image[0:unit, 0:unit].reshape(-1)
    # roi_2 = image[0:unit, cent - int(unit / 2):cent + int(unit / 2)].reshape(-1)
    # roi_3 = image[0:unit, -unit:-1].reshape(-1)
    # roi_4 = image[-unit:-1, 0:unit].reshape(-1)
    # roi_5 = image[-unit:-1, cent - int(unit / 2):cent + int(unit / 2)].reshape(-1)
    # roi_6 = image[-unit:-1, -unit:-1].reshape(-1)
    # rois = np.concatenate((roi_1, roi_2, roi_3, roi_4, roi_5, roi_6))
    # rms = np.nanstd(rois)
    rms1 = np.abs(np.percentile(image, 16) - np.percentile(image, 50))
    rms2 = np.abs(np.percentile(image, 84) - np.percentile(image, 50))
    rms = (rms1 + rms2) / 2
    return rms


def make_cntr(image, rms=None, contour_snr=3, scale=np.sqrt(2)):
    """
    Set contour levels for the input image.
        Arguments:
            image (2D-array): input image
            countre_snr (float): signal-to-noise ratio for starting contour
            scale (float): scaling factor for contour levels
        Returns:
            tuple: (positive_contour_levels, negative_contour_levels),
                pos_cntr_level (list): positive contour levels
                neg_cntr_level (list): negative contour levels
    """
    fmax = np.max(image)
    fmin = np.min(image)

    if rms is None:
        statistical_rms = cal_rms(image)
    else:
        statistical_rms = rms

    if abs(fmax) >= abs(fmin):
        neg_cntr_level = [-contour_snr * statistical_rms]
        pos_cntr_level = [+contour_snr * statistical_rms]
        while pos_cntr_level[-1] < fmax:
            cntr_inner = pos_cntr_level[-1] * scale
            pos_cntr_level.append(cntr_inner)
    else:
        neg_cntr_level = [-contour_snr * statistical_rms]
        pos_cntr_level = [+contour_snr * statistical_rms]
        while abs(neg_cntr_level[-1]) < abs(fmin):
            cntr_inner = neg_cntr_level[-1] * scale
            neg_cntr_level.append(cntr_inner)
        neg_cntr_level.sort()
    return (pos_cntr_level, neg_cntr_level)


def sarray(data, field, dtype):
    """
    Set structured array for the input data.
        Arguments:
            data (list): input data
            field (list): field names for the structured array
            dtype (list): data types for the structured array
        Returns:
            structured array: structured array for the input data
    """
    data = np.array(data)
    sarray_ = np.zeros(data.shape[1:], dtype=list(zip(field, dtype)))
    for nf,field in enumerate(field):
        sarray_[field] = data[nf]
    return sarray_


def cc2d(image1, image2, shift, psize, mrng,
    f1=False, f2=False, mask_thick=False, plotimg=True,
    save_path=False, save_name=False, save_form="pdf"
):
    """
    Compute the 2-D cross-correlation of the input images.
        Arguments:
            image1 (2D-array): input image 1
            image2 (2D-array): input image 2
            shift (float): shift range for the cross-correlation
            psize (float): pixel size of the input images
            mrng (float): map range for the cross-correlation
            f1 (float): frequency of the input image 1
            f2 (float): frequency of the input image 2
            mask_thick (bool): mask thickness for the cross-correlation
            plotimg (bool): plot the cross-correlation image
            save_path (str): save path for the cross-correlation image
            save_name (str): save name for the cross-correlation image
            save_form (str): save format for the cross-correlation image
        Returns:
            ceff: 2-D cross-correlation coefficient
            ra: RA axis for the cross-correlation
            dec: Dec axis for the cross-correlation
            peakra: RA shift for the peak of the cross-correlation
            peakdec: Dec shift for the peak of the cross-correlation
    """
    if save_path:
        mkdir(save_path)
    if mask_thick:
        alphamap = np.log(image1/image2)/np.log(f1/f2)
        image1_mask = np.where(alphamap > 0, np.nan, image1)
        image2_mask = np.where(alphamap > 0, np.nan, image2)
    else:
        image1_mask = image1.copy()
        image2_mask = image2.copy()
    mean1 = np.nanmean(image1)
    mean2 = np.nanmean(image2)

    delxy = int(shift/psize)
    maproi = int((mrng-shift)/psize)
    center = int(image1.shape[0]/2)
    shift1 = np.arange(-delxy, +delxy, 1)
    shift2 = np.arange(-delxy, +delxy, 1)

    ceff = np.zeros((shift1.shape[0], shift2.shape[0]))
    image1_roi = image1_mask.copy()[center-maproi:center+maproi, center-maproi:center+maproi]
    for i, x in enumerate(shift1):
        center1 = center+x
        for j, y in enumerate(shift2):
            center2 = center+y
            image2_roi = image2_mask.copy()[center2-maproi:center2+maproi, center1-maproi:center1+maproi]

            numer = np.nansum((image1_roi-mean1)*(image2_roi-mean2))
            denom1 = np.nansum((image1_roi-mean1)**2)
            denom2 = np.nansum((image2_roi-mean2)**2)

            rxy = numer/np.sqrt(denom1*denom2)
            ceff[i,j] = rxy
    ra = shift1*psize
    dec = shift2*psize
    ra, dec = np.meshgrid(ra, dec)

    peakloc = np.where(ceff == np.max(ceff))
    peakra = ra [peakloc][0]
    peakdec = dec[peakloc][0]
    fig_2dcc, ax_2dcc = plt.subplots(1, 1, figsize=(8, 8))
    ax_2dcc.set_aspect("equal")
    ax_2dcc.contourf(ra, dec, ceff, levels=101)
    ax_2dcc.axvline(x=peakra, c="red", ls="--")
    ax_2dcc.axhline(y=peakdec, c="red", ls="--")
    ax_2dcc.set_xlabel(r"$\rm \Delta R.A~(mas)$", fontsize=15, fontweight="bold")
    ax_2dcc.set_ylabel(r"$\rm \Delta Dec~(mas)$", fontsize=15, fontweight="bold")
    ax_2dcc.tick_params("both", labelsize=13)
    ax_2dcc.set_title(f"RA={-peakra:+.3f} | Dec={-peakdec:+.3f} (@{f1:.3f} GHz)", fontsize=15, fontweight="bold")
    if save_path and save_name:
        fig_2dcc.savefig(f"{save_path}" + f"{save_name}.{save_form}", format=save_form, dpi=300)
    if plotimg:
        plt.show()
    close_figure(fig_2dcc)
    return ceff, ra, dec, -peakra, -peakdec


def print_stats(uvf, uvcomb, k, logz, dlogz, ftype):
    """
    Print the statistics of the input UVF data.
        Arguments:
            uvf: uv-fits data
            uvcomb: uv-combination data
            k: number of parameters
            logz: log-likelihood
            dlogz: uncertainty of the log-likelihood
            ftype: fit type
        Returns:
            tuple: (data_terms, chi, aic, bic),
                data_terms (list): data terms
                chi (list): chi-squared values
                aic (list): AIC values
                bic (list): BIC values
    """
    nvis_ = uvf.data.shape[0]
    out_fty = []
    out_chi = []
    out_aic = []
    out_bic = []

    # add visibility statistics first
    obs = uvf.data["vis"]
    mod = uvf.data["vism"]
    sig = uvf.data["sigma"]
    chi =\
        0.5 * np.mean((np.abs(mod - obs) / sig)**2)
    aic =\
        np.sum(
            (np.abs(mod - obs) / sig)**2 +\
            np.log(2 * np.pi * sig**2)
        ) +\
        k * 2
    bic =\
        np.sum(
            (np.abs(mod - obs) / sig)**2 +\
            np.log(2 * np.pi * sig**2)
        ) +\
        k * np.log(nvis_)
    out_fty.append("vis")
    out_chi.append(chi)
    out_aic.append(aic)
    out_bic.append(bic)
    for nft, ft in enumerate(ftype):
        if ft == "vis":
            print("# (vis)" + " "*(10-len("vis")) + f"| Chi2 : {chi:-10.2f} | AIC : {aic:-10.2f} | BIC : {bic:-10.2f}")
            continue
        elif ft == "amp":
            obs = np.abs(uvf.data["vis"])
            mod = np.abs(uvf.data["vism"])
            sig = uvf.data["sigma"]
            chi =\
                0.5 * np.mean((np.abs(mod - obs) / sig)**2)
            aic =\
                np.sum(
                    (np.abs(mod - obs) / sig)**2 +\
                    np.log(2 * np.pi * sig**2)
                ) +\
                k * 2
            bic =\
                np.sum(
                    (np.abs(mod - obs) / sig)**2 +\
                    np.log(2 * np.pi * sig**2)
                ) +\
                k * np.log(nvis_)
        elif ft == "phs":
            obs = np.angle(uvf.data["vis"])
            mod = np.angle(uvf.data["vism"])
            sig = uvf.data["sigma"] / np.abs(uvf.data["vis"])
            chi =\
                0.5 *\
                np.mean(
                    (np.abs(np.exp(1j * mod) - np.exp(1j * obs)) / sig)**2
                )
            aic =\
                np.sum(
                    (np.abs(np.exp(1j * mod) - np.exp(1j * obs)) / sig)**2 +\
                    np.log(2 * np.pi * sig**2)
                ) +\
                k * 2
            bic =\
                np.sum(
                    (np.abs(np.exp(1j * mod) - np.exp(1j * obs)) / sig)**2 +\
                    np.log(2 * np.pi * sig**2)
                ) +\
                k * np.log(nvis_)
        elif ft in ["clamp", "clphs"]:
            clq_obs = (uvcomb[0], uvcomb[1])
            clq_mod = set_closure(
                uvf.data["u"], uvf.data["v"], uvf.data["vism"],
                np.zeros(uvf.data["vism"].shape[0]), uvf.data["ant_name1"], uvf.data["ant_name2"],
                uvcomb[4], uvcomb[5]
            )
            clq_sig = (uvcomb[2], uvcomb[3])
            if ft == "clamp":
                obs = clq_obs[0]
                mod = clq_mod[0]
                sig = clq_sig[0]
                try:
                    nclq = obs.shape[0]
                    chi =\
                        0.5 * np.mean((np.abs(mod - obs) / sig)**2)
                    aic =\
                        np.sum(
                            (np.abs(mod - obs) / sig)**2 +\
                            np.log(2 * np.pi * sig**2)
                        ) +\
                        k * 2
                    bic =\
                        np.sum(
                            (np.abs(mod - obs) / sig)**2 +\
                            np.log(2 * np.pi * sig**2)
                        ) +\
                        k * np.log(nvis_)
                except:
                    print("! Absence of closure amplitude data.")
                    print("! Replace with NaNs.")
                    chi = np.nan
                    aic = np.nan
                    bic = np.nan
            if ft == "clphs":
                obs = clq_obs[1]
                mod = clq_mod[1]
                sig = clq_sig[1]
                nclq = obs.shape[0]
                try:
                    chi =\
                        0.5 * np.mean((np.abs(np.exp(1j * mod) - np.exp(1j * obs)) / sig)**2)
                    aic =\
                        np.sum(
                            (np.abs(np.exp(1j * mod) - np.exp(1j * obs)) / sig)**2 +\
                            np.log(2 * np.pi * sig**2)
                        ) +\
                        k * 2
                    bic =\
                        np.sum(
                            (np.abs(np.exp(1j * mod) - np.exp(1j * obs)) / sig)**2 +\
                            np.log(2 * np.pi * sig**2)
                        ) +\
                        k * np.log(nvis_)
                except:
                    print("! Absence of closure phase data.")
                    print("! Replace with NaNs.")
                    chi = np.nan
                    aic = np.nan
                    bic = np.nan

        if ft in ["vis", "amp", "phs", "clphs"]:
            outft = ft
        elif ft in ["clamp"]:
            outft = "log.clamp"
        print(f"# ({outft:9s}) " + f"| Chi2 : {chi:-10.2f} | AIC : {aic:-10.2f} | BIC : {bic:-10.2f}")
        out_fty.append(outft)
        out_chi.append(chi)
        out_aic.append(aic)
        out_bic.append(bic)
    print(f"# logz : {logz:-8.2f} +/- {dlogz:-8.2f}")
    return (out_fty, out_chi, out_aic, out_bic)


def get_fwght(ftype, vdat, tmpl_clamp, tmpl_clphs):
    """
    Get the weight for the input data based on the number of data points
        Arguments:
            ftype (list): fit type
            vdat (2D-array): visibility data
            tmpl_clamp (structured array): closure amplitude template
            tmpl_clphs (structured array): closure phase template
        Returns:
            list: weight for the input
    """
    fwght_ = np.array([])
    nvis = vdat.shape[0]
    namp = vdat.shape[0]
    nphs = vdat.shape[0]
    ncamp = tmpl_clamp.shape[0]
    ncphs = tmpl_clphs.shape[0]
    wght_tot = 0
    for i in range(len(ftype)):
        if ftype[i] == "vis":
            wght_tot += 1 / nvis
            fwght_ = np.append(fwght_, 1 / nvis)
        elif ftype[i] == "amp":
            wght_tot += 1 / namp
            fwght_ = np.append(fwght_, 1 / namp)
        elif ftype[i] == "phs":
            wght_tot += 1 / nphs
            fwght_ = np.append(fwght_, 1 / nphs)
        elif ftype[i] == "clamp":
            wght_tot += 1 / ncamp
            fwght_ = np.append(fwght_, 1 / ncamp)
        elif ftype[i] == "clphs":
            wght_tot += 1 / ncphs
            fwght_ = np.append(fwght_, 1 / ncphs)
    fwght_ = fwght_ / wght_tot
    fwght_ = np.round(3 * fwght_, 3).tolist()
    return fwght_


def fit_beam(uvc):
    """
    Fit the beam parameters for the input UVF data
     (imported from eht-imaging)
     (https://achael.github.io/eht-imaging/; Chael+2018, ApJ, 857, 23C)
        Arguments:
            uvc (structured array): uv coverage data
        Returns:
            array: beam parameters
    """
    def fit_chisq(beamparams, db_coeff):
        (fwhm_maj2, fwhm_min2, theta) = beamparams
        a = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_min2 + np.sin(theta)**2 / fwhm_maj2)
        b = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_maj2 + np.sin(theta)**2 / fwhm_min2)
        c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1.0 / fwhm_maj2 - 1.0 / fwhm_min2)
        gauss_coeff = np.array((a, b, c))
        chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)
        return chisq

    uu = uvc["u"]
    vv = uvc["v"]

    wfn = np.ones(uvc.shape)

    abc = np.array([np.sum(wfn * uu**2),
                    np.sum(wfn * vv**2),
                    2 * np.sum(wfn * uu * vv)])
    abc *= (2. * np.pi**2 / np.sum(wfn))
    abc *= 1e-20    # Decrease size of coefficients

    # Fit the beam
    guess = [(50)**2, (50)**2, 0.0]
    params = optimize.minimize(fit_chisq, guess, args=(abc,), method="Powell")

    if params.x[0] > params.x[1]:
        fwhm_maj = 1e-10 * np.sqrt(params.x[0])
        fwhm_min = 1e-10 * np.sqrt(params.x[1])
        theta = np.mod(params.x[2], np.pi)
    else:
        fwhm_maj = 1e-10 * np.sqrt(params.x[1])
        fwhm_min = 1e-10 * np.sqrt(params.x[0])
        theta = np.mod(params.x[2] + np.pi / 2.0, np.pi)

    gparams = np.array((fwhm_maj, fwhm_min, theta))
    gparams[0] *= u.rad.to(u.mas)
    gparams[1] *= u.rad.to(u.mas)
    gparams[2] *= u.rad.to(u.deg)
    return gparams


def set_uvf(dataset, type="sf"):
    """
    Set the uv-fits data for the input dataset
        Arguments:
            dataset (list): input dataset
            type (str): type of the dataset ('sf': single-frequency, 'mf': multi-frequency)
        Returns:
            uvf: uv-fits data
    """
    out = copy.deepcopy(dataset[0])
    ndat = len(dataset)
    for i in range(ndat):
        data_ = dataset[i].data
        clamp_ = dataset[i].clamp
        clphs_ = dataset[i].clphs
        tmpl_clamp_ = dataset[i].tmpl_clamp
        tmpl_clphs_ = dataset[i].tmpl_clphs
        uant = np.unique(np.append(data_["ant_name1"], data_["ant_name2"]))
        if ndat == 1:
            data = data_
            clamp = clamp_
            clphs = clphs_
            tmpl_clamp = tmpl_clamp_
            tmpl_clphs = tmpl_clphs_
        else:
            if i == 0:
                data = data_
                clamp = clamp_
                clphs = clphs_
                tmpl_clamp = tmpl_clamp_
                tmpl_clphs = tmpl_clphs_
            else:
                data = rfn.stack_arrays((data, data_))
                if len(uant) >= 4:
                    clamp = rfn.stack_arrays((clamp, clamp_))
                    tmpl_clamp = rfn.stack_arrays((tmpl_clamp, tmpl_clamp_))
                if len(uant) >= 3:
                    clphs = rfn.stack_arrays((clphs, clphs_))
                    tmpl_clphs = rfn.stack_arrays((tmpl_clphs, tmpl_clphs_))
    out.ufreq = [dataset[i].freq for i in range(ndat)]
    out.data = data
    out.clamp = clamp
    out.clphs = clphs
    out.tmpl_clamp = tmpl_clamp
    out.tmpl_clphs = tmpl_clphs
    out.fit_beam(uvw=dataset[0].uvw)
    out.ploter.clq_obs = (clamp, clphs)
    if type == "sf":
        out.select = dataset[0].select
    elif type == "mf":
        out.freq = nan
        out.select = "mf"
    return out


def set_closure(
    data_u, data_v, data_vis, data_sig, data_ant1, data_ant2, clamp_uvcomb, clphs_uvcomb
):
    """
    Set the closure quantities for the input dataset
        Arguments:
            data_u (1D-array): u-coordinate of the visibility data
            data_v (1D-array): v-coordinate of the visibility data
            data_vis (1D-array): visibility data
            data_sig (1D-array): uncertainty of the visibility data
            data_ant1 (1D-array): antenna 1 of the visibility data
            data_ant2 (1D-array): antenna 2 of the visibility data
            clamp_uvcomb (tuple): closure amplitude uv-combination
            clphs_uvcomb (tuple): closure phase uv-combination
        Returns:
            tuple: (closure_amplitude, closure_phase),
    """
    keys = np.column_stack((data_u, data_v))
    uvvis = dict(zip(map(tuple, keys), data_vis))
    Nant = len(np.unique(np.append(data_ant1, data_ant2)))

    clamp = np.array([])
    clphs = np.array([])
    clamp_sig = np.array([])
    clphs_sig = np.array([])

    if Nant >= 4:
        amp12 = np.abs(np.array(list(map(uvvis.get, clamp_uvcomb[1]))))
        amp34 = np.abs(np.array(list(map(uvvis.get, clamp_uvcomb[2]))))
        amp13 = np.abs(np.array(list(map(uvvis.get, clamp_uvcomb[3]))))
        amp24 = np.abs(np.array(list(map(uvvis.get, clamp_uvcomb[4]))))
        clamp = (amp12 * amp34) / (amp13 * amp24)

    if Nant >= 3:
        phs12 = np.angle(np.array(list(map(uvvis.get, clphs_uvcomb[1]))))
        phs23 = np.angle(np.array(list(map(uvvis.get, clphs_uvcomb[2]))))
        phs31 = np.angle(np.array(list(map(uvvis.get, clphs_uvcomb[3]))).conj())
        clphs = phs12 + phs23 + phs31
        clphs = np.where(clphs > +np.pi, clphs - 2 * np.pi, clphs)
        clphs = np.where(clphs < -np.pi, clphs + 2 * np.pi, clphs)

    if Nant >= 3:
        return clamp, clphs
    else:
        raise ValueError("There are no valid closure quantities")


def set_uvcombination(vdat, tmpl_clamp, tmpl_clphs):
    """
    Set the uv-combination for the input dataset
        Arguments:
            vdat (structured array): visibility data
            tmpl_clamp (structured array): closure amplitude template
            tmpl_clphs (structured array): closure phase template
        Returns:
            tuple: (uv_closure_amplitude, uv_closure_phase),
    """
    Nant = len(np.unique(np.append(vdat["ant_name1"], vdat["ant_name2"])))
    if Nant >= 4:
        clamp_uv12 = tuple(zip(np.ma.getdata(tmpl_clamp["u12"]), np.ma.getdata(tmpl_clamp["v12"])))
        clamp_uv34 = tuple(zip(np.ma.getdata(tmpl_clamp["u34"]), np.ma.getdata(tmpl_clamp["v34"])))
        clamp_uv13 = tuple(zip(np.ma.getdata(tmpl_clamp["u13"]), np.ma.getdata(tmpl_clamp["v13"])))
        clamp_uv24 = tuple(zip(np.ma.getdata(tmpl_clamp["u24"]), np.ma.getdata(tmpl_clamp["v24"])))
        clphs_uv12 = tuple(zip(np.ma.getdata(tmpl_clphs["u12"]), np.ma.getdata(tmpl_clphs["v12"])))
        clphs_uv23 = tuple(zip(np.ma.getdata(tmpl_clphs["u23"]), np.ma.getdata(tmpl_clphs["v23"])))
        clphs_uv31 = tuple(zip(np.ma.getdata(tmpl_clphs["u31"]), np.ma.getdata(tmpl_clphs["v31"])))
        clamp_comb = (np.ma.getdata(tmpl_clamp["freq"]), clamp_uv12, clamp_uv34, clamp_uv13, clamp_uv24)
        clphs_comb = (np.ma.getdata(tmpl_clphs["freq"]), clphs_uv12, clphs_uv23, clphs_uv31)
    if Nant == 3:
        clphs_uv12 = tuple(zip(np.ma.getdata(tmpl_clphs["u12"]), np.ma.getdata(tmpl_clphs["v12"])))
        clphs_uv23 = tuple(zip(np.ma.getdata(tmpl_clphs["u23"]), np.ma.getdata(tmpl_clphs["v23"])))
        clphs_uv31 = tuple(zip(np.ma.getdata(tmpl_clphs["u31"]), np.ma.getdata(tmpl_clphs["v31"])))
        clamp_comb = (np.nan, np.nan, np.nan, np.nan, np.nan)
        clphs_comb = (np.ma.getdata(tmpl_clphs["freq"]), clphs_uv12, clphs_uv23, clphs_uv31)
    return clamp_comb, clphs_comb


def set_boundary(
    nmod=1, spectrum="single", select="I", zblf=1,
    width=5, mrng=10, bnd_l=[-10, +10], bnd_m=[-10, +10], bnd_f=[13.5, 140]
):
    """
    Set the boundary for the a priori
        Arguments:
            nmod (int): the number of models
            spectrum (str): type of spectrum ('single', 'spl', 'cpl', 'ssa')
            select (str): polarization ('I', 'RR', 'LL', 'P', 'Q', 'U', 'V')
            zblf (float): zero-baseline flux
            mrng (float): map range
            bnd_l (list): boundary for the RA-direction
            bnd_m (list): boundary for the DEC-direction
            bnd_f (list): boundary for the frequency
        Returns:
            tuple: set of boundaries
                (in_bnd_S, in_bnd_a, in_bnd_l, in_bnd_m, in_bnd_f, in_bnd_i),
    """

    if select.upper() in ["I", "RR", "LL", "P"]:
        # zblf (float): zero-baseline flux
        in_bnd_S = [[+0.1 * zblf, +1.0 * zblf]]
        in_bnd_a = [[+0.00, +1.00]]
        in_bnd_l = [[+0.00, +mrng / 4]]
        in_bnd_m = [[+0.00, +mrng / 4]]
        in_bnd_f = [bnd_f]
        in_bnd_i = [[-4.00, +0.00]]

        if nmod >= 2:
            nmod_ = nmod - 1

            if spectrum in ["single", "spl"]:
                for i in range(nmod_):
                    in_bnd_S += [[+0.00, +zblf]]
                    in_bnd_l += [bnd_l]
                    in_bnd_m += [bnd_m]
                    in_bnd_f += [bnd_f]
                    in_bnd_i += [[-4.00, +3.00]]
                    in_bnd_a += [[+0.00, +width]]
            elif spectrum in ["cpl", "ssa"]:
                for i in range(nmod_):
                    in_bnd_S += [[+0.00, +zblf]]
                    in_bnd_l += [bnd_l]
                    in_bnd_m += [bnd_m]
                    in_bnd_f += [bnd_f]
                    in_bnd_i += [[-4.00, +0.00]]
                    in_bnd_a += [[+0.00, +width]]
            else:
                raise Exception(f"Given spectrum ({spectrum}) cannot be assigned. (available options are 'single', 'spl', 'cpl', 'ssa')")
        return (in_bnd_S, in_bnd_a, in_bnd_l, in_bnd_m, in_bnd_f, in_bnd_i)

    elif select.upper() in ["Q", "U", "V"]:
        # zblf (list): model flux
        nmod = len(zblf)
        in_bnd_S = [[-zblf[i], +zblf[i]] for i in range(nmod)]
        return in_bnd_S
