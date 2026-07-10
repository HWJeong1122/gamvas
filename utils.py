
import os
import sys
import gc
import copy
import warnings

import pynufft
import numpy as np

import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.time import Time as atime
from astropy import units as au

import gamvas as gv

nan = np.nan
r2m = au.rad.to(au.mas)
d2m = au.deg.to(au.mas)
d2r = au.deg.to(au.rad)


def avg_cgain(cg, axis=0):
    amp_avg = np.nanmean(np.abs(cg), axis=axis)     # scalar average (amp)
    phs_avg = np.angle(np.nanmean(cg, axis=axis))   # vector average (phase)
    return amp_avg * np.exp(1j * phs_avg)


def cal_evpa(
    p=None, q=None, u=None, rms=None, snr=3, mapfov=10, npix=1024,
    evpa_length=1, evpa_width=1
):
    """
    Compute electric vector position angle (EVPA) data
    Args:
        p (float, 2D): Linear polarization image
        q (float, 2D): Stokes P image
        u (float, 2D): Stokes P image
        rms (float): root-mean-square (RMS) value of the polarization image
        snr (float): signal-to-noise ratio for masking
        mapfov (float): field-of-view (FOV) of the map in mas
        npix (int): number of pixels in the map
        evpa_length (float): length of the EVPA in mas
        evpa_width (float): width of the EVPA in mas
    """
    psize = mapfov / npix
    evpal_ength = evpa_length
    evpa_width = evpa_width * psize

    scale = 1 / evpa_length
    evpa_set = dict(
        color="black", pivot="middle", units="xy",
        scale=scale, width=evpa_width,
        headlength=0, headwidth=0, headaxislength=0
    )

    evpa = 0.5 * np.arctan2(u, q)
    evpa_x = np.sin(evpa)
    evpa_y = -np.cos(evpa)

    mask = p >= snr * rms

    evpa = np.where(mask, evpa, np.nan)
    evpa_x = np.where(mask, evpa_x, np.nan)
    evpa_y = np.where(mask, evpa_y, np.nan)

    return (evpa, evpa_x, evpa_y, evpa_set)

def cal_rms(image, roi=False):
    """
    Compute the root-mean-square (RMS) value of the input image
    Args:
        image (float, 2D): input image
    Returns:
        float: statistical RMS value of the input image
    """

    if roi:
        cent = image.size
        unit = int(cent / 10)
        roi_1 = image[0:unit, 0:unit].reshape(-1)
        roi_2 = image[
            0:unit,
            cent - int(unit / 2):cent + int(unit / 2)
        ].reshape(-1)

        roi_3 = image[0:unit, -unit:-1].reshape(-1)
        roi_4 = image[-unit:-1, 0:unit].reshape(-1)
        roi_5 = image[
            -unit:-1,
            cent - int(unit / 2):cent + int(unit / 2)
        ].reshape(-1)

        roi_6 = image[-unit:-1, -unit:-1].reshape(-1)
        rois = np.concatenate((roi_1, roi_2, roi_3, roi_4, roi_5, roi_6))
        rms = np.nanstd((rois**2).mean())
    else:
        rms1 = np.abs(np.percentile(image, 16) - np.percentile(image, 50))
        rms2 = np.abs(np.percentile(image, 84) - np.percentile(image, 50))
        rms = (rms1 + rms2) / 2
        if np.isnan(rms):
            cent = image.size
            unit = int(cent / 10)
            roi_1 = image[0:unit, 0:unit].reshape(-1)
            roi_2 = image[
                0:unit,
                cent - int(unit / 2):cent + int(unit / 2)
            ].reshape(-1)

            roi_3 = image[0:unit, -unit:-1].reshape(-1)
            roi_4 = image[-unit:-1, 0:unit].reshape(-1)
            roi_5 = image[
                -unit:-1,
                cent - int(unit / 2):cent + int(unit / 2)
            ].reshape(-1)

            roi_6 = image[-unit:-1, -unit:-1].reshape(-1)
            rois = np.concatenate((roi_1, roi_2, roi_3, roi_4, roi_5, roi_6))
            rms = np.nanstd((rois**2).mean())
    return rms

def convert_mapfov_unit(in_mapfov, in_mapunit, out_mapunit):
    traceback_mapunit = [
        "uas", "micro-arcsecond",
        "mas", "milli-arcsecond",
        "as", "arcsecond",
        "am", "arcminute", "arcmin"
    ]

    if in_mapunit not in traceback_mapunit:
        raise ValueError(f"Unknown input map unit: {in_mapunit!r}")
    if out_mapunit not in traceback_mapunit:
        raise ValueError(f"Unknown output map unit: {out_mapunit!r}")

    if in_mapunit.lower() in ["uas", "micro-arcsecond"]:
        in_unit = au.uas
    elif in_mapunit.lower() in ["mas", "milli-arcsecond"]:
        in_unit = au.mas
    elif in_mapunit.lower() in ["as", "arcsecond"]:
        in_unit = au.arcsecond
    else:
        in_unit = au.arcmin

    if out_mapunit.lower() in ["uas", "micro-arcsecond"]:
        out_unit = au.uas
    elif out_mapunit.lower() in ["mas", "milli-arcsecond"]:
        out_unit = au.mas
    elif out_mapunit.lower() in ["as", "arcsecond"]:
        out_unit = au.arcsecond
    else:
        out_unit = au.arcmin

    return in_mapfov * in_unit.to(out_unit), out_mapunit

def close_figure(fig):
    """
    Close all figures and free memory.
    """
    plt.close(fig)
    plt.close("all")
    gc.collect()

def dft_fits(path="", file="", uvcov=None, dotype="component"):
    """
    Perform a 2D Fourier Transform on the image.
    Args:
        path (str): Path to the FITS file.
        file (str): Name of the FITS file.
        uvcov (np.ndarray): The UV coverage array.
        dotype (str): Type of data to process, either "image" or "component".

    Returns:
        visibility (np.ndarray): The visibility data.
    """
    fitsfile = os.path.join(path, file)
    if not fitsfile:
        raise ValueError("'path' and 'file' must be specified.!")
    if not os.path.exists(fitsfile):
        raise ValueError(f"FITS file not found: {fitsfile}")
    if uvcov is None:
        raise ValueError("'uvcov' must be provided!")
    if dotype not in ["image", "component"]:
        raise ValueError("'dotype' must be 'image' or 'component'!")

    model = gv.utils.get_fits(path=path, file=file, dotype=dotype)

    u = np.asarray(uvcov["u"]).flatten()
    v = np.asarray(uvcov["v"]).flatten()

    if dotype == "image":
        image = model[0]
        npix = image.shape[0]

        mapfov = model[1] * au.mas.to(au.rad)
        dx = mapfov / npix

        om = np.stack(
            [
                2 * np.pi * dx * u,
                2 * np.pi * dx * v
            ],
            axis=1
        )

        nfft = pynufft.NUFFT()
        nfft.plan(om, (npix, npix), (2 * npix, 2 * npix), (6, 6))

        visibility = nfft.forward(image).flatten()

    else:
        n = len(uvcov)
        nmod = len(model)
        visibility = np.zeros(n, dtype="complex64")
        for i in range(nmod):
            s = model["FLUX"][i]
            a_maj = model["MAJOR AX"][i]
            a_min = model["MINOR AX"][i]
            l = model["DELTAX"][i]
            m = model["DELTAY"][i]
            pa = model["POSANGLE"][i]


            visibility += gv.functions.gvis_elipse(
                (u, v), s, a_maj, a_min, l, m, pa
            )
            # visibility += gv.functions.gvis_el((u, v), s, l, m)

    return visibility

def dft_vis(uvf, plot_resi=False, uvw="u", npix=1024):
    """
    Perform a 2D Fourier Transform on the visibility data.
    Args:
        uvf (UVF): UVF object containing visibility data.
        plot_resi (bool, optional): Whether to plot the residual after
            subtracting the model.
        uvw (str, optional): Which UVW coordinate to use for the
            transform.
        npix (int, optional): Number of pixels in the output image.

    Returns:
        beam (np.ndarray): The reconstructed beam image.
        dirty (np.ndarray): The dirty image.
    """
    mapfov = uvf.mapfov.value
    mapunit = uvf.mapunit
    psize = mapfov / npix * mapunit.to(au.rad)

    _ = uvf.get_data(dotype="vis")
    mask_nan = np.isnan(np.abs(_))

    vis = uvf.get_data(dotype="vis")[~mask_nan].flatten()
    sig = uvf.get_data(dotype="sig")[~mask_nan].flatten()
    u = uvf.get_data(dotype="u")[~mask_nan].flatten()
    v = uvf.get_data(dotype="v")[~mask_nan].flatten()

    if plot_resi:
        vism = uvf.get_data(dotype="vism")[~mask_nan].flatten()
        vis = vis - vism

    if uvw in ["w"]:
        wfn = 1 / sig**2
        weight = "w"
    else:
        wfn = np.ones(vis.shape)
        weight = "u"

    om = np.stack(
        [
            2 * np.pi * v * psize,
            2 * np.pi * u * psize
        ], axis=1
    )

    nfft = pynufft.NUFFT()
    nfft.plan(om, (npix, npix), (2 * npix, 2 * npix), (6, 6))

    beam_raw = nfft.adjoint(wfn * np.ones_like(vis)).real
    dirty_raw = nfft.adjoint(wfn * vis).real

    beam = beam_raw / beam_raw.max()
    dirty = dirty_raw / beam_raw.max()

    return beam, dirty

def fit_beam(uvc, sig=None, uvw="u"):
    """
    Compute beam parameters under Gaussian approximation.
    Given observed (u,v)-coordinates, the beam response near its peak:
        B_0(l,m) \approx
            1 - 2*pi^2/N * (l^2*sum(u^2) - 2*l*m*sum(u*v) + m^2*sum(v^2))
        (TMS, Interferometry and Synthesis, 2017, Thompson, Moran, Swenson)

    This is
    Beam parameters can be determined by fitting the latter one to
    a 2D Gaussian function:
        F(x,y) = exp(-(a*x^2 + b*y^2 + c*x*y)),
    where (a,b,c) are functions of beam parameters, (b_min, b_maj, b_pa).

    Args:
        uvw (str): UV weighting option
            - w: weighting by visibility weight
            - u: unity weighting
    """
    u = uvc["u"]
    v = uvc["v"]

    if uvw == "u":
        wfn = np.ones_like(u)
    elif uvw == "w":
        if sig is None:
            raise ValueError(f"Invalid 'sig' value is presented: {sig}.")
        wfn = 1 / sig**2
    else:
        raise ValueError("Invalid uv-weighting type.\nAvailable: ['u', 'w']")

    k = 2 * np.pi**2 / np.sum(wfn)
    A = k * np.sum(wfn * u**2)
    B = k * np.sum(wfn * v**2)
    C = k * np.sum(wfn * u * v)

    M = np.array([[A, C], [C, B]])
    lam, vec = np.linalg.eigh(M)
    fwhm = 2 * np.sqrt(np.log(2) / lam)
    fwhm_maj, fwhm_min = fwhm[0], fwhm[1]

    ev = vec[:, 0]  # eigenvector of major axis
    theta = np.mod(np.arctan2(ev[0], ev[1]), np.pi)

    return np.array([
        fwhm_maj * au.rad.to(au.mas),
        fwhm_min * au.rad.to(au.mas),
        theta    * au.rad.to(au.deg),
    ])


def get_data(uvf, dotype=None, flatten=False):
    """
    Return a plain (mask-stripped) ndarray for the requested data type.

    Args:
        uvf: uvf object.
        dotype (str): one of the entries in `_GET_DATA_AVAILABLE`.
    """
    if dotype is None or dotype not in _GET_DATA_AVAILABLE:
        raise ValueError(
            f"Invalid data type: {dotype!r}.\n"
            f"availables: {sorted(_GET_DATA_AVAILABLE)}"
        )

    _shape = (
        uvf.data_shape if uvf.data_shape is not None else uvf.r_1.shape
    )

    # tarr columns
    if dotype in _GET_DATA_TARR_FIELDS:
        out = np.ma.getdata(uvf.tarr[_GET_DATA_TARR_FIELDS[dotype]])

    # antenna / baseline names
    elif dotype in _GET_DATA_NAME_FIELDS:
        out = _get_name_field(uvf, dotype, _shape)

    # mount / parallactic angle
    elif dotype in _GET_DATA_PANGLE_FIELDS:
        out = _get_pangle_field(uvf, dotype, _shape)

    # model visibility (no deepcopy, no set_data)
    elif dotype == "vism":
        if "vism" not in uvf.data.dtype.names:
            raise ValueError("Model visibility is not found.")
        out = np.ma.getdata(uvf.data["vism"]).reshape(_shape)

    else:
        # everything below needs a fresh set_data on a deepcopy
        _uvf = _silent_deepcopy(uvf)
        _uvf.set_data(prt=False)

        # derived: amp / phs / snr (and their sig_* variants)
        if dotype in _GET_DATA_DERIVED_FIELDS:
            out = _get_derived_field(_uvf.data, dotype).reshape(_shape)
        else:
            out = np.ma.getdata(_uvf.data[dotype]).reshape(_shape)

    # apply flatten regardless of dotype
    if flatten:
        out = np.asarray(out).flatten()
    return out

def get_fits(path="", file="", unit="mas", dotype="component"):
    """
    Load image-FITS file
    Args:
        path (str): Path to the FITS file.
        file (str): Name of the FITS file.
        unit (str): Coordinate unit.
        dotype (str): Type of data to process, either "image" or "component".

    Returns:
        visibility (np.ndarray): The visibility data.
    """
    fitsfile = os.path.join(path, file)
    if not fitsfile:
        raise ValueError("'path' and 'file' must be specified.!")
    if not os.path.exists(fitsfile):
        raise ValueError(f"File not found: {fitsfile}")
    if dotype not in ["image", "component"]:
        raise ValueError("'dotype' must be 'image' or 'component'!")


    _fits = fits.open(fitsfile)

    _primary = _fits["PRIMARY"]
    _cc = _fits["AIPS CC"]

    dict_unit = {
        "deg": au.deg, "degree": au.deg, "degrees": au.deg,
        "arcsec": au.arcsec, "arcsecond": au.arcsec, "arcseconds": au.arcsec,
        "arcmin": au.arcmin, "arcminute": au.arcmin, "arcminutes": au.arcmin,
        "mas": au.mas, "milliarcsecond": au.mas, "milliarcseconds": au.mas
    }

    if dotype == "image":
        _image = _primary.data

        _dx = abs(_primary.header["CDELT1"])
        _nx = _primary.header["NAXIS1"]

        _unit = _primary.header.get("CUNIT1", "deg").lower()

        _bmin = _primary.header["BMIN"] * dict_unit[_unit].to(dict_unit[unit])
        _bmaj = _primary.header["BMAJ"] * dict_unit[_unit].to(dict_unit[unit])
        _beam_area = (
            (np.pi * _bmaj * _bmin)
            / (4 * np.log(2))
            / (_dx * dict_unit[_unit].to(dict_unit[unit]))**2
        )
        _image_corrected = _image / _beam_area

        _mapfov = (_dx * _nx) * dict_unit[_unit].to(dict_unit[unit])

        _fits.close()
        return _image_corrected[0][0], _mapfov

    else:
        _comp = _cc.data
        _unit_x = _cc.columns["DELTAX"].unit.lower()
        _unit_y = _cc.columns["DELTAY"].unit.lower()
        _unit_maj = _cc.columns["MAJOR AX"].unit.lower()
        _unit_min = _cc.columns["MAJOR AX"].unit.lower()
        _unit_pa = _cc.columns["POSANGLE"].unit.lower()
        _comp["DELTAX"] *= dict_unit[_unit_x].to(dict_unit[unit])
        _comp["DELTAY"] *= dict_unit[_unit_y].to(dict_unit[unit])
        _comp["MAJOR AX"] *= dict_unit[_unit_maj].to(dict_unit[unit])
        _comp["MINOR AX"] *= dict_unit[_unit_min].to(dict_unit[unit])
        _comp["POSANGLE"] *= dict_unit[_unit_pa].to(au.rad)
        _fits.close()
        return _comp

def get_fwght(dotype, vdat, tmpl_clamp, tmpl_clphs):
    """
    Get the weight for the input data based on the number of data points
    Args:
        dotype (list): fit type
        vdat (2D-array): visibility data
        tmpl_clamp (structured array): closure amplitude template
        tmpl_clphs (structured array): closure phase template
    Returns:
        list: weight for the input
    """
    out_wght = np.array([])
    nvis = vdat.shape[0]
    namp = vdat.shape[0]
    nphs = vdat.shape[0]
    ncamp = tmpl_clamp.shape[0]
    ncphs = tmpl_clphs.shape[0]
    wght_tot = 0
    for nt, _type in enumerate(dotype):
        if _type == "vis":
            _wght = 1 / nvis
            wght_tot += _wght
            out_wght = np.append(out_wght, _wght)
        elif _type == "amp":
            _wght = 1 / namp
            wght_tot += _wght
            out_wght = np.append(out_wght, _wght)
        elif _type == "phs":
            _wght = 1 / nphs
            wght_tot += _wght
            out_wght = np.append(out_wght, _wght)
        elif _type == "clamp":
            _wght = 1 / ncamp
            wght_tot += _wght
            out_wght = np.append(out_wght, _wght)
        elif _type == "clphs":
            _wght = 1 / ncphs
            wght_tot += _wght
            out_wght = np.append(out_wght, _wght)
    out_wght = out_wght / wght_tot
    out_wght = np.round(3 * out_wght, 3).tolist()
    return out_wght

def get_xygrid(uvf, npix):
    mapfov = uvf.mapfov.value

    xlist = -np.linspace(
        -mapfov / 2, +mapfov / 2, npix
    )

    ygrid, xgrid = np.meshgrid(xlist, xlist)
    return (xgrid, ygrid)


def make_cntr(image, rms=None, contour_snr=3, scale=np.sqrt(2)):
    """
    Set contour levels for the input image.
    Args:
        image (2D-array): input image
        contour_snr (float): signal-to-noise ratio for starting contour
        scale (float): scaling factor for contour levels
    Returns:
        tuple: (positive_contour_levels, negative_contour_levels),
            - pos_cntr_level (list): positive contour levels
            - neg_cntr_level (list): negative contour levels
    """
    fmax = image.max()
    fmin = image.min()

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


def mkdir(path):
    """
    Make a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)

def model_visibility_append(args, theta, mask):
    u = args[0]
    v = args[1]
    freq_ref = args[2]
    freq = args[3]
    model = args[4]
    spectrum = args[5]
    dshape = args[6]
    dtypes = args[7]

    mask_parallel = mask[0]
    mask_cross = mask[1]

    if mask_parallel:
        vism = np.zeros(dshape, dtype="c8")

        nmod = round(float(theta["nmod"]))
        if model == "gaussian":
            if spectrum == "flat":  # gaussian with flat spectrum
                _fn = gv.functions.gvis
                for i in range(nmod):
                    idx_s = f"{i + 1}_S"
                    idx_a = f"{i + 1}_a"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"

                    has_l = idx_l in dtypes
                    has_m = idx_m in dtypes

                    args = (u, v)

                    _s = theta[idx_s]
                    _a = theta[idx_a]

                    if has_l and has_m:
                        _l = theta[idx_l]
                        _m = theta[idx_m]
                    else:
                        _l = 0
                        _m = 0

                    _theta = [_s, _a, _l, _m]
                    vism += _fn(args, *_theta)

            else:   # gaussian with spectrum model
                for i in range(nmod):
                    idx_s = f"{i + 1}_S"
                    idx_a = f"{i + 1}_a"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"
                    idx_i = f"{i + 1}_alpha"
                    idx_f = f"{i + 1}_freq"
                    idx_t = f"{i + 1}_thick"

                    has_l = idx_l in dtypes
                    has_m = idx_m in dtypes

                    if has_l and has_m:
                        _l = theta[idx_l]
                        _m = theta[idx_m]
                    else:
                        _l = 0
                        _m = 0

                    _s = theta[idx_s]
                    _a = theta[idx_a]
                    _i = theta[idx_i]

                    if spectrum == "spl":
                        args = (freq_ref, freq, u, v)
                        _theta = [_s, _a, _l, _m, _i]
                        _fn = gv.functions.gvis_spl

                    elif spectrum in ["cpl", "ssa"]:
                        if i == 0:
                            _f = theta[idx_f]
                            args = (freq, u, v)
                            _theta = [_s, _a, _l, _m, _i, _f]
                            if spectrum == "cpl":
                                _fn = gv.functions.gvis_cpl
                            else:
                                _fn = gv.functions.gvis_ssa
                        else:
                            mask_thick = round(float(theta[idx_t])) == 0

                            if mask_thick:
                                args = (freq_ref, freq, u, v)
                                _theta = [_s, _a, _l, _m, _i]
                                _fn = gv.functions.gvis_spl
                            else:
                                _f = theta[idx_f]
                                args = (freq, u, v)
                                _theta = [_s, _a, _l, _m, _i, _f]
                                if spectrum == "cpl":
                                    _fn = gv.functions.gvis_cpl
                                elif spectrum == "ssa":
                                    _fn = gv.functions.gvis_ssa
                                elif spectrum == "quad":
                                    raise NotImplementedError(
                                        "To be updated."
                                    )

                    vism += _fn(args, *_theta)

        elif model == "delta":
            if spectrum == "flat":  # delta-function with flat spectrum
                _fn = gv.functions.dvis
                for i in range(nmod):
                    idx_s = f"{i + 1}_S"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"

                    has_l = idx_l in dtypes
                    has_m = idx_m in dtypes

                    args = (u, v)

                    _s = theta[idx_s]

                    if has_l and has_m:
                        _l = theta[idx_l]
                        _m = theta[idx_m]
                    else:
                        _l = 0
                        _m = 0

                    _theta = [_s, _l, _m]
                    vism += _fn(args, *_theta)

            else:   # delta-function with spectrum model
                for i in range(nmod):
                    idx_s = f"{i + 1}_S"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"
                    idx_i = f"{i + 1}_alpha"
                    idx_f = f"{i + 1}_freq"
                    idx_t = f"{i + 1}_thick"

                    has_l = idx_l in dtypes
                    has_m = idx_m in dtypes

                    if has_l and has_m:
                        _l = theta[idx_l]
                        _m = theta[idx_m]
                    else:
                        _l = 0
                        _m = 0

                    _s = theta[idx_s]
                    _i = theta[idx_i]

                    if spectrum == "spl":
                        args = (freq_ref, freq, u, v)
                        _theta = [_s, _l, _m, _i]
                        _fn = gv.functions.dvis_spl

                    elif spectrum in ["cpl", "ssa"]:
                        if i == 0:
                            _f = theta[idx_f]
                            args = (freq, u, v)
                            _theta = [_s, _l, _m, _i, _f]
                            if spectrum == "cpl":
                                _fn = gv.functions.dvis_cpl
                            else:
                                _fn = gv.functions.dvis_ssa
                        else:
                            mask_thick = round(float(theta[idx_t])) == 0

                            if mask_thick:
                                args = (freq_ref, freq, u, v)
                                _theta = [_s, _l, _m, _i]
                                _fn = gv.functions.dvis_spl
                            else:
                                _f = theta[idx_f]
                                args = (freq, u, v)
                                _theta = [_s, _l, _m, _i, _f]
                                if spectrum == "cpl":
                                    _fn = gv.functions.dvis_cpl
                                elif spectrum == "ssa":
                                    _fn = gv.functions.dvis_ssa
                                elif spectrum == "quad":
                                    raise Exception("To be updated.")

                    vism += _fn(args, *_theta)
        return vism

    if mask_cross:
        vism_q = np.zeros(dshape, dtype="c8")
        vism_u = np.zeros(dshape, dtype="c8")

        nmod = round(float(theta["nmod"]))
        if model == "gaussian":
            if spectrum == "flat":  # gaussian with flat spectrum
                _fn = gv.polarization.functions.gvis
                for i in range(nmod):
                    idx_q = f"{i + 1}_Sq"
                    idx_u = f"{i + 1}_Su"
                    idx_a = f"{i + 1}_a"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"

                    args = (u, v)

                    _q = theta[idx_q]
                    _u = theta[idx_u]
                    _a = theta[idx_a]
                    _l = theta[idx_l]
                    _m = theta[idx_m]

                    _theta = [_q, _u, _a, _l, _m]
                    vism = _fn(args, *_theta)

                    vism_q += vism[0]
                    vism_u += vism[1]
        return vism_q, vism_u

def nanaverage(value=None, weight=None, axis=None, returned=False):
    if value is None:
        raise ValueError("value cannot be None")
    if weight is None:
        raise ValueError("weight cannot be None")
    if axis is None:
        raise ValueError("axis cannot be None")

    mask = ~(np.isnan(value) | np.isnan(weight))
    _weight = np.where(mask, weight, 0)

    out_weight = np.sum(_weight, axis=axis)

    out_value = np.where(
        out_weight > 0,
        np.nansum(value * _weight, axis=axis)
        / np.where(out_weight > 0, out_weight, 1),
        np.nan
    )

    if returned:
        return out_value, out_weight
    else:
        return out_value


def print_stats(
    uvf=None, uvcomb=None, k=None, logz=0, dlogz=0, dotype=None,
    reduced=True, pol=False, prt=True
):
    """
    Print the statistics of the input UVF data.
    Args:
        uvf: uv-fits data
        uvcomb: uv-combination data
        k: number of free parameters
        logz: log-likelihood
        dlogz: uncertainty of the log-likelihood
        dotype: fit type
        pol: whether to combine Stokes (P, Q, U) for the vis term
    Returns:
        tuple: (data_terms, chi, aic, bic),
            data_terms (list): data terms
            chi (list): chi-squared values
            aic (list): AIC values
            bic (list): BIC values
    """
    if not isinstance(dotype, list):
        dotype = [dotype]

    nvis = uvf.data.shape[0]
    out_fty, out_chi, out_aic, out_bic = [], [], [], []

    def cal_stats(residual, sig, is_vis=False, reduced=True):
        res2 = np.abs(residual) ** 2
        sig2 = sig ** 2
        log_term = np.log(2 * np.pi * sig2)

        if is_vis:
            chi2 = res2 / sig2 * 0.5
            nll = np.nansum(chi2 + log_term)
        else:
            chi2 = res2 / sig2
            nll = 0.5 * np.nansum(chi2 + log_term)

        if reduced:
            chi2 = np.nansum(chi2) / (len(residual) - k)
        else:
            chi2 = np.nanmean(chi2)
        aic = 2 * nll + 2 * k
        bic = 2 * nll + k * np.log(nvis)
        return chi2, aic, bic

    def emit(label, chi, aic, bic, width=10):
        if not prt:
            return
        print(
            f"# ({label:9s}) | "
            f"R.chi2 : {chi:-{width}.2f} | "
            f"AIC    : {aic:-{width}.2f} | "
            f"BIC    : {bic:-{width}.2f}"
        )

    def store(label, chi, aic, bic):
        out_fty.append(label)
        out_chi.append(chi)
        out_aic.append(aic)
        out_bic.append(bic)

    # vis stats: always computed and stored, regardless of requested dotype
    if pol:
        obs = np.concatenate(
            [uvf.get_data(dotype="vis_p"),
             uvf.get_data(dotype="vis_q"),
             uvf.get_data(dotype="vis_u")], axis=0
        )
        mod = np.concatenate([uvf.vism_p, uvf.vism_q, uvf.vism_u], axis=0)
        sig = np.concatenate(
            [uvf.get_data(dotype="sig_p"),
             uvf.get_data(dotype="sig_q"),
             uvf.get_data(dotype="sig_u")], axis=0
        )
    else:
        obs = uvf.get_data(dotype="vis")
        mod = uvf.get_data(dotype="vism")
        sig = uvf.get_data(dotype="sig")

    vis_chi, vis_aic, vis_bic = cal_stats(
        mod - obs, sig, is_vis=True, reduced=reduced
    )

    vism = uvf.get_data(dotype="vism")
    vis = uvf.get_data(dotype="vis")
    amp = uvf.get_data(dotype="amp")
    sig_amp = uvf.get_data(dotype="sig")
    sig_phs = uvf.get_data(dotype="sig_phs")

    for _type in dotype:
        if _type == "vis":
            emit("vis", vis_chi, vis_aic, vis_bic)
            store("vis", vis_chi, vis_aic, vis_bic)
            continue

        elif _type == "amp":
            residual = np.abs(vism) - amp
            chi, aic, bic = cal_stats(residual, sig_amp, reduced=reduced)

        elif _type == "phs":
            residual = np.angle(vism / vis)
            chi, aic, bic = cal_stats(residual, sig_phs, reduced=reduced)

        elif _type in ["clamp", "clphs"]:
            clq_obs = (copy.deepcopy(uvcomb[0]), copy.deepcopy(uvcomb[1]))
            clq_mod = set_closure(
                uvf, uvf.data["u"], uvf.data["v"], uvf.data["vism"],
                np.zeros(uvf.data["vism"].shape[0]),
                uvf.data["ant1"], uvf.data["ant2"],
                uvcomb[4], uvcomb[5]
            )
            clq_sig = (uvcomb[2], uvcomb[3])

            if _type == "clphs":
                complex_clpm = np.exp(1j * np.ma.getdata(clq_mod[1]))
                complex_clp = np.exp(1j * np.ma.getdata(clq_obs[1]))
                residual = np.angle(complex_clpm * np.conj(complex_clp))
                sig = np.ma.getdata(clq_sig[1])
            else:   # clamp
                log_clam = np.log(np.ma.getdata(clq_mod[0]))
                log_cla = np.log(np.ma.getdata(clq_obs[0]))
                residual = log_clam - log_cla
                sig = np.ma.getdata(clq_sig[0])

            try:
                chi, aic, bic = cal_stats(residual, sig, reduced=reduced)
            except Exception:
                label = (
                    "closure amplitude" if _type == "clamp"
                    else "closure phase"
                )
                warnings.warn(f"Empty {label} data. Replace with NaNs.")
                chi = aic = bic = np.nan

        label = "clamp" if "clamp" in _type else _type
        emit(label, chi, aic, bic)
        store(label, chi, aic, bic)

    if prt:
        print(f"# logz : {logz:-8.2f} +/- {dlogz:-8.2f}")

    return (out_fty, out_chi, out_aic, out_bic)


def rd_theta(path="", file="", id_core=1, emode="mean"):
    _file = f"{path}{file}"

    if not os.path.isfile(_file):
        raise FileNotFoundError(f"File not found: {_file}")

    theta = pd.read_excel(f"{path}{file}", index_col=[0])

    data = theta["value"]
    field = theta["idx"]
    dtype = ["f8" for a in range(len(theta))]

    vtheta = gv.utils.structured_array(data, field, dtype)
    elims = np.stack((theta["lolim"], theta["uplim"]))
    if emode == "min":
        etheta = gv.utils.structured_array(elims.min(axis=0), field, dtype)
    else:
        etheta = gv.utils.structured_array(elims.mean(axis=0), field, dtype)
    nmod = int(np.round(vtheta["nmod"]))
    if id_core != 1:
        core_vra = vtheta[f"{id_core}_l"].copy()
        core_vdec = vtheta[f"{id_core}_m"].copy()
        core_era = etheta[f"{id_core}_l"].copy()
        core_edec = etheta[f"{id_core}_m"].copy()
        for a in range(nmod):
            if a != 0:
                if a + 1 == id_core:
                    vtheta[f"{a + 1}_l"] = -core_vra
                    vtheta[f"{a + 1}_m"] = -core_vdec
                    etheta[f"{a + 1}_l"] = -core_era
                    etheta[f"{a + 1}_m"] = -core_edec
                else:
                    vtheta[f"{a + 1}_l"] = vtheta[f"{a + 1}_l"] - core_vra
                    vtheta[f"{a + 1}_m"] = vtheta[f"{a + 1}_m"] - core_vdec
                    etheta[f"{a + 1}_l"] = etheta[f"{a + 1}_l"] - core_era
                    etheta[f"{a + 1}_m"] = etheta[f"{a + 1}_m"] - core_edec

        core_vs = vtheta[f"{id_core}_S"].copy()
        core_va = vtheta[f"{id_core}_a"].copy()
        core_ss = etheta[f"{id_core}_S"].copy()
        core_sa = etheta[f"{id_core}_a"].copy()

        vtheta[f"{id_core}_S"] = vtheta[f"1_S"].copy()
        vtheta[f"{id_core}_a"] = vtheta[f"1_a"].copy()
        etheta[f"{id_core}_S"] = etheta[f"1_S"].copy()
        etheta[f"{id_core}_a"] = etheta[f"1_a"].copy()

        vtheta["1_S"] = core_vs
        vtheta["1_a"] = core_va
        etheta["1_S"] = core_ss
        etheta["1_a"] = core_sa

    nmod = int(np.round(theta["value"][0]))
    mask_thick = np.array(list(map(lambda x: "thick" in x, theta["idx"])))
    mask_thick = np.round(theta["value"].values[mask_thick]).astype(int)
    k = len(theta) - 2 * (len(mask_thick) - np.sum(mask_thick)) - 1

    return vtheta, etheta, k

def save_cgain(uvf=None, save_path="", save_name=""):
    outfile = os.path.join(save_path, save_name)

    if uvf is None:
        raise ValueError("'uvf' must be specified.!")
    if not outfile:
        raise ValueError("'save_path' and 'save_name' must be specified.!")
    if outfile and not outfile.endswith(".csv"):
        outfile += ".csv"

    time = uvf.get_data(dotype="time", flatten=True)
    freq = uvf.get_data(dotype="frequency", flatten=True)
    ant1 = uvf.get_data(dotype="ant1_name", flatten=True)
    ant2 = uvf.get_data(dotype="ant2_name", flatten=True)
    cg_pol1_ant1 = uvf.cg_pol1_ant1.flatten()
    cg_pol1_ant2 = uvf.cg_pol1_ant2.flatten()
    cg_pol2_ant1 = uvf.cg_pol2_ant1.flatten()
    cg_pol2_ant2 = uvf.cg_pol2_ant2.flatten()

    data = [
        time, freq, ant1, ant2,
        cg_pol1_ant1, cg_pol1_ant2,
        cg_pol2_ant1, cg_pol2_ant2
    ]

    field = [
        "time", "frequency", "ant1", "ant2",
        "gain_pol1_ant1", "gain_pol1_ant2",
        "gain_pol2_ant1", "gain_pol2_ant2"
    ]

    dtype = [
        "f8", "f8", "U32", "U32",
        "c8", "c8",
        "c8", "c8"
    ]

    out = pd.DataFrame(gv.utils.structured_array(
        data=data,
        field=field,
        dtype=dtype
    ))

    out.to_csv(outfile)

def save_imgfits(
    uvf=None, save_path=None, save_name=None, image=None, telescope="VLBI",
    instrument="VLBI", component=None, ra=None, dec=None, freq=None,
    mapfov=None, source=None, date=None, bmaj=None, bmin=None, bpa=None,
    unit=None
):
    """
    Save Gaussian model fitting result as a difmap-style FITS file
    (PRIMARY image + AIPS CC table). Symmetric to 'get_fits'.

    Args:
        fitsfile (str): output path
        image (np.ndarray): 2D image (npix, npix), Jy/beam
        components (dict | structured array): model components with keys
            "FLUX"      [Jy]
            "DELTAX"    [unit]  RA-direction offset from phase center
            "DELTAY"    [unit]  DEC-direction offset
            "MAJOR AX"  [unit]  FWHM major axis  (0 if delta)
            "MINOR AX"  [unit]  FWHM minor axis
            "POSANGLE"  [deg]   position angle (N→E)
            "TYPE OBJ"  [int]   0 = delta, 1 = Gaussian
        ra, dec (float): phase center, J2000 [deg]
        freq (float): observation frequency [Hz]
        mapfov: image FOV. Scalar in `unit` or astropy.Quantity.
        obj_name, date_obs: OBJECT, DATE-OBS header values
        bmaj, bmin (float): synthesized beam FWHM in `unit`
        bpa (float): beam position angle [deg]
        unit (str): spatial unit for axes and CC columns (default "mas")
    """

    def _resolve(value, attr=None, default=None, required=False):
        if value is not None:
            return value

        if uvf is not None and attr is not None:
            v = getattr(uvf, attr, None)
            if v is not None:
                return v

        if default is not None:
            return default

        if required:
            raise ValueError(
                f"'{attr or 'value'}' must be provided directly "
                f"or via uvf.{attr}"
            )

        return None

    ra = _resolve(ra, "ra", default=None, required=False)
    dec = _resolve(dec, "dec", default=None, required=False)
    freq = _resolve(freq, "freq_mean", default=None, required=True)
    mapfov = _resolve(mapfov, "mapfov", default=None, required=True)
    source = _resolve(source, "source", default=None, required=False)
    date = _resolve(date, "date", default=None, required=False)

    bmaj = _resolve(bmaj, "bmaj", default=None, required=False)
    bmin = _resolve(bmin, "bmin", default=None, required=False)
    bpa = _resolve(bpa, "bpa", default=None, required=False)
    unit = _resolve(unit, "mapunit", default=None, required=False)

    if isinstance(component, bool) and not component:
        component = None
    else:
        component = _resolve(
            component, "component", default=None, required=False
        )

    dict_unit = {
        "deg": au.deg, "degree": au.deg, "degrees": au.deg,
        "arcsec": au.arcsec, "arcsecond": au.arcsec, "arcseconds": au.arcsec,
        "arcmin": au.arcmin, "arcminute": au.arcmin, "arcminutes": au.arcmin,
        "mas": au.mas, "milliarcsecond": au.mas, "milliarcseconds": au.mas,
    }

    image = np.asarray(image, dtype="f4")
    assert (
        image.ndim == 2
        and image.shape[0] == image.shape[1]
    ), "image must be square 2D"

    npix = image.shape[0]

    if not hasattr(unit, "to"):
        unit = dict_unit[unit]

    if hasattr(mapfov, "to"):
        mapfov_val = mapfov.to(unit).value
    else:
        mapfov_val = float(mapfov)

    psize = mapfov_val / npix * unit.to(au.deg) # pixel size in 'degree'

    # ==================================================
    # make PRIMARY table: image
    # ==================================================
    img_4d = image.reshape(1, 1, npix, npix)
    hdu_pr = fits.PrimaryHDU(img_4d)
    h = hdu_pr.header

    h["ORIGIN"]   = "GaMVAs-generated"
    h["BSCALE"]   = 1.0
    h["BZERO"]    = 0.0
    h["BUNIT"]    = "JY/BEAM"
    h["EQUINOX"]  = 2000.0
    h["OBJECT"]   = source
    h["TELESCOP"] = telescope
    h["INSTRUME"] = instrument
    h["DATE-OBS"] = date

    if str(unit).upper() == "DEG":
        cunit = "DEGREES"
    elif str(unit).upper() == "RAD":
        cunit = "RADIANS"
    else:
        cunit = str(unit).upper()

    # axis 1 = RA (decrement)
    h["CTYPE1"] = "RA---SIN"
    h["CRPIX1"] = npix / 2 + 1
    h["CRVAL1"] = float(ra)
    h["CDELT1"] = -psize
    h["CROTA1"] = 0.0
    h["CUNIT1"] = "DEGREES"

    # axis 2 = DEC
    h["CTYPE2"] = "DEC--SIN"
    h["CRPIX2"] = npix / 2 + 1
    h["CRVAL2"] = float(dec)
    h["CDELT2"] = +psize
    h["CROTA2"] = 0.0
    h["CUNIT2"] = "DEGREES"

    # axis 3 = FREQ
    h["CTYPE3"] = "FREQ"
    h["CRPIX3"] = 1.0
    h["CRVAL3"] = float(freq)
    h["CDELT3"] = 1.0
    h["CROTA3"] = 0.0

    # axis 4 = STOKES
    h["CTYPE4"] = "STOKES"
    h["CRPIX4"] = 1.0
    h["CRVAL4"] = 1.0
    h["CDELT4"] = 1.0
    h["CROTA4"] = 0.0

    if bmaj is not None:
        h["BMAJ"] = float(bmaj) * unit.to(au.deg)
        h["BMIN"] = float(bmin) * unit.to(au.deg)
        h["BPA"]  = float(bpa)

    # ==================================================
    # make AIPS CC table: model component (if presented)
    # ==================================================
    if component is not None:
        keys  = [
            "FLUX", "DELTAX", "DELTAY", "MAJOR AX", "MINOR AX",
            "POSANGLE", "TYPE OBJ"
        ]
        units = [
            "JY", "DEGREES", "DEGREES", "DEGREES", "DEGREES",
            "DEGREES", ""
        ]

        cols = []
        for k, u in zip(keys, units):
            if u == "DEGREES":
                cf = unit.to(au.deg)
            else:
                cf = 1.0

            arr = np.asarray(component[k] * cf, dtype="f4")
            cols.append(fits.Column(
                name=k, format="1E", unit=u if u else None, array=arr
            ))

        hdu_cc = fits.BinTableHDU.from_columns(cols)
        hdu_cc.name = "AIPS CC"
        hdu_cc.header["EXTNAME"] = "AIPS CC"
        hdu_cc.header["EXTVER"]  = 1

        hdul = fits.HDUList([hdu_pr, hdu_cc])
    else:
        hdul = fits.HDUList([hdu_pr])

    # ==================================================
    # save image FITS file
    # ==================================================
    hdul.writeto(
        f"{save_path}/{save_name}",
        overwrite=True, output_verify="warn"
    )


def set_boundary(
    nmod=1, spectrum="flat", select_pol="I", sblf=1, relmod=True, bnd_a=5,
    bnd_l=[-10, +10], bnd_m=[-10, +10], bnd_f=[13.5, 140], bnd_rm=[-1e6, +1e6]
):
    """
    Set the boundary for the a priori
    Args:
        nmod (int): the number of models
        spectrum (str): type of spectrum ('flat', 'spl', 'cpl', 'ssa')
        select_pol (str): polarization ('I', 'RR', 'LL', 'P')
        sblf (float): shortest-baseline flux
        bnd_l (list): boundary for the RA-direction
        bnd_m (list): boundary for the DEC-direction
        bnd_f (list): boundary for the frequency
    Returns:
        tuple: set of boundaries
            (in_bnd_S, in_bnd_a, in_bnd_l, in_bnd_m, in_bnd_f, in_bnd_i),
    """

    if select_pol.upper() in ["I", "RR", "LL"]:
        in_bnd_S = [[+0.0, +sblf]]
        in_bnd_a = [[+0.00, +bnd_a]]

        if relmod:
            in_bnd_l = [[-1.00, +1.00]]
            in_bnd_m = [[-1.00, +1.00]]
        else:
            in_bnd_l = [bnd_l]
            in_bnd_m = [bnd_m]

        in_bnd_f = [bnd_f]

        if spectrum in ["flat", "spl", "quad"]:
            in_bnd_i = [[-3.00, +3.00]]
        else:
            in_bnd_i = [[-3.00, +0.00]]

        if nmod >= 2:
            nmod_ = nmod - 1

            if spectrum in ["flat", "spl", "quad"]:
                for i in range(nmod_):
                    in_bnd_S += [[+0.00, +sblf]]
                    in_bnd_l += [bnd_l]
                    in_bnd_m += [bnd_m]
                    in_bnd_f += [bnd_f]
                    in_bnd_i += [[-3.00, +3.00]]
                    in_bnd_a += [[+0.00, +bnd_a]]

            elif spectrum in ["cpl", "ssa"]:
                for i in range(nmod_):
                    in_bnd_S += [[+0.00, +sblf]]
                    in_bnd_l += [bnd_l]
                    in_bnd_m += [bnd_m]
                    in_bnd_f += [bnd_f]
                    in_bnd_i += [[-3.00, +0.00]]
                    in_bnd_a += [[+0.00, +bnd_a]]

            else:
                availables = ["flat", "spl", "cpl", "ssa"]
                raise ValueError(
                    f"Invalid spectrum type: {spectrum!r}.\n"
                    f"Availables: {availables}"
                )

        return (in_bnd_S, in_bnd_a, in_bnd_l, in_bnd_m, in_bnd_f, in_bnd_i)

    elif select_pol.upper() == "P":
        in_bnd_Sq = [[-sblf, +sblf] for i in range(nmod)]
        in_bnd_Su = [[-sblf, +sblf] for i in range(nmod)]
        in_bnd_a = [[+0.0, +bnd_a] for i in range(nmod)]
        in_bnd_l = [bnd_l for i in range(nmod)]
        in_bnd_m = [bnd_m for i in range(nmod)]
        in_bnd_i = [[-3.00, +3.00] for i in range(nmod)]
        in_bnd_f = [bnd_f for i in range(nmod)]
        in_bnd_rm = [bnd_rm for i in range(nmod)]
        return (
            in_bnd_Sq, in_bnd_Su, in_bnd_a, in_bnd_l, in_bnd_m, in_bnd_i,
            in_bnd_f, in_bnd_rm
        )

    else:
        availables = ["I", "RR", "LL", "P"]
        raise ValueError(
            f"Invalid polarization type: {select_pol!r}.\n"
            f"Availables: {availables}"
        )


def set_closure(
    uvf, data_u, data_v, data_vis, data_sig, data_ant1, data_ant2,
    clamp_uvcomb, clphs_uvcomb
):

    # strip mask to avoid ComplexWarning from np.abs / .conj on complex MA
    data_vis = np.ma.getdata(data_vis)

    nant = len(np.unique(np.append(data_ant1, data_ant2)))

    uv_coord = np.column_stack((data_u, data_v))
    uv_coord = uv_coord.reshape(len(uv_coord), -1)

    clamp = np.array([])
    clphs = np.array([])
    clamp_sig = np.array([])
    clphs_sig = np.array([])

    mask_clamp = uvf.clamp_check
    mask_clphs = uvf.clphs_check

    if mask_clamp:
        mask_amp12 = clamp_uvcomb[1].reshape(len(clamp_uvcomb[1]), -1)
        mask_amp34 = clamp_uvcomb[2].reshape(len(clamp_uvcomb[2]), -1)
        mask_amp13 = clamp_uvcomb[3].reshape(len(clamp_uvcomb[3]), -1)
        mask_amp24 = clamp_uvcomb[4].reshape(len(clamp_uvcomb[4]), -1)
        mask_amp12 = np.argmax(
            (mask_amp12[:, None, :] == uv_coord[None, :, :]).all(axis=2),
            axis=1
        )

        mask_amp34 = np.argmax(
            (mask_amp34[:, None, :] == uv_coord[None, :, :]).all(axis=2),
            axis=1
        )
        mask_amp13 = np.argmax(
            (mask_amp13[:, None, :] == uv_coord[None, :, :]).all(axis=2),
            axis=1
        )
        mask_amp24 = np.argmax(
            (mask_amp24[:, None, :] == uv_coord[None, :, :]).all(axis=2),
            axis=1
        )

        amp12 = np.abs(data_vis[mask_amp12])
        amp34 = np.abs(data_vis[mask_amp34])
        amp13 = np.abs(data_vis[mask_amp13])
        amp24 = np.abs(data_vis[mask_amp24])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            clamp = (amp12 * amp34) / (amp13 * amp24)
    else:
        clamp = np.array([np.nan])

    if mask_clphs:
        mask_phs12 = clphs_uvcomb[1].reshape(len(clphs_uvcomb[1]), -1)
        mask_phs23 = clphs_uvcomb[2].reshape(len(clphs_uvcomb[2]), -1)
        mask_phs31 = clphs_uvcomb[3].reshape(len(clphs_uvcomb[3]), -1)
        mask_phs12 = np.argmax(
            (mask_phs12[:, None, :] == uv_coord[None, :, :]).all(axis=2),
            axis=1
        )
        mask_phs23 = np.argmax(
            (mask_phs23[:, None, :] == uv_coord[None, :, :]).all(axis=2),
            axis=1
        )
        mask_phs31 = np.argmax(
            (mask_phs31[:, None, :] == uv_coord[None, :, :]).all(axis=2),
            axis=1
        )

        vis12 = data_vis[mask_phs12]
        vis23 = data_vis[mask_phs23]
        vis31 = data_vis[mask_phs31].conj()
        clphs = np.angle(vis12 * vis23 * vis31)
    else:
        clphs = np.array([np.nan])

    if mask_clamp or mask_clphs:
        return clamp, clphs
    else:
        raise Exception("There are no valid closure quantities")

def set_scan(time=None, gaptime=None, scanlen=None):
    """
    Compute scan numbers from a time array, splitting on time
    gaps and length limits.

    Accepts time of shape (N,) or (N, M, 1) where M is IF channel.
    Scan is a property of the time axis only, so the non-time
    axes are collapsed (the time value is assumed constant across
    IF/channel for a given visibility).

    Args:
        time: array of times in seconds.
        gaptime: gap threshold (s). If None, falls back to 60s.
        scanlen: maximum scan length (s). If None,
            falls back to 0 (disables the length cut).

    Returns:
        np.ndarray of scan indices ((2d, 1d)).
    """

    time = np.asarray(time)

    if time.ndim > 1:
        time_1d = time.flatten()
    else:
        time_1d = time

    if np.all(time_1d < 24):
        raise ValueError("given 'time' values seem not to be in seconds.")

    if gaptime is None:
        warnings.warn("'gaptime' is not set. Use 60s as default.")
        gaptime = 60.0

    if scanlen is None:
        warnings.warn("'scanlen' is not set. Disable scan length cut.")
        scanlen = 0.0

    n = len(time_1d)
    order = np.argsort(time_1d)
    t = time_1d[order]

    scannum_1d = np.zeros(n, dtype=int)
    current_num = 0
    scan_start = t[0]

    for i in range(1, n):
        next = False
        if t[i] - t[i - 1] > gaptime:
            next = True

        if scanlen > 0 and (t[i] - scan_start) > scanlen:
            next = True

        if next:
            current_num += 1
            scan_start = t[i]

        scannum_1d[order[i]] = current_num

    scannum_1d[order[0]] = 0

    if time.ndim > 1:
        scannum = scannum_1d.reshape(time.shape)
    else:
        scannum = scannum_1d

    return scannum, scannum_1d

def set_uvcombination(uvf):
    """
    Set the uv-combination for the input dataset
    Args:
        uvf (gamvas.load.open_fits): input dataset
    Returns:
        tuple: (uv_closure_amplitude, uv_closure_phase)
    """
    ant1 = uvf.ant1
    ant2 = uvf.ant2
    tmpl_clamp = uvf.tmpl_clamp
    tmpl_clphs = uvf.tmpl_clphs

    uant = np.unique(np.append(ant1, ant2))

    nant = len(uant)

    mask_clamp = uvf.clamp_check
    mask_clphs = uvf.clphs_check
    if mask_clamp:
        clamp_uv12 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clamp["u12"]),
                np.ma.getdata(tmpl_clamp["v12"])
            ))
        )

        clamp_uv34 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clamp["u34"]),
                np.ma.getdata(tmpl_clamp["v34"])
            ))
        )

        clamp_uv13 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clamp["u13"]),
                np.ma.getdata(tmpl_clamp["v13"])
            ))
        )

        clamp_uv24 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clamp["u24"]),
                np.ma.getdata(tmpl_clamp["v24"])
            ))
        )

        clphs_uv12 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clphs["u12"]),
                np.ma.getdata(tmpl_clphs["v12"])
            ))
        )

        clphs_uv23 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clphs["u23"]),
                np.ma.getdata(tmpl_clphs["v23"])
            ))
        )

        clphs_uv31 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clphs["u31"]),
                np.ma.getdata(tmpl_clphs["v31"])
            ))
        )

        clamp_comb = (
            np.ma.getdata(tmpl_clamp["freq"]),
            clamp_uv12, clamp_uv34, clamp_uv13, clamp_uv24
        )
    else:
        clamp_comb = (
            np.nan,
            np.nan, np.nan, np.nan, np.nan
        )

    if mask_clphs:
        clphs_uv12 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clphs["u12"]),
                np.ma.getdata(tmpl_clphs["v12"])
            ))
        )

        clphs_uv23 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clphs["u23"]),
                np.ma.getdata(tmpl_clphs["v23"])
            ))
        )

        clphs_uv31 = np.array(
            list(zip(
                np.ma.getdata(tmpl_clphs["u31"]),
                np.ma.getdata(tmpl_clphs["v31"])
            ))
        )

        clphs_comb = (
            np.ma.getdata(tmpl_clphs["freq"]),
            clphs_uv12, clphs_uv23, clphs_uv31
        )
    else:
        clphs_comb = (
            np.nan,
            np.nan, np.nan, np.nan, np.nan
        )

    return clamp_comb, clphs_comb

def set_uvf(dataset, dotype="sf", closure=True):
    """
    Set the uv-fits data for the input dataset
    Args:
        dataset (list): input dataset
        dotype (str): type of the dataset
            - 'sf': single-frequency
            - 'mf': multi-frequency
        closure (bool): whether to compute closure phases
    Returns:
        uvf: uv-fits data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = copy.deepcopy(dataset[0])

    ndat = len(dataset)

    ufreq = np.array([])

    clamp = None
    clphs = None
    tmpl_clamp = None
    tmpl_clphs = None

    has_vism = all(d.vism is not None for d in dataset)
    has_cgain = all(d.cg_pol1_ant1 is not None for d in dataset)

    for i in range(ndat):
        dataset[i].check_w0()
        if i == 0:
            mjd = dataset[i].mjd
            time = dataset[i].time
            freq = dataset[i].freq
            ufreq = dataset[i].ufreq
            baseline = dataset[i].baseline
            ant1 = dataset[i].ant1
            ant2 = dataset[i].ant2
            u = dataset[i].u
            v = dataset[i].v
            w = dataset[i].w
            r_1 = dataset[i].r_1
            r_2 = dataset[i].r_2
            r_3 = dataset[i].r_3
            r_4 = dataset[i].r_4
            i_1 = dataset[i].i_1
            i_2 = dataset[i].i_2
            i_3 = dataset[i].i_3
            i_4 = dataset[i].i_4
            w_1 = dataset[i].w_1
            w_2 = dataset[i].w_2
            w_3 = dataset[i].w_3
            w_4 = dataset[i].w_4
            w0_1 = dataset[i].w0_1
            w0_2 = dataset[i].w0_2
            w0_3 = dataset[i].w0_3
            w0_4 = dataset[i].w0_4
            if has_vism:
                vism = dataset[i].vism
            if has_cgain:
                cg_pol1_ant1 = dataset[i].cg_pol1_ant1
                cg_pol1_ant2 = dataset[i].cg_pol1_ant2
                cg_pol2_ant1 = dataset[i].cg_pol2_ant1
                cg_pol2_ant2 = dataset[i].cg_pol2_ant2
        else:
            mjd = np.concatenate([mjd, dataset[i].mjd], axis=0)
            time = np.concatenate([time, dataset[i].time], axis=0)
            freq = np.concatenate([freq, dataset[i].freq], axis=0)
            ufreq = np.concatenate([ufreq, dataset[i].ufreq], axis=0)
            baseline = np.concatenate([baseline, dataset[i].baseline], axis=0)
            ant1 = np.concatenate([ant1, dataset[i].ant1], axis=0)
            ant2 = np.concatenate([ant2, dataset[i].ant2], axis=0)
            u = np.concatenate([u, dataset[i].u], axis=0)
            v = np.concatenate([v, dataset[i].v], axis=0)
            w = np.concatenate([w, dataset[i].w], axis=0)
            r_1 = np.concatenate([r_1, dataset[i].r_1], axis=0)
            r_2 = np.concatenate([r_2, dataset[i].r_2], axis=0)
            r_3 = np.concatenate([r_3, dataset[i].r_3], axis=0)
            r_4 = np.concatenate([r_4, dataset[i].r_4], axis=0)
            i_1 = np.concatenate([i_1, dataset[i].i_1], axis=0)
            i_2 = np.concatenate([i_2, dataset[i].i_2], axis=0)
            i_3 = np.concatenate([i_3, dataset[i].i_3], axis=0)
            i_4 = np.concatenate([i_4, dataset[i].i_4], axis=0)
            w_1 = np.concatenate([w_1, dataset[i].w_1], axis=0)
            w_2 = np.concatenate([w_2, dataset[i].w_2], axis=0)
            w_3 = np.concatenate([w_3, dataset[i].w_3], axis=0)
            w_4 = np.concatenate([w_4, dataset[i].w_4], axis=0)
            w0_1 = np.concatenate([w0_1, dataset[i].w0_1], axis=0)
            w0_2 = np.concatenate([w0_2, dataset[i].w0_2], axis=0)
            w0_3 = np.concatenate([w0_3, dataset[i].w0_3], axis=0)
            w0_4 = np.concatenate([w0_4, dataset[i].w0_4], axis=0)

            # concat model visibility
            if has_vism:
                vism = np.concatenate([vism, dataset[i].vism], axis=0)

            # concat complex gain
            if has_cgain:
                cg_pol1_ant1 = np.concatenate(
                    [cg_pol1_ant1, dataset[i].cg_pol1_ant1], axis=0
                )
                cg_pol1_ant2 = np.concatenate(
                    [cg_pol1_ant2, dataset[i].cg_pol1_ant2], axis=0
                )
                cg_pol2_ant1 = np.concatenate(
                    [cg_pol2_ant1, dataset[i].cg_pol2_ant1], axis=0
                )
                cg_pol2_ant2 = np.concatenate(
                    [cg_pol2_ant2, dataset[i].cg_pol2_ant2], axis=0
                )

        data_ = dataset[i].data
        tarr_ = dataset[i].tarr

        uant = np.unique(
            np.append(data_["ant1"], data_["ant2"])
        )

        # concat data & antenna table
        if i == 0:
            data = data_
            tarr = tarr_
        else:
            data = rfn.stack_arrays((data, data_))
            for nant in range(len(uant)):
                if tarr_["name"][nant] not in tarr["name"]:
                    tarr = rfn.stack_arrays((tarr, tarr_[nant]))

        # concat closure relations (skip bands without valid closures)
        if closure:
            clamp_ = dataset[i].clamp
            clphs_ = dataset[i].clphs
            tmpl_clamp_ = dataset[i].tmpl_clamp
            tmpl_clphs_ = dataset[i].tmpl_clphs

            if dataset[i].clamp_check:
                if clamp is None:
                    clamp = clamp_
                    tmpl_clamp = tmpl_clamp_
                else:
                    clamp = rfn.stack_arrays((clamp, clamp_))
                    tmpl_clamp = rfn.stack_arrays((tmpl_clamp, tmpl_clamp_))
            if dataset[i].clphs_check:
                if clphs is None:
                    clphs = clphs_
                    tmpl_clphs = tmpl_clphs_
                else:
                    clphs = rfn.stack_arrays((clphs, clphs_))
                    tmpl_clphs = rfn.stack_arrays((tmpl_clphs, tmpl_clphs_))
    out.ufreq = np.array(ufreq)
    out.avg_timebin = dataset[0].avg_timebin
    out.data = data
    out.tarr = tarr

    out.freq0 = ufreq.min()
    if dotype == "sf":
        out.freq_mean = ufreq.mean()
    else:
        out.freq_mean = "mf"

    out.mjd = mjd
    out.time = time
    out.freq = freq
    out.ufreq = ufreq
    out.baseline = baseline
    out.ant1 = ant1
    out.ant2 = ant2
    out.u = u
    out.v = v
    out.w = w
    out.r_1 = r_1
    out.r_2 = r_2
    out.r_3 = r_3
    out.r_4 = r_4
    out.i_1 = i_1
    out.i_2 = i_2
    out.i_3 = i_3
    out.i_4 = i_4
    out.w_1 = w_1
    out.w_2 = w_2
    out.w_3 = w_3
    out.w_4 = w_4
    out.w0_1 = w0_1
    out.w0_2 = w0_2
    out.w0_3 = w0_3
    out.w0_4 = w0_4

    out.data_shape = r_1.shape

    if has_vism:
        out.vism = vism
    else:
        out.vism = None

    if has_cgain:
        out.cg_pol1_ant1 = cg_pol1_ant1
        out.cg_pol1_ant2 = cg_pol1_ant2
        out.cg_pol2_ant1 = cg_pol2_ant1
        out.cg_pol2_ant2 = cg_pol2_ant2
    else:
        out.cg_pol1_ant1 = None
        out.cg_pol1_ant2 = None
        out.cg_pol2_ant1 = None
        out.cg_pol2_ant2 = None

    if closure:
        # fall back to the first band if no band has valid closures
        if clamp is None:
            clamp = dataset[0].clamp
            tmpl_clamp = dataset[0].tmpl_clamp
        if clphs is None:
            clphs = dataset[0].clphs
            tmpl_clphs = dataset[0].tmpl_clphs

        out.clamp = clamp
        out.clphs = clphs
        out.tmpl_clamp = tmpl_clamp
        out.tmpl_clphs = tmpl_clphs
        out.clamp_check = any(d.clamp_check for d in dataset)
        out.clphs_check = any(d.clphs_check for d in dataset)

        if out.vism is not None:
            clamp_uvcomb, clphs_uvcomb = gv.utils.set_uvcombination(out)

            out.ploter.clq_mod = gv.utils.set_closure(
                out, out.u.flatten(), out.v.flatten(), out.vism.flatten(),
                np.zeros(vism.size),
                out.ant1.flatten(), out.ant2.flatten(),
                clamp_uvcomb, clphs_uvcomb
            )

    uvc = out.set_uvcov(flatten=True, returned=True)
    sig = out.get_data(dotype="sig")
    uvw = out.uvw
    out.bprms = fit_beam(uvc=uvc, sig=sig, uvw=uvw)

    if closure:
        out.ploter.clq_obs = (
            copy.deepcopy(clamp),
            copy.deepcopy(clphs)
        )

    select_pol = dataset[0].select_pol.split(".")[-1]
    if dotype == "sf":
        out.select_pol = f"sf.{select_pol}"
    elif dotype == "mf":
        out.select_pol = f"mf.{select_pol}"

    return out

def structured_array(data, field, dtype):
    sarr = np.zeros(
        np.asarray(data[0]).shape,
        dtype=list(zip(field, dtype))
    )

    for f, x in zip(field, data):
        sarr[f] = np.asarray(x)

    return sarr

def _silent_deepcopy(uvf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return copy.deepcopy(uvf)


def _get_name_field(uvf, dotype, shape):
    _uvf = _silent_deepcopy(uvf)
    ant1_name = np.array(
        list(map(_uvf.ant_dict_num2name.get, _uvf.ant1.ravel()))
    ).reshape(shape)
    ant2_name = np.array(
        list(map(_uvf.ant_dict_num2name.get, _uvf.ant2.ravel()))
    ).reshape(shape)
    if dotype == "ant1_name":
        return ant1_name
    if dotype == "ant2_name":
        return ant2_name
    return np.char.add(np.char.add(ant1_name, "-"), ant2_name)


def _get_pangle_field(uvf, dotype, shape):
    _uvf = _silent_deepcopy(uvf)
    _uvf.set_data(prt=False)
    data_mount = _uvf.cal_pangle()
    return np.ma.getdata(
        data_mount[_GET_DATA_PANGLE_FIELDS[dotype]]
    ).reshape(shape)


def _get_derived_field(uvf_data=None, dotype=None):
    # parse polarization suffix
    if "_" in dotype:
        suffix = dotype.rsplit("_", 1)[-1]
        pol = f"_{suffix}" if suffix in _GET_DATA_POL_SUFFIXES else ""
    else:
        pol = ""

    # strip masks once before any complex op to avoid ComplexWarning
    _vis = np.ma.getdata(uvf_data[f"vis{pol}"])
    _sig = np.ma.getdata(uvf_data[f"sig{pol}"])

    is_sig = "sig" in dotype
    if "amp" in dotype:
        return _vis if is_sig else np.abs(_vis)
    if "phs" in dotype:
        return _sig / np.abs(_vis) if is_sig else np.angle(_vis)
    return np.abs(_vis) / _sig    # snr

_GET_DATA_TARR_FIELDS = {
    "ant_name":   "name",
    "ant_number": "number",
    "ant_x":      "x",
    "ant_y":      "y",
    "ant_z":      "z",
}

_GET_DATA_NAME_FIELDS = ("baseline_name", "ant1_name", "ant2_name")

_GET_DATA_PANGLE_FIELDS = {
    "ant1_azimuth":   "az1",
    "ant1_elevation": "el1",
    "ant1_pangle":    "pangle1",
    "ant2_azimuth":   "az2",
    "ant2_elevation": "el2",
    "ant2_pangle":    "pangle2",
}

_GET_DATA_DERIVED_FIELDS = frozenset([
    "amp", "amp_rr", "amp_ll", "amp_rl", "amp_lr",
    "sig_amp", "sig_amp_rr", "sig_amp_ll", "sig_amp_rl", "sig_amp_lr",
    "amp_i", "amp_q", "amp_u", "amp_v",
    "sig_amp_i", "sig_amp_q", "sig_amp_u", "sig_amp_v",
    "phs", "phs_rr", "phs_ll", "phs_rl", "phs_lr",
    "sig_phs", "sig_phs_rr", "sig_phs_ll", "sig_phs_rl", "sig_phs_lr",
    "phs_i", "phs_q", "phs_u", "phs_v",
    "sig_phs_i", "sig_phs_q", "sig_phs_u", "sig_phs_v",
    "snr", "snr_rr", "snr_ll", "snr_rl", "snr_lr",
    "snr_i", "snr_q", "snr_u", "snr_v",
])

_GET_DATA_POL_SUFFIXES = frozenset(
    ("rr", "ll", "rl", "lr", "i", "q", "u", "v")
)

_GET_DATA_AVAILABLE = frozenset(
    list(_GET_DATA_TARR_FIELDS)
    + list(_GET_DATA_NAME_FIELDS)
    + list(_GET_DATA_PANGLE_FIELDS)
    + list(_GET_DATA_DERIVED_FIELDS)
    + [
        "mjd", "time", "frequency", "scan",
        "baseline", "ant1", "ant2",
        "u", "v", "w",
        "r_1", "i_1", "w_1", "w0_1", "r_2", "i_2", "w_2", "w0_2",
        "r_3", "i_3", "w_3", "w0_3", "r_4", "i_4", "w_4", "w0_4",
        "vis", "vis_rr", "vis_ll", "vis_rl", "vis_lr",
        "sig", "sig_rr", "sig_ll", "sig_rl", "sig_lr",
        "sig0", "sig0_rr", "sig0_ll", "sig0_rl", "sig0_lr",
        "vis_i", "vis_q", "vis_u", "vis_v", "vis_p",
        "sig_i", "sig_q", "sig_u", "sig_v", "sig_p",
        "sig0_i", "sig0_q", "sig0_u", "sig0_v", "sig0_p",
        "vism",
    ]
)
