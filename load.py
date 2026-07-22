
import gc
import os
import sys
import copy
import itertools as it
import warnings

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.special import comb
from sklearn.cluster import DBSCAN as dbs
from astropy import constants as ac
from astropy import units as au
from astropy.time import Time as atime
from astropy.io import fits
from astropy.modeling import models
from astropy.modeling import fitting
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import Angle
from astropy.coordinates import get_body_barycentric
from astropy.coordinates import get_body
from uncertainties import ufloat
from uncertainties import unumpy as unp

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

import gamvas as gv

cr = (1.00, 0.25, 0.25)
cg = (0.10, 0.90, 0.10)
cb = (0.25, 0.25, 1.00)

nan = np.nan
r2m = au.rad.to(au.mas)
d2m = au.deg.to(au.mas)
d2r = au.deg.to(au.rad)
m2d = au.mas.to(au.deg)
m2r = au.mas.to(au.rad)


if _HAS_NUMBA:
    @njit(cache=True)
    def _selfcal_nll_amp(theta, y, yerr, model, idx1, idx2, nant):
        n = y.shape[0]
        grad = np.zeros(nant)
        nll = 0.0
        for k in range(n):
            a1 = theta[idx1[k]]
            a2 = theta[idx2[k]]
            mg = model[k] * a1 * a2
            r = mg - y[k]
            w = yerr[k] * yerr[k]
            val = (r.real * r.real + r.imag * r.imag) / w
            if not np.isnan(val):
                nll += 0.5 * val
            c1 = (mg / a1 * np.conj(r)).real / w
            c2 = (mg / a2 * np.conj(r)).real / w
            if not np.isnan(c1):
                grad[idx1[k]] += c1
            if not np.isnan(c2):
                grad[idx2[k]] += c2
        return nll, grad

    @njit(cache=True)
    def _selfcal_nll_full(theta, y, yerr, model, idx1, idx2, nant):
        n = y.shape[0]
        grad = np.zeros(2 * nant)
        nll = 0.0
        for k in range(n):
            a1 = theta[idx1[k]]
            a2 = theta[idx2[k]]
            p1 = theta[nant + idx1[k]]
            p2 = theta[nant + idx2[k]]
            g1 = a1 * np.exp(1j * p1)
            g2 = a2 * np.exp(1j * p2)
            mg = model[k] * g1 * np.conj(g2)
            r = mg - y[k]
            w = yerr[k] * yerr[k]
            val = (r.real * r.real + r.imag * r.imag) / w
            if not np.isnan(val):
                nll += 0.5 * val
            cA1 = (mg / a1 * np.conj(r)).real / w
            cA2 = (mg / a2 * np.conj(r)).real / w
            cP1 = (1j * mg * np.conj(r)).real / w
            cP2 = (-1j * mg * np.conj(r)).real / w
            if not np.isnan(cA1):
                grad[idx1[k]] += cA1
            if not np.isnan(cA2):
                grad[idx2[k]] += cA2
            if not np.isnan(cP1):
                grad[nant + idx1[k]] += cP1
            if not np.isnan(cP2):
                grad[nant + idx2[k]] += cP2
        return nll, grad

    @njit(cache=True)
    def _selfcal_nll_phs(theta, y, yerr, model, idx1, idx2, nant):
        n = y.shape[0]
        grad = np.zeros(nant)
        nll = 0.0
        for k in range(n):
            g1 = np.exp(1j * theta[idx1[k]])
            g2 = np.exp(1j * theta[idx2[k]])
            mg = model[k] * g1 * np.conj(g2)
            r = mg - y[k]
            w = yerr[k] * yerr[k]
            val = (r.real * r.real + r.imag * r.imag) / w
            if not np.isnan(val):
                nll += 0.5 * val
            c1 = (1j * mg * np.conj(r)).real / w
            c2 = (-1j * mg * np.conj(r)).real / w
            if not np.isnan(c1):
                grad[idx1[k]] += c1
            if not np.isnan(c2):
                grad[idx2[k]] += c2
        return nll, grad

class open_fits:
    def __init__(self, path=None, file=None, mapfov=10, mapunit="mas"):
        self.path = path
        self.file = file

        self.data = None
        self.data_shape = None
        self.avg_timebin = None

        self.clamp_check = False
        self.clphs_check = False
        self.clamp = None
        self.clphs = None
        self.tmpl_clamp = None
        self.tmpl_clphs = None

        self.intervals = None

        self.vism = None

        self.cg_pol1_ant1 = None   # complex gain of polarization 1 (antenna1)
        self.cg_pol1_ant2 = None   # complex gain of polarization 1 (antenna2)
        self.cg_pol2_ant1 = None   # complex gain of polarization 2 (antenna1)
        self.cg_pol2_ant2 = None   # complex gain of polarization 2 (antenna2)

        self.w0_1 = None
        self.w0_2 = None
        self.w0_3 = None
        self.w0_4 = None
        self.empty_w0 = True

        if mapunit.lower() in ["uas", "micro-arcsecond"]:
            self.mapunit = au.uas
            self.mapfov = mapfov * self.mapunit
        elif mapunit.lower() in ["mas", "milli-arcsecond"]:
            self.mapunit = au.mas
            self.mapfov = mapfov * self.mapunit
        elif mapunit.lower() in ["as", "arcsecond"]:
            self.mapunit = au.arcsecond
            self.mapfov = mapfov * self.mapunit
        elif mapunit.lower() in ["am", "arcminute", "arcmin"]:
            self.mapunit = au.arcmin
            self.mapfov = mapfov * self.mapunit

        self.mask_flag = None

        if self.path is not None and self.file is not None:
            if not os.path.isdir(self.path):
                raise FileNotFoundError(f"Directory {self.path!r} not found!")

            if not os.path.isfile(self.path + self.file):
                raise FileNotFoundError(
                    f"File {self.file!r} not found in {self.path!r}!"
                )
            self.uvf_original = fits.open(self.path + self.file)
        else:
            self.uvf_original = None

        self.ploter = gv.plotting.plotter()
        self.modeling = gv.modeling.modeling()

    def __deepcopy__(self, memo):
        """
        Deep copy the object
        """
        cls = self.__class__.__new__(self.__class__)
        memo[id(self)] = cls
        for k, v in self.__dict__.items():
            if k == "hdu":
                setattr(cls, k, v)
            else:
                setattr(cls, k, copy.deepcopy(v, memo))
        return cls

    def _resolve_ant(self, v):
        """
        Resolve an antenna name or number to its antenna number
        """

        if isinstance(v, int):
            return v
        if isinstance(v, str):
            if v.lstrip('-').isdigit():
                return int(v)
            return self.ant_dict_name2num[v.upper()]
        raise ValueError(f"Cannot resolve antenna: {v!r}")

    def _invalidate_closure(self):
        """Invalidate closure quantities after visibility data change."""

        self.clamp_check = False
        self.clphs_check = False
        self.clamp = None
        self.clphs = None
        self.tmpl_clamp = None
        self.tmpl_clphs = None
        self.ploter.clq_obs = (None, None)
        self.ploter.clq_mod = None

    def average(
        self,
        dotype="time", weighted=True, value=60, mode="day", prt=True
    ):
        """
        Average the visibility data over time or IF channel
        """

        if isinstance(value, str):
            if value.upper() == "SCAN":
                value = self.scanlen
            elif value.upper() == "NONE":
                value = None
            else:
                raise ValueError("Invalid value for average: {}".format(value))

        mjd = self.mjd
        time = self.time
        freq = self.freq
        bsli = self.baseline
        ant1 = self.ant1
        ant2 = self.ant2

        u = self.u
        v = self.v
        w = self.w

        r_1 = self.r_1
        r_2 = self.r_2
        r_3 = self.r_3
        r_4 = self.r_4
        i_1 = self.i_1
        i_2 = self.i_2
        i_3 = self.i_3
        i_4 = self.i_4
        w_1 = self.w_1
        w_2 = self.w_2
        w_3 = self.w_3
        w_4 = self.w_4

        self.check_w0()
        w0_1 = self.w0_1
        w0_2 = self.w0_2
        w0_3 = self.w0_3
        w0_4 = self.w0_4

        mask_cg = (
            self.cg_pol1_ant1 is None
            or self.cg_pol1_ant2 is None
            or self.cg_pol2_ant1 is None
            or self.cg_pol2_ant2 is None
        )

        if not mask_cg:
            cg_pol1_ant1 = self.cg_pol1_ant1
            cg_pol1_ant2 = self.cg_pol1_ant2
            cg_pol2_ant1 = self.cg_pol2_ant1
            cg_pol2_ant2 = self.cg_pol2_ant2

        if dotype == "ifchan":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                # single-channel IF
                if r_1.shape[2] == 1:
                    if weighted:
                        r_1 = gv.utils.nanaverage(r_1, w_1, axis=1).flatten()
                        r_2 = gv.utils.nanaverage(r_2, w_2, axis=1).flatten()
                        r_3 = gv.utils.nanaverage(r_3, w_3, axis=1).flatten()
                        r_4 = gv.utils.nanaverage(r_4, w_4, axis=1).flatten()
                        i_1 = gv.utils.nanaverage(i_1, w_1, axis=1).flatten()
                        i_2 = gv.utils.nanaverage(i_2, w_2, axis=1).flatten()
                        i_3 = gv.utils.nanaverage(i_3, w_3, axis=1).flatten()
                        i_4 = gv.utils.nanaverage(i_4, w_4, axis=1).flatten()
                    else:
                        r_1 = np.nanmean(r_1, axis=1).flatten()
                        r_2 = np.nanmean(r_2, axis=1).flatten()
                        r_3 = np.nanmean(r_3, axis=1).flatten()
                        r_4 = np.nanmean(r_4, axis=1).flatten()
                        i_1 = np.nanmean(i_1, axis=1).flatten()
                        i_2 = np.nanmean(i_2, axis=1).flatten()
                        i_3 = np.nanmean(i_3, axis=1).flatten()
                        i_4 = np.nanmean(i_4, axis=1).flatten()

                    w_1 = np.nansum(w_1.squeeze(-1), axis=1)
                    w_2 = np.nansum(w_2.squeeze(-1), axis=1)
                    w_3 = np.nansum(w_3.squeeze(-1), axis=1)
                    w_4 = np.nansum(w_4.squeeze(-1), axis=1)
                    w0_1 = np.nansum(w0_1.squeeze(-1), axis=1)
                    w0_2 = np.nansum(w0_2.squeeze(-1), axis=1)
                    w0_3 = np.nansum(w0_3.squeeze(-1), axis=1)
                    w0_4 = np.nansum(w0_4.squeeze(-1), axis=1)

                    if not mask_cg:
                        cg_pol1_ant1 = gv.utils.avg_cgain(
                            cg_pol1_ant1.squeeze(-1), axis=1
                        )
                        cg_pol1_ant2 = gv.utils.avg_cgain(
                            cg_pol1_ant2.squeeze(-1), axis=1
                        )
                        cg_pol2_ant1 = gv.utils.avg_cgain(
                            cg_pol2_ant1.squeeze(-1), axis=1
                        )
                        cg_pol2_ant2 = gv.utils.avg_cgain(
                            cg_pol2_ant2.squeeze(-1), axis=1
                        )
                # multi-channel IF
                else:
                    if weighted:
                        r_1 = gv.utils.nanaverage(r_1, w_1, axis=(1, 2))
                        r_2 = gv.utils.nanaverage(r_2, w_2, axis=(1, 2))
                        r_3 = gv.utils.nanaverage(r_3, w_3, axis=(1, 2))
                        r_4 = gv.utils.nanaverage(r_4, w_4, axis=(1, 2))
                        i_1 = gv.utils.nanaverage(i_1, w_1, axis=(1, 2))
                        i_2 = gv.utils.nanaverage(i_2, w_2, axis=(1, 2))
                        i_3 = gv.utils.nanaverage(i_3, w_3, axis=(1, 2))
                        i_4 = gv.utils.nanaverage(i_4, w_4, axis=(1, 2))
                    else:
                        r_1 = np.nanmean(r_1, axis=(1, 2))
                        r_2 = np.nanmean(r_2, axis=(1, 2))
                        r_3 = np.nanmean(r_3, axis=(1, 2))
                        r_4 = np.nanmean(r_4, axis=(1, 2))
                        i_1 = np.nanmean(i_1, axis=(1, 2))
                        i_2 = np.nanmean(i_2, axis=(1, 2))
                        i_3 = np.nanmean(i_3, axis=(1, 2))
                        i_4 = np.nanmean(i_4, axis=(1, 2))
                    w_1 = np.nansum(np.nansum(w_1, axis=2), axis=1)
                    w_2 = np.nansum(np.nansum(w_2, axis=2), axis=1)
                    w_3 = np.nansum(np.nansum(w_3, axis=2), axis=1)
                    w_4 = np.nansum(np.nansum(w_4, axis=2), axis=1)
                    w0_1 = np.nansum(np.nansum(w0_1, axis=2), axis=1)
                    w0_2 = np.nansum(np.nansum(w0_2, axis=2), axis=1)
                    w0_3 = np.nansum(np.nansum(w0_3, axis=2), axis=1)
                    w0_4 = np.nansum(np.nansum(w0_4, axis=2), axis=1)

                    if not mask_cg:
                        cg_pol1_ant1 = gv.utils.avg_cgain(
                            gv.utils.avg_cgain(cg_pol1_ant1, axis=2),
                            axis=1
                        )
                        cg_pol1_ant2 = gv.utils.avg_cgain(
                            gv.utils.avg_cgain(cg_pol1_ant2, axis=2),
                            axis=1
                        )
                        cg_pol2_ant1 = gv.utils.avg_cgain(
                            gv.utils.avg_cgain(cg_pol2_ant1, axis=2),
                            axis=1
                        )
                        cg_pol2_ant2 = gv.utils.avg_cgain(
                            gv.utils.avg_cgain(cg_pol2_ant2, axis=2),
                            axis=1
                        )

            # replace negative weights to NaN
            w_1 = np.where(w_1 <= 0, np.nan, w_1)
            w_2 = np.where(w_2 <= 0, np.nan, w_2)
            w_3 = np.where(w_3 <= 0, np.nan, w_3)
            w_4 = np.where(w_4 <= 0, np.nan, w_4)
            w0_1 = np.where(w0_1 <= 0, np.nan, w0_1)
            w0_2 = np.where(w0_2 <= 0, np.nan, w0_2)
            w0_3 = np.where(w0_3 <= 0, np.nan, w0_3)
            w0_4 = np.where(w0_4 <= 0, np.nan, w0_4)

            # write averaged values
            self.mjd = np.mean(mjd, axis=1)[..., None]
            self.time = np.mean(time, axis=1)[..., None]
            self.freq = np.mean(freq, axis=1)[..., None]
            self.baseline = np.mean(bsli, axis=1)[..., None]
            self.ant1 = np.mean(ant1, axis=1)[..., None]
            self.ant2 = np.mean(ant2, axis=1)[..., None]
            self.u = np.mean(u, axis=1)[..., None]
            self.v = np.mean(v, axis=1)[..., None]
            self.w = np.mean(w, axis=1)[..., None]
            self.r_1 = r_1[..., None, None]
            self.r_2 = r_2[..., None, None]
            self.r_3 = r_3[..., None, None]
            self.r_4 = r_4[..., None, None]
            self.i_1 = i_1[..., None, None]
            self.i_2 = i_2[..., None, None]
            self.i_3 = i_3[..., None, None]
            self.i_4 = i_4[..., None, None]
            self.w_1 = w_1[..., None, None]
            self.w_2 = w_2[..., None, None]
            self.w_3 = w_3[..., None, None]
            self.w_4 = w_4[..., None, None]
            self.w0_1 = w0_1[..., None, None]
            self.w0_2 = w0_2[..., None, None]
            self.w0_3 = w0_3[..., None, None]
            self.w0_4 = w0_4[..., None, None]

            if not mask_cg:
                self.cg_pol1_ant1 = cg_pol1_ant1[..., None, None]
                self.cg_pol1_ant2 = cg_pol1_ant2[..., None, None]
                self.cg_pol2_ant1 = cg_pol2_ant1[..., None, None]
                self.cg_pol2_ant2 = cg_pol2_ant2[..., None, None]

            self.freq0 = float(self.ufreq.mean())
            self.ufreq = np.array([self.freq0])

        elif dotype == "time" and value is not None and not (
            isinstance(value, str) and value.upper() == "NONE"
        ):
            self.avg_timebin = value

            freq = self.freq.astype("f4")
            ufreq = np.unique(freq)

            s1, s2, s3 = time.shape

            time_sec = time * 3600

            mjd_avg = []
            time_avg = []
            freq_avg = []
            bsli_avg = []
            ant1_avg = []
            ant2_avg = []
            u_avg = []
            v_avg = []
            w_avg = []
            r_1_avg = []
            r_2_avg = []
            r_3_avg = []
            r_4_avg = []
            i_1_avg = []
            i_2_avg = []
            i_3_avg = []
            i_4_avg = []
            w_1_avg = []
            w_2_avg = []
            w_3_avg = []
            w_4_avg = []
            w0_1_avg = []
            w0_2_avg = []
            w0_3_avg = []
            w0_4_avg = []
            cg_pol1_ant1_avg = []
            cg_pol1_ant2_avg = []
            cg_pol2_ant1_avg = []
            cg_pol2_ant2_avg = []

            # observation MJD
            mjd_obs = int(atime(self.date).mjd)

            # floating point processing
            time_sec = np.round(time_sec * 10) / 10

            if mode == "day":
                # set time binning block
                tbin_idx = np.floor(time_sec / value).astype(np.int64)
                utbin_idx = np.unique(tbin_idx)

                # unique baseline
                ubsli = np.unique(bsli)

                for nt, _tbin_idx in enumerate(utbin_idx):
                    mask_tbin = (tbin_idx == _tbin_idx)

                    for nb, _bsli in enumerate(ubsli):
                        mask_bsli = (bsli == _bsli)

                        mask = mask_tbin & mask_bsli

                        if mask.sum() == 0:
                            continue

                        _tbin_center = _tbin_idx * value + value / 2
                        _mjd0 = mjd_obs + _tbin_center / 3600.0 / 24.0

                        _freq = freq[mask].reshape(-1, s2, s3)
                        _bsli = bsli[mask].reshape(-1, s2, s3)
                        _ant1 = ant1[mask].reshape(-1, s2, s3)
                        _ant2 = ant2[mask].reshape(-1, s2, s3)
                        _u = u[mask].reshape(-1, s2, s3)
                        _v = v[mask].reshape(-1, s2, s3)
                        _w = w[mask].reshape(-1, s2, s3)
                        _r_1 = r_1[mask].reshape(-1, s2, s3)
                        _r_2 = r_2[mask].reshape(-1, s2, s3)
                        _r_3 = r_3[mask].reshape(-1, s2, s3)
                        _r_4 = r_4[mask].reshape(-1, s2, s3)
                        _i_1 = i_1[mask].reshape(-1, s2, s3)
                        _i_2 = i_2[mask].reshape(-1, s2, s3)
                        _i_3 = i_3[mask].reshape(-1, s2, s3)
                        _i_4 = i_4[mask].reshape(-1, s2, s3)
                        _w_1 = w_1[mask].reshape(-1, s2, s3)
                        _w_2 = w_2[mask].reshape(-1, s2, s3)
                        _w_3 = w_3[mask].reshape(-1, s2, s3)
                        _w_4 = w_4[mask].reshape(-1, s2, s3)
                        _w0_1 = w0_1[mask].reshape(-1, s2, s3)
                        _w0_2 = w0_2[mask].reshape(-1, s2, s3)
                        _w0_3 = w0_3[mask].reshape(-1, s2, s3)
                        _w0_4 = w0_4[mask].reshape(-1, s2, s3)
                        if not mask_cg:
                            _cg_pol1_ant1 = cg_pol1_ant1[mask].reshape(
                                -1, s2, s3
                            )
                            _cg_pol1_ant2 = cg_pol1_ant2[mask].reshape(
                                -1, s2, s3
                            )
                            _cg_pol2_ant1 = cg_pol2_ant1[mask].reshape(
                                -1, s2, s3
                            )
                            _cg_pol2_ant2 = cg_pol2_ant2[mask].reshape(
                                -1, s2, s3
                            )

                        mjd_avg.append(_mjd0)
                        time_avg.append(_tbin_center)
                        freq_avg.append(_freq[0])
                        bsli_avg.append(_bsli[0])
                        ant1_avg.append(_ant1[0])
                        ant2_avg.append(_ant2[0])
                        u_avg.append(np.nanmean(_u, axis=0))
                        v_avg.append(np.nanmean(_v, axis=0))
                        w_avg.append(np.nanmean(_w, axis=0))

                        if weighted:
                            r_1_avg.append(gv.utils.nanaverage(
                                _r_1, _w_1, axis=0
                            ))
                            r_2_avg.append(gv.utils.nanaverage(
                                _r_2, _w_2, axis=0
                            ))
                            r_3_avg.append(gv.utils.nanaverage(
                                _r_3, _w_3, axis=0
                            ))
                            r_4_avg.append(gv.utils.nanaverage(
                                _r_4, _w_4, axis=0
                            ))
                            i_1_avg.append(gv.utils.nanaverage(
                                _i_1, _w_1, axis=0
                            ))
                            i_2_avg.append(gv.utils.nanaverage(
                                _i_2, _w_2, axis=0
                            ))
                            i_3_avg.append(gv.utils.nanaverage(
                                _i_3, _w_3, axis=0
                            ))
                            i_4_avg.append(gv.utils.nanaverage(
                                _i_4, _w_4, axis=0
                            ))
                        else:
                            r_1_avg.append(np.nanmean(_r_1, axis=0))
                            r_2_avg.append(np.nanmean(_r_2, axis=0))
                            r_3_avg.append(np.nanmean(_r_3, axis=0))
                            r_4_avg.append(np.nanmean(_r_4, axis=0))
                            i_1_avg.append(np.nanmean(_i_1, axis=0))
                            i_2_avg.append(np.nanmean(_i_2, axis=0))
                            i_3_avg.append(np.nanmean(_i_3, axis=0))
                            i_4_avg.append(np.nanmean(_i_4, axis=0))
                        w_1_avg.append(np.nansum(_w_1, axis=0))
                        w_2_avg.append(np.nansum(_w_2, axis=0))
                        w_3_avg.append(np.nansum(_w_3, axis=0))
                        w_4_avg.append(np.nansum(_w_4, axis=0))
                        w0_1_avg.append(np.nansum(_w0_1, axis=0))
                        w0_2_avg.append(np.nansum(_w0_2, axis=0))
                        w0_3_avg.append(np.nansum(_w0_3, axis=0))
                        w0_4_avg.append(np.nansum(_w0_4, axis=0))

                        if not mask_cg:
                            cg_pol1_ant1_avg.append(
                                gv.utils.avg_cgain(_cg_pol1_ant1, axis=0)
                            )
                            cg_pol1_ant2_avg.append(
                                gv.utils.avg_cgain(_cg_pol1_ant2, axis=0)
                            )
                            cg_pol2_ant1_avg.append(
                                gv.utils.avg_cgain(_cg_pol2_ant1, axis=0)
                            )
                            cg_pol2_ant2_avg.append(
                                gv.utils.avg_cgain(_cg_pol2_ant2, axis=0)
                            )
            elif mode == "scan":
                # make scan information
                self.set_scan(
                    time=time_sec, gaptime=self.gaptime, scanlen=self.scanlen
                )

                scannum = self.scannum
                uscan = np.unique(scannum)

                # use each scan start as the binning reference
                scan_start = np.zeros_like(time_sec, dtype="f8")
                for _scan in uscan:
                    _mask = (scannum == _scan)
                    scan_start[_mask] = time_sec[_mask].min()

                # set time binning block relative to each scan start
                tbin_idx = np.floor(
                    (time_sec - scan_start) / value
                ).astype(np.int64)

                # unique (scan, within-scan time bin) blocks
                ublock = np.unique(
                    np.stack([scannum.ravel(), tbin_idx.ravel()], axis=1),
                    axis=0
                )

                # unique baseline
                ubsli = np.unique(bsli)

                for nt, (_scan, _tbin_idx) in enumerate(ublock):
                    mask_tbin = (scannum == _scan) & (tbin_idx == _tbin_idx)

                    _scan_start = scan_start[mask_tbin][0]
                    _tbin_center = _scan_start + _tbin_idx * value + value / 2
                    _mjd0 = mjd_obs + _tbin_center / 3600.0 / 24.0

                    for nb, _bsli in enumerate(ubsli):
                        mask_bsli = (bsli == _bsli)

                        mask = mask_tbin & mask_bsli

                        if mask.sum() == 0:
                            continue

                        _freq = freq[mask].reshape(-1, s2, s3)
                        _bsli = bsli[mask].reshape(-1, s2, s3)
                        _ant1 = ant1[mask].reshape(-1, s2, s3)
                        _ant2 = ant2[mask].reshape(-1, s2, s3)
                        _u = u[mask].reshape(-1, s2, s3)
                        _v = v[mask].reshape(-1, s2, s3)
                        _w = w[mask].reshape(-1, s2, s3)
                        _r_1 = r_1[mask].reshape(-1, s2, s3)
                        _r_2 = r_2[mask].reshape(-1, s2, s3)
                        _r_3 = r_3[mask].reshape(-1, s2, s3)
                        _r_4 = r_4[mask].reshape(-1, s2, s3)
                        _i_1 = i_1[mask].reshape(-1, s2, s3)
                        _i_2 = i_2[mask].reshape(-1, s2, s3)
                        _i_3 = i_3[mask].reshape(-1, s2, s3)
                        _i_4 = i_4[mask].reshape(-1, s2, s3)
                        _w_1 = w_1[mask].reshape(-1, s2, s3)
                        _w_2 = w_2[mask].reshape(-1, s2, s3)
                        _w_3 = w_3[mask].reshape(-1, s2, s3)
                        _w_4 = w_4[mask].reshape(-1, s2, s3)
                        _w0_1 = w0_1[mask].reshape(-1, s2, s3)
                        _w0_2 = w0_2[mask].reshape(-1, s2, s3)
                        _w0_3 = w0_3[mask].reshape(-1, s2, s3)
                        _w0_4 = w0_4[mask].reshape(-1, s2, s3)
                        if not mask_cg:
                            _cg_pol1_ant1 = cg_pol1_ant1[mask].reshape(
                                -1, s2, s3
                            )
                            _cg_pol1_ant2 = cg_pol1_ant2[mask].reshape(
                                -1, s2, s3
                            )
                            _cg_pol2_ant1 = cg_pol2_ant1[mask].reshape(
                                -1, s2, s3
                            )
                            _cg_pol2_ant2 = cg_pol2_ant2[mask].reshape(
                                -1, s2, s3
                            )

                        mjd_avg.append(_mjd0)
                        time_avg.append(_tbin_center)
                        freq_avg.append(_freq[0])
                        bsli_avg.append(_bsli[0])
                        ant1_avg.append(_ant1[0])
                        ant2_avg.append(_ant2[0])
                        u_avg.append(np.nanmean(_u, axis=0))
                        v_avg.append(np.nanmean(_v, axis=0))
                        w_avg.append(np.nanmean(_w, axis=0))

                        if weighted:
                            r_1_avg.append(gv.utils.nanaverage(
                                _r_1, _w_1, axis=0
                            ))
                            r_2_avg.append(gv.utils.nanaverage(
                                _r_2, _w_2, axis=0
                            ))
                            r_3_avg.append(gv.utils.nanaverage(
                                _r_3, _w_3, axis=0
                            ))
                            r_4_avg.append(gv.utils.nanaverage(
                                _r_4, _w_4, axis=0
                            ))
                            i_1_avg.append(gv.utils.nanaverage(
                                _i_1, _w_1, axis=0
                            ))
                            i_2_avg.append(gv.utils.nanaverage(
                                _i_2, _w_2, axis=0
                            ))
                            i_3_avg.append(gv.utils.nanaverage(
                                _i_3, _w_3, axis=0
                            ))
                            i_4_avg.append(gv.utils.nanaverage(
                                _i_4, _w_4, axis=0
                            ))
                        else:
                            r_1_avg.append(np.nanmean(_r_1, axis=0))
                            r_2_avg.append(np.nanmean(_r_2, axis=0))
                            r_3_avg.append(np.nanmean(_r_3, axis=0))
                            r_4_avg.append(np.nanmean(_r_4, axis=0))
                            i_1_avg.append(np.nanmean(_i_1, axis=0))
                            i_2_avg.append(np.nanmean(_i_2, axis=0))
                            i_3_avg.append(np.nanmean(_i_3, axis=0))
                            i_4_avg.append(np.nanmean(_i_4, axis=0))
                        w_1_avg.append(np.nansum(_w_1, axis=0))
                        w_2_avg.append(np.nansum(_w_2, axis=0))
                        w_3_avg.append(np.nansum(_w_3, axis=0))
                        w_4_avg.append(np.nansum(_w_4, axis=0))
                        w0_1_avg.append(np.nansum(_w0_1, axis=0))
                        w0_2_avg.append(np.nansum(_w0_2, axis=0))
                        w0_3_avg.append(np.nansum(_w0_3, axis=0))
                        w0_4_avg.append(np.nansum(_w0_4, axis=0))

                        if not mask_cg:
                            cg_pol1_ant1_avg.append(
                                gv.utils.avg_cgain(_cg_pol1_ant1, axis=0)
                            )
                            cg_pol1_ant2_avg.append(
                                gv.utils.avg_cgain(_cg_pol1_ant2, axis=0)
                            )
                            cg_pol2_ant1_avg.append(
                                gv.utils.avg_cgain(_cg_pol2_ant1, axis=0)
                            )
                            cg_pol2_ant2_avg.append(
                                gv.utils.avg_cgain(_cg_pol2_ant2, axis=0)
                            )
            else:
                raise ValueError(f"Invalid mode for time-average: {mode!r}")


            new_shape = np.array(r_1_avg).shape
            # self.mjd = np.array(mjd_avg)
            self.mjd = np.broadcast_to(
                np.array(mjd_avg)[:, None, None], new_shape
            ).copy()
            self.time = np.broadcast_to(
                np.array(time_avg)[:, None, None] / 3600.0, new_shape
            ).copy()
            self.freq = np.array(freq_avg)
            self.baseline = np.array(bsli_avg)
            self.ant1 = np.array(ant1_avg)
            self.ant2 = np.array(ant2_avg)
            self.u = np.array(u_avg)
            self.v = np.array(v_avg)
            self.w = np.array(w_avg)
            self.r_1 = np.array(r_1_avg)
            self.r_2 = np.array(r_2_avg)
            self.r_3 = np.array(r_3_avg)
            self.r_4 = np.array(r_4_avg)
            self.i_1 = np.array(i_1_avg)
            self.i_2 = np.array(i_2_avg)
            self.i_3 = np.array(i_3_avg)
            self.i_4 = np.array(i_4_avg)
            self.w_1 = np.array(w_1_avg)
            self.w_2 = np.array(w_2_avg)
            self.w_3 = np.array(w_3_avg)
            self.w_4 = np.array(w_4_avg)
            self.w0_1 = np.array(w0_1_avg)
            self.w0_2 = np.array(w0_2_avg)
            self.w0_3 = np.array(w0_3_avg)
            self.w0_4 = np.array(w0_4_avg)

            if not mask_cg:
                self.cg_pol1_ant1 = np.array(cg_pol1_ant1_avg)
                self.cg_pol1_ant2 = np.array(cg_pol1_ant2_avg)
                self.cg_pol2_ant1 = np.array(cg_pol2_ant1_avg)
                self.cg_pol2_ant2 = np.array(cg_pol2_ant2_avg)

        self.set_data(prt=False)

    def cal_pangle(self):
        """
        Compute the parallactic angle
        """

        ndata = self.r_1.size

        tarr = self.tarr.copy()
        x = tarr["x"]
        y = tarr["y"]
        z = tarr["z"]
        r = (x**2 + y**2 + z**2)**0.5

        ant_geocent = EarthLocation(
            x=x * au.m,
            y=y * au.m,
            z=z * au.m
        )
        lon, lat, h = ant_geocent.to_geodetic()

        lat = lat.degree
        lon = lon.degree
        height = h.value

        mask_llh = [
            field not in tarr.dtype.names
            for field in ["lat", "lon", "height"]
        ]

        if np.all(mask_llh):
            dtype_ = np.dtype({
                "names": ["lat", "lon", "height"],
                "formats": ["f8", "f8", "f8"]
            })
            dtype_ = np.dtype((tarr.dtype.descr + dtype_.descr))
            tarr_ = []
            for i in range(tarr.size):
                tarr_.append(
                    np.array(
                        tuple(
                            [tarr[i][j] for j in range(len(tarr[i]))]
                            + [lat[i], lon[i], height[i]]
                        ),
                        dtype=dtype_
                    )
                )

            tarr = np.array(tarr_)
            self.tarr = tarr

        _shape = self.mjd.shape
        obstime = atime(self.mjd.flatten(), format="mjd").iso

        src_coord = SkyCoord(
            ra=self.ra * au.deg,
            dec=self.dec * au.deg
        )

        ants = self.tarr["name"]

        ant1 = self.get_data(dotype="ant1_name").flatten()
        ant2 = self.get_data(dotype="ant2_name").flatten()
        lat1 = np.zeros(ndata)
        lon1 = np.zeros(ndata)
        lat2 = np.zeros(ndata)
        lon2 = np.zeros(ndata)
        az1 = np.zeros(ndata)
        el1 = np.zeros(ndata)
        az2 = np.zeros(ndata)
        el2 = np.zeros(ndata)

        for tidx in range(len(ants)):
            _ant = tarr["name"][tidx]
            _lat = tarr["lat"][tidx]
            _lon = tarr["lon"][tidx]
            _h = tarr["height"][tidx]

            ant_position = EarthLocation(
                lat=_lat * au.deg,
                lon=_lon * au.deg,
                height=_h * au.m
            )

            src_azel = src_coord.transform_to(
                AltAz(obstime=obstime, location=ant_position)
            )

            az1 = np.where(ant1 == _ant, src_azel.az.degree, az1)
            el1 = np.where(ant1 == _ant, src_azel.alt.degree, el1)
            az2 = np.where(ant2 == _ant, src_azel.az.degree, az2)
            el2 = np.where(ant2 == _ant, src_azel.alt.degree, el2)
            lat1 = np.where(ant1 == _ant, _lat, lat1)
            lon1 = np.where(ant1 == _ant, _lon, lon1)
            lat2 = np.where(ant2 == _ant, _lat, lat2)
            lon2 = np.where(ant2 == _ant, _lon, lon2)

        ra = np.pi / 180 * self.ra
        dec = np.pi / 180 * self.dec
        lat1_rad = np.pi / 180 * lat1
        lat2_rad = np.pi / 180 * lat2
        az1_rad = np.pi / 180 * az1
        az2_rad = np.pi / 180 * az2
        el1_rad = np.pi / 180 * el1
        el2_rad = np.pi / 180 * el2
        sin_pa1 = np.cos(lat1_rad) / np.cos(dec) * np.sin(az1_rad)

        cos_pa1 = (
            (np.sin(lat1_rad) - np.sin(dec) * np.sin(el1_rad))
            / (np.cos(dec) * np.cos(el1_rad))
        )

        sin_pa2 = (
            np.cos(lat2_rad) / np.cos(dec) * np.sin(az2_rad)
        )

        cos_pa2 = (
            (np.sin(lat2_rad) - np.sin(dec) * np.sin(el2_rad))
            / (np.cos(dec) * np.cos(el2_rad))
        )

        p_angle1 = -np.angle(cos_pa1 + 1j * sin_pa1)
        p_angle2 = -np.angle(cos_pa2 + 1j * sin_pa2)

        data_mount = gv.utils.structured_array(
            data=[self.mjd.flatten(), az1, az2, el1, el2, p_angle1, p_angle2],
            field=["time", "az1", "az2", "el1", "el2", "pangle1", "pangle2"],
            dtype=["f8", "f8", "f8", "f8", "f8", "f8", "f8"]
        )

        return data_mount

    def check_dims(self):
        """
        Check the dimensions of the visibility data and add a channel
        dimension if necessary
        """

        if self.r_1.ndim == 2:
            self.r_1 = self.r_1[..., None]
            self.i_1 = self.i_1[..., None]
            self.w_1 = self.w_1[..., None]
            self.r_2 = self.r_2[..., None]
            self.i_2 = self.i_2[..., None]
            self.w_2 = self.w_2[..., None]
            self.r_3 = self.r_3[..., None]
            self.i_3 = self.i_3[..., None]
            self.w_3 = self.w_3[..., None]
            self.r_4 = self.r_4[..., None]
            self.i_4 = self.i_4[..., None]
            self.w_4 = self.w_4[..., None]

    def check_w0(self):
        """
        Check if w0 is None and set it to w if so
        """

        mask_w0 = (
            self.w0_1 is None
            or self.w0_2 is None
            or self.w0_3 is None
            or self.w0_4 is None
        )

        if mask_w0:
            self.w0_1 = self.w_1.copy()
            self.w0_2 = self.w_2.copy()
            self.w0_3 = self.w_3.copy()
            self.w0_4 = self.w_4.copy()
        self.empty_w0 = False

    def debias_amplitude(self):
        """
        Debias the amplitude of the visibility data
        """

        r_1 = self.get_data("r_1")
        r_2 = self.get_data("r_2")
        r_3 = self.get_data("r_3")
        r_4 = self.get_data("r_4")
        i_1 = self.get_data("i_1")
        i_2 = self.get_data("i_2")
        i_3 = self.get_data("i_3")
        i_4 = self.get_data("i_4")
        w_1 = self.get_data("w_1")
        w_2 = self.get_data("w_2")
        w_3 = self.get_data("w_3")
        w_4 = self.get_data("w_4")

        v_1 = r_1 + 1j * i_1
        v_2 = r_2 + 1j * i_2
        v_3 = r_3 + 1j * i_3
        v_4 = r_4 + 1j * i_4

        with np.errstate(divide="ignore", invalid="ignore"):
            s_1 = np.sqrt(1 / w_1)
            s_2 = np.sqrt(1 / w_2)
            s_3 = np.sqrt(1 / w_3)
            s_4 = np.sqrt(1 / w_4)

        amp_1 = np.abs(v_1)
        amp_2 = np.abs(v_2)
        amp_3 = np.abs(v_3)
        amp_4 = np.abs(v_4)

        phs_1 = np.angle(v_1)
        phs_2 = np.angle(v_2)
        phs_3 = np.angle(v_3)
        phs_4 = np.angle(v_4)

        valid_1 = amp_1 > s_1
        valid_2 = amp_2 > s_2
        valid_3 = amp_3 > s_3
        valid_4 = amp_4 > s_4

        amp_1 = np.sqrt(np.maximum(np.abs(v_1)**2 - s_1**2, 0.0))
        amp_2 = np.sqrt(np.maximum(np.abs(v_2)**2 - s_2**2, 0.0))
        amp_3 = np.sqrt(np.maximum(np.abs(v_3)**2 - s_3**2, 0.0))
        amp_4 = np.sqrt(np.maximum(np.abs(v_4)**2 - s_4**2, 0.0))

        v_1_debias = amp_1 * np.exp(1j * phs_1)
        v_2_debias = amp_2 * np.exp(1j * phs_2)
        v_3_debias = amp_3 * np.exp(1j * phs_3)
        v_4_debias = amp_4 * np.exp(1j * phs_4)

        self.r_1 = np.where(valid_1, v_1_debias.real, np.nan)
        self.r_2 = np.where(valid_2, v_2_debias.real, np.nan)
        self.r_3 = np.where(valid_3, v_3_debias.real, np.nan)
        self.r_4 = np.where(valid_4, v_4_debias.real, np.nan)

        self.i_1 = np.where(valid_1, v_1_debias.imag, np.nan)
        self.i_2 = np.where(valid_2, v_2_debias.imag, np.nan)
        self.i_3 = np.where(valid_3, v_3_debias.imag, np.nan)
        self.i_4 = np.where(valid_4, v_4_debias.imag, np.nan)

        self.w_1 = np.where(valid_1, w_1, -np.abs(w_1))
        self.w_2 = np.where(valid_2, w_2, -np.abs(w_2))
        self.w_3 = np.where(valid_3, w_3, -np.abs(w_3))
        self.w_4 = np.where(valid_4, w_4, -np.abs(w_4))

        self.set_data(prt=False)

    def flag_data(
        self,
        dotype=None, value=None, unit="m", timerange="all", prt=True
    ):
        """
        Flag data based on the given type, value, unit, and timerange

        Args:
            dotype (str): Type of data to flag
            value (float): Flagging criterion
            unit (str, dotype='uvr' only): Unit of uv-radius
            timerange (str): Timerange to flag
            prt (bool): Print flagging summary
        """

        _dshape = self.data_shape

        obs = self.flat_data()

        time = obs["time"]
        freq = obs["frequency"]
        ant1 = obs["ant1"]
        ant2 = obs["ant2"]
        u = obs["u"]
        v = obs["v"]

        ndat = len(time)
        uvdist = np.sqrt(u**2 + v**2)

        availables = [
            "time", "sig", "sigma", "snr", "ant", "antenna", "nant",
            "bsli", "baseline", "uvr", "uvradius"
        ]

        if dotype is None or dotype not in availables:
            raise ValueError(
                f"Given type {dotype!r} cannot be assigned. "
                f"Please give appropriate type!\n"
                f"Available: {availables}"
            )

        if value is None:
            raise ValueError("Please give appropriate value for flagging!")
        else:
            if dotype in ["time"] and not isinstance(value, (list, tuple)):
                raise ValueError(
                    "Please give flagging value as a list of floats! "
                    "([float1, float2])"
                )

            elif (
                dotype in ["sigma", "snr", "nant"]
                and not isinstance(value, (int, float))
            ):
                raise ValueError(
                    "Please give flagging value as a float or an integer!"
                )

            elif (
                dotype in ["bsli", "baseline"]
                and not isinstance(value, (list, tuple))
            ):
                raise ValueError(
                    "Please give flagging value as a list/tuple "
                    "of strings or ints!"
                )

            elif (
                dotype in ["ant", "antenna"]
                and not isinstance(value, (str, int, list, tuple))
            ):
                raise ValueError(
                    "Please give flagging value as a string, int,"
                    " or a list/tuple of strings or ints!"
                )

            elif (
                dotype in ["uvr", "uvradius"]
                and not isinstance(value, (list, tuple))
            ):
                raise ValueError(
                    "Please give flagging value as a list of floats! "
                    "([float1, float2])"
                )

        if dotype in ["time"]:
            mask = (value[0] <= time) & (time <= value[1])

        elif dotype in ["sig", "sigma"]:
            sig = self.get_data(dotype="sig").flatten()
            mask = sig >= value

        elif dotype in ["snr"]:
            vis = self.get_data(dotype="vis").flatten()
            sig = self.get_data(dotype="sig").flatten()
            snr = np.abs(vis) / sig
            mask = snr <= value

        elif dotype in ["bsli", "baseline"]:
            ant_num = []
            for ant in value:
                ant_num.append(self._resolve_ant(ant))

            mask_bsli1 = (ant1 == ant_num[0]) & (ant2 == ant_num[1])
            mask_bsli2 = (ant1 == ant_num[1]) & (ant2 == ant_num[0])
            mask = mask_bsli1 | mask_bsli2

        elif dotype in ["ant", "antenna"]:
            if isinstance(value, (str, int)):
                value = [value]

            mask = np.zeros(ndat, dtype=bool)
            for ant in value:
                ant_num = self._resolve_ant(ant)
                mask_ = (ant1 == ant_num) | (ant2 == ant_num)
                if np.sum(mask_) == 0 and prt:
                    print(
                        f"No visibility for antenna {ant!r}."
                        " Skip flagging on this station."
                    )
                else:
                    mask = mask | mask_

        elif dotype in ["nant"]:
            mask = np.zeros(ndat, dtype=bool)
            vis = self.get_data(dotype="vis").flatten()
            sig = self.get_data(dotype="sig").flatten()
            live = (
                np.isfinite(vis.real)
                & np.isfinite(vis.imag)
                & np.isfinite(sig)
                & (sig > 0)
            )

            time_freq = np.unique(
                np.stack([time, freq], axis=1), axis=0
            )
            min_baseline = value * (value - 1) / 2
            for time_, freq_ in time_freq:
                mask_group = (time == time_) & (freq == freq_)
                mask_live = mask_group & live

                pairs_live = np.stack(
                    [ant1[mask_live], ant2[mask_live]], axis=1
                )
                if pairs_live.size:
                    pairs_live = np.unique(
                        np.sort(pairs_live, axis=1), axis=0
                    )
                    ants_live = np.unique(pairs_live)
                else:
                    pairs_live = np.empty((0, 2), dtype=int)
                    ants_live = np.array([], dtype=int)

                if (
                    len(ants_live) < value
                    or len(pairs_live) < min_baseline
                ):
                    mask[mask_group] = True

        elif dotype in ["uvr", "uvradius"]:
            if unit.lower() in ["k", "kilo"]:
                factor = 1e3
            elif unit.lower() in ["m", "mega"]:
                factor = 1e6
            elif unit.lower() in ["g", "giga"]:
                factor = 1e9
            else:
                raise ValueError(f"Invalid unit: {unit!r}")

            uvdist_ = uvdist / factor
            mask = ((value[0] <= uvdist_) & (uvdist_ <= value[1]))

        if dotype != "time":
            if timerange == "all":
                mask_timerange = np.ones_like(time, dtype=bool)
            else:
                mask_timerange = (
                    (timerange[0] <= time)
                    & (time <= timerange[1])
                )

            mask = mask_timerange & mask

        if prt:
            if dotype in ["uvr", "uvradius"]:
                print(
                    f"# Flag {np.sum(mask)}/{ndat} visibilities"
                    f" (type={dotype!r}, value={value}, unit={unit!r})"
                )

            else:
                if timerange == "all":
                    print(
                        f"# Flag {np.sum(mask)}/{ndat} visibilities"
                        f" (type={dotype!r}, value={value})"
                    )
                else:
                    print(
                        f"# Flag {np.sum(mask)}/{ndat} visibilities"
                        f" (type={dotype!r}, value={value},"
                        f" timerange={timerange!r})"
                    )

        self.r_1 = np.where(mask, np.nan, obs["r_1"]).reshape(_dshape)
        self.r_2 = np.where(mask, np.nan, obs["r_2"]).reshape(_dshape)
        self.r_3 = np.where(mask, np.nan, obs["r_3"]).reshape(_dshape)
        self.r_4 = np.where(mask, np.nan, obs["r_4"]).reshape(_dshape)
        self.i_1 = np.where(mask, np.nan, obs["i_1"]).reshape(_dshape)
        self.i_2 = np.where(mask, np.nan, obs["i_2"]).reshape(_dshape)
        self.i_3 = np.where(mask, np.nan, obs["i_3"]).reshape(_dshape)
        self.i_4 = np.where(mask, np.nan, obs["i_4"]).reshape(_dshape)
        self.w_1 = np.where(mask, np.nan, obs["w_1"]).reshape(_dshape)
        self.w_2 = np.where(mask, np.nan, obs["w_2"]).reshape(_dshape)
        self.w_3 = np.where(mask, np.nan, obs["w_3"]).reshape(_dshape)
        self.w_4 = np.where(mask, np.nan, obs["w_4"]).reshape(_dshape)

        self.check_w0()
        if not self.empty_w0:
            w0_1_flat = self.w0_1.flatten()
            w0_2_flat = self.w0_2.flatten()
            w0_3_flat = self.w0_3.flatten()
            w0_4_flat = self.w0_4.flatten()

            self.w0_1 = np.where(
                mask, np.nan, w0_1_flat
            ).reshape(_dshape)
            self.w0_2 = np.where(
                mask, np.nan, w0_2_flat
            ).reshape(_dshape)
            self.w0_3 = np.where(
                mask, np.nan, w0_3_flat
            ).reshape(_dshape)
            self.w0_4 = np.where(
                mask, np.nan, w0_4_flat
            ).reshape(_dshape)

        # propagate flags to complex gains (same flat order as the data)
        for _cg_attr in [
            "cg_pol1_ant1", "cg_pol1_ant2",
            "cg_pol2_ant1", "cg_pol2_ant2"
        ]:
            _cg = getattr(self, _cg_attr)
            if _cg is not None and _cg.size == ndat:
                _cg_shape = _cg.shape
                _cg_flat = _cg.flatten()
                _cg_flat[mask] = np.nan
                setattr(self, _cg_attr, _cg_flat.reshape(_cg_shape))

        self.set_data(prt=False)

    def flat_data(self):
        """
        Flatten the data and return it as a structured array
        """

        mjd = self.mjd.flatten()
        time = self.time.flatten()
        freq = self.freq.flatten()
        bsli = self.baseline.flatten()
        ant1 = self.ant1.flatten()
        ant2 = self.ant2.flatten()

        u = self.u.flatten()
        v = self.v.flatten()
        w = self.w.flatten()

        r_1 = self.r_1.flatten()
        r_2 = self.r_2.flatten()
        r_3 = self.r_3.flatten()
        r_4 = self.r_4.flatten()
        i_1 = self.i_1.flatten()
        i_2 = self.i_2.flatten()
        i_3 = self.i_3.flatten()
        i_4 = self.i_4.flatten()
        w_1 = self.w_1.flatten()
        w_2 = self.w_2.flatten()
        w_3 = self.w_3.flatten()
        w_4 = self.w_4.flatten()
        w0_1 = self.w0_1.flatten()
        w0_2 = self.w0_2.flatten()
        w0_3 = self.w0_3.flatten()
        w0_4 = self.w0_4.flatten()

        _data = [
            mjd, time, freq, bsli,
            ant1, ant2, u, v, w,
            r_1, r_2, r_3, r_4,
            i_1, i_2, i_3, i_4,
            w_1, w_2, w_3, w_4,
            w0_1, w0_2, w0_3, w0_4
        ]

        _name = [
            "mjd", "time", "frequency", "baseline",
            "ant1", "ant2", "u", "v", "w",
            "r_1", "r_2", "r_3", "r_4",
            "i_1", "i_2", "i_3", "i_4",
            "w_1", "w_2", "w_3", "w_4",
            "w0_1", "w0_2", "w0_3", "w0_4"
        ]

        _type = [
            "f8", "f8", "f8", "i4",
            "i4", "i4", "f8", "f8", "f8",
            "f8", "f8", "f8", "f8",
            "f8", "f8", "f8", "f8",
            "f8", "f8", "f8", "f8",
            "f8", "f8", "f8", "f8"
        ]

        out = gv.utils.structured_array(
            data=_data,
            field=_name,
            dtype=_type
        )
        return out

    def get_data(self, dotype=None, flatten=False):
        """
        Return the called data as a structured array
        """

        if dotype is None:
            raise ValueError("Data type must be specified.")

        out = gv.utils.get_data(
            uvf=self, dotype=dotype, flatten=flatten
        )

        return out

    def get_lblf(self):
        """
        Extract the longest-baseline flux level
        """

        data = self.data

        uvd = np.sqrt(data["u"]**2 + data["v"]**2)

        mask = uvd == uvd.max()
        lbl = data[mask]

        lbl_ant1 = lbl["ant1"][0]
        lbl_ant2 = lbl["ant2"][0]

        mask_lbl = (
            (data["frequency"] == data["frequency"].max())
            & (data["ant1"] == lbl_ant1)
            & (data["ant2"] == lbl_ant2)
        )

        lbl = data[mask_lbl]
        lblf = np.nanmedian(np.abs(lbl["vis"]))
        self.lblf = lblf
        self.lbl = (lbl_ant1, lbl_ant2)
        self.lbl_mask = mask_lbl
        return self.lblf, self.lbl, self.lbl_mask

    def get_sblf(self):
        """
        Extract the shortest-baseline flux level
        """

        data = self.data

        uvd = np.sqrt(data["u"]**2 + data["v"]**2)

        mask = uvd == uvd.min()
        sbl = data[mask]

        sbl_ant1 = sbl["ant1"][0]
        sbl_ant2 = sbl["ant2"][0]

        mask_sbl = (
            (data["frequency"] == data["frequency"].min())
            & (data["ant1"] == sbl_ant1)
            & (data["ant2"] == sbl_ant2)
        )

        sbl = data[mask_sbl]
        sblf = np.nanmedian(np.abs(sbl["vis"]))
        self.sblf = sblf
        self.sbl = (sbl_ant1, sbl_ant2)
        self.sbl_mask = mask_sbl
        return self.sblf, self.sbl, self.sbl_mask

    def inflate_sigma_fractional(self, inflate=0.0, timerange=None):
        """
        Inflate visibility sigma values fractionally to the visibility
        amplitude

        Args:
            inflate (float | dict): fraction of the error to be added
        """

        had_closure = self.clamp_check or self.clphs_check

        # geometric mean of visibility amplitudes
        def gmean_amp(x):
            x = x[np.isfinite(x) & (x > 0)]
            if x.size == 0:
                return 0.0
            return np.exp(np.nanmean(np.log(x)))

        self.check_w0()

        # get visibility amplitudes
        amp_1 = np.abs(self.r_1 + 1j * self.i_1)
        amp_2 = np.abs(self.r_2 + 1j * self.i_2)
        amp_3 = np.abs(self.r_3 + 1j * self.i_3)
        amp_4 = np.abs(self.r_4 + 1j * self.i_4)

        time = self.get_data("time")
        ant1 = self.get_data("ant1")
        ant2 = self.get_data("ant2")

        # scan masking
        scans, scans_1d = self.set_scan(
            time=time * 3600.0, gaptime=self.gaptime, scanlen=self.scanlen,
            returned=True
        )
        uscan = np.unique(scans)

        # time masking
        if timerange is None:
            mask_timer = np.ones_like(time, dtype=bool)
        else:
            if not isinstance(timerange, (list, tuple)):
                raise TypeError(
                    "timerange should be a list or tuple "
                    "of [start, end] times."
                )
            mask_timer = (time >= timerange[0]) & (time <= timerange[1])

        # baseline masking
        baseline = self.baseline
        ubaseline = np.unique(baseline)

        if isinstance(inflate, (int, float)):
            for ns, scan in enumerate(uscan):
                mask_scan = scans == scan
                for nb, baseline in enumerate(ubaseline):

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning
                        )

                        sig_1 = 1 / self.w_1**0.5
                        sig_2 = 1 / self.w_2**0.5
                        sig_3 = 1 / self.w_3**0.5
                        sig_4 = 1 / self.w_4**0.5

                    mask_baseline = self.baseline == baseline
                    mask = mask_scan & mask_baseline & mask_timer

                    if mask.sum() == 0:
                        continue

                    add_1 = gmean_amp(amp_1[mask]) * inflate
                    add_2 = gmean_amp(amp_2[mask]) * inflate
                    add_3 = gmean_amp(amp_3[mask]) * inflate
                    add_4 = gmean_amp(amp_4[mask]) * inflate

                    sig_1_new = sig_1[mask] + add_1
                    sig_2_new = sig_2[mask] + add_2
                    sig_3_new = sig_3[mask] + add_3
                    sig_4_new = sig_4[mask] + add_4

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning
                        )
                        self.w_1[mask] = 1 / sig_1_new**2
                        self.w_2[mask] = 1 / sig_2_new**2
                        self.w_3[mask] = 1 / sig_3_new**2
                        self.w_4[mask] = 1 / sig_4_new**2

            self.set_data(prt=False)

        elif isinstance(inflate, dict):
            antenna = list(inflate.keys())
            factor = list(inflate.values())

            for ns, scan in enumerate(uscan):
                mask_scan = scans == scan

                for na, _a in enumerate(antenna):

                    if not isinstance(_a, (list, tuple)):
                        _a = [_a]

                    if len(_a) > 2:
                        raise ValueError(
                            f"Invalid antenna list: {_a!r}. "
                            "Antenna list should have at most 2 elements."
                        )

                    if len(_a) == 1:
                        mask_ant1 = ant1 == self._resolve_ant(_a[0])
                        mask_ant2 = ant2 == self._resolve_ant(_a[0])
                        mask_ant = (mask_ant1 | mask_ant2)
                    else:
                        mask_ant = (
                            (
                                (ant1 == self._resolve_ant(_a[0]))
                                & (ant2 == self._resolve_ant(_a[1]))
                            )
                            | (
                                (ant1 == self._resolve_ant(_a[1]))
                                & (ant2 == self._resolve_ant(_a[0]))
                            )
                        )

                    for nb, baseline in enumerate(ubaseline):

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                category=RuntimeWarning
                            )

                            sig_1 = 1 / self.w_1**0.5
                            sig_2 = 1 / self.w_2**0.5
                            sig_3 = 1 / self.w_3**0.5
                            sig_4 = 1 / self.w_4**0.5

                        mask_baseline = self.baseline == baseline
                        mask = (
                            mask_scan & mask_baseline & mask_timer & mask_ant
                        )

                        if mask.sum() == 0:
                            continue

                        _f = factor[na]

                        add_1 = gmean_amp(amp_1[mask]) * _f
                        add_2 = gmean_amp(amp_2[mask]) * _f
                        add_3 = gmean_amp(amp_3[mask]) * _f
                        add_4 = gmean_amp(amp_4[mask]) * _f

                        sig_1_new = sig_1[mask] + add_1
                        sig_2_new = sig_2[mask] + add_2
                        sig_3_new = sig_3[mask] + add_3
                        sig_4_new = sig_4[mask] + add_4

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                category=RuntimeWarning
                            )
                            self.w_1[mask] = 1 / sig_1_new**2
                            self.w_2[mask] = 1 / sig_2_new**2
                            self.w_3[mask] = 1 / sig_3_new**2
                            self.w_4[mask] = 1 / sig_4_new**2

            self.set_data(prt=False)

        else:
            raise ValueError(
                f"Invalid inflate value: {type(inflate)!r}"
            )

    def load_uvf(
        self,
        select_pol="i", select_if="all", uvw="u", minclq=True, gaptime=None,
        scanlen=None, prt=True
    ):
        """
        Load uv-fits file and extract the information

        Args:
            select_pol (str): Polarization type
            select_if (str): Number(s) of IFs
            uvw (str): UV-weighting option
                - 'w': Weighting by visibility weight
                - 'u': Unity weighting
            minclq (bool): Compute minimum set of closures
            gaptime (float): Gaptime between scans [sec]
            scanlen (float): Scan length [sec]
            prt (bool): Print summarized information
        """
        available_pol = ["rr", "ll", "rl", "lr", "i", "q", "u", "v"]
        if select_pol not in available_pol:
            raise ValueError(f"Invalid polarization {select_pol!r}!")

        self.select_pol = select_pol.lower()
        self.select_if = select_if
        self.uvw = uvw
        self.minclq = minclq

        if gaptime is None:
            warnings.warn("'gaptime' is not set. Use 60s as default.")
            gaptime = 60.0

        if scanlen is None:
            warnings.warn("'scanlen' is not set. Disable scan length cut.")
            scanlen = 0.0
        self.gaptime = gaptime
        self.scanlen = scanlen

        mask_uvfdata = self.uvf_original is None

        # === Load UVF data

        # if self.uvf_original is None (e.g., synthetic data), load from file
        if mask_uvfdata:
            r_1 = self.r_1
            i_1 = self.i_1
            w_1 = self.w_1
            r_2 = self.r_2
            i_2 = self.i_2
            w_2 = self.w_2
            r_3 = self.r_3
            i_3 = self.i_3
            w_3 = self.w_3
            r_4 = self.r_4
            i_4 = self.i_4
            w_4 = self.w_4

            date = self.date
            mjd = self.mjd

        # if self.uvf_original exists (e.g., real data)
        else:
            uvf_original = self.uvf_original

            # load headers: PRIMARY, AIPS FQ, AIPS AN
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                h1 = uvf_original["PRIMARY"].copy()
                h2 = uvf_original["AIPS FQ"].copy()
                h3 = uvf_original["AIPS AN"].copy()

            stokes = int(h1.header["CRVAL3"])

            # get header data
            data1 = h1.data
            data2 = h2.data
            data3 = h3.data

            # get source name
            self.source = h1.header["OBJECT"]

            # get IF channels
            if isinstance(self.select_if, (list, tuple)):
                ifs = np.array(self.select_if) - 1
            elif self.select_if == "all":
                ifs = np.arange(h2.header["NO_IF"])
            else:
                ifs = np.array([self.select_if]).astype(int) - 1

            # get frequency information
            if_freq = data2.field("IF FREQ").reshape(1, -1)
            self.freq0 = h1.header["CRVAL4"] / 1e9

            self.freq = (
                self.freq0 + if_freq / 1e9
            )[0][ifs].reshape(1, -1)

            self.freq_mean = self.freq.mean()
            self.ufreq = np.unique(self.freq.astype("f8"))

            if self.freq.ndim == 1:
                self.freq = self.freq[:, None]

            self.no_if_original = ifs.size
            self.no_if = ifs.size

            # right ascension, declination, number of visibility, reference GST
            self.ra = h1.header["CRVAL6"]
            self.dec = h1.header["CRVAL7"]
            self.nvis = h1.header["GCOUNT"]
            self.refGST = h3.header["GSTIA0"]

            # ensure right ascension to 0-360 range
            if self.ra < 0:
                self.ra += 360

            # get (u, v, w) from data
            list_u = ["UU---SIN", "UU--", "UU"]
            list_v = ["VV---SIN", "VV--", "VV"]
            list_w = ["WW---SIN", "WW--", "WW"]
            for i in range(len(list_u)):
                if list_u[i] in data1.dtype.names:
                    idx_u = list_u[i]
            for i in range(len(list_v)):
                if list_v[i] in data1.dtype.names:
                    idx_v = list_v[i]
            for i in range(len(list_w)):
                if list_w[i] in data1.dtype.names:
                    idx_w = list_w[i]

            self.u = data1[idx_u][:, None, None] * self.freq[:, :, None] * 1e9
            self.v = data1[idx_v][:, None, None] * self.freq[:, :, None] * 1e9
            self.w = data1[idx_w][:, None, None] * self.freq[:, :, None] * 1e9

            # get time information from data
            self.DATE = data1["DATE"]
            self._DATE = data1["_DATE"]

            # get telescope information from AN table
            ants = np.array(list(map(
                lambda x: x.replace(" ", ""), data3.field(0)
            )))
            xpos = data3.field(1)[:, 0]
            ypos = data3.field(1)[:, 1]
            zpos = data3.field(1)[:, 2]
            anum = data3.field(3)
            ptya = data3.field(6)
            pola = data3.field(7)
            cala = data3.field(8)
            ptyb = data3.field(9)
            polb = data3.field(10)
            calb = data3.field(11)

            dr = np.zeros(len(data3), dtype="c8")
            dl = np.zeros(len(data3), dtype="c8")
            sefdr = np.zeros(len(data3))
            sefdl = np.zeros(len(data3))

            # set telescope array
            tarr_data = [
                ants, anum,
                xpos, ypos, zpos,
                dr, dl, sefdr, sefdl
            ]
            tarr_name = [
                "name", "number",
                "x", "y", "z",
                "d_r", "d_l", "sefd_r", "sefd_l"
            ]
            tarr_dtys = [
                "U32", "i4",
                "f8", "f8", "f8",
                "c8", "c8", "f8", "f8"
            ]

            self.tarr = gv.utils.structured_array(
                data=tarr_data,
                field=tarr_name,
                dtype=tarr_dtys
            )

            # set time information (jd, date, mjd)
            jd = self.DATE.astype("f8") + self._DATE.astype("f8")
            mjd = atime(jd, format="jd").mjd
            self.date = atime(jd.min(), format="jd").iso[:10]

            jd_min = int(jd.min() - 2400000.5)

            # time data (in hour)
            self.time = (jd - 2400000.5 - jd_min) * 24

            # set baseline information
            self.baseline = data1["BASELINE"].astype(int)
            self.ant1 = self.baseline // 256
            self.ant2 = self.baseline - self.ant1 * 256

            """
            Data axis description // e.g., dim = (1, 1, 4, 1, 4, 3) :
                1 : DEC
                2 : RA
                3 : IF
                4 : FREQ
                5 : STOKES
                    CRPIX=+1 to +4: I, Q, U, V
                    CRPIX=-1 to -4: RR, LL, RL, LR
                    CRPIX=-5 to -8: XX, YY, XY, YX
                6 : COMPLEX // (real, imag, weight for visibility measurement)
                * Please see the AIPS MEMO 117 for the details
                ** weight<=0 indicates that the visibility measurement
                    is flagged and that the values may not be
                    in any way meaningful
            """
            dict_stokes = {
                -8:"YX", -7:"XY", -6:"YY", -5:"XX",
                -4:"LR", -3:"RL", -2:"LL", -1:"RR",
                +1:"I", +2:"Q", +3:"U", +4:"V",
            }

            # get Stokes information
            self.nstokes = data1["DATA"].shape[5]
            self.stokes = dict_stokes[stokes]

            # get visibility data
            r_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 0]
            i_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 1]
            w_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 2]
            if self.nstokes == 2:
                r_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 0]
                i_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 1]
                w_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 2]
                r_3 = r_1 * 0.0
                i_3 = i_1 * 0.0
                w_3 = w_1 * 0.0
                r_4 = r_1 * 0.0
                i_4 = i_1 * 0.0
                w_4 = w_1 * 0.0
            elif self.nstokes == 4:
                r_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 0]
                i_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 1]
                w_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 2]
                r_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 0]
                i_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 1]
                w_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 2]
                r_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 0]
                i_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 1]
                w_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 2]
            else:
                r_2 = r_1 * 0.0
                i_2 = i_1 * 0.0
                w_2 = w_1 * 0.0
                r_3 = r_1 * 0.0
                i_3 = i_1 * 0.0
                w_3 = w_1 * 0.0
                r_4 = r_1 * 0.0
                i_4 = i_1 * 0.0
                w_4 = w_1 * 0.0

            if_freq = self.ufreq.reshape(1, -1)
            if isinstance(self.select_if, (list, tuple)):
                ifs = np.array(self.select_if) - 1
            elif self.select_if == "all":
                ifs = np.arange(len(self.ufreq))
            else:
                ifs = np.array([self.select_if]).astype(int) - 1

        # set antenna dictionary (name -> number, number -> name)
        self.ant_dict_name2num = dict(zip(
            self.tarr["name"], self.tarr["number"]
        ))
        self.ant_dict_num2name = dict(zip(
            self.tarr["number"], self.tarr["name"]
        ))

        # set visibility data into 'self'
        self.r_1 = r_1.reshape(self.nvis, self.no_if, 1)
        self.i_1 = i_1.reshape(self.nvis, self.no_if, 1)
        self.w_1 = w_1.reshape(self.nvis, self.no_if, 1)
        self.r_2 = r_2.reshape(self.nvis, self.no_if, 1)
        self.i_2 = i_2.reshape(self.nvis, self.no_if, 1)
        self.w_2 = w_2.reshape(self.nvis, self.no_if, 1)
        self.r_3 = r_3.reshape(self.nvis, self.no_if, 1)
        self.i_3 = i_3.reshape(self.nvis, self.no_if, 1)
        self.w_3 = w_3.reshape(self.nvis, self.no_if, 1)
        self.r_4 = r_4.reshape(self.nvis, self.no_if, 1)
        self.i_4 = i_4.reshape(self.nvis, self.no_if, 1)
        self.w_4 = w_4.reshape(self.nvis, self.no_if, 1)

        _amp_1 = np.sqrt(self.r_1**2 + self.i_1**2)
        _amp_2 = np.sqrt(self.r_2**2 + self.i_2**2)
        # zero/negative weights are flagged data, masked out below
        with np.errstate(divide="ignore", invalid="ignore"):
            _sig_1 = np.sqrt(1 / self.w_1)
            _sig_2 = np.sqrt(1 / self.w_2)

        maskn_1 = (
            (np.isnan(self.r_1)) | (np.isinf(self.r_1))
            | (np.isnan(self.i_1)) | (np.isinf(self.i_1))
            | (np.isnan(self.w_1)) | (np.isinf(self.w_1)) | (self.w_1 <= 0)
        )

        maskn_2 = (
            (np.isnan(self.r_2)) | (np.isinf(self.r_2))
            | (np.isnan(self.i_2)) | (np.isinf(self.i_2))
            | (np.isnan(self.w_2)) | (np.isinf(self.w_2)) | (self.w_2 <= 0)
        )

        if self.nstokes >= 2:
            mask = (maskn_1) | (maskn_2)
        elif self.nstokes == 1:
            mask = (maskn_1)

        self.mask = mask

        self.r_1[mask] = np.nan
        self.r_2[mask] = np.nan
        self.r_3[mask] = np.nan
        self.r_4[mask] = np.nan
        self.i_1[mask] = np.nan
        self.i_2[mask] = np.nan
        self.i_3[mask] = np.nan
        self.i_4[mask] = np.nan
        self.w_1[mask] = np.nan
        self.w_2[mask] = np.nan
        self.w_3[mask] = np.nan
        self.w_4[mask] = np.nan

        self.check_dims()

        mask_ = np.sum(np.sum(self.mask, axis=2), axis=1) == self.no_if
        self.mjd = np.broadcast_to(
            mjd[:, None, None], self.r_1.shape
        )[~mask_, :, :].copy()
        self.time = np.broadcast_to(
            self.time[:, None, None], self.r_1.shape
        )[~mask_, :, :].copy()
        self.freq = np.broadcast_to(
            self.freq[:, :, None], self.r_1.shape
        )[~mask_, :, :].copy()
        self.baseline = np.broadcast_to(
            self.baseline[:, None, None], self.r_1.shape
        )[~mask_, :, :].copy()
        self.ant1 = np.broadcast_to(
            self.ant1[:, None, None], self.r_1.shape
        )[~mask_, :, :].copy()
        self.ant2 = np.broadcast_to(
            self.ant2[:, None, None], self.r_1.shape
        )[~mask_, :, :].copy()

        self.u = self.u[~mask_]
        self.v = self.v[~mask_]
        self.w = self.w[~mask_]

        self.r_1 = self.r_1[~mask_]
        self.r_2 = self.r_2[~mask_]
        self.r_3 = self.r_3[~mask_]
        self.r_4 = self.r_4[~mask_]
        self.i_1 = self.i_1[~mask_]
        self.i_2 = self.i_2[~mask_]
        self.i_3 = self.i_3[~mask_]
        self.i_4 = self.i_4[~mask_]
        self.w_1 = self.w_1[~mask_]
        self.w_2 = self.w_2[~mask_]
        self.w_3 = self.w_3[~mask_]
        self.w_4 = self.w_4[~mask_]
        self.w0_1 = self.w_1
        self.w0_2 = self.w_2
        self.w0_3 = self.w_3
        self.w0_4 = self.w_4

        self.cg_pol1_ant1 = np.ones(self.time.shape, dtype="c8")
        self.cg_pol1_ant2 = np.ones(self.time.shape, dtype="c8")
        self.cg_pol2_ant1 = np.ones(self.time.shape, dtype="c8")
        self.cg_pol2_ant2 = np.ones(self.time.shape, dtype="c8")

        data = self.u
        self.sort_data(dotype=["freq", "time", "ant"])

        uvc = self.set_uvcov(flatten=True, returned=True)
        self.bprms = gv.utils.fit_beam(uvc=uvc, sig=None, uvw=self.uvw)
        self.bmaj = self.bprms[0]
        self.bmin = self.bprms[1]
        self.bpa = self.bprms[2]
        self.ploter.bprms = self.bprms

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.original = copy.deepcopy(self)

        if prt:
            cf_mapunits = au.mas.to(self.mapunit)
            print(
                f"\nfound {self.r_1.size} visibilities...\n"
                f"\t: {self.date}, "
                f"{self.source!r}, "
                f"{self.freq.mean():.3f} GHz\n"
                f"\t: [{self.bprms[0] * au.mas.to(self.mapunit):.2f}"
                f" \u00D7 "
                f"{self.bprms[1] * cf_mapunits:.2f} {self.mapunit.name}, "
                f"{self.bprms[2]:.1f} deg], "
                f"{self.no_if} IF chans, "
                f"{self.nstokes} Stokes parameters"
            )

    def model_visibility_append(
        self,
        vism=None, freq_ref=None, theta=None, model="gaussian",
        spectrum="flat", closure=True
    ):
        """
        Append the model visibility to the data

        Args:
            vism (np.ndarray): Model visibility to append
            freq_ref (float): Reference frequency [GHz]
            theta (dict): Model parameters
            pol (bool): Assumes polarization model
            spectrum (str): Spectrum type
                - flat: Flat spectrum (no spectrum model)
                - spl: Simple power-law
                - cpl: Curved power-law
                - ssa: Synchrotron self-absorption
                - poly: Logarithmic 2nd-order polynomial
        """
        if vism is None:
            self.spectrum = spectrum
            self.theta = theta
            self.ploter.spectrum = spectrum
            self.ploter.theta = theta

            if theta is None:
                raise ValueError(
                    f"Invalid 'theta' value: {theta}. "
                    f"Provide appropriate model parameters."
                )

            available_spectrum = ["flat", "spl", "cpl", "ssa", "poly"]
            if spectrum not in available_spectrum:
                raise ValueError(
                    f"Invalid spectrum: {spectrum!r}. "
                    f"Availables: {available_spectrum}."
                )

            dshape = self.data_shape
            dtypes = theta.dtype.names

            freq = self.freq

            if dshape is None:
                print(
                    "\nNo data is established. "
                    "Run 'set_data()' to load data.\n"
                )

                self.set_data(prt=False)

                dshape = self.data_shape
                dtypes = theta.dtype.names

            u = self.u
            v = self.v

            mask_parallel = self.select_pol in ["rr", "ll", "i"]
            mask_cross = self.select_pol in ["q", "u", "p"]

            in_args = (u, v, freq_ref, freq, model, spectrum, dshape, dtypes)
            in_mask = (mask_parallel, mask_cross)

            if mask_parallel:
                self.vism = gv.utils.model_visibility_append(
                    in_args, theta, in_mask
                )

            if mask_cross:
                vism_q, vism_u = gv.utils.model_visibility_append(
                    in_args, theta, in_mask
                )

                self.vism_p = vism_q + 1j * vism_u
                self.vism_q = vism_q
                self.vism_u = vism_u

                if self.select_pol == "p":
                    self.vism = self.vism_p

                elif self.select_pol == "q":
                    self.vism = self.vism_q

                elif self.select_pol == "u":
                    self.vism = self.vism_u
        else:
            self.vism = np.asarray(vism).reshape(self.data_shape)

        self.set_data(prt=False)

        # model closure quantities
        if closure:
            if self.clamp is None or self.clphs is None:
                self.set_closure(self.minclq)

            clamp_uvcomb, clphs_uvcomb = gv.utils.set_uvcombination(self)

            self.ploter.clq_mod = gv.utils.set_closure(
                self, self.u.flatten(), self.v.flatten(), self.vism.flatten(),
                np.zeros(self.vism.size),
                self.ant1.flatten(), self.ant2.flatten(),
                clamp_uvcomb, clphs_uvcomb
            )

    def model_visibility_drop(self):
        """
        Drop the model visibility from the data
        """

        self.vism = None

        uvdat = self.data

        if "vism" in uvdat.dtype.names:
            uvdat = rfn.drop_fields(uvdat, "vism")

        self.data = uvdat

    def rescale_flux(
        self,
        rescale=None, timerange=None
    ):
        """
        Rescale flux density of visibilities

        Args:
            rescale (float | dict): Rescaling factor(s)
                - float: Global factor applied to all visibilities
                - {antenna: factor}: Factor for baselines with the antenna
                  (overlapping keys accumulate multiplicatively)
                - {(ant1, ant2): factor}: Factor for a single baseline
            timerange (list, tuple): Time range ([start, end]) in hours
                (None = all)
        """

        had_closure = self.clamp_check or self.clphs_check

        time = self.get_data("time")
        r_1 = self.get_data("r_1")
        r_2 = self.get_data("r_2")
        r_3 = self.get_data("r_3")
        r_4 = self.get_data("r_4")
        i_1 = self.get_data("i_1")
        i_2 = self.get_data("i_2")
        i_3 = self.get_data("i_3")
        i_4 = self.get_data("i_4")
        w_1 = self.get_data("w_1")
        w_2 = self.get_data("w_2")
        w_3 = self.get_data("w_3")
        w_4 = self.get_data("w_4")
        w0_1 = self.get_data("w0_1")
        w0_2 = self.get_data("w0_2")
        w0_3 = self.get_data("w0_3")
        w0_4 = self.get_data("w0_4")

        ant1 = self.get_data("ant1")
        ant2 = self.get_data("ant2")

        # Traceback messages
        if not isinstance(rescale, (int, float, dict)):
            raise ValueError(
                "rescale (float or dictionary) must be provided. "
                "Available format: float (global factor), "
                "{antenna: factor}, or {(ant1, ant2): factor}"
            )

        # time masking
        if timerange is None:
            mask_timer = np.ones_like(time, dtype=bool)
        else:
            if not isinstance(timerange, (list, tuple)):
                raise TypeError(
                    "timerange should be a list or tuple "
                    "of [start, end] times."
                )
            mask_timer = (time >= timerange[0]) & (time <= timerange[1])

        def _apply_scale(mask, _f):
            r_1[mask] = r_1[mask] * _f
            r_2[mask] = r_2[mask] * _f
            r_3[mask] = r_3[mask] * _f
            r_4[mask] = r_4[mask] * _f
            i_1[mask] = i_1[mask] * _f
            i_2[mask] = i_2[mask] * _f
            i_3[mask] = i_3[mask] * _f
            i_4[mask] = i_4[mask] * _f
            w_1[mask] = w_1[mask] / _f ** 2
            w_2[mask] = w_2[mask] / _f ** 2
            w_3[mask] = w_3[mask] / _f ** 2
            w_4[mask] = w_4[mask] / _f ** 2
            w0_1[mask] = w0_1[mask] / _f ** 2
            w0_2[mask] = w0_2[mask] / _f ** 2
            w0_3[mask] = w0_3[mask] / _f ** 2
            w0_4[mask] = w0_4[mask] / _f ** 2

        if isinstance(rescale, (int, float)):
            # global rescaling
            _apply_scale(mask_timer, rescale)

        else:
            antenna = list(rescale.keys())
            factor = list(rescale.values())

            for na, _a in enumerate(antenna):
                if not isinstance(_a, (list, tuple)):
                    _a = [_a]

                if len(_a) > 2:
                    raise ValueError(
                        f"Invalid antenna list: {_a!r}. "
                        "Antenna list should have at most 2 elements."
                    )

                _f = factor[na]

                if len(_a) == 1:
                    mask_ant1 = ant1 == self._resolve_ant(_a[0])
                    mask_ant2 = ant2 == self._resolve_ant(_a[0])
                    mask = mask_ant1 | mask_ant2
                else:
                    mask = (
                        (
                            (ant1 == self._resolve_ant(_a[0]))
                            & (ant2 == self._resolve_ant(_a[1]))
                        )
                        | (
                            (ant1 == self._resolve_ant(_a[1]))
                            & (ant2 == self._resolve_ant(_a[0]))
                        )
                    )

                mask = mask & mask_timer

                _apply_scale(mask, _f)

        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3
        self.r_4 = r_4
        self.i_1 = i_1
        self.i_2 = i_2
        self.i_3 = i_3
        self.i_4 = i_4
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.w_4 = w_4
        self.w0_1 = w0_1
        self.w0_2 = w0_2
        self.w0_3 = w0_3
        self.w0_4 = w0_4

        self.set_data(prt=False)

    def rescale_sigma(self, rescale=1, timerange=None):
        """
        Rescale visibility sigma by a factor
        (weights are divided by the squared factor)

        Args:
            rescale (float | dict): rescaling factor(s)
                - float: global factor applied to all visibilities
                - {antenna: factor}: factor for baselines with the antenna
                  (overlapping keys accumulate multiplicatively)
                - {(ant1, ant2): factor}: factor for a single baseline
            timerange (list, tuple): Time range ([start, end]) in hours
                (None = all)
        """

        had_closure = self.clamp_check or self.clphs_check

        self.check_w0()

        time = self.get_data("time")
        ant1 = self.get_data("ant1")
        ant2 = self.get_data("ant2")

        # Traceback messages
        if not isinstance(rescale, (int, float, dict)):
            raise ValueError(
                "rescale (float or dictionary) must be provided. "
                "Available format: float (global factor), "
                "{antenna: factor}, or {(ant1, ant2): factor}"
            )

        # time masking
        if timerange is None:
            mask_timer = np.ones_like(time, dtype=bool)
        else:
            if not isinstance(timerange, (list, tuple)):
                raise TypeError(
                    "timerange should be a list or tuple "
                    "of [start, end] times."
                )
            mask_timer = (time >= timerange[0]) & (time <= timerange[1])

        def _apply_scale(mask, _f):
            # sigma * f == weight / f**2; direct division keeps
            # flagged (non-positive or NaN) weights intact
            self.w_1[mask] = self.w_1[mask] / _f ** 2
            self.w_2[mask] = self.w_2[mask] / _f ** 2
            self.w_3[mask] = self.w_3[mask] / _f ** 2
            self.w_4[mask] = self.w_4[mask] / _f ** 2

        if isinstance(rescale, (int, float)):
            # global rescaling
            _apply_scale(mask_timer, rescale)

        else:
            antenna = list(rescale.keys())
            factor = list(rescale.values())

            for na, _a in enumerate(antenna):
                if not isinstance(_a, (list, tuple)):
                    _a = [_a]

                if len(_a) > 2:
                    raise ValueError(
                        f"Invalid antenna list: {_a!r}. "
                        "Antenna list should have at most 2 elements."
                    )

                _f = factor[na]

                if len(_a) == 1:
                    mask_ant1 = ant1 == self._resolve_ant(_a[0])
                    mask_ant2 = ant2 == self._resolve_ant(_a[0])
                    mask_ant = mask_ant1 | mask_ant2
                else:
                    mask_ant = (
                        (
                            (ant1 == self._resolve_ant(_a[0]))
                            & (ant2 == self._resolve_ant(_a[1]))
                        )
                        | (
                            (ant1 == self._resolve_ant(_a[1]))
                            & (ant2 == self._resolve_ant(_a[0]))
                        )
                    )

                mask = mask_ant & mask_timer

                _apply_scale(mask, _f)

        self.set_data(prt=False)

    def reset_weight(self):
        """
        Reset the weight to original data
        (remove systematatics & added fraction/factor)
        """

        had_closure = self.clamp_check or self.clphs_check

        if not self.empty_w0:
            self.w_1 = self.w0_1
            self.w_2 = self.w0_2
            self.w_3 = self.w0_3
            self.w_4 = self.w0_4
            self.set_data(prt=False)
        else:
            warnings.warn("No weight to reset. Pass 'reset_weight()'.")

    def save_cgain(self, save_path="", save_name=""):
        """
        Save the complex gain to a csv file

        Args:
            save_name (str): Name of the new fits file
            save_path (str): Path of the new fits file
        """
        gv.utils.save_cgain(
            uvf=self,
            save_path=save_path,
            save_name=save_name
        )

    def save_uvfits(self, save_path=None, save_name=None):
        """
        Save new fits file by rebuilding all HDUs from scratch
        (PRIMARY, AIPS AN, AIPS FQ, AIPS NX)

        Args:
            save_name (str): Name of the new fits file
            save_path (str): Path of the new fits file
        """
        if save_path is None:
            save_path = self.path or ""

        if save_name is None:
            save_name = f"newsave.{self.file or 'sim.uvf'}"

        hdul_orig = getattr(self, "uvf_original", None)
        if hdul_orig is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                hdul_orig = copy.deepcopy(hdul_orig)

        def _pr_hdr_get(key, default):
            if hdul_orig is None:
                return default
            try:
                return hdul_orig["PRIMARY"].header.get(key, default)
            except KeyError:
                return default

        def _an_hdr_get(key, default):
            if hdul_orig is None:
                return default
            try:
                return hdul_orig["AIPS AN"].header.get(key, default)
            except KeyError:
                return default

        def _fq_hdr_get(key, default):
            if hdul_orig is None:
                return default
            try:
                return hdul_orig["AIPS FQ"].header.get(key, default)
            except KeyError:
                return default

        def convert_nan_to_zero(data):
            return np.nan_to_num(data, nan=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _uvf = copy.deepcopy(self)

        _uvf.sort_data(dotype=["freq", "time", "ant"])
        _uvf.set_data(prt=False)

        time = _uvf.get_data(dotype="time").astype("f8")[:, 0, 0]
        freq = _uvf.get_data(dotype="frequency").astype("f8")[:, 0, 0]
        baseline = _uvf.get_data(dotype="baseline")[:, 0, 0]
        ant1 = _uvf.get_data(dotype="ant1")[:, 0, 0].astype("i4")
        ant2 = _uvf.get_data(dotype="ant2")[:, 0, 0].astype("i4")
        u = _uvf.get_data(dotype="u")[:, 0, 0] / (self.freq0 * 1e9)
        v = _uvf.get_data(dotype="v")[:, 0, 0] / (self.freq0 * 1e9)
        w = _uvf.get_data(dotype="w")[:, 0, 0] / (self.freq0 * 1e9)

        ufreq = np.unique(freq)
        set_mf = "mf" in self.select_pol

        if set_mf:
            nant_orig = len(self.tarr)
            for i in range(len(ufreq)):
                mask_freq = freq == ufreq[i]
                if i != 0:
                    ant1[mask_freq] = ant1[mask_freq] + nant_orig * i
                    ant2[mask_freq] = ant2[mask_freq] + nant_orig * i

                if i == 0:
                    tarr = self.tarr.copy()
                    tarr["name"] = np.char.add(tarr["name"], "1")
                else:
                    tarr_ = self.tarr.copy()
                    tarr_["name"] = np.char.add(tarr_["name"], f"{i + 1}")
                    tarr = np.append(tarr, tarr_)
            baseline = (ant1 * 256 + ant2).astype("f4")
        else:
            tarr = self.tarr
        nant = len(tarr)

        # Build vis data array
        nstokes = self.nstokes
        ndata = _uvf.data_shape[0]
        nif = _uvf.data_shape[1]
        dims_pr = (ndata, 1, 1, nif, 1, nstokes, 3)

        r_1 = convert_nan_to_zero(_uvf.r_1)
        r_2 = convert_nan_to_zero(_uvf.r_2)
        r_3 = convert_nan_to_zero(_uvf.r_3)
        r_4 = convert_nan_to_zero(_uvf.r_4)

        i_1 = convert_nan_to_zero(_uvf.i_1)
        i_2 = convert_nan_to_zero(_uvf.i_2)
        i_3 = convert_nan_to_zero(_uvf.i_3)
        i_4 = convert_nan_to_zero(_uvf.i_4)

        w_1 = convert_nan_to_zero(_uvf.w_1)
        w_2 = convert_nan_to_zero(_uvf.w_2)
        w_3 = convert_nan_to_zero(_uvf.w_3)
        w_4 = convert_nan_to_zero(_uvf.w_4)

        w0_1 = convert_nan_to_zero(_uvf.w0_1)
        w0_2 = convert_nan_to_zero(_uvf.w0_2)
        w0_3 = convert_nan_to_zero(_uvf.w0_3)
        w0_4 = convert_nan_to_zero(_uvf.w0_4)

        lst_real = [r_1, r_2, r_3, r_4]
        lst_imag = [i_1, i_2, i_3, i_4]
        if self.empty_w0:
            lst_wght = [w_1, w_2, w_3, w_4]
        else:
            lst_wght = [w0_1, w0_2, w0_3, w0_4]

        data_ = np.empty(dims_pr, dtype="f4")
        for _nif in range(nif):
            for _nst in range(nstokes):
                data_[:, 0, 0, _nif, 0, _nst, :] = np.stack(
                    [
                        lst_real[_nst][:, _nif, 0],
                        lst_imag[_nst][:, _nif, 0],
                        lst_wght[_nst][:, _nif, 0]
                    ],
                    axis=-1
                )

        # Random parameters (DATE / _DATE / INTTIM)
        jd_full = atime(self.date, format="iso").jd + time / 24.0
        jd_ref = int(np.floor(jd_full.min() - 0.5))

        DATE_par = np.zeros(ndata, dtype="f8")
        _DATE_par = (jd_full - jd_ref).astype("f8")

        if self.avg_timebin is None:
            INTTIM = np.full(ndata, 1.0, dtype="f4")
        else:
            INTTIM = np.full(ndata, self.avg_timebin, dtype="f4")

        # Detect u/v/w parameter names from the original (preserve projection)
        if hdul_orig is not None:
            orig_parnames = list(hdul_orig["PRIMARY"].data.parnames)
        else:
            orig_parnames = []
        ulabel = next(
            (p for p in orig_parnames if p.startswith("UU")), "UU---SIN"
        )
        vlabel = next(
            (p for p in orig_parnames if p.startswith("VV")), "VV---SIN"
        )
        wlabel = next(
            (p for p in orig_parnames if p.startswith("WW")), "WW---SIN"
        )

        parnames = [
            ulabel, vlabel, wlabel, "BASELINE", "DATE", "DATE", "INTTIM"
        ]

        pardata = [
            u.astype("f4"), v.astype("f4"), w.astype("f4"),
            baseline.astype("f4"),
            DATE_par, _DATE_par,
            INTTIM
        ]

        cols_pr_ = fits.GroupData(
            data_, bitpix=-32,
            parnames=parnames,
            pardata=pardata
        )

        hdu_pr = fits.GroupsHDU(cols_pr_)
        hdu_pr.name = "PRIMARY"

        # PRIMARY header
        hdr_pr = hdu_pr.header

        hdr_pr["BSCALE"] = 1.0
        hdr_pr["BZERO"] = 0.0
        hdr_pr["BUNIT"] = "JY"
        hdr_pr["EQUINOX"] = float(_pr_hdr_get("EQUINOX", 2000.0))

        if hdul_orig is not None:
            orig_pr_hdr = hdul_orig["PRIMARY"].header
            for key in (
                "OBJECT", "TELESCOP", "INSTRUME", "OBSERVER",
                "DATE-OBS", "DATE-MAP"
            ):
                if key in orig_pr_hdr:
                    hdr_pr[key] = orig_pr_hdr[key]
        if "OBJECT" not in hdr_pr:
            hdr_pr["OBJECT"] = getattr(self, "source", "") or ""
        if "DATE-OBS" not in hdr_pr:
            hdr_pr["DATE-OBS"] = str(getattr(self, "date", ""))
        if "TELESCOP" not in hdr_pr:
            hdr_pr["TELESCOP"] = "VLBI"

        # World coordinate axes (FITS NAXIS2..NAXIS7)
        crval3_orig = float(_pr_hdr_get("CRVAL3", -1.0))
        cdelt3_orig = float(_pr_hdr_get("CDELT3", -1.0))
        crpix3_orig = float(_pr_hdr_get("CRPIX3", 1.0))

        # Channel width from original FQ (used for CDELT4)
        if hdul_orig is not None:
            fq_data = hdul_orig["AIPS FQ"].data
            fq_names = list(fq_data.dtype.names)
        else:
            fq_data = None
            fq_names = []

        def _fq_field(name, default=None):
            if fq_data is None or name not in fq_names:
                return default
            return np.atleast_2d(fq_data[name])[0]

        ch_width_orig = _fq_field("CH WIDTH", np.array([1e6]))
        if ch_width_orig is None or len(ch_width_orig) == 0:
            ch_width_default = float(_pr_hdr_get("CDELT4", 1e6))
        else:
            ch_width_default = float(ch_width_orig[0])

        ra_val = getattr(self, "ra", None)
        dec_val = getattr(self, "dec", None)
        if (ra_val is None or dec_val is None) and \
                getattr(self, "source_coord", None) is not None:
            ra_val = self.source_coord.ra.deg
            dec_val = self.source_coord.dec.deg
        if ra_val is None:
            ra_val = float(_pr_hdr_get("CRVAL6", 0.0))
        if dec_val is None:
            dec_val = float(_pr_hdr_get("CRVAL7", 0.0))

        axes_spec = [
            ("COMPLEX", 1.0, 1.0, 1.0, 0.0),
            ("STOKES", crval3_orig, cdelt3_orig, crpix3_orig, 0.0),
            ("FREQ", self.freq0 * 1e9, ch_width_default, 1.0, 0.0),
            ("IF", 1.0, 1.0, 1.0, 0.0),
            ("RA", float(ra_val), 1.0, 1.0, 0.0),
            ("DEC", float(dec_val), 1.0, 1.0, 0.0),
        ]
        for n, (ctype, crval, cdelt, crpix, crota) in enumerate(
            axes_spec, start=2
        ):
            hdr_pr[f"CTYPE{n}"] = ctype
            hdr_pr[f"CRVAL{n}"] = crval
            hdr_pr[f"CDELT{n}"] = cdelt
            hdr_pr[f"CRPIX{n}"] = crpix
            hdr_pr[f"CROTA{n}"] = crota

        # Random parameter scaling
        for k in range(1, 8):
            hdr_pr[f"PSCAL{k}"] = 1.0
            hdr_pr[f"PZERO{k}"] = 0.0
        hdr_pr["PZERO5"] = float(jd_ref)

        # AIPS AN table
        # Map per-IF columns (POLCALA / POLCALB) from original to new IF count
        an_no_if_orig = int(_an_hdr_get(
            "NO_IF",
            _fq_hdr_get("NO_IF", nif)
        ))
        polcal_nelem_orig = 2  # default 2 (real, imag) per IF
        if hdul_orig is not None:
            try:
                an_orig_data = hdul_orig["AIPS AN"].data
                shp = np.atleast_2d(an_orig_data["POLCALA"]).shape
                if an_no_if_orig > 0 and shp[1] % an_no_if_orig == 0:
                    polcal_nelem_orig = shp[1] // an_no_if_orig
            except Exception:
                pass
        polcal_nelem = polcal_nelem_orig * nif

        ANNAME = tarr["name"].astype("U8")
        STABXYZ = np.column_stack(
            [tarr["x"], tarr["y"], tarr["z"]]
        ).astype("f8")
        NOSTA = np.arange(1, nant + 1, dtype="i4")
        MNTSTA = np.zeros(nant, dtype="i4")
        STAXOF = np.zeros(nant, dtype="f4")
        POLTYA = np.array(["R"] * nant)
        POLAA = np.zeros(nant, dtype="f4")
        POLCALA = np.zeros((nant, polcal_nelem), dtype="f4")
        POLTYB = np.array(["L"] * nant)
        POLAB = np.zeros(nant, dtype="f4")
        POLCALB = np.zeros((nant, polcal_nelem), dtype="f4")
        ORBPARM = np.zeros((nant, 0), dtype="f8")

        cols_an = [
            fits.Column(name="ANNAME", format="8A", array=ANNAME),
            fits.Column(
                name="STABXYZ", format="3D", unit="METERS", array=STABXYZ
            ),
            fits.Column(name="ORBPARM", format="0D", array=ORBPARM),
            fits.Column(name="NOSTA", format="1J", array=NOSTA),
            fits.Column(name="MNTSTA", format="1J", array=MNTSTA),
            fits.Column(
                name="STAXOF", format="1E", unit="METERS", array=STAXOF
            ),
            fits.Column(name="POLTYA", format="1A", array=POLTYA),
            fits.Column(
                name="POLAA", format="1E", unit="DEGREES", array=POLAA
            ),
            fits.Column(
                name="POLCALA", format=f"{polcal_nelem}E", array=POLCALA
            ),
            fits.Column(name="POLTYB", format="1A", array=POLTYB),
            fits.Column(
                name="POLAB", format="1E", unit="DEGREES", array=POLAB
            ),
            fits.Column(
                name="POLCALB", format=f"{polcal_nelem}E", array=POLCALB
            ),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            hdu_an = fits.BinTableHDU.from_columns(cols_an)
        hdu_an.name = "AIPS AN"

        # GSTIA0 fallback: derive from RDATE if self.refGST not set
        if getattr(self, "refGST", None) is not None:
            gstia0 = float(self.refGST)
        else:
            try:
                rdate_str = _an_hdr_get("RDATE", str(getattr(self, "date", "")))
                gstia0 = atime(
                    rdate_str, format="iso"
                ).sidereal_time("apparent", "greenwich").deg
            except Exception:
                gstia0 = 0.0

        # FREQ fallback: use freq_mean or freq0
        if getattr(self, "freq_mean", None) is not None:
            an_freq = float(self.freq_mean * 1e9)
        else:
            an_freq = float(self.freq0 * 1e9)

        hdr_an = hdu_an.header
        hdr_an["EXTVER"] = 1
        hdr_an["EXTNAME"] = "AIPS AN"
        hdr_an["ARRAYX"] = float(_an_hdr_get("ARRAYX", 0.0))
        hdr_an["ARRAYY"] = float(_an_hdr_get("ARRAYY", 0.0))
        hdr_an["ARRAYZ"] = float(_an_hdr_get("ARRAYZ", 0.0))
        hdr_an["GSTIA0"] = gstia0
        hdr_an["DEGPDY"] = float(_an_hdr_get("DEGPDY", 360.9856449733))
        hdr_an["FREQ"] = an_freq
        hdr_an["RDATE"] = _an_hdr_get("RDATE", str(getattr(self, "date", "")))
        hdr_an["POLARX"] = float(_an_hdr_get("POLARX", 0.0))
        hdr_an["POLARY"] = float(_an_hdr_get("POLARY", 0.0))
        hdr_an["UT1UTC"] = float(_an_hdr_get("UT1UTC", 0.0))
        hdr_an["IATUTC"] = float(_an_hdr_get("IATUTC", 0.0))
        hdr_an["TIMSYS"] = _an_hdr_get("TIMSYS", "UTC")
        hdr_an["ARRNAM"] = _an_hdr_get("ARRNAM", "VLBI")
        hdr_an["NUMORB"] = 0
        hdr_an["NOPCAL"] = polcal_nelem_orig
        hdr_an["POLTYPE"] = _an_hdr_get("POLTYPE", "        ")
        hdr_an["NO_IF"] = nif
        hdr_an["XYZHAND"] = _an_hdr_get("XYZHAND", "RIGHT")
        hdr_an["FRAME"] = _an_hdr_get("FRAME", "ITRF")

        # AIPS FQ table
        # Per-IF frequency of the data being saved.
        # _uvf.freq has shape (ndata, nif, 1); take row 0.
        if hasattr(_uvf, "freq") and _uvf.freq.ndim >= 2:
            data_if_freq = (
                np.asarray(_uvf.freq)[0, :, 0].astype("f8") * 1e9
            )
        else:
            data_if_freq = np.full(nif, self.freq_mean * 1e9)

        # Read original FQ defaults (for CH WIDTH / TOTAL BANDWIDTH /
        # SIDEBAND we fall back to original-file values; if IF count
        # changed (e.g. averaging), broadcast/aggregate as needed).
        orig_chw = _fq_field("CH WIDTH", np.array([ch_width_default]))
        orig_tbw = _fq_field("TOTAL BANDWIDTH", orig_chw)
        orig_sb = _fq_field("SIDEBAND", np.array([1], dtype="i4"))

        def _take_or_broadcast(arr, idx, n):
            """Index arr by idx if shape matches, else broadcast first elem."""
            arr = np.atleast_1d(arr)
            if idx.max() < arr.size:
                out = arr[idx]
                if out.size == n:
                    return out
            return np.full(n, arr.flat[0])

        if set_mf:
            # Multi-frequency: each unique freq becomes one IF
            if_freq_new = (ufreq * 1e9 - self.freq0 * 1e9).astype("f8")
            ch_width_new = np.full(nif, float(orig_chw[0]), dtype="f4")
            tot_bw_new = np.full(nif, float(orig_tbw[0]), dtype="f4")
            sideband_new = np.full(nif, int(orig_sb[0]), dtype="i4")
        else:
            select_if = getattr(self, "select_if", "all")
            no_if_original = getattr(self, "no_if_original", nif)
            if isinstance(select_if, (list, tuple)):
                ifs_orig = np.array(select_if) - 1
            elif select_if == "all":
                ifs_orig = np.arange(no_if_original)
            else:
                ifs_orig = np.array([select_if]).astype(int) - 1

            if_freq_new = (
                data_if_freq - self.freq0 * 1e9
            ).astype("f8")

            if nif == ifs_orig.size:
                # Same IF count as selection: use original FQ values
                # (broadcast if defaults are length-1)
                ch_width_new = _take_or_broadcast(
                    orig_chw, ifs_orig, nif
                ).astype("f4")
                tot_bw_new = _take_or_broadcast(
                    orig_tbw, ifs_orig, nif
                ).astype("f4")
                sideband_new = _take_or_broadcast(
                    orig_sb, ifs_orig, nif
                ).astype("i4")
            else:
                # IFs were averaged/changed; synthesize from data
                if ifs_orig.max() < orig_chw.size:
                    summed_chw = float(orig_chw[ifs_orig].sum())
                    summed_tbw = float(orig_tbw[ifs_orig].sum())
                else:
                    summed_chw = float(orig_chw.flat[0]) * nif
                    summed_tbw = float(orig_tbw.flat[0]) * nif
                ch_width_new = np.full(
                    nif, summed_chw / nif, dtype="f4"
                )
                tot_bw_new = np.full(
                    nif, summed_tbw / nif, dtype="f4"
                )
                sideband_new = np.full(
                    nif, int(np.atleast_1d(orig_sb).flat[0]),
                    dtype="i4"
                )

        cols_fq = [
            fits.Column(
                name="FRQSEL", format="1J",
                array=np.array([1], dtype="i4")
            ),
            fits.Column(
                name="IF FREQ", format=f"{nif}D", unit="HZ",
                array=if_freq_new.reshape(1, nif)
            ),
            fits.Column(
                name="CH WIDTH", format=f"{nif}E", unit="HZ",
                array=ch_width_new.reshape(1, nif)
            ),
            fits.Column(
                name="TOTAL BANDWIDTH", format=f"{nif}E", unit="HZ",
                array=tot_bw_new.reshape(1, nif)
            ),
            fits.Column(
                name="SIDEBAND", format=f"{nif}J",
                array=sideband_new.reshape(1, nif)
            ),
        ]
        hdu_fq = fits.BinTableHDU.from_columns(cols_fq)
        hdu_fq.name = "AIPS FQ"
        hdu_fq.header["EXTVER"] = 1
        hdu_fq.header["EXTNAME"] = "AIPS FQ"
        hdu_fq.header["NO_IF"] = nif

        # AIPS NX table (scan index)
        hdu_nx = None
        try:
            scangap = self.gaptime if self.gaptime is not None else 60.0
            scan_inttim = (
                self.avg_timebin if self.avg_timebin is not None else 1.0
            )

            time_sec = (jd_full - jd_ref) * 86400.0
            scan_break = np.zeros(ndata, dtype=bool)
            scan_break[1:] = (
                (np.diff(time_sec) > scangap)
                | (np.diff(freq) != 0)
            )
            scan_starts = np.where(scan_break)[0]
            scan_starts = np.concatenate(([0], scan_starts))
            scan_ends = np.concatenate(
                (scan_starts[1:] - 1, [ndata - 1])
            )
            nscan = len(scan_starts)

            nx_time = np.zeros(nscan, dtype="f8")
            nx_tint = np.zeros(nscan, dtype="f4")
            nx_srcid = np.ones(nscan, dtype="i4")
            nx_subarr = np.ones(nscan, dtype="i4")
            nx_fqid = np.ones(nscan, dtype="i4")
            nx_startvis = np.zeros(nscan, dtype="i4")
            nx_endvis = np.zeros(nscan, dtype="i4")

            for i in range(nscan):
                i1 = scan_starts[i]
                i2 = scan_ends[i]
                t1 = _DATE_par[i1]
                t2 = _DATE_par[i2]
                nx_time[i] = 0.5 * (t1 + t2)
                nx_tint[i] = float(
                    max(t2 - t1, 0.0) + scan_inttim / 86400.0
                )
                nx_startvis[i] = i1 + 1
                nx_endvis[i] = i2 + 1

            cols_nx = [
                fits.Column(
                    name="TIME", format="1D", unit="DAYS", array=nx_time
                ),
                fits.Column(
                    name="TIME INTERVAL", format="1E", unit="DAYS",
                    array=nx_tint
                ),
                fits.Column(
                    name="SOURCE ID", format="1J", array=nx_srcid
                ),
                fits.Column(
                    name="SUBARRAY", format="1J", array=nx_subarr
                ),
                fits.Column(
                    name="FREQ ID", format="1J", array=nx_fqid
                ),
                fits.Column(
                    name="START VIS", format="1J", array=nx_startvis
                ),
                fits.Column(
                    name="END VIS", format="1J", array=nx_endvis
                ),
            ]
            hdu_nx = fits.BinTableHDU.from_columns(cols_nx)
            hdu_nx.name = "AIPS NX"
            hdu_nx.header["EXTVER"] = 1
            hdu_nx.header["EXTNAME"] = "AIPS NX"
        except Exception as e:
            warnings.warn(f"Could not build AIPS NX table: {e}")
            hdu_nx = None

        # Write
        hdul_new = fits.HDUList([hdu_pr, hdu_an, hdu_fq])
        if hdu_nx is not None:
            hdul_new.append(hdu_nx)

        hdul_new.writeto(
            f"{save_path}/{save_name}",
            overwrite=True, output_verify="warn"
        )

    def selfcal(
        self,
        dotype="phs", intervals=None, lm=(0, 0), selfflag=True,
        zero_cp=False, bound_amp=[0.5, 1.5], bound_phs=[-np.pi, np.pi],
        prt=True
    ):
        """
        Self-calibration using model visibility

        Args:
            dotype (str): Self-calibration type
                - availables: 'amp', 'phs', 'a&p', 'gscale', 'startmod'
            intervals (list, float): interval times [minute]
            lm (tuple): Position of the point source
            selfflag (bool): Scans with three stations or less are not
                flagged out
            zero_cp (bool): Assume zero circular polarization
                (RR = LL = Stokes I) and solve antenna gains for each
                polarization independently. If False, solve a single
                antenna gain from the Stokes I visibility
                (= (RR + LL) / 2; the available polarization is used
                for single-polarization data) and apply it to both
                polarizations
            prt (bool): Print summarized information
        """

        # print messages
        availables = ["amp", "phs", "a&p", "gscale", "startmod"]
        if dotype not in availables:
            raise ValueError(
                f"Invalid self-calibration type: {dotype!r}."
                f"Availables: {availables}"
            )

        if prt:
            out_txt = f"\n# Self-calibration (type={dotype!r})"

            if dotype == "startmod":
                out_txt += (
                    f"\tStartmod : Self-calibrating"
                    f" to 1 Jy point source at ({lm[0]}, {lm[1]})"
                )

            print(out_txt)

        def get_gain_amp(_x, _ant1, _ant2):
            _uant = np.unique(np.append(_ant1, _ant2))
            _gamp = dict(zip(_uant, _x))
            _gain1 = np.array(list(map(_gamp.get, _ant1)))
            _gain2 = np.array(list(map(_gamp.get, _ant2)))
            return _gain1, _gain2

        def get_gain_full(_x, _ant1, _ant2):
            _uant = np.unique(np.append(_ant1, _ant2))
            _nant = len(_uant)
            _gamp = dict(zip(_uant, _x[:_nant]))
            _gphs = dict(zip(_uant, _x[_nant:]))
            _gain1 = (
                np.array(list(map(_gamp.get, _ant1)))
                * np.exp(1j * np.array(list(map(_gphs.get, _ant1))))
            )
            _gain2 = (
                np.array(list(map(_gamp.get, _ant2)))
                * np.exp(1j * np.array(list(map(_gphs.get, _ant2))))
            )
            return _gain1, _gain2

        def get_gain_phs(_x, _ant1, _ant2):
            _uant = np.unique(np.append(_ant1, _ant2))
            _gphs = dict(zip(_uant, _x))
            _gain1 = np.exp(1j * np.array(list(map(_gphs.get, _ant1))))
            _gain2 = np.exp(1j * np.array(list(map(_gphs.get, _ant2))))
            return _gain1, _gain2

        def cal_nll_amp(theta, _y, _yerr, _model, idx1, idx2, _nant):
            # antenna gains mapped by integer index (idx1/idx2 precomputed)
            _a1 = theta[idx1]
            _a2 = theta[idx2]
            _mg = _model * (_a1 * _a2)
            _residual = _mg - _y
            _w = _yerr**2
            nll = 0.5 * np.nansum(np.abs(_residual)**2 / _w)
            # gradient via scatter-add (nan -> 0, equivalent to nansum)
            c1 = np.nan_to_num(np.real(_mg / _a1 * _residual.conj()) / _w)
            c2 = np.nan_to_num(np.real(_mg / _a2 * _residual.conj()) / _w)
            grad = np.zeros(_nant)
            np.add.at(grad, idx1, c1)
            np.add.at(grad, idx2, c2)
            return nll, grad

        def cal_nll_full(theta, _y, _yerr, _model, idx1, idx2, _nant):
            # amp in theta[:_nant], phase in theta[_nant:]
            _a1 = theta[idx1]
            _a2 = theta[idx2]
            _p1 = theta[_nant + idx1]
            _p2 = theta[_nant + idx2]
            _gain1 = _a1 * np.exp(1j * _p1)
            _gain2 = _a2 * np.exp(1j * _p2)
            _mg = _model * (_gain1 * _gain2.conj())
            _residual = _mg - _y
            _w = _yerr**2
            nll = 0.5 * np.nansum(np.abs(_residual)**2 / _w)
            # amp/phase gradients via scatter-add (nan -> 0, equiv to nansum)
            cA1 = np.nan_to_num(np.real(_mg / _a1 * _residual.conj()) / _w)
            cA2 = np.nan_to_num(np.real(_mg / _a2 * _residual.conj()) / _w)
            cP1 = np.nan_to_num(np.real( 1j * _mg * _residual.conj()) / _w)
            cP2 = np.nan_to_num(np.real(-1j * _mg * _residual.conj()) / _w)
            grad = np.zeros(2 * _nant)
            np.add.at(grad, idx1, cA1)
            np.add.at(grad, idx2, cA2)
            np.add.at(grad, _nant + idx1, cP1)
            np.add.at(grad, _nant + idx2, cP2)
            return nll, grad

        def cal_nll_phs(theta, _y, _yerr, _model, idx1, idx2, _nant):
            _gain1 = np.exp(1j * theta[idx1])
            _gain2 = np.exp(1j * theta[idx2])
            _mg = _model * (_gain1 * _gain2.conj())
            _residual = _mg - _y
            _w = _yerr**2
            nll = 0.5 * np.nansum(np.abs(_residual)**2 / _w)
            # gradient via scatter-add (nan -> 0, equivalent to nansum)
            c1 = np.nan_to_num(np.real( 1j * _mg * _residual.conj()) / _w)
            c2 = np.nan_to_num(np.real(-1j * _mg * _residual.conj()) / _w)
            grad = np.zeros(_nant)
            np.add.at(grad, idx1, c1)
            np.add.at(grad, idx2, c2)
            return nll, grad

        def set_caltype(nant, dotype, selfflag):
            if nant < 3:    # nant < 3
                doampcal = False
                dophscal = False

            elif nant == 3: # nant = 3
                if selfflag:    # nant = 3 & selfflag = True
                    if dotype == "amp":
                        doampcal = False
                        dophscal = False

                    elif dotype == "phs":
                        doampcal = False
                        dophscal = True

                    elif dotype == "a&p":
                        doampcal = False
                        dophscal = True

                    elif dotype == "gscale":
                        doampcal = True
                        dophscal = False

                    elif dotype == "startmod":
                        doampcal = False
                        dophscal = True

                else:   # nant = 3 & selfflag = False
                    if dotype == "amp":
                        doampcal = True
                        dophscal = False

                    elif dotype == "phs":
                        doampcal = False
                        dophscal = True

                    elif dotype == "a&p":
                        doampcal = True
                        dophscal = True

                    elif dotype == "gscale":
                        doampcal = True
                        dophscal = False

                    elif dotype == "startmod":
                        doampcal = False
                        dophscal = True

            else:   # nant >= 4
                if dotype == "amp":
                    doampcal = True
                    dophscal = False

                elif dotype == "phs":
                    doampcal = False
                    dophscal = True

                elif dotype == "a&p":
                    doampcal = True
                    dophscal = True

                elif dotype == "gscale":
                    doampcal = True
                    dophscal = False

                elif dotype == "startmod":
                    doampcal = False
                    dophscal = True

            return doampcal, dophscal

        # define time interval for self-calibration
        if intervals is None:
            if self.intervals is None:
                intervals = [
                    2880, 1920, 1280, 853, 569, 379, 253, 169, 113, 75, 50,
                    33, 22, 15, 10, 7, 5, 3, 2, 1, 30/60, 10/60, 5/60, 3/60,
                    2/60, 1.5/60, 0.5/60
                ]

                if dotype == "gscale":
                    obs_span = (
                        np.nanmax(self.time) - np.nanmin(self.time)
                    ) * 60

                    intervals = list(filter(
                        lambda x: x >= obs_span / 3,
                        intervals)
                    )

                    if self.avg_timebin is None:
                        tdiff = np.diff(self.time[:, 0, 0] * 3600)
                        tdiff = tdiff[tdiff > 0]
                        t_sep = np.round(tdiff.min()) / 60 / 2
                    else:
                        t_sep = self.avg_timebin / 60 / 2

                    intervals = list(filter(lambda x: x >= t_sep, intervals))
            else:
                intervals = self.intervals

        if isinstance(intervals, list):
            intervals = np.array(intervals)
        else:
            intervals = np.array([intervals])

        dshape = self.data_shape
        if dshape is None:
            print(
                "\nNo data is established. "
                "Run 'set_data()' to load data.\n"
            )

            self.set_data(prt=False)

            dshape = self.data_shape

        freq = self.freq.astype("f4")
        ufreq = np.unique(freq)

        time = self.time.astype("f4")
        utime = np.unique(time)

        self.check_w0()

        # get real term
        r_1 = self.r_1.copy()
        r_2 = self.r_2.copy()
        r_3 = self.r_3.copy()
        r_4 = self.r_4.copy()

        # get imaginary term
        i_1 = self.i_1.copy()
        i_2 = self.i_2.copy()
        i_3 = self.i_3.copy()
        i_4 = self.i_4.copy()

        # get weight term
        w_1 = self.w_1.copy()
        w_2 = self.w_2.copy()
        w_3 = self.w_3.copy()
        w_4 = self.w_4.copy()

        # get weight0 term
        w0_1 = self.w0_1.copy()
        w0_2 = self.w0_2.copy()
        w0_3 = self.w0_3.copy()
        w0_4 = self.w0_4.copy()

        # get antennas
        ant1 = self.ant1.copy()
        ant2 = self.ant2.copy()

        mask_cg = (
            (self.cg_pol1_ant1 is None)
            & (self.cg_pol1_ant2 is None)
            & (self.cg_pol2_ant1 is None)
            & (self.cg_pol2_ant2 is None)
        )

        if mask_cg:
            out_cg_pol1_ant1 = np.ones(dshape) * np.exp(1j * 0)
            out_cg_pol1_ant2 = np.ones(dshape) * np.exp(1j * 0)
            out_cg_pol2_ant1 = np.ones(dshape) * np.exp(1j * 0)
            out_cg_pol2_ant2 = np.ones(dshape) * np.exp(1j * 0)
        else:
            out_cg_pol1_ant1 = self.cg_pol1_ant1.copy()
            out_cg_pol1_ant2 = self.cg_pol1_ant2.copy()
            out_cg_pol2_ant1 = self.cg_pol2_ant1.copy()
            out_cg_pol2_ant2 = self.cg_pol2_ant2.copy()

        # scan numbers depend only on time (invariant over freq/interval loops)
        scans, scans_1d = self.set_scan(
            time=time * 3600.0, gaptime=self.gaptime,
            scanlen=self.scanlen, returned=True
        )
        scans_1d = scans_1d.reshape(dshape)

        # by frequency
        for nf, f in enumerate(ufreq):
            if_idx = np.where(freq[0, :, 0].astype("f4") == f)[0][0]
            mask_freq = (freq == f)

            if prt:
                print(
                    f"\t calibrating {nf + 1}/{len(ufreq)} IF channels "
                    f"({f:.3f} GHz)"
                )

            # by time bin
            for nt, t in enumerate(intervals):
                vis_1 = r_1 + 1j * i_1
                vis_2 = r_2 + 1j * i_2
                vis_3 = r_3 + 1j * i_3
                vis_4 = r_4 + 1j * i_4

                _out_cg_pol1_ant1 = np.ones(dshape) * np.exp(1j * 0)
                _out_cg_pol1_ant2 = np.ones(dshape) * np.exp(1j * 0)
                _out_cg_pol2_ant1 = np.ones(dshape) * np.exp(1j * 0)
                _out_cg_pol2_ant2 = np.ones(dshape) * np.exp(1j * 0)

                # by time bin (absolute time across the whole observation)
                time_norm = (time - time.min()) * 60
                time_range = np.arange(0, time_norm.max() + 2 * t, t)
                ntime = len(time_range)

                for _ntime in range(ntime - 1):
                    mask_time = (
                        (time_range[_ntime + 0] <= time_norm)
                        & (time_norm < time_range[_ntime + 1])
                    )

                    mask = (mask_freq & mask_time)

                    # skip if no data
                    if mask.sum() == 0:
                        continue

                    # get complex visibility & visibility sigma
                    _vis1 = r_1[mask] + 1j * i_1[mask]
                    _vis2 = r_2[mask] + 1j * i_2[mask]

                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            category=RuntimeWarning
                        )

                        _sig1 = np.sqrt(1 / w0_1[mask])
                        _sig2 = np.sqrt(1 / w0_2[mask])

                    # get unique antennas & number of antennas
                    _ant1 = ant1[mask]
                    _ant2 = ant2[mask]
                    _uant = np.unique(
                        np.append(_ant1, _ant2)
                    )

                    _nant = len(_uant)

                    # number of scans falling in this time bin
                    _nscan = len(np.unique(scans_1d[mask]))

                    # map antennas to integer indices (for vectorized NLL)
                    idx1 = np.searchsorted(_uant, _ant1)
                    idx2 = np.searchsorted(_uant, _ant2)

                    # set calibration types
                    doampcal, dophscal = set_caltype(
                        _nant, dotype, selfflag
                    )

                    # a time bin with fewer than 3 scans and fewer than 4
                    # antennas is too poorly constrained for amplitude cal;
                    # fall back to phase-only there.
                    if doampcal and _nscan < 3 and _nant < 4:
                        doampcal = False

                    if not doampcal and not dophscal:
                        continue

                    # define model visibility
                    if dotype == "startmod":
                        _vism = np.ones(dshape, dtype="c8")
                    else:
                        if self.vism is None:
                            raise ValueError(
                                "Invalid model visibility is given."
                            )
                        else:
                            _vism = self.vism

                    if doampcal and dophscal:
                        nll_func = (
                            _selfcal_nll_full if _HAS_NUMBA
                            else cal_nll_full
                        )
                        get_gain = get_gain_full
                        init = np.append(
                            np.ones(_nant), np.zeros(_nant)
                        )
                        bounds = [
                            [bound_amp[0], bound_amp[1]]
                            for _ in range(_nant)
                        ] + [
                            [bound_phs[0], bound_phs[1]]
                            for _ in range(_nant)
                        ]
                    elif dophscal:
                        nll_func = (
                            _selfcal_nll_phs if _HAS_NUMBA
                            else cal_nll_phs
                        )
                        get_gain = get_gain_phs
                        init = np.zeros(_nant)
                        bounds = [
                            [bound_phs[0], bound_phs[1]]
                            for _ in range(_nant)
                        ]
                    else:
                        nll_func = (
                            _selfcal_nll_amp if _HAS_NUMBA
                            else cal_nll_amp
                        )
                        get_gain = get_gain_amp
                        init = np.ones(_nant)
                        bounds = [
                            [bound_amp[0], bound_amp[1]]
                            for _ in range(_nant)
                        ]

                    if zero_cp or self.nstokes < 2:
                        # assume zero circular polarization (RR = LL);
                        # solve antenna gains per polarization
                        args_pol1 = (
                            _vis1.astype("c16"), _sig1.astype("f8"),
                            _vism[mask].astype("c16"), idx1, idx2, _nant
                        )

                        soln_pol1 = optimize.minimize(
                            nll_func, init, args=args_pol1,
                            bounds=bounds, method="L-BFGS-B", jac=True
                        )

                        if self.nstokes >= 2:
                            args_pol2 = (
                                _vis2.astype("c16"), _sig2.astype("f8"),
                                _vism[mask].astype("c16"), idx1, idx2, _nant
                            )
                            soln_pol2 = optimize.minimize(
                                nll_func, init, args=args_pol2,
                                bounds=bounds, method="L-BFGS-B", jac=True
                            )

                        _cg_pol1 = get_gain(soln_pol1.x, _ant1, _ant2)
                        _out_cg_pol1_ant1[mask] = _cg_pol1[0]
                        _out_cg_pol1_ant2[mask] = _cg_pol1[1]

                        if self.nstokes >= 2:
                            _cg_pol2 = get_gain(soln_pol2.x, _ant1, _ant2)
                            _out_cg_pol2_ant1[mask] = _cg_pol2[0]
                            _out_cg_pol2_ant2[mask] = _cg_pol2[1]
                    else:
                        # solve a single antenna gain from Stokes I
                        # (= (RR + LL) / 2) and apply it to both
                        # polarizations; samples with one invalid
                        # polarization fall back to the valid one
                        _bad1 = np.isnan(_vis1) | ~np.isfinite(_sig1)
                        _bad2 = np.isnan(_vis2) | ~np.isfinite(_sig2)

                        _visi = np.where(
                            _bad1, _vis2,
                            np.where(
                                _bad2, _vis1,
                                0.5 * (_vis1 + _vis2)
                            )
                        )
                        _sigi = np.where(
                            _bad1, _sig2,
                            np.where(
                                _bad2, _sig1,
                                0.5 * np.sqrt(_sig1**2 + _sig2**2)
                            )
                        )

                        args_poli = (
                            _visi.astype("c16"), _sigi.astype("f8"),
                            _vism[mask].astype("c16"), idx1, idx2, _nant
                        )

                        soln_poli = optimize.minimize(
                            nll_func, init, args=args_poli,
                            bounds=bounds, method="L-BFGS-B", jac=True
                        )

                        _cg_poli = get_gain(soln_poli.x, _ant1, _ant2)
                        _out_cg_pol1_ant1[mask] = _cg_poli[0]
                        _out_cg_pol1_ant2[mask] = _cg_poli[1]
                        _out_cg_pol2_ant1[mask] = _cg_poli[0]
                        _out_cg_pol2_ant2[mask] = _cg_poli[1]

                vis_1[:, if_idx, 0] /= (
                    _out_cg_pol1_ant1[:, if_idx, 0]
                    * _out_cg_pol1_ant2[:, if_idx, 0].conj()
                )

                vis_2[:, if_idx, 0] /= (
                    _out_cg_pol2_ant1[:, if_idx, 0]
                    * _out_cg_pol2_ant2[:, if_idx, 0].conj()
                )

                vis_3[:, if_idx, 0] /= (
                    _out_cg_pol1_ant1[:, if_idx, 0]
                    * _out_cg_pol2_ant2[:, if_idx, 0].conj()
                )

                vis_4[:, if_idx, 0] /= (
                    _out_cg_pol2_ant1[:, if_idx, 0]
                    * _out_cg_pol1_ant2[:, if_idx, 0].conj()
                )

                # real term
                r_1[:, if_idx, 0] = vis_1[:, if_idx, 0].real
                r_2[:, if_idx, 0] = vis_2[:, if_idx, 0].real
                r_3[:, if_idx, 0] = vis_3[:, if_idx, 0].real
                r_4[:, if_idx, 0] = vis_4[:, if_idx, 0].real

                # imaginary term
                i_1[:, if_idx, 0] = vis_1[:, if_idx, 0].imag
                i_2[:, if_idx, 0] = vis_2[:, if_idx, 0].imag
                i_3[:, if_idx, 0] = vis_3[:, if_idx, 0].imag
                i_4[:, if_idx, 0] = vis_4[:, if_idx, 0].imag

                # weight term
                w_1[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol1_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol1_ant2[:, if_idx, 0])
                )**2
                w_2[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol2_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol2_ant2[:, if_idx, 0])
                )**2
                w_3[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol1_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol2_ant2[:, if_idx, 0])
                )**2
                w_4[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol2_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol1_ant2[:, if_idx, 0])
                )**2

                # weight0 term
                w0_1[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol1_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol1_ant2[:, if_idx, 0])
                )**2
                w0_2[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol2_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol2_ant2[:, if_idx, 0])
                )**2
                w0_3[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol1_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol2_ant2[:, if_idx, 0])
                )**2
                w0_4[:, if_idx, 0] *= (
                    np.abs(_out_cg_pol2_ant1[:, if_idx, 0])
                    * np.abs(_out_cg_pol1_ant2[:, if_idx, 0])
                )**2

                out_cg_pol1_ant1[:, if_idx, 0] /= (
                    _out_cg_pol1_ant1[:, if_idx, 0]
                )
                out_cg_pol1_ant2[:, if_idx, 0] /= (
                    _out_cg_pol1_ant2[:, if_idx, 0]
                )
                out_cg_pol2_ant1[:, if_idx, 0] /= (
                    _out_cg_pol2_ant1[:, if_idx, 0]
                )
                out_cg_pol2_ant2[:, if_idx, 0] /= (
                    _out_cg_pol2_ant2[:, if_idx, 0]
                )

        if self.nstokes == 1:
            out_cg_pol2_ant1 *= np.nan
            out_cg_pol2_ant2 *= np.nan

        # assign real term
        self.r_1 = r_1
        self.r_2 = r_2
        self.r_3 = r_3
        self.r_4 = r_4

        # assign imaginary term
        self.i_1 = i_1
        self.i_2 = i_2
        self.i_3 = i_3
        self.i_4 = i_4

        # assign weight term
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.w_4 = w_4

        # assign weight0 term
        self.w0_1 = w0_1
        self.w0_2 = w0_2
        self.w0_3 = w0_3
        self.w0_4 = w0_4

        self.cg_pol1_ant1 = out_cg_pol1_ant1
        self.cg_pol1_ant2 = out_cg_pol1_ant2
        self.cg_pol2_ant1 = out_cg_pol2_ant1
        self.cg_pol2_ant2 = out_cg_pol2_ant2

        self.set_data(prt=False)

    def set_beamprm(self, bprms):
        """
        Set user-defined beam parameters
        Args:
            bprms (list, tuple): Beam parameters to set
                - bprms[0] (float): FWHM of the minor axis
                - bprms[1] (float): FWHM of the major axis
                - bprms[2] (float): Position angle of the beam (degrees)
        """

        self.bprms = bprms
        self.bmaj = bprms[0]
        self.bmin = bprms[1]
        self.bpa = bprms[2]
        self.ploter.bprms = bprms

    def set_closure(self, minclq=None):
        """
        Set the closure quantities of minimal complete

        Args:
            minclq (bool): Compute full closure
        """

        if minclq is None:
            minclq = self.minclq

        if not minclq:
            self.set_closure_full()
        else:
            if self.data is None:
                self.set_data(prt=False)

            time = np.asarray(self.get_data("time")).flatten()
            freq = np.asarray(self.get_data("frequency")).flatten()
            ant1 = self.get_data("ant1").flatten()
            ant2 = self.get_data("ant2").flatten()
            u = self.get_data("u").flatten()
            v = self.get_data("v").flatten()
            vis = self.get_data("vis").flatten()
            sig = self.get_data("sig").flatten()

            valid = (
                np.isfinite(vis.real)
                & np.isfinite(vis.imag)
                & (np.abs(vis) > 0)
                & np.isfinite(sig)
                & (sig > 0)
                & np.isfinite(u)
                & np.isfinite(v)
            )

            utime = np.unique(time)
            ufreq = np.unique(freq)

            field_cav = f"clamp"
            field_cas = f"sig_logclamp"
            field_cpv = f"clphs"
            field_cps = f"sig_clphs"

            uvvis = {}
            uvsig = {}

            valid_keys = zip(
                time[valid].tolist(), freq[valid].tolist(),
                u[valid].tolist(), v[valid].tolist()
            )

            for key, _vis, _sig in zip(
                valid_keys, vis[valid], sig[valid]
            ):
                uvvis.setdefault(key, _vis)
                uvsig.setdefault(key, _sig)

            # set field & dtype for closure amplitude
            field_amp = [
                "time", "freq", "quadrangle",
                "u12", "v12", "u34", "v34", "u13", "v13", "u24", "v24",
                "vis12", "vis34", "vis13", "vis24",
                "sig12", "sig34", "sig13", "sig24",
                field_cav, field_cas
            ]

            dtype_amp = [
                "f8", "f8", "U32",
                "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8",
                "c16", "c16", "c16", "c16",
                "f8", "f8", "f8", "f8",
                "f8", "f8"
            ]

            # set field & dtype for closure phase
            field_phs = [
                "time", "freq", "triangle",
                "u12", "v12", "u23", "v23", "u31", "v31",
                "vis12", "vis23", "vis31",
                "sig12", "sig23", "sig31",
                field_cpv, field_cps
            ]

            dtype_phs = [
                "f8", "f8", "U32",
                "f8", "f8", "f8", "f8", "f8", "f8",
                "c16", "c16", "c16",
                "f8", "f8", "f8",
                "f8", "f8"
            ]

            clamp_records = []
            clphs_records = []
            clamp_basis_cache = {}
            clphs_basis_cache = {}

            # set closure amplitude & phase for each time/freq pair
            for nt, _time in enumerate(tqdm(
                utime,
                desc="Setting closure relations",
                leave=False,
                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}"
            )):
                for nf, _freq in enumerate(ufreq):
                    mask_time = time == _time
                    mask_freq = freq == _freq
                    mask = mask_time & mask_freq & valid

                    _ant1 = ant1[mask]
                    _ant2 = ant2[mask]
                    _u = u[mask]
                    _v = v[mask]
                    _pairs = np.sort(
                        np.stack([_ant1, _ant2], axis=1), axis=1
                    )

                    _uant = np.unique(np.append(_ant1, _ant2))
                    _nant = len(_uant)

                    # set closure amplitude if more than three antennas
                    if _nant >= 4:
                        pairs_obs = np.unique(_pairs, axis=0)
                        basis_key = (
                            tuple(_uant.tolist()),
                            tuple(map(tuple, pairs_obs.tolist()))
                        )
                        if basis_key not in clamp_basis_cache:
                            clamp_basis_cache[basis_key] = (
                                set_min_matrix_clamp(
                                    _nant, _uant, pairs_obs=pairs_obs,
                                    return_relations=True
                                )
                            )

                        _, _, relations = clamp_basis_cache[basis_key]
                        for pair_add, pair_sub, pair_ants in relations:
                            pair_add = np.asarray(pair_add)
                            pair_sub = np.asarray(pair_sub)
                            pair_ants = np.asarray(pair_ants)

                            mask_uv1 = np.all(
                                _pairs == pair_add[0], axis=1
                            )
                            mask_uv2 = np.all(
                                _pairs == pair_add[1], axis=1
                            )
                            mask_uv3 = np.all(
                                _pairs == pair_sub[0], axis=1
                            )
                            mask_uv4 = np.all(
                                _pairs == pair_sub[1], axis=1
                            )

                            if not all((
                                mask_uv1.any(), mask_uv2.any(),
                                mask_uv3.any(), mask_uv4.any()
                            )):
                                continue

                            out_quadrangle = "-".join(list(map(
                                self.ant_dict_num2name.get, pair_ants
                            )))
                            out_uv1 = (_u[mask_uv1][0], _v[mask_uv1][0])
                            out_uv2 = (_u[mask_uv2][0], _v[mask_uv2][0])
                            out_uv3 = (_u[mask_uv3][0], _v[mask_uv3][0])
                            out_uv4 = (_u[mask_uv4][0], _v[mask_uv4][0])

                            record = gv.utils.structured_array(
                                data=[
                                    _time, _freq, out_quadrangle,
                                    out_uv1[0], out_uv1[1],
                                    out_uv2[0], out_uv2[1],
                                    out_uv3[0], out_uv3[1],
                                    out_uv4[0], out_uv4[1],
                                    0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0
                                ],
                                field=field_amp,
                                dtype=dtype_amp
                            )
                            clamp_records.append(record.reshape(-1)[0])
                    if _nant >= 3:
                        pairs_obs = np.unique(_pairs, axis=0)
                        basis_key = (
                            tuple(_uant.tolist()),
                            tuple(map(tuple, pairs_obs.tolist()))
                        )
                        if basis_key not in clphs_basis_cache:
                            clphs_basis_cache[basis_key] = (
                                set_min_matrix_clphs(
                                    _nant, _uant, pairs_obs=pairs_obs,
                                    return_relations=True
                                )
                            )

                        _, _, relations = clphs_basis_cache[basis_key]
                        for pair_add, pair_sub, pair_ants in relations:
                            pair_add = np.asarray(pair_add)
                            pair_sub = np.asarray(pair_sub)
                            pair_ants = np.asarray(pair_ants)

                            mask_uv1 = np.all(
                                _pairs == pair_add[0], axis=1
                            )
                            mask_uv2 = np.all(
                                _pairs == pair_add[1], axis=1
                            )
                            mask_uv3 = np.all(
                                _pairs == pair_sub[0], axis=1
                            )
                            if not all((
                                mask_uv1.any(), mask_uv2.any(),
                                mask_uv3.any()
                            )):
                                continue

                            out_triangle = "-".join(list(map(
                                self.ant_dict_num2name.get, pair_ants
                            )))
                            out_uv1 = (_u[mask_uv1][0], _v[mask_uv1][0])
                            out_uv2 = (_u[mask_uv2][0], _v[mask_uv2][0])
                            out_uv3 = (_u[mask_uv3][0], _v[mask_uv3][0])

                            record = gv.utils.structured_array(
                                data=[
                                    _time, _freq, out_triangle,
                                    out_uv1[0], out_uv1[1],
                                    out_uv2[0], out_uv2[1],
                                    out_uv3[0], out_uv3[1],
                                    0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0
                                ],
                                field=field_phs,
                                dtype=dtype_phs
                            )
                            clphs_records.append(record.reshape(-1)[0])

            flag_clamp = False
            flag_clphs = False

            dtype_amp_np = np.dtype(list(zip(field_amp, dtype_amp)))
            dtype_phs_np = np.dtype(list(zip(field_phs, dtype_phs)))
            tmpl_clamp = np.asarray(
                clamp_records, dtype=dtype_amp_np
            ).reshape(-1)
            tmpl_clphs = np.asarray(
                clphs_records, dtype=dtype_phs_np
            ).reshape(-1)

            if tmpl_clamp.size == 0:
                tmpl_clamp = [np.nan] * len(field_amp)
                tmpl_clamp = gv.utils.structured_array(
                    data=tmpl_clamp,
                    field=field_amp,
                    dtype=dtype_amp
                )

                flag_clamp = True
            if tmpl_clphs.size == 0:
                tmpl_clphs = [np.nan] * len(field_phs)
                tmpl_clphs = gv.utils.structured_array(
                    data=tmpl_clphs,
                    field=field_phs,
                    dtype=dtype_phs
                )

                flag_clphs = True

            if not flag_clamp:
                clamp_ = tmpl_clamp.copy()
                clamp_["vis12"] = list(map(
                    uvvis.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u12"], clamp_["v12"]
                    ))
                ))

                clamp_["vis34"] = list(map(
                    uvvis.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u34"], clamp_["v34"]
                    ))
                ))

                clamp_["vis13"] = list(map(
                    uvvis.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u13"], clamp_["v13"]
                    ))
                ))

                clamp_["vis24"] = list(map(
                    uvvis.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u24"], clamp_["v24"]
                    ))
                ))

                clamp_["sig12"] = list(map(
                    uvsig.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u12"], clamp_["v12"]
                    ))
                ))

                clamp_["sig34"] = list(map(
                    uvsig.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u34"], clamp_["v34"]
                    ))
                ))

                clamp_["sig13"] = list(map(
                    uvsig.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u13"], clamp_["v13"]
                    ))
                ))

                clamp_["sig24"] = list(map(
                    uvsig.get,
                    tuple(zip(
                        clamp_["time"], clamp_["freq"],
                        clamp_["u24"], clamp_["v24"]
                    ))
                ))

                vis12 = np.array(clamp_["vis12"], dtype="c16")
                vis34 = np.array(clamp_["vis34"], dtype="c16")
                vis13 = np.array(clamp_["vis13"], dtype="c16")
                vis24 = np.array(clamp_["vis24"], dtype="c16")

                sig12 = np.array(clamp_["sig12"])
                sig34 = np.array(clamp_["sig34"])
                sig13 = np.array(clamp_["sig13"])
                sig24 = np.array(clamp_["sig24"])

                amp12 = np.abs(vis12)
                amp34 = np.abs(vis34)
                amp13 = np.abs(vis13)
                amp24 = np.abs(vis24)

                with np.errstate(divide="ignore", invalid="ignore"):
                    snr12 = amp12 / sig12
                    snr34 = amp34 / sig34
                    snr13 = amp13 / sig13
                    snr24 = amp24 / sig24
                    clamp_[field_cav] = (
                        (amp12 * amp34) / (amp13 * amp24)
                    )
                    clamp_[field_cas] = (
                        snr12**-2 + snr34**-2
                        + snr13**-2 + snr24**-2
                    )**0.5

            if not flag_clphs:
                clphs_ = tmpl_clphs.copy()
                clphs_["vis12"] = list(map(
                    uvvis.get, tuple(zip(
                        clphs_["time"], clphs_["freq"],
                        clphs_["u12"], clphs_["v12"]
                    ))
                ))

                clphs_["vis23"] = list(map(
                    uvvis.get, tuple(zip(
                        clphs_["time"], clphs_["freq"],
                        clphs_["u23"], clphs_["v23"]
                    ))
                ))

                clphs_["vis31"] = list(map(
                    uvvis.get, tuple(zip(
                        clphs_["time"], clphs_["freq"],
                        clphs_["u31"], clphs_["v31"]
                    ))
                ))

                clphs_["sig12"] = list(map(
                    uvsig.get, tuple(zip(
                        clphs_["time"], clphs_["freq"],
                        clphs_["u12"], clphs_["v12"]
                    ))
                ))

                clphs_["sig23"] = list(map(
                    uvsig.get, tuple(zip(
                        clphs_["time"], clphs_["freq"],
                        clphs_["u23"], clphs_["v23"]
                    ))
                ))

                clphs_["sig31"] = list(map(
                    uvsig.get, tuple(zip(
                        clphs_["time"], clphs_["freq"],
                        clphs_["u31"], clphs_["v31"]
                    ))
                ))

                vis12 = np.array(clphs_["vis12"], dtype="c16")
                vis23 = np.array(clphs_["vis23"], dtype="c16")
                vis31 = np.array(clphs_["vis31"], dtype="c16").conj()

                with np.errstate(divide="ignore", invalid="ignore"):
                    snr12 = np.abs(vis12) / np.abs(clphs_["sig12"])
                    snr23 = np.abs(vis23) / np.abs(clphs_["sig23"])
                    snr31 = np.abs(vis31) / np.abs(clphs_["sig31"])

                    bispectrum = vis12 * vis23 * vis31
                    clphs_[field_cpv] = np.angle(bispectrum)
                    clphs_[field_cps] = (
                        snr12**-2 + snr23**-2 + snr31**-2
                    )**0.5

            if not flag_clamp:
                valid_clamp = (
                    np.isfinite(clamp_[field_cav])
                    & (clamp_[field_cav] > 0)
                    & np.isfinite(clamp_[field_cas])
                    & (clamp_[field_cas] > 0)
                )
                tmpl_clamp = tmpl_clamp[valid_clamp]
                clamp_ = clamp_[valid_clamp]
                if tmpl_clamp.size == 0:
                    tmpl_clamp = gv.utils.structured_array(
                        data=[np.nan] * len(field_amp),
                        field=field_amp,
                        dtype=dtype_amp
                    )
                    flag_clamp = True

            if not flag_clphs:
                valid_clphs = (
                    np.isfinite(clphs_[field_cpv])
                    & np.isfinite(clphs_[field_cps])
                    & (clphs_[field_cps] > 0)
                )
                tmpl_clphs = tmpl_clphs[valid_clphs]
                clphs_ = clphs_[valid_clphs]
                if tmpl_clphs.size == 0:
                    tmpl_clphs = gv.utils.structured_array(
                        data=[np.nan] * len(field_phs),
                        field=field_phs,
                        dtype=dtype_phs
                    )
                    flag_clphs = True

            self.tmpl_clamp = copy.deepcopy(tmpl_clamp)
            self.tmpl_clphs = copy.deepcopy(tmpl_clphs)

            if not flag_clamp:
                fields = ["time", "quadrangle", "freq", field_cav, field_cas]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                datas = [clamp_[fields[nf]] for nf in range(len(fields))]
                clamp_ = gv.utils.structured_array(
                    data=datas,
                    field=fields,
                    dtype=dtypes
                )

            else:
                fields = ["time", "quadrangle", "freq", field_cav, field_cas]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                clamp_ = gv.utils.structured_array(
                    data=[np.nan for i in range(len(fields))],
                    field=fields,
                    dtype=dtypes
                )

            if not flag_clphs:
                fields = ["time", "triangle", "freq", field_cpv, field_cps]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                datas = [clphs_[fields[nf]] for nf in range(len(fields))]
                clphs_ = gv.utils.structured_array(
                    data=datas,
                    field=fields,
                    dtype=dtypes
                )
            else:
                fields = ["time", "triangle", "freq", field_cpv, field_cps]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                clphs_ = gv.utils.structured_array(
                    data=[np.nan for i in range(len(fields))],
                    field=fields,
                    dtype=dtypes
                )

            clamp = clamp_
            clphs = clphs_

            if not flag_clamp:
                self.clamp = clamp
                self.clamp_check = True

            else:
                fields = ["time", "quadrangle", "freq", "clamp", "sig_logclamp"]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]

                self.clamp = gv.utils.structured_array(
                    data=[np.nan for i in range(len(fields))],
                    field=fields,
                    dtype=dtypes
                )

                self.clamp_check = False

            if not flag_clphs:
                self.clphs = clphs
                self.clphs_check = True

            else:
                fields = ["time", "triangle", "freq", "clphs", "sig_clphs"]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]

                self.clphs = gv.utils.structured_array(
                    data=[np.nan for i in range(len(fields))],
                    field=fields,
                    dtype=dtypes
                )

                self.clphs_check = False

            self.ploter.clq_obs = (
                copy.deepcopy(self.clamp),
                copy.deepcopy(self.clphs)
            )

    def set_closure_full(self):
        """
        Set full closure quantities
        """

        if "vis" not in self.data.dtype.names:
            self.set_uvvis()

        data = self.data
        tarr = self.tarr

        times = np.unique(data["time"])

        torder_num = {
            int(idx + 1): idx
            for idx, tel in enumerate(tarr["name"])
        }
        torder_name = {tel: idx for idx, tel in enumerate(tarr["name"])}

        ant_nums = np.unique(np.append(data["ant1"], data["ant2"]))

        field_vis = f"vis"
        field_sig = f"sig"
        field_cav = f"clamp"
        field_cas = f"sig_logclamp"
        field_cpv = f"clphs"
        field_cps = f"sig_clphs"

        uvvis = dict(zip(
            tuple(zip(data["u"].tolist(), data["v"].tolist())),
            data[field_vis]
        ))

        uvsig = dict(zip(
            tuple(zip(data["u"].tolist(), data["v"].tolist())),
            data[field_sig]
        ))

        utimes = np.unique(data["time"])
        ufreqs = np.unique(data["frequency"])

        field_amp = [
            "time", "freq", "quadrangle",
            "u12", "v12", "u34", "v34", "u13", "v13", "u24", "v24",
            "vis12", "vis34", "vis13", "vis24",
            "sig12", "sig34", "sig13", "sig24",
            field_cav, field_cas
        ]

        dtype_amp = [
            "f8", "f8", "U32",
            "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8",
            "c8", "c8", "c8", "c8",
            "f8", "f8", "f8", "f8",
            "f8", "f8"
        ]

        field_phs = [
            "time", "freq", "triangle",
            "u12", "v12", "u23", "v23", "u31", "v31",
            "vis12", "vis23", "vis31",
            "sig12", "sig23", "sig31",
            field_cpv, field_cps
        ]

        dtype_phs = [
            "f8", "f8", "U32",
            "f8", "f8", "f8", "f8", "f8", "f8",
            "c8", "c8", "c8",
            "f8", "f8", "f8",
            "f8", "f8"
        ]

        outca_time, outca_freq, outca_quad = [], [], []
        outca_u12, outca_v12 = [], []
        outca_u34, outca_v34 = [], []
        outca_u13, outca_v13 = [], []
        outca_u24, outca_v24 = [], []

        outcp_time, outcp_freq, outcp_tri = [], [], []
        outcp_u12, outcp_v12 = [], []
        outcp_u23, outcp_v23 = [], []
        outcp_u31, outcp_v31 = [], []

        flag_clamp = False
        flag_clphs = False

        for ut, time in enumerate(tqdm(
            utimes,
            desc="Setting closure relations",
            leave=False,
            bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}"
        )):
            for freq in ufreqs:
                mask_time = (
                    (data["time"] == time) &
                    (data["frequency"] == freq)
                )
                data_ = data[mask_time]

                if len(data_) == 0:
                    continue

                ant_nums_ = np.array(sorted(
                    np.unique(
                        np.append(data_["ant1"], data_["ant2"])
                    ),
                    key=lambda x: torder_num[x]
                ))

                nant_ = len(ant_nums_)
                if nant_ >= 4:
                    quadrangles_num = list(
                        it.combinations(ant_nums_.tolist(), 4)
                    )
                    for nquad, quadrangle in enumerate(quadrangles_num):
                        for nvert in range(2):
                            if nvert == 0:
                                quadra_ = [
                                    quadrangles_num[nquad][0],
                                    quadrangles_num[nquad][1],
                                    quadrangles_num[nquad][2],
                                    quadrangles_num[nquad][3]
                                ]

                                mask_uv12 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[0])
                                    & (data["ant2"] == quadrangle[1])
                                )

                                mask_uv34 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[2])
                                    & (data["ant2"] == quadrangle[3])
                                )

                                mask_uv13 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[0])
                                    & (data["ant2"] == quadrangle[2])
                                )

                                mask_uv24 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[1])
                                    & (data["ant2"] == quadrangle[3])
                                )

                            else:
                                quadra_ = [
                                    quadrangles_num[nquad][0],
                                    quadrangles_num[nquad][2],
                                    quadrangles_num[nquad][1],
                                    quadrangles_num[nquad][3]
                                ]

                                mask_uv12 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[0])
                                    & (data["ant2"] == quadrangle[2])
                                )

                                mask_uv34 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[1])
                                    & (data["ant2"] == quadrangle[3])
                                )

                                mask_uv13 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[0])
                                    & (data["ant2"] == quadrangle[3])
                                )

                                mask_uv24 = (
                                    (mask_time)
                                    & (data["ant1"] == quadrangle[1])
                                    & (data["ant2"] == quadrangle[2])
                                )

                            mask_uv1234 = (
                                np.any(mask_uv12)
                                and np.any(mask_uv34)
                                and np.any(mask_uv13)
                                and np.any(mask_uv24)
                            )

                            if mask_uv1234:
                                out_quadrangle = "-".join(
                                    list(map(
                                        self.ant_dict_num2name.get,
                                        quadra_
                                    ))
                                )

                                out_uv12 = (
                                    data["u"][mask_uv12][0],
                                    data["v"][mask_uv12][0]
                                )

                                out_uv34 = (
                                    data["u"][mask_uv34][0],
                                    data["v"][mask_uv34][0]
                                )

                                out_uv13 = (
                                    data["u"][mask_uv13][0],
                                    data["v"][mask_uv13][0]
                                )

                                out_uv24 = (
                                    data["u"][mask_uv24][0],
                                    data["v"][mask_uv24][0]
                                )

                                outca_time.append(time)
                                outca_freq.append(freq)
                                outca_quad.append(out_quadrangle)
                                outca_u12.append(out_uv12[0])
                                outca_v12.append(out_uv12[1])
                                outca_u34.append(out_uv34[0])
                                outca_v34.append(out_uv34[1])
                                outca_u13.append(out_uv13[0])
                                outca_v13.append(out_uv13[1])
                                outca_u24.append(out_uv24[0])
                                outca_v24.append(out_uv24[1])

                if nant_ >= 3:
                    triangles_num = list(
                        it.combinations(ant_nums_.tolist(), 3)
                    )

                    for ntri, triangle in enumerate(triangles_num):

                        mask_uv12 = (
                            (mask_time)
                            & (data["ant1"] == triangle[0])
                            & (data["ant2"] == triangle[1])
                        )

                        mask_uv23 = (
                            (mask_time)
                            & (data["ant1"] == triangle[1])
                            & (data["ant2"] == triangle[2])
                        )

                        mask_uv31 = (
                            (mask_time)
                            & (data["ant1"] == triangle[0])
                            & (data["ant2"] == triangle[2])
                        )

                        mask_uv123 = (
                            np.any(mask_uv12)
                            and np.any(mask_uv23)
                            and np.any(mask_uv31)
                        )

                        if mask_uv123:
                            out_frequency = freq
                            out_triangle = "-".join(
                                list(map(
                                    self.ant_dict_num2name.get,
                                    triangle
                                ))
                            )

                            out_uv12 = (
                                data["u"][mask_uv12][0],
                                data["v"][mask_uv12][0]
                            )

                            out_uv23 = (
                                data["u"][mask_uv23][0],
                                data["v"][mask_uv23][0]
                            )

                            out_uv31 = (
                                data["u"][mask_uv31][0],
                                data["v"][mask_uv31][0]
                            )

                            outcp_time.append(time)
                            outcp_freq.append(out_frequency)
                            outcp_tri.append(out_triangle)
                            outcp_u12.append(out_uv12[0])
                            outcp_v12.append(out_uv12[1])
                            outcp_u23.append(out_uv23[0])
                            outcp_v23.append(out_uv23[1])
                            outcp_u31.append(out_uv31[0])
                            outcp_v31.append(out_uv31[1])

        if len(outca_time) == 0:
            tmpl_clamp = gv.utils.structured_array(
                data=[[np.nan] for i in range(len(field_amp))],
                field=field_amp,
                dtype=dtype_amp
            )

            flag_clamp = True

        else:
            tmpl_clamp = gv.utils.structured_array(
                data=[
                    outca_time, outca_freq, outca_quad,
                    outca_u12, outca_v12, outca_u34, outca_v34,
                    outca_u13, outca_v13, outca_u24, outca_v24,
                    np.zeros(len(outca_time), dtype="c8"),
                    np.zeros(len(outca_time), dtype="c8"),
                    np.zeros(len(outca_time), dtype="c8"),
                    np.zeros(len(outca_time), dtype="c8"),
                    np.zeros(len(outca_time), dtype="f8"),
                    np.zeros(len(outca_time), dtype="f8"),
                    np.zeros(len(outca_time), dtype="f8"),
                    np.zeros(len(outca_time), dtype="f8"),
                    np.zeros(len(outca_time), dtype="f8"),
                    np.zeros(len(outca_time), dtype="f8")
                ],
                field=field_amp,
                dtype=dtype_amp
            )

        if len(outcp_time) == 0:
            tmpl_clphs = gv.utils.structured_array(
                data=[[np.nan] for i in range(len(field_phs))],
                field=field_phs,
                dtype=dtype_phs
            )

            flag_clphs = True

        else:
            tmpl_clphs = gv.utils.structured_array(
                data=[
                    outcp_time, outcp_freq, outcp_tri,
                    outcp_u12, outcp_v12, outcp_u23, outcp_v23,
                    outcp_u31, outcp_v31,
                    np.zeros(len(outcp_time), dtype="c8"),
                    np.zeros(len(outcp_time), dtype="c8"),
                    np.zeros(len(outcp_time), dtype="c8"),
                    np.zeros(len(outcp_time), dtype="f8"),
                    np.zeros(len(outcp_time), dtype="f8"),
                    np.zeros(len(outcp_time), dtype="f8"),
                    np.zeros(len(outcp_time), dtype="f8"),
                    np.zeros(len(outcp_time), dtype="f8")
                ],
                field=field_phs,
                dtype=dtype_phs
            )

        tmpl_clamp = tmpl_clamp.reshape(-1)
        tmpl_clphs = tmpl_clphs.reshape(-1)

        if tmpl_clamp.size == 0:
            tmpl_clamp = [np.nan] * len(field_amp)
            tmpl_clamp = gv.utils.structured_array(
                data=tmpl_clamp,
                field=field_amp,
                dtype=dtype_amp
            )

            flag_clamp = True

        if tmpl_clphs.size == 0:
            tmpl_clphs = [np.nan] * len(field_phs)
            tmpl_clphs = gv.utils.structured_array(
                data=tmpl_clphs,
                field=field_phs,
                dtype=dtype_phs
            )

            flag_clphs = True

        self.tmpl_clamp = copy.deepcopy(tmpl_clamp)
        self.tmpl_clphs = copy.deepcopy(tmpl_clphs)

        if not flag_clamp:
            clamp_ = tmpl_clamp
            clamp_["vis12"] = list(map(
                uvvis.get,
                tuple(zip(clamp_["u12"], clamp_["v12"]))
            ))

            clamp_["vis34"] = list(map(
                uvvis.get,
                tuple(zip(clamp_["u34"], clamp_["v34"]))
            ))

            clamp_["vis13"] = list(map(
                uvvis.get,
                tuple(zip(clamp_["u13"], clamp_["v13"]))
            ))

            clamp_["vis24"] = list(map(
                uvvis.get,
                tuple(zip(clamp_["u24"], clamp_["v24"]))
            ))

            clamp_["sig12"] = list(map(
                uvsig.get,
                tuple(zip(clamp_["u12"], clamp_["v12"]))
            ))

            clamp_["sig34"] = list(map(
                uvsig.get,
                tuple(zip(clamp_["u34"], clamp_["v34"]))
            ))

            clamp_["sig13"] = list(map(
                uvsig.get,
                tuple(zip(clamp_["u13"], clamp_["v13"]))
            ))

            clamp_["sig24"] = list(map(
                uvsig.get,
                tuple(zip(clamp_["u24"], clamp_["v24"]))
            ))

            amp12 = np.abs(clamp_["vis12"])
            amp34 = np.abs(clamp_["vis34"])
            amp13 = np.abs(clamp_["vis13"])
            amp24 = np.abs(clamp_["vis24"])

            snr12 = amp12 / np.abs(clamp_["sig12"])
            snr34 = amp34 / np.abs(clamp_["sig34"])
            snr13 = amp13 / np.abs(clamp_["sig13"])
            snr24 = amp24 / np.abs(clamp_["sig24"])
            clamp_[field_cav] = (amp12 * amp34) / (amp13 * amp24)
            clamp_[field_cas] = (
                snr12**-2 + snr34**-2 + snr13**-2 + snr24**-2
            )**0.5

        if not flag_clphs:
            clphs_ = tmpl_clphs
            clphs_["vis12"] = list(map(
                uvvis.get,
                tuple(zip(clphs_["u12"], clphs_["v12"]))
            ))

            clphs_["vis23"] = list(map(
                uvvis.get,
                tuple(zip(clphs_["u23"], clphs_["v23"]))
            ))

            clphs_["vis31"] = list(map(
                uvvis.get,
                tuple(zip(clphs_["u31"], clphs_["v31"]))
            ))

            clphs_["sig12"] = list(map(
                uvsig.get,
                tuple(zip(clphs_["u12"], clphs_["v12"]))
            ))

            clphs_["sig23"] = list(map(
                uvsig.get,
                tuple(zip(clphs_["u23"], clphs_["v23"]))
            ))

            clphs_["sig31"] = list(map(
                uvsig.get,
                tuple(zip(clphs_["u31"], clphs_["v31"]))
            ))

            vis12 = np.array(clphs_["vis12"], dtype="c16")
            vis23 = np.array(clphs_["vis23"], dtype="c16")
            vis31 = np.array(clphs_["vis31"], dtype="c16").conj()

            snr12 = np.abs(vis12) / np.abs(clphs_["sig12"])
            snr23 = np.abs(vis23) / np.abs(clphs_["sig23"])
            snr31 = np.abs(vis31) / np.abs(clphs_["sig31"])

            bispectrum = vis12 * vis23 * vis31
            clphs_v = np.angle(bispectrum)
            clphs_[field_cpv] = clphs_v
            clphs_[field_cps] = (
                snr12**-2 + snr23**-2 + snr31**-2
            )**0.5

        if not flag_clamp:
            fields = ["time", "quadrangle", "freq", field_cav, field_cas]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            datas = [clamp_[fields[nf]] for nf in range(len(fields))]
            clamp_ = gv.utils.structured_array(
                data=datas,
                field=fields,
                dtype=dtypes
            )
        else:
            fields = ["time", "quadrangle", "freq", field_cav, field_cas]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            clamp_ = gv.utils.structured_array(
                data=[np.nan for i in range(len(fields))],
                field=fields,
                dtype=dtypes
            )

        if not flag_clphs:
            fields = ["time", "triangle", "freq", field_cpv, field_cps]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            datas = [clphs_[fields[nf]] for nf in range(len(fields))]
            clphs_ = gv.utils.structured_array(
                data=datas,
                field=fields,
                dtype=dtypes
            )
        else:
            fields = ["time", "triangle", "freq", field_cpv, field_cps]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            clphs_ = gv.utils.structured_array(
                data=[np.nan for i in range(len(fields))],
                field=fields,
                dtype=dtypes
            )

        clamp = clamp_
        clphs = clphs_

        if not flag_clamp:
            self.clamp = clamp
            self.clamp_check = True
        else:
            fields = ["time", "quadrangle", "freq", "clamp", "sig_logclamp"]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]

            self.clamp = gv.utils.structured_array(
                data=[np.nan for i in range(len(fields))],
                field=fields,
                dtype=dtypes)
            self.clamp_check = False

        if not flag_clphs:
            self.clphs = clphs
            self.clphs_check = True
        else:
            fields = ["time", "triangle", "freq", "clphs", "sig_clphs"]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]

            self.clphs = gv.utils.structured_array(
                data=[np.nan for i in range(len(fields))],
                field=fields,
                dtype=dtypes
            )

            self.clphs_check = False

        self.ploter.clq_obs = (
            copy.deepcopy(self.clamp),
            copy.deepcopy(self.clphs)
        )

    def set_data(self, prt=True):
        """
        Set the data
        """

        obs = self.flat_data()

        obs["w_1"] = np.where(obs["w_1"] <= 0, np.nan, obs["w_1"])
        obs["w_2"] = np.where(obs["w_2"] <= 0, np.nan, obs["w_2"])
        obs["w_3"] = np.where(obs["w_3"] <= 0, np.nan, obs["w_3"])
        obs["w_4"] = np.where(obs["w_4"] <= 0, np.nan, obs["w_4"])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            v_1 = obs["r_1"] + 1j * obs["i_1"]
            v_2 = obs["r_2"] + 1j * obs["i_2"]
            v_3 = obs["r_3"] + 1j * obs["i_3"]
            v_4 = obs["r_4"] + 1j * obs["i_4"]
            e_1 = (1 / obs["w_1"])**0.5
            e_2 = (1 / obs["w_2"])**0.5
            e_3 = (1 / obs["w_3"])**0.5
            e_4 = (1 / obs["w_4"])**0.5
            e0_1 = ((1 / self.w0_1)**0.5).flatten()
            e0_2 = ((1 / self.w0_2)**0.5).flatten()
            e0_3 = ((1 / self.w0_3)**0.5).flatten()
            e0_4 = ((1 / self.w0_4)**0.5).flatten()

            v_i = 0.5 * (v_1 + v_2)
            v_q = 0.5 * (v_3 + v_4)
            v_u = 0.5 * (v_3 - v_4) / 1j
            v_v = 0.5 * (v_1 - v_2)
            v_p = 1.0 * (v_q + 1j * v_u)
            e_i = 0.5 * (e_1**2 + e_2**2)**0.5
            e_q = 0.5 * (e_3**2 + e_4**2)**0.5
            e_u = 0.5 * (e_3**2 + e_4**2)**0.5
            e_v = 0.5 * (e_1**2 + e_2**2)**0.5
            e_p = 1.0 * (e_q**2 + e_u**2)**0.5
            e0_i = 0.5 * (e0_1**2 + e0_2**2)**0.5
            e0_q = 0.5 * (e0_3**2 + e0_4**2)**0.5
            e0_u = 0.5 * (e0_3**2 + e0_4**2)**0.5
            e0_v = 0.5 * (e0_1**2 + e0_2**2)**0.5
            e0_p = 1.0 * (e0_q**2 + e0_u**2)**0.5

        if self.nstokes == 4:
            v_rr = v_1.copy()
            v_ll = v_2.copy()
            v_rl = v_3.copy()
            v_lr = v_4.copy()
            e_rr = e_1.copy()
            e_ll = e_2.copy()
            e_rl = e_3.copy()
            e_lr = e_4.copy()
            e0_rr = e0_1.copy()
            e0_ll = e0_2.copy()
            e0_rl = e0_3.copy()
            e0_lr = e0_4.copy()
            if self.select_pol in ["rr", "sf.rr", "mf.rr"]:
                v = v_rr
                e = e_rr
                e0 = e0_rr
            elif self.select_pol in ["ll", "sf.ll", "mf.ll"]:
                v = v_ll
                e = e_ll
                e0 = e0_ll
            elif self.select_pol in ["rl", "sf.rl", "mf.rl"]:
                v = v_rl
                e = e_rl
                e0 = e0_rl
            elif self.select_pol in ["lr", "sf.lr", "mf.lr"]:
                v = v_lr
                e = e_lr
                e0 = e0_lr
            elif self.select_pol in ["i", "sf.i", "mf.i"]:
                v = v_i
                e = e_i
                e0 = e0_i
            elif self.select_pol in ["q", "sf.q", "mf.q"]:
                v = v_q
                e = e_q
                e0 = e0_q
            elif self.select_pol in ["u", "sf.u", "mf.u"]:
                v = v_u
                e = e_u
                e0 = e0_u
            elif self.select_pol in ["v", "sf.v", "mf.v"]:
                v = v_v
                e = e_v
                e0 = e0_v
            elif self.select_pol in ["p", "sf.p", "mf.p"]:
                v = v_p
                e = e_p
                e0 = e0_p
        elif self.nstokes == 2:
            v_rr = v_1.copy()
            v_ll = v_2.copy()
            v_rl = v_3.copy()
            v_lr = v_4.copy()
            e_rr = e_1.copy()
            e_ll = e_2.copy()
            e_rl = e_3.copy()
            e_lr = e_4.copy()
            e0_rr = e0_1.copy()
            e0_ll = e0_2.copy()
            e0_rl = e0_3.copy()
            e0_lr = e0_4.copy()

            if self.select_pol in ["rr", "sf.rr", "mf.rr"]:
                v = v_1
                e = e_1
                e0 = e0_1
            elif self.select_pol in ["ll", "sf.ll", "mf.ll"]:
                v = v_2
                e = e_2
                e0 = e0_2
            elif self.select_pol in ["i", "sf.i", "mf.i"]:
                v = v_i
                e = e_i
                e0 = e0_i
            elif self.select_pol in [
                "rl", "lr", "q", "u", "v",
                "sf.rl", "sf.lr", "sf.q", "sf.u", "sf.v",
                "mf.rl", "mf.lr", "mf.q", "mf.u", "mf.v"
            ]:
                raise ValueError(
                    f"None of full-polarization data "
                    f"(given polarization: {self.select_pol.upper()!r})"
                )

        elif self.nstokes == 1:
            v_rl = v_3.copy()
            v_lr = v_4.copy()
            e_rl = e_3.copy()
            e_lr = e_4.copy()
            e0_rl = e0_3.copy()
            e0_lr = e0_4.copy()
            if self.stokes.lower() == "rr":
                v_rr = v_1.copy()
                v_ll = v_2.copy()
                e_rr = e_1.copy()
                e_ll = e_2.copy()
                e0_rr = e0_1.copy()
                e0_ll = e0_2.copy()
            elif self.stokes.lower() == "ll":
                v_rr = v_2.copy()
                v_ll = v_1.copy()
                e_rr = e_2.copy()
                e_ll = e_1.copy()
                e0_rr = e0_2.copy()
                e0_ll = e0_1.copy()
            else:
                raise ValueError(
                    f"Invalid polarization type is given: {self.stokes!r}"
                )

            if self.select_pol.split(".")[-1] != self.stokes.lower():
                warnings.warn(
                    f"\nInvalid polarization type "
                    f"(given: {self.select_pol.upper()!r}, "
                    f"data: {self.stokes!r}).\n"
                    f"\tInstead load {self.stokes!r}.\n",
                    UserWarning
                )
                self.select_pol = self.stokes.lower()

            v = v_1
            e = e_1
            e0 = e0_1
        else:
            raise ValueError(
                f"Invalid number of stokes is given: {self.nstokes}."
            )

        _data = [
            obs["mjd"], obs["time"], obs["frequency"], obs["baseline"],
            obs["ant1"], obs["ant2"], obs["u"], obs["v"], obs["w"], v, e, e0,
            obs["r_1"], obs["r_2"], obs["r_3"], obs["r_4"],
            obs["i_1"], obs["i_2"], obs["i_3"], obs["i_4"],
            obs["w_1"], obs["w_2"], obs["w_3"], obs["w_4"],
            obs["w0_1"], obs["w0_2"], obs["w0_3"], obs["w0_4"],
            v_rr, v_ll, v_rl, v_lr,
            e_rr, e_ll, e_rl, e_lr,
            e0_rr, e0_ll, e0_rl, e0_lr,
            v_i, v_q, v_u, v_v, v_p,
            e_i, e_q, e_u, e_v, e_p,
            e0_i, e0_q, e0_u, e0_v, e0_p
        ]

        _name = [
            "mjd", "time", "frequency", "baseline",
            "ant1", "ant2", "u", "v", "w", "vis", "sig", "sig0",
            "r_1", "r_2", "r_3", "r_4",
            "i_1", "i_2", "i_3", "i_4",
            "w_1", "w_2", "w_3", "w_4",
            "w0_1", "w0_2", "w0_3", "w0_4",
            "vis_rr", "vis_ll", "vis_rl", "vis_lr",
            "sig_rr", "sig_ll", "sig_rl", "sig_lr",
            "sig0_rr", "sig0_ll", "sig0_rl", "sig0_lr",
            "vis_i", "vis_q", "vis_u", "vis_v", "vis_p",
            "sig_i", "sig_q", "sig_u", "sig_v", "sig_p",
            "sig0_i", "sig0_q", "sig0_u", "sig0_v", "sig0_p"
        ]

        _type = [
            "f8", "f4", "f4", "i4",
            "i4", "i4", "f8", "f8", "f8", "c8", "f4", "f4",
            "f4", "f4", "f4", "f4",
            "f4", "f4", "f4", "f4",
            "f4", "f4", "f4", "f4",
            "f4", "f4", "f4", "f4",
            "c8", "c8", "c8", "c8",
            "f4", "f4", "f4", "f4",
            "f4", "f4", "f4", "f4",
            "c8", "c8", "c8", "c8", "c8",
            "f4", "f4", "f4", "f4", "f4",
            "f4", "f4", "f4", "f4", "f4"
        ]

        if self.vism is not None and len(obs) == self.vism.size:
            _data += [self.vism.flatten()]
            _name += ["vism"]
            _type += ["c8"]

        data = gv.utils.structured_array(
            data=_data,
            field=_name,
            dtype=_type
        )

        self.data_shape = self.r_1.shape

        time_sec = self.time * 3600.0
        self.set_scan(
            time=time_sec, gaptime=self.gaptime, scanlen=self.scanlen
        )

        data = rfn.append_fields(
            data, "scan", self.scannum_1d, usemask=False
        )

        idx = [
            "mjd", "time", "frequency", "scan", "baseline",
            "ant1", "ant2", "u", "v", "w", "vis", "sig", "sig0",
            "r_1", "r_2", "r_3", "r_4",
            "i_1", "i_2", "i_3", "i_4",
            "w_1", "w_2", "w_3", "w_4",
            "w0_1", "w0_2", "w0_3", "w0_4",
            "vis_rr", "vis_ll", "vis_rl", "vis_lr",
            "sig_rr", "sig_ll", "sig_rl", "sig_lr",
            "sig0_rr", "sig0_ll", "sig0_rl", "sig0_lr",
            "vis_i", "vis_q", "vis_u", "vis_v", "vis_p",
            "sig_i", "sig_q", "sig_u", "sig_v", "sig_p",
            "sig0_i", "sig0_q", "sig0_u", "sig0_v", "sig0_p"
        ]

        if "vism" in data.dtype.names:
            idx += ["vism"]

        data = data[idx]

        self.data = data

        self.no_if = self.data_shape[1]

        if self.no_if < self.no_if_original:
            txt_tail = ", IF-averaged"
        elif self.no_if == self.no_if_original:
            txt_tail = ""
        else:
            raise ValueError(
                f"Invalid number of IF channel is given: {self.no_if}"
            )

        if prt:
            print(
                f"# set {len(self.data)} visibilities... "
                f"({self.no_if} IF chans{txt_tail})"
            )

        # refresh uv coverage from the data table built above;
        # do NOT use set_uvcov() here: it calls get_data(), which
        # deep-copies the object and re-runs set_data(), causing
        # infinite recursion
        self.uvcov = gv.utils.structured_array(
            data=[
                np.ma.getdata(self.data["u"]).flatten(),
                np.ma.getdata(self.data["v"]).flatten(),
                np.ma.getdata(self.data["vis"]).flatten()
            ],
            field=["u", "v", "vis"],
            dtype=["f8", "f8", "c8"]
        )
        self.bprms = gv.utils.fit_beam(uvc=self.uvcov, sig=None, uvw=self.uvw)
        self.bmaj = self.bprms[0]
        self.bmin = self.bprms[1]
        self.bpa = self.bprms[2]
        self.ploter.bprms = self.bprms

    def set_scan(
        self,
        time=None, gaptime=None, scanlen=None, returned=False
    ):
        """
        Compute scan numbers from a time array, splitting on time
        gaps and length limits

        Args:
            time: Array of times in seconds, shape (N,) or (N, M, 1).
                If None, uses 'self.time * 3600'
            gaptime: Gap threshold (s). If None, falls back to
                'self.gaptime'; if that is also None, defaults to 60.
            scanlen: Maximum scan length (s). If None, falls back to
                'self.scanlen'; if that is also None, defaults to 0
                (disables the length cut)
            returned: Return the scan numbers as well as writing to
                'self.scannum'

        Returns:
            np.ndarray of scan indices ((2d, 1d); if returned).
        """
        if time is None:
            time = self.time * 3600.0
        if gaptime is None:
            gaptime = self.gaptime
        if scanlen is None:
            scanlen = self.scanlen

        scannum, scannum_1d = gv.utils.set_scan(
            time=time, gaptime=gaptime, scanlen=scanlen
        )

        if returned:
            return scannum, scannum_1d
        else:
            self.scannum = scannum
            self.scannum_1d = scannum_1d

    def set_uvcov(self, flatten=False, returned=False):
        """
        Set UV-coverage
        """

        data = [self.get_data("u"), self.get_data("v"), self.get_data("vis")]
        if flatten:
            data = [d.flatten() for d in data]

        field = ["u", "v", "vis"]
        dtype = ["f8", "f8", "c8"]

        self.uvcov = gv.utils.structured_array(
            data=data, field=field, dtype=dtype
        )

        if returned:
            return self.uvcov

    def sort_data(self, dotype=["freq", "time", "ant"], reverse=False):
        """
        Sort data by frequency, time, or antenna
        """

        sort_types = []

        availables = ["freq", "frequency", "time", "ant", "antenna", "snr"]
        if isinstance(dotype, str):
            dotype = [dotype]

        for _type in dotype:
            if _type not in availables:
                raise ValueError(
                    f"Invalid type: {_type!r}.\n"
                    f"Availables: {availables}"
                )

            if _type in ["freq", "frequency"]:
                data = [self.freq[:, 0, 0]]
            elif _type in ["time"]:
                data = [self.time[:, 0, 0]]
            elif _type in ["ant", "antenna"]:
                data = [self.ant1[:, 0, 0], self.ant2[:, 0, 0]]
            elif _type in ["snr"]:
                vis = self.get_data(dotype="vis")
                sig = self.get_data(dotype="sig")
                snr = np.nanmean(np.abs(vis) / sig, axis=(1, 2))
                data = [snr]

            sort_types += data

        sort_types.reverse()

        order = np.lexsort(sort_types)

        if reverse:
            order = order[::-1]

        self.time = self.time[order]
        self.mjd = self.mjd[order]
        self.freq = self.freq[order]
        self.baseline = self.baseline[order]
        self.ant1 = self.ant1[order]
        self.ant2 = self.ant2[order]
        self.u = self.u[order]
        self.v = self.v[order]
        self.w = self.w[order]
        self.r_1 = self.r_1[order]
        self.r_2 = self.r_2[order]
        self.r_3 = self.r_3[order]
        self.r_4 = self.r_4[order]
        self.i_1 = self.i_1[order]
        self.i_2 = self.i_2[order]
        self.i_3 = self.i_3[order]
        self.i_4 = self.i_4[order]
        self.w_1 = self.w_1[order]
        self.w_2 = self.w_2[order]
        self.w_3 = self.w_3[order]
        self.w_4 = self.w_4[order]

        if not self.empty_w0:
            self.w0_1 = self.w0_1[order]
            self.w0_2 = self.w0_2[order]
            self.w0_3 = self.w0_3[order]
            self.w0_4 = self.w0_4[order]

        if self.vism is not None:
            self.vism = self.vism[order]

        if self.cg_pol1_ant1 is not None:
            self.cg_pol1_ant1 = self.cg_pol1_ant1[order]
            self.cg_pol1_ant2 = self.cg_pol1_ant2[order]
            self.cg_pol2_ant1 = self.cg_pol2_ant1[order]
            self.cg_pol2_ant2 = self.cg_pol2_ant2[order]

        self.set_data(prt=False)

    def systematics_apply(self, dotype=None, d=0.0, m=0.0):
        """
        Apply the estimated systematics to the data
        """

        availables = ["vis", "logclamp", "clphs"]

        if dotype is None:
            raise ValueError(
                f"Invalid type is given: {dotype!r}.\n"
                f"Availables: {availables}"
            )

        for nty, _type in enumerate(dotype):
            if _type == "vis":
                data = self.get_data(dotype=f"vis_{self.select_pol}")
                time = self.time
                bsli = self.baseline.astype(str)
                systematics = self.systematics_vis

            elif _type in ["clamp", "logclamp"]:
                data = self.clamp
                time = self.clamp["time"]
                bsli = self.clamp["quadrangle"].astype(str)
                systematics = self.systematics_clamp

            elif _type == "clphs":
                data = self.clphs
                time = self.clphs["time"]
                bsli = self.clphs["triangle"].astype(str)
                systematics = self.systematics_clphs

            out = np.zeros(data.shape)
            count = np.zeros(data.shape)

            for ns in range(len(systematics)):
                time1_ = systematics["time_beg"][ns]
                time2_ = systematics["time_end"][ns]
                bsli_ = systematics["baseline"][ns].astype(str)
                systematics_ = systematics["systematics"][ns]

                mask_time1 = time1_ <= time
                mask_time2 = time <= time2_
                mask_bsli = bsli == bsli_
                mask = mask_time1 & mask_time2 & mask_bsli

                if mask.sum() == 0:
                    continue
                else:
                    out[mask] = systematics_
                    count[mask] += 1

            if count.max() >= 2:
                warnings.warn(
                    f" Duplicates are found in applying systematics."
                    f" (type: {_type!r})",
                    UserWarning
                )

            if _type == "vis":
                vis = self.get_data(dotype=f"vis_{self.select_pol}")

                self.check_w0()

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)

                    self.w_1 = 1 / (
                        (
                            1 / self.w_1
                            + out.reshape(self.data_shape)**2
                        )**0.5
                        + np.abs(vis) * np.sqrt(2 * 2) * d * m
                    )**2

                    self.w_2 = 1 / (
                        (
                            1 / self.w_2
                            + out.reshape(self.data_shape)**2
                        )**0.5
                        + np.abs(vis) * np.sqrt(2 * 2) * d * m
                    )**2

                    if self.nstokes == 4:
                        self.w_3 = 1 / (
                            (
                                1 / self.w_3
                                + out.reshape(self.data_shape)**2
                            )**0.5
                            + np.abs(vis) * np.sqrt(2 * 2) * d * m
                        )**2

                        self.w_4 = 1 / (
                            (
                                1 / self.w_4
                                + out.reshape(self.data_shape)**2
                            )**0.5
                            + np.abs(vis) * np.sqrt(2 * 2) * d * m
                        )**2

            elif _type == "logclamp":
                self.clamp["sig_logclamp"] = (
                    np.sqrt(self.clamp["sig_logclamp"]**2 + out**2)
                    + (
                        np.sqrt(2 * 4) * d * m
                        * np.abs(np.log(self.clamp["clamp"]))
                    )
                )

                self.ploter.clq_obs = (
                    copy.deepcopy(self.clamp), copy.deepcopy(self.clphs)
                )

            elif _type == "clphs":
                self.clphs["sig_clphs"] = (
                    np.sqrt(self.clphs["sig_clphs"]**2 + out**2)
                    + (np.sqrt(2 * 3) * d * m)
                )

                self.ploter.clq_obs = (
                    copy.deepcopy(self.clamp), copy.deepcopy(self.clphs)
                )

        self.set_data(prt=False)

    def systematics_cal(self, dotype=None):
        """
        Compute the statistical systematics through the median absolute
        deviation
        """

        availables = ["vis", "logclamp", "clphs"]

        if dotype is None:
            raise ValueError(
                f"Invalid type is given: {dotype!r}.\n"
                f"Availables: {availables}"
            )

        if self.clamp is not None or self.clphs is not None:
            self.set_closure(self.minclq)

        if (
            self.clamp is None and self.clphs is None
            and ("logclamp" in dotype or "clphs" in dotype)
        ):
            raise ValueError("No closure data available.")

        if self.clamp is None and "logclamp" in dotype:
            raise ValueError("No logclamp data available.")

        def cal_s(s, X, sig_thermal, circular=False):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                if circular:
                    center = np.angle(np.nanmean(np.exp(1j * X)))
                    resid = np.angle(np.exp(1j * (X - center)))
                else:
                    center = np.nanmedian(X)
                    resid = X - center
                Y = resid / np.sqrt(sig_thermal**2 + s**2)

                mad = np.nanmedian(np.abs(Y - np.nanmedian(Y)))
                out = np.abs(mad - 1 / 1.483)

            if np.isnan(out):
                return np.inf
            else:
                return out

        for nty, _type in enumerate(dotype):
            if _type in ["vis", "amp", "phs"]:
                data = self.data
                X = self.get_data("amp")
                bsli = self.get_data("baseline")
                time = self.get_data("time")
                sig_thermal = self.get_data("sig")

            elif _type == "logclamp":
                data = self.clamp
                X = np.log(data["clamp"])
                bsli = data["quadrangle"]
                time = data["time"]
                sig_thermal = data["sig_logclamp"]

            elif _type == "clphs":
                data = self.clphs
                X = data["clphs"]
                bsli = data["triangle"]
                time = data["time"]
                sig_thermal = data["sig_clphs"]

            else:
                raise ValueError(
                    f"Invalid type is given: {dotype!r}.\n"
                    f"Availables: {availables}"
                )

            utime_sec = np.unique(time * 3600)

            if not np.all(np.isnan(time)):
                scans, scans_1d = self.set_scan(
                    time=time * 3600.0,
                    gaptime=self.gaptime,
                    scanlen=self.scanlen,
                    returned=True
                )
                uscan = np.unique(scans)
                ubsli = np.unique(bsli)

                out_systematics = []
                out_bsli = []
                out_time1 = []
                out_time2 = []

                for ns, _scan in enumerate(uscan):
                    for nbsli, bsli_ in enumerate(ubsli):
                        mask = (
                            ~np.isnan(X)
                            & (bsli == bsli_)
                            & (scans == _scan)
                        )

                        if np.sum(mask) == 0:
                            continue

                        _circular = (_type == "clphs")

                        x_masked = X[mask]
                        sig_masked = sig_thermal[mask]

                        if x_masked.size < 2:
                            _fsys = 0.0
                        else:
                            _fsys = optimize.minimize_scalar(
                                lambda s: cal_s(
                                    s,
                                    x_masked,
                                    sig_masked,
                                    circular=_circular
                                ),
                                bounds=(0.0, 1.0),
                                method="bounded",
                            ).x

                        out_time1.append(time[mask].min())
                        out_time2.append(time[mask].max())
                        out_bsli.append(bsli_)
                        out_systematics.append(np.abs(_fsys))

                out = gv.utils.structured_array(
                    data=[out_time1, out_time2, out_bsli, out_systematics],
                    dtype=["f8", "f8", "U32", "f8"],
                    field=["time_beg", "time_end", "baseline", "systematics"]
                )

                if _type == "vis":
                    self.systematics_vis = out

                elif _type == "logclamp":
                    self.systematics_clamp = out

                elif _type == "clphs":
                    self.systematics_clphs = out

    def uvshift(self, dl=0, dm=0):
        """
        Shift the uv-visibility data
        Args:
            deltal (float, mas): shift in the RA-direction
            deltam (float, mas): shift in the DEC-direction
        """
        dl = dl * self.mapunit.to(au.rad)
        dm = dm * self.mapunit.to(au.rad)

        u = self.get_data("u")
        v = self.get_data("v")

        v_1 = self.get_data("r_1") + 1j * self.get_data("i_1")
        v_2 = self.get_data("r_2") + 1j * self.get_data("i_2")
        v_3 = self.get_data("r_3") + 1j * self.get_data("i_3")
        v_4 = self.get_data("r_4") + 1j * self.get_data("i_4")

        v_1 *= (np.exp(+2j * np.pi * u * dl) * np.exp(+2j * np.pi * v * dm))
        v_2 *= (np.exp(+2j * np.pi * u * dl) * np.exp(+2j * np.pi * v * dm))
        v_3 *= (np.exp(+2j * np.pi * u * dl) * np.exp(+2j * np.pi * v * dm))
        v_4 *= (np.exp(+2j * np.pi * u * dl) * np.exp(+2j * np.pi * v * dm))

        self.r_1 = v_1.real
        self.r_2 = v_2.real
        self.r_3 = v_3.real
        self.r_4 = v_4.real

        self.i_1 = v_1.imag
        self.i_2 = v_2.imag
        self.i_3 = v_3.imag
        self.i_4 = v_4.imag

def set_matrix_visphs(N):
    """
    Set the matrix for the visibility phase

    Args:
        N (int): Number of antennas

    Returns:
        out (np.array): Matrix for the visibility phase
    """

    out = np.array([[1, -1]])
    if N == 2:
        return out
    else:
        for i in range(3, N + 1):
            m1 = np.ones((i - 1, 1))
            Is = np.eye(i - 1)
            m0 = np.zeros((int(comb(i - 1, 2)), 1))
            phi = out

            upper = np.concatenate((m1, -Is), axis=1)
            lower = np.concatenate((m0, phi), axis=1)
            out = np.concatenate((upper, lower), axis=0)
        return out

def _closure_live_pairs(pairs_full, pairs_obs=None):
    """Return canonical live pairs, validating them against the array."""

    if pairs_obs is None:
        return {tuple(pair) for pair in pairs_full.tolist()}

    pairs_obs = np.asarray(pairs_obs, dtype=int)
    if pairs_obs.size == 0:
        return set()
    pairs_obs = np.unique(
        np.sort(pairs_obs.reshape(-1, 2), axis=1), axis=0
    )

    pairs_all = {tuple(pair) for pair in pairs_full.tolist()}
    pairs_live = {tuple(pair) for pair in pairs_obs.tolist()}
    unknown = pairs_live - pairs_all
    if unknown:
        raise ValueError(
            f"Observed baselines are inconsistent with antennas: {unknown}"
        )
    return pairs_live


def _append_independent_closure(rows, row):
    """Append a closure row only when it increases matrix rank."""

    if not rows:
        rows.append(row)
        return True

    trial = np.vstack((*rows, row))
    if np.linalg.matrix_rank(trial) > len(rows):
        rows.append(row)
        return True
    return False


def _legacy_matrix_clamp(N, ant_nums):
    """Return the pre-adaptive minimal closure-amplitude basis."""

    npair = N * (N - 1) // 2
    if N < 4:
        return np.empty((0, npair), dtype=float)

    out = np.array([
        [0, 1, -1, -1, 1, 0],
        [1, 0, -1, -1, 0, 1],
    ], dtype=float)
    if N == 4:
        return out

    for i in range(5, N + 1):
        ants = np.asarray(ant_nums[:i])
        first = np.concatenate((
            np.eye(i - 2), -np.ones((i - 2, 1))
        ), axis=1)
        tail_pairs = list(it.combinations(ants.tolist(), 2))[i - 1:]
        tail_idx = {tuple(pair): idx for idx, pair in enumerate(tail_pairs)}
        second = np.zeros((i - 2, len(tail_pairs)))

        for j in range(i - 2):
            if j != i - 3:
                pair1 = (ants[j + 1], ants[j + 2])
                pair2 = (ants[j + 2], ants[-1])
            else:
                pair1 = (ants[1], ants[-2])
                pair2 = (ants[1], ants[-1])
            second[j, tail_idx[tuple(pair1)]] = -1
            second[j, tail_idx[tuple(pair2)]] = +1

        upper = np.concatenate((first, second), axis=1)
        lower = np.concatenate((
            np.zeros((out.shape[0], i - 1)), out
        ), axis=1)
        out = np.concatenate((upper, lower), axis=0)

    return out


def _legacy_matrix_clphs(N):
    """Return the pre-adaptive minimal closure-phase basis."""

    npair = N * (N - 1) // 2
    if N < 3:
        return np.empty((0, npair), dtype=float)
    if N == 3:
        return np.array([[1, -1, 1]], dtype=float)

    phi = set_matrix_visphs(N - 1)
    identity = np.eye((N - 1) * (N - 2) // 2)
    return np.concatenate((phi, identity), axis=1)


def _clamp_relation(row, pairs, legacy=False):
    """Convert one four-edge row to uv-pair and label metadata."""

    pair_add = tuple(map(tuple, pairs[row == +1].tolist()))
    pair_sub = tuple(map(tuple, pairs[row == -1].tolist()))
    ants = sorted(set(it.chain.from_iterable(pair_add + pair_sub)))
    if len(pair_add) != 2 or len(pair_sub) != 2 or len(ants) != 4:
        raise ValueError("Invalid closure-amplitude basis row.")

    if legacy:
        # Historical set_closure labels were the two positive pairs flattened
        # in pairs_full order. Preserve them for retained legacy rows.
        return pair_add, pair_sub, tuple(it.chain.from_iterable(pair_add))

    a, b, c, d = ants
    p0 = frozenset(((a, b), (c, d)))
    p1 = frozenset(((a, c), (b, d)))
    p2 = frozenset(((a, d), (b, c)))
    relation_order = {
        (p0, p1): (a, b, d, c),
        (p1, p2): (a, c, b, d),
        (p0, p2): (a, b, c, d),
    }
    try:
        label_order = relation_order[
            (frozenset(pair_add), frozenset(pair_sub))
        ]
    except KeyError as exc:
        raise ValueError("Unsupported closure-amplitude row orientation.") from exc
    return pair_add, pair_sub, label_order


def _clphs_relation(row, pairs):
    """Convert one triangular row to uv-pair and label metadata."""

    pair_add = tuple(map(tuple, pairs[row == +1].tolist()))
    pair_sub = tuple(map(tuple, pairs[row == -1].tolist()))
    if len(pair_add) != 2 or len(pair_sub) != 1:
        raise ValueError("Invalid closure-phase basis row.")

    shared = set(pair_add[0]).intersection(pair_add[1])
    if len(shared) != 1:
        raise ValueError("Closure-phase positive pairs do not form a path.")
    middle = shared.pop()
    ends = sorted(set(pair_sub[0]))
    label_order = (ends[0], middle, ends[1])
    return pair_add, pair_sub, label_order


def set_min_matrix_clamp(
    N, ant_nums, pairs_obs=None, return_relations=False
):
    """
    Minimal complete set of closure amplitudes.

    Args:
        N (int): Number of antennas
        ant_nums (array-like): Antenna numbers (length N)
        pairs_obs (array-like): Live antenna pairs. If None, use all pairs.
        return_relations (bool): Also return ordered pair/label metadata.

    Returns:
        out (ndarray): Independent closure-amplitude rows
        pairs (ndarray): All antenna pairs, shape (N * (N - 1) / 2, 2)
        relations (list, optional): Numerator, denominator, and label order
    """

    ant_nums = np.asarray(ant_nums)
    if len(ant_nums) != N:
        raise ValueError(
            f"N={N} is inconsistent with {len(ant_nums)} antennas."
        )

    pairs = np.array(list(it.combinations(ant_nums.tolist(), 2)), dtype=int)
    pair_idx = {tuple(p): k for k, p in enumerate(pairs.tolist())}
    pairs_live = _closure_live_pairs(pairs, pairs_obs)
    npair = len(pairs)
    target = N * (N - 3) // 2

    rows = []
    relations = []

    # Retain every still-live row of the historical basis first. Complete
    # arrays therefore preserve the same rows, signs, and order.
    for row in _legacy_matrix_clamp(N, ant_nums):
        used_pairs = pairs[row != 0]
        if all(tuple(pair) in pairs_live for pair in used_pairs):
            rows.append(row.copy())
            relations.append(_clamp_relation(row, pairs, legacy=True))

    if len(rows) < target:
        for quad in it.combinations(ant_nums.tolist(), 4):
            a, b, c, d = quad
            p0 = ((a, b), (c, d))
            p1 = ((a, c), (b, d))
            p2 = ((a, d), (b, c))

            for addp, subp in (
                (p1, p2), (p0, p2), (p0, p1)
            ):
                if len(rows) >= target:
                    break
                if not all(pair in pairs_live for pair in addp + subp):
                    continue

                row = np.zeros(npair, dtype=float)
                for pair in addp:
                    row[pair_idx[pair]] += 1.0
                for pair in subp:
                    row[pair_idx[pair]] -= 1.0

                if _append_independent_closure(rows, row):
                    relations.append(_clamp_relation(row, pairs))
            if len(rows) >= target:
                break

    out = np.asarray(rows, dtype=float).reshape(-1, npair)
    if return_relations:
        return out, pairs, relations
    return out, pairs


def set_min_matrix_clphs(
    N, ant_nums, pairs_obs=None, return_relations=False
):
    """
    Set the minimum matrix for the closure phase

    Args:
        N (int): Number of antennas
        ant_nums (array-like): Antenna numbers (length N)
        pairs_obs (array-like): Live antenna pairs. If None, use all pairs.
        return_relations (bool): Also return ordered pair/label metadata.

    Returns:
        out (ndarray): Independent closure-phase rows
        pairs (ndarray): All antenna pairs, shape (N * (N - 1) / 2, 2)
        relations (list, optional): Positive, negative, and triangle order
    """

    ant_nums = np.asarray(ant_nums)
    if len(ant_nums) != N:
        raise ValueError(
            f"N={N} is inconsistent with {len(ant_nums)} antennas."
        )

    pairs = np.array(list(it.combinations(ant_nums.tolist(), 2)), dtype=int)
    pair_idx = {tuple(p): k for k, p in enumerate(pairs.tolist())}
    pairs_live = _closure_live_pairs(pairs, pairs_obs)
    target = (N - 1) * (N - 2) // 2

    rows = []
    relations = []

    for row in _legacy_matrix_clphs(N):
        used_pairs = pairs[row != 0]
        if all(tuple(pair) in pairs_live for pair in used_pairs):
            rows.append(row.copy())
            relations.append(_clphs_relation(row, pairs))

    if len(rows) < target:
        for a, b, c in it.combinations(ant_nums.tolist(), 3):
            pair_add = ((a, b), (b, c))
            pair_sub = ((a, c),)
            if not all(pair in pairs_live for pair in pair_add + pair_sub):
                continue

            row = np.zeros(len(pairs), dtype=float)
            for pair in pair_add:
                row[pair_idx[pair]] += 1.0
            row[pair_idx[pair_sub[0]]] -= 1.0

            if _append_independent_closure(rows, row):
                relations.append(_clphs_relation(row, pairs))
            if len(rows) >= target:
                break

    out = np.asarray(rows, dtype=float).reshape(-1, len(pairs))
    if return_relations:
        return out, pairs, relations
    return out, pairs
