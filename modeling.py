
import os
import sys
import gc
import copy
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import itertools as it
from astropy import units as u
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import dynesty
from dynesty import NestedSampler
from dynesty.pool import Pool
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty.utils import quantile as dyquan

import gamvas

nan = np.nan
r2m = u.rad.to(u.mas)
d2m = u.rad.to(u.mas)

fields_sf = ["S", "a", "l", "m"]
dtypes_sf = ["f8", "f8", "f8", "f8", "f8", "f8"]

fields_mf = ["S", "a", "l", "m", "freq", "alpha"]
dtypes_mf = ["f8", "f8", "f8", "f8", "f8", "f8"]

class modeling:
    """
    NOTE: This modeling is based on 'dynesty' which is implementing Bayesian nested sampling
    (Web site: https://dynesty.readthedocs.io/en/stable/api.html#api)
    (NASA/ADS: https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract)

    Attributes:
        uvfs (list): A list of uvf objects
        select (str): The Stokes parameters (I, Q, U, and V) or parallel/cross-hand polarization (RR, LL, RL, LR)
        x (tuple): A tuple of x-arguments
        y (tuple): A tuple of y-arguments
        yerr (tuple): A tuple of y-error-arguments
        args (tuple): Arguments set
        factor_zblf (float): A factor zero-baseline flux density
        sampler (str): The sampling method in 'dynesty' (availables: 'rwalk', 'rslice', 'slice')
        bound (str): The bounding condition in 'dynesty'
        runfit_set (str): The modeling frequency setting (availables: 'sf' for single-frequency; 'mf' for multi-frequency)
        runfit_sf (bool): The toggle option if to run single-frequency modeling
        runfit_mf (bool): The toggle option if to run multi-frequency modeling
        runfit_pol (bool): The toggle option if to run polarization modeling
        ftype (list, str): The fitting data terms (availables: 'vis', 'amp' 'phs', 'clamp', 'clphs')
        fwght (list, float): The fitting weights for the given data terms
        boundset (2D-list): The list of boundary conditions for priors
        bnd_l (list): The priori boundary condition in Right Ascension (RA)
        bnd_m (list): The priori boundary condition in Declination (DEC)
        bnd_f (list): The priori boundary condition of turnover frequency (when 'spectrum' == 'cpl' | 'ssa')
        ufreq (list, float): The unique frequency of 'uvfs'
        bands (list, str): The frquency-band names to save the results
        spectrum (str): The modeling spectrum (availables: 'single', 'spl', 'cpl', 'ssa')
        uvw (str): The uv-weighting option ('n' for natural-weighting; 'u' for uniform-weighting)
        shift ((float, float)): The amount of shift of the uvf data in (RA, DEC)-direction
        fixnmod (bool): The toggle option if to fix the number of models to the 'maxn'
        maxn (int): The maximum number of models to be allowed
        npix (int): The number of pixels in resultant images
        mindr (float): The minimum dynamic range to plot a contour in resultant images
        mrng (float): The map range
        dogscale (bool): The toggle option to run a antenna gain-scaling
        doampcal (bool): The toggle option to run visibility amplitude self-calibration
        dophscal (bool): The toggle option to run visibility phase self-calibration
        path_fig (str): The path to save the resultant figures
        source (str): The source name
        date (str): The observation date
        cgain_truth (DataFrame): The truth complex antenna gain values (this option is for simulation)
        ncpu (int): The number of CPU to run the modeling
    """
    def __init__(self,
        uvfs=None, select="i", x=None, y=None, yerr=None, args=None,
        factor_zblf=1.0, sampler="rwalk", bound="multi",
        runfit_set="mf", runfit_sf=False, runfit_mf=True, runfit_pol=False,
        niter=1, ftype=None, fwght=None, re_wamp_mf=None, re_ftype=None, re_fwght=None,
        boundset=False, width=5, bnd_l=None, bnd_m=None, bnd_f=None, bnd_pa=None,
        ufreq=None, bands=None, spectrum=None, model="gaussian", uvw=None, shift=None,
        fixnmod=False, maxn=None, npix=None, mindr=3, mrng=None,
        gacalerr=0, dognorm=True, dogscale=False, doampcal=False, dophscal=True,
        path_fig=None, source=None, date=None, cgain_truth=None, ncpu=1
    ):
        self.uvfs = uvfs
        self.select = select
        self.x = x
        self.y = y
        self.yerr = yerr
        self.args = args

        self.factor_zblf = factor_zblf
        self.sampler = sampler
        self.bound = bound

        self.runfit_set = runfit_set
        self.runfit_sf = runfit_sf
        self.runfit_mf = runfit_mf
        self.runfit_pol = runfit_pol

        self.niter = niter
        self.ftype = ftype
        if ftype is not None and fwght is not None:
            self.fwght = fwght
            self.fdict = dict(zip(self.ftype, self.fwght))
        else:
            self.fwght = fwght
            self.fdict = None
        self.re_wamp_mf = re_wamp_mf
        self.re_ftype = re_ftype
        self.re_fwght = re_fwght

        self.boundset = boundset
        self.width = width
        self.bnd_l = bnd_l
        self.bnd_m = bnd_m
        self.bnd_f = bnd_f
        self.bnd_pa = bnd_pa

        self.ufreq = ufreq
        self.bands = bands
        self.spectrum = spectrum
        self.model = model.lower()
        self.uvw = uvw
        self.shift = shift

        self.fixnmod = fixnmod
        self.maxn = maxn
        self.npix = npix
        self.mindr = mindr
        self.mrng = mrng

        self.gacalerr = gacalerr
        self.dognorm  = dognorm
        self.dogscale = dogscale
        self.doampcal = doampcal
        self.dophscal = dophscal

        self.path_fig = path_fig
        self.source = source
        self.date = date
        self.cgain_truth = cgain_truth
        self.ncpu = ncpu

        self.pol = gamvas.polarization.modeling.polarization(ncpu=self.ncpu)


    def objective_function(self,
        theta, x, y, yerr, args
    ):
        """
        Compute objective function (Bayesian Information Criterion)
            Arguments:
                theta (list): A list of parameters
                x (tuple): A tuple of x-arguments
                y (tuple): A tuple of y-arguments
                yerr (tuple): A tuple of y-error-arguments
                args (tuple): Arguments set
            Returns:
                Bayesian Information Criterion value (float)
        """
        N = len(x[1])
        model = np.zeros(N, dtype="c8")
        ufreq = np.unique(x[1])

        nidx = 1 if self.set_spectrum else 0
        nmprm = 0

        nmod = int(np.round(theta[0]))
        mask_pa = False
        if self.model == "gaussian":
            for i in range(nmod):
                if self.ifsingle:
                    if i == 0:
                        model +=\
                            gamvas.functions.gvis0(
                                (x[2], x[3]),
                                theta[nidx+1],
                                theta[nidx+2]
                            )
                        nidx += 2
                        nmprm += 2
                    else:
                        if not self.bnd_pa is None:
                            pa = np.angle(theta[nidx+4] + 1j * theta[nidx+3], deg=True)
                            if pa < 0 :
                                pa += 360
                            if self.bnd_pa[0] < 0:
                                self.bnd_pa[0] += 360
                            if self.bnd_pa[1] < 0:
                                self.bnd_pa[1] += 360
                            if self.bnd_pa[0] > pa or pa > self.bnd_pa[1]:
                                mask_pa = True
                        model +=\
                            gamvas.functions.gvis(
                                (x[2], x[3]),
                                theta[nidx+1],
                                theta[nidx+2],
                                theta[nidx+3],
                                theta[nidx+4]
                            )
                        nidx += 4
                        nmprm += 4
                else:
                    if self.set_spectrum:
                        if i == 0:
                            if self.spectrum == "spl":
                                model +=\
                                    gamvas.functions.gvis_spl0(
                                        (x[0], x[1], x[2], x[3]),
                                        theta[nidx+0],
                                        theta[nidx+1],
                                        theta[nidx+2]
                                    )
                                nidx += 4
                                nmprm += 3
                            elif self.spectrum == "cpl":
                                model +=\
                                    gamvas.functions.gvis_cpl0(
                                        (x[1], x[2], x[3]),
                                        theta[nidx+0],
                                        theta[nidx+1],
                                        theta[nidx+2],
                                        theta[nidx+3]
                                    )
                                nidx += 4
                                nmprm += 4
                            elif self.spectrum == "ssa":
                                model +=\
                                    gamvas.functions.gvis_ssa0(
                                        (x[1], x[2], x[3]),
                                        theta[nidx+0],
                                        theta[nidx+1],
                                        theta[nidx+2],
                                        theta[nidx+3]
                                    )
                                nidx += 4
                                nmprm += 4
                        else:
                            if not self.bnd_pa is None:
                                pa = np.angle(theta[nidx+4] + 1j * theta[nidx+3], deg=True)
                                if pa < 0 :
                                    pa += 360
                                if self.bnd_pa[0] < 0:
                                    self.bnd_pa[0] += 360
                                if self.bnd_pa[1] < 0:
                                    self.bnd_pa[1] += 360
                                if self.bnd_pa[0] > pa or pa > self.bnd_pa[1]:
                                    mask_pa = True
                            if int(np.round(theta[nidx])) == 0 or self.spectrum == "spl":
                                model +=\
                                    gamvas.functions.gvis_spl(
                                        (x[0], x[1], x[2], x[3]),
                                        theta[nidx+1],
                                        theta[nidx+2],
                                        theta[nidx+3],
                                        theta[nidx+4],
                                        theta[nidx+5]
                                    )
                                nidx += 7
                                nmprm += 5
                            else:
                                if self.spectrum == "cpl":
                                    model +=\
                                        gamvas.functions.gvis_cpl(
                                            (x[1], x[2], x[3]),
                                            theta[nidx+1],
                                            theta[nidx+2],
                                            theta[nidx+3],
                                            theta[nidx+4],
                                            theta[nidx+5],
                                            theta[nidx+6]
                                        )
                                    nidx += 7
                                    nmprm += 6
                                elif self.spectrum == "ssa":
                                    model +=\
                                        gamvas.functions.gvis_ssa(
                                            (x[1], x[2], x[3]),
                                            theta[nidx+1],
                                            theta[nidx+2],
                                            theta[nidx+3],
                                            theta[nidx+4],
                                            theta[nidx+5],
                                            theta[nidx+6]
                                        )
                                    nidx += 7
                                    nmprm += 6
                    else:
                        if i == 0:
                            model +=\
                                gamvas.functions.gvis0(
                                    (x[2], x[3]),
                                    theta[nidx+1],
                                    theta[nidx+2]
                                )
                            nidx += 2
                            nmprm += 2
                        else:
                            if not self.bnd_pa is None:
                                pa = np.angle(theta[nidx+4] + 1j * theta[nidx+3], deg=True)
                                if pa < 0 :
                                    pa += 360
                                if self.bnd_pa[0] < 0:
                                    self.bnd_pa[0] += 360
                                if self.bnd_pa[1] < 0:
                                    self.bnd_pa[1] += 360
                                if self.bnd_pa[0] > pa or pa > self.bnd_pa[1]:
                                    mask_pa = True
                            model +=\
                                gamvas.functions.gvis(
                                    (x[2], x[3]),
                                    theta[nidx+1],
                                    theta[nidx+2],
                                    theta[nidx+3],
                                    theta[nidx+4]
                                )
                            nidx += 4
                            nmprm += 4
        elif self.model == "delta":
            for i in range(nmod):
                if self.ifsingle:
                    if i == 0:
                        model +=\
                            gamvas.functions.dvis0(
                                (x[2]),
                                theta[nidx+1]
                            )
                        nidx += 1
                        nmprm += 12
                    else:
                        if not self.bnd_pa is None:
                            pa = np.angle(theta[nidx+4] + 1j * theta[nidx+3], deg=True)
                            if pa < 0 :
                                pa += 360
                            if self.bnd_pa[0] < 0:
                                self.bnd_pa[0] += 360
                            if self.bnd_pa[1] < 0:
                                self.bnd_pa[1] += 360
                            if self.bnd_pa[0] > pa or pa > self.bnd_pa[1]:
                                mask_pa = True
                        model +=\
                            gamvas.functions.dvis(
                                (x[2], x[3]),
                                theta[nidx+1],
                                theta[nidx+2],
                                theta[nidx+3]
                            )
                        nidx += 3
                        nmprm += 3

        nasum = np.nansum(np.abs(model))
        ftypes = list(self.fdict.keys())

        # compute objective functions
        if not np.isnan(nasum) and not nasum == 0:
            objective = 0

            def compute_bic(in_res, in_sig2, in_type, in_nobs, in_nmprm):
                penalty = in_nmprm * np.log(in_nobs)
                if in_type == "vis":
                    nll =\
                        0.5 *\
                        (np.nansum(0.5 * (in_res**2 / in_sig2) + np.log(2 * np.pi * in_sig2)))
                else:
                    nll =\
                        0.5 *\
                        (np.nansum(1.0 * (in_res**2 / in_sig2) + np.log(2 * np.pi * in_sig2)))
                return 2 * nll + penalty

            if "vis" in ftypes:
                vis_obs = y[0]
                vis_mod = model
                vis_res = np.abs(vis_mod - vis_obs)
                nobs = len(y[0])
                vis_sig2 = yerr[0]**2
                objective -=\
                    self.fdict["vis"] *\
                    compute_bic(vis_res, vis_sig2, "vis", nobs, nmprm)

            if "amp" in ftypes:
                nobs = len(y[0])
                amp_obs = np.abs(y[0])
                amp_obs = np.where(amp_obs <= yerr[0], 0, np.sqrt(amp_obs**2 - yerr[0]**2))
                amp_mod = np.abs(model)
                amp_sig2 = yerr[0]**2
                amp_res = amp_mod - amp_obs
                objective -=\
                    self.fdict["amp"] *\
                    compute_bic(amp_res, amp_sig2, "amp", nobs, nmprm)

            if "phs" in ftypes:
                nobs = len(y[0])
                phs_obs = np.angle(y[0])
                phs_mod = np.angle(model)
                phs_sig2 = (yerr[0] / np.abs(y[0]))**2
                phs_res = np.abs(np.exp(1j * phs_mod) - np.exp(1j * phs_obs))
                objective -=\
                    self.fdict["phs"] *\
                    compute_bic(phs_res, phs_sig2, "phs", nobs, nmprm)

            if "clamp" in ftypes or "clphs" in ftypes:
                clqm =\
                    gamvas.utils.set_closure(
                        x[2],
                        x[3],
                        model,
                        np.zeros(model.shape[0]),
                        args[0],
                        args[1],
                        y[3],
                        y[4]
                    )

                if "clamp" in ftypes:
                    nobs = len(y[2])
                    clamp_obs = y[1]
                    clamp_mod = clqm[0]
                    clamp_sig2 = yerr[1]**2
                    clamp_res = np.abs( np.log(clamp_mod) - np.log(clamp_obs) )
                    objective -=\
                        self.fdict["clamp"] *\
                        compute_bic(clamp_res, clamp_sig2, "clamp", nobs, nmprm)

                if "clphs" in ftypes:
                    nobs = len(y[2])
                    clphs_obs = y[2]
                    clphs_mod = clqm[1]
                    clphs_sig2 = yerr[2]**2
                    clphs_res = np.abs( np.exp(1j * clphs_mod) - np.exp(1j * clphs_obs) )
                    objective -=\
                        self.fdict["clphs"] *\
                        compute_bic(clphs_res, clphs_sig2, "clphs", nobs, nmprm)
            if mask_pa:
                objective = -np.inf
        else:
            objective = -np.inf
        return objective


    def prior_transform(self,
        theta
    ):
        """
        Transform priori boundary conditions
        (a boundary between A to B: [B - A] * x + A)
            Arguments:
                theta (list): A list of parameters
            Returns:
                Bayesian Information Criterion value (float)
        """
        bounds = self.boundset
        results = []
        ndim = 0
        if self.fixnmod:
            results.append(1.0 * theta[0] + self.maxn - 0.5)
        else:
            results.append((self.maxn - 0.01) * theta[0] + 0.5)

        if self.ifsingle:
            for i in range(self.nmod):
                self.set_field(nmod=i+1)
                for nfield, field in enumerate(self.fields):
                    results.append(
                        (bounds[field][i][1] - bounds[field][i][0]) *\
                        theta[1 + ndim + nfield] +\
                        bounds[field][i][0]
                    )
                ndim += self.dims
        else:
            if self.set_spectrum:
                for i in range(self.nmod):
                    if i != 0:
                        results.append(+1.98 * theta[ndim] - 0.49)
                    self.set_field(nmod=i+1)
                    for nfield, field in enumerate(self.fields):
                        results.append(
                            (bounds[field][i][1] - bounds[field][i][0]) *\
                            theta[1 + ndim + nfield] +\
                            bounds[field][i][0]
                        )
                    ndim += self.dims
            else:
                for i in range(self.nmod):
                    self.set_field(nmod=i+1)
                    for nfield, field in enumerate(self.fields):
                        results.append(
                            (bounds[field][i][1] - bounds[field][i][0]) *\
                            theta[1 + ndim + nfield] +\
                            bounds[field][i][0]
                        )
                    ndim += self.dims
        return results


    def set_field(self,
        nmod=1
    ):
        """
        Set field names and dimensions
            Arguments:
                nmod (int): The number of models
        """
        if self.ifsingle:
            if self.model == "gaussian":
                if nmod == 1:
                    self.dims = 2
                    self.fields =\
                        ["S", "a"]
                else:
                    self.dims = 4
                    self.fields =\
                        ["S", "a", "l", "m"]
            elif self.model == "delta":
                if nmod == 1:
                    self.dims = 1
                    self.fields =\
                        ["S"]
                else:
                    self.dims = 3
                    self.fields =\
                        ["S", "l", "m"]
        else:
            if self.set_spectrum:
                if self.model == "gaussian":
                    if nmod == 1:
                        self.dims = 4
                        self.fields =\
                            ["S", "a", "alpha", "freq"]
                    else:
                        self.dims = 7
                        self.fields =\
                            ["S", "a", "l", "m", "alpha", "freq"]
                elif self.model == "delta":
                    if nmod == 1:
                        self.dims = 3
                        self.fields =\
                            ["S", "alpha", "freq"]
                    else:
                        self.dims = 6
                        self.fields =\
                            ["S", "l", "m", "alpha", "freq"]
            else:
                if self.model == "gaussian":
                    if nmod == 1:
                        self.dims = 2
                        self.fields =\
                            ["S", "a"]
                    else:
                        self.dims = 4
                        self.fields =\
                            ["S", "a", "l", "m"]
                elif self.model == "delta":
                    if nmod == 1:
                        self.dims = 1
                        self.fields =\
                            ["S"]
                    else:
                        self.dims = 3
                        self.fields =\
                            ["S", "l", "m"]


    def set_ndim(self,
        nmod=1
    ):
        """
        Set the number of dimensions
            Arguments:
                nmod (int): The number of models
        """
        ndim = 1
        self.ufreq = np.unique(self.x[1])
        for i in range(nmod):
            nmod_ = i + 1
            self.set_field(nmod=nmod_)
            ndim += self.dims
        self.ndim = ndim


    def set_index(self):
        """
        Set field index
        """
        index_ = ["nmod"]
        for i in range(self.nmod):
            if i != 0 and self.set_spectrum:
                index_ = index_ + [f"{i + 1}_thick"]

            self.set_field(nmod=i+1)
            nums = np.full(self.dims, i + 1)
            fields = self.fields
            index_list = ["_".join([str(x), y]) for x, y in zip(nums, fields)]
            index_ = index_ + index_list
        self.index = index_


    def get_results(self,
        qs=(0.025, 0.500, 0.975), save_path=False, save_name=False, save_xlsx=False
    ):
        """
        Get the modeling results (parameters)
            Arguments:
                qs (tuple, flaot): The quantile values
                save_path (str): The path to save the results
                save_name (str): The name of the file to save the results
                save_xlsx (bool): The toggle option if to save the results in xlsx format
        """
        samples = self.samples
        weights = self.weights
        nprms = samples.shape[1]
        qls = np.array([])
        qms = np.array([])
        qhs = np.array([])
        for i in range(nprms):
            ql, qm, qh = dyquan(samples[:,i], qs, weights=weights)
            ql = qm - ql
            qh = qh - qm
            qls = np.append(qls, ql)
            qms = np.append(qms, qm)
            qhs = np.append(qhs, qh)
        self.ql = qls
        self.qm = qms
        self.qh = qhs
        self.prms = np.array([qls, qms, qhs])

        if save_xlsx:
            nmod = int(np.round(qms[0]))
            idxn = np.array(list(map(lambda x: x.split("_")[0], self.index[1:])), dtype=int)
            mask_nmod = idxn <= nmod
            mask_nmod = np.append(np.array([True]), mask_nmod)

            qls_ = qls[mask_nmod]
            qms_ = qms[mask_nmod]
            qhs_ = qhs[mask_nmod]
            idx_ = np.array(self.index)[mask_nmod]
            prms_ = np.array([qls_, qms_, qhs_])

            out_xlsx = pd.DataFrame(prms_, index=["lolim", "value", "uplim"]).T
            out_xlsx["idx"] = idx_
            out_xlsx.to_excel(f"{save_path}{save_name}")


    def get_nmprms(self):
        """
        Get the number of parameters
        """
        mprms = self.mprms.copy()
        nmprms = 0
        nmod = int(np.round(mprms["nmod"]))

        if self.model == "gaussian":
            if self.set_spectrum:
                for i in range(nmod):
                    nmod_ = int(i + 1)
                    if nmod_ == 1:
                        if self.spectrum == "spl":
                            nmprms += 3
                        else:
                            nmprms += 4
                    else:
                        nmprms_ = 6
                        spectrum = bool(np.round(mprms[f"{nmod_}_thick"]))
                        if not spectrum:
                            nmprms_ -= 1
                        nmprms += nmprms_
            else:
                if nmod == 1:
                    nmprms += 2
                else:
                    nmprms += 4
        elif self.model == "delta":
            if self.set_spectrum:
                for i in range(nmod):
                    nmod_ = int(i + 1)
                    if nmod_ == 1:
                        if self.spectrum == "spl":
                            nmprms += 2
                        else:
                            nmprms += 3
                    else:
                        nmprms_ = 5
                        spectrum = bool(np.round(mprms[f"{nmod_}_thick"]))
                        if not spectrum:
                            nmprms_ -= 1
                        nmprms += nmprms_
            else:
                if nmod == 1:
                    nmprms += 1
                else:
                    nmprms += 3

        self.nmprms = nmprms


    def run_util(self,
        nmod=1, sample="rwalk", bound="multi", boundset=None, run_type=None,
        save_path=None, save_name=None, save_xlsx=False
    ):
        """
        Run 'dynesty' utilies
            Arguments:
                sample (str): The sampling method in 'dynesty' (availables: 'rwalk', 'rslice', 'slice')
                bound (str): The bounding condition in 'dynesty'
                boundset (2D-list): The list of boundary conditions for priors
                run_type (str): The modeling frequency setting (availables: 'sf' for single-frequency; 'mf' for multi-frequency)
                save_path (str): The path to save the results
                save_name (str): The name of the file to save the results
                save_xlsx (bool): The toggle option if to save the results in xlsx format

        """
        self.set_index()
        args = self.args
        ndim = self.ndim

        if not self.boundset is None:
            boundset=self.boundset
        else:
            raise Exception("Boundary conditions for priors are not given.")

        self.sampler = sample
        self.bound = bound
        # run dynesty
        with Pool(
            self.ncpu,
            loglike=self.objective_function,
            prior_transform=self.prior_transform,
            logl_args=args) as pool:
                sampler =\
                    dynesty.DynamicNestedSampler(
                        pool.loglike,
                        pool.prior_transform,
                        ndim,
                        sample=sample,
                        bound=bound,
                        pool=pool
                    )
                sampler.run_nested()

        # extract dynesty restuls
        results = sampler.results
        samples = results.samples
        weights = results.importance_weights()
        self.results = results
        self.samples = samples
        self.weights = weights
        self.get_results(
            save_path=save_path,
            save_name=save_name,
            save_xlsx=save_xlsx
        )

        fields = self.index
        dtypes = ["f8"] + ["f8" for i in range(len(fields)-1)]
        mprms =\
            gamvas.utils.sarray(
                self.prms[1].copy(),
                fields,
                dtypes
            )
        self.errors = (self.prms[0] + self.prms[2]) / 2
        self.mprms = mprms
        self.get_nmprms()


    def run_sf(self):
        """
        Run single-frequency model-fit
        """

        if self.maxn is None:
            self.maxn = 5
        print(f"# Setting maximum number of models to {self.maxn}")

        if len(self.uvfs) != len(self.bands):
            raise Exception("The number of uvf files and bands are not matched.")
        else:
            nfreq = len(self.uvfs)

        for nband in range(nfreq):
            uvfs = copy.deepcopy(self.uvfs)
            uvf = gamvas.utils.set_uvf([copy.deepcopy(uvfs[nband])], type="sf")
            cgain1 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)
            cgain2 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)

            for niter_ in range(self.niter):
                uvfs = copy.deepcopy(self.uvfs)
                uvall = gamvas.utils.set_uvf(uvfs, type="mf")

                # set fit weights
                if niter_ == 0:
                    ftype = self.ftype.copy()
                    if self.fwght is None:
                        fwght = gamvas.utils.get_fwght(ftype, uvf.data, uvf.clamp["clamp"], uvf.clphs["clphs"])
                    else:
                        fwght = self.fwght.copy()
                    self.fdict = dict(zip(ftype, fwght))
                else:
                    if self.re_ftype is None and self.re_fwght is None:
                        ftype = self.ftype.copy()
                        fwght = [1 for i in range(len(ftype))]
                        self.fdict = dict(zip(ftype, fwght))
                    elif not self.re_ftype is None and self.re_fwght is None:
                        ftype = self.re_ftype.copy()[niter_ - 1]
                        if self.fwght is None:
                            fwght = gamvas.utils.get_fwght(ftype, uvf.data, uvf.clamp["clamp"], uvf.clphs["clphs"])
                        else:
                            fwght = self.fwght.copy()
                        self.fdict = dict(zip(ftype, fwght))
                    elif not self.re_ftype is None and not self.re_fwght is None:
                        ftype = self.re_ftype.copy()[niter_ - 1]
                        fwght = self.re_fwght.copy()[niter_ - 1]
                        self.fdict = dict(zip(ftype, fwght))

                band = self.bands[nband]
                uvf = gamvas.utils.set_uvf([copy.deepcopy(uvfs[nband])], type="sf")
                zblf = self.factor_zblf * uvf.get_zblf()[0]
                freq = uvf.freq
                nmod = self.maxn

                path_fig = self.path_fig + f"{freq:.1f}/"
                gamvas.utils.mkdir(path_fig)

                if niter_ == 0:
                    selfcal_ = "initial"
                else:
                    selfcal_ = "selfcal"
                uvf.ploter.draw_tplot(
                    uvf, plotimg=False, show_title=False,
                    save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.{selfcal_}.tplot",
                    save_form="pdf"
                )

                uvf.ploter.draw_radplot(
                    uvf, plotimg=False, show_title=False,
                    save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.{selfcal_}.radplot",
                    save_form="pdf"
                )

                uvf.ploter.draw_uvcover(
                    uvf, plotimg=False, show_title=False,
                    save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.{selfcal_}.uvcover",
                    save_form="pdf"
                )

                uvf.ploter.draw_dirtymap(
                    uvf, plotimg=False, show_title=False,
                    npix=self.npix, uvw=self.uvw,
                    save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.{selfcal_}.dirtmap",
                    save_form="pdf"
                )

                data = uvf.data
                if len(uvfs) >= 3:
                    if freq <= np.mean(self.ufreq):
                        width_ = self.width
                    else:
                        width_ = self.width / 2
                else:
                    width_ = self.width

                bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i =\
                    gamvas.utils.set_boundary(
                        nmod=nmod, select=self.select, spectrum="single", zblf=zblf,
                        width=width_, mrng=self.mrng, bnd_l=self.bnd_l, bnd_m=self.bnd_m
                    )

                bnds =\
                    gamvas.utils.sarray(
                        (bnd_S, bnd_a, bnd_l, bnd_m),
                        field=fields_sf,
                        dtype=dtypes_sf
                    )

                # set frequency information on closure quantities
                if uvf.clamp_check:
                    f_clamp = np.ma.getdata(uvf.tmpl_clamp["freq"])
                else:
                    f_clamp = []

                if uvf.clphs_check:
                    f_clphs = np.ma.getdata(uvf.tmpl_clphs["freq"])
                else:
                    f_clphs = []

                clamp_uvcomb, clphs_uvcomb =\
                    gamvas.utils.set_uvcombination(
                        uvf.data,
                        uvf.tmpl_clamp,
                        uvf.tmpl_clphs
                    )

                # set x parameters
                self.x =\
                (
                    uvf.freq,
                    np.ma.getdata(uvf.data["freq"]),
                    np.ma.getdata(uvf.data["u"]),
                    np.ma.getdata(uvf.data["v"])
                )

                # set y parameters
                self.y =\
                (
                    np.ma.getdata(uvf.data["vis"]),
                    np.ma.getdata(uvf.clamp["clamp"]),
                    np.ma.getdata(uvf.clphs["clphs"]),
                    clamp_uvcomb,
                    clphs_uvcomb
                )

                # set yerr parameters
                self.yerr =\
                (
                    np.ma.getdata(uvf.data["sigma"]),
                    np.ma.getdata(uvf.clamp["sigma_clamp"]),
                    np.ma.getdata(uvf.clphs["sigma_clphs"])
                )

                self.args =\
                (
                    self.x, self.y, self.yerr,
                    (
                        np.ma.getdata(uvf.data["ant_name1"]),
                        np.ma.getdata(uvf.data["ant_name2"])
                    )
                )

                self.boundset = bnds

                # count the number of visibility data
                self.nvis = uvf.data["vis"].shape[0]
                if uvf.clamp_check:
                    self.ncamp = uvf.clamp["clamp"].shape[0]
                if uvf.clphs_check:
                    self.ncphs = uvf.clphs["clphs"].shape[0]

                # set the number of free parameters
                self.nmod = nmod
                self.set_ndim(nmod=nmod)

                # set sampler
                if self.sampler is None:
                    if nmod < 3:
                        insample = "rwalk"
                    elif 3 <= nmod < 11:
                        insample = "rslice"
                    elif 11 <= nmod:
                        insample = "slice"
                else:
                    insample = self.sampler

                # print running information
                runtxt = f"\n# Running {uvf.freq:.1f} GHz ... "
                runtxt += f"(Pol {uvf.select.upper()}, MaxN_model={nmod}, sampler='{insample}', bound='multi')"
                if self.relmod:
                    runtxt += " // ! relative position"
                print(runtxt)
                print(f"# Fit-parameters : {self.fdict}")

                # run dynesty
                self.run_util(
                    nmod=nmod,
                    sample=insample,
                    bound=self.bound,
                    run_type="sf",
                    save_path=path_fig,
                    save_name="model_params.xlsx",
                    save_xlsx=True
                )

                if self.runfit_set == "sf":
                    if not "vis" in ftype and not "amp" in ftype:
                        self.rsc_amplitude([uvf])

                # extract statistical values
                logz_v = float(self.results.logz[-1])
                logz_d = float(self.results.logzerr[-1])
                prms = self.mprms
                nmod_ = int(np.round(prms["nmod"]))

                # add model visibility
                uvf.append_visibility_model(
                    freq_ref=uvf.freq, freq=uvf.freq,
                    theta=prms, fitset=self.runfit_set, model=self.model,
                    spectrum=self.spectrum, set_spectrum=self.set_spectrum
                )

                if self.dogscale:
                    if self.dophscal:
                        uvf.selfcal(type="phs", gnorm=self.dognorm)
                        cgain1 *= uvf.cgain1
                        cgain2 *= uvf.cgain2
                        uvf.selfcal(type="gscale", gnorm=self.dognorm)
                        cgain1 *= uvf.cgain1
                        cgain2 *= uvf.cgain2
                    else:
                        uvf.selfcal(type="gscale", gnorm=self.dognorm)
                        cgain1 *= uvf.cgain1
                        cgain2 *= uvf.cgain2
                else:
                    if self.doampcal:
                        if self.dophscal:
                            uvf.selfcal(type="phs", gnorm=self.dognorm)
                            cgain1 *= uvf.cgain1
                            cgain2 *= uvf.cgain2
                            uvf.selfcal(type="a&p", gnorm=self.dognorm)
                            cgain1 *= uvf.cgain1
                            cgain2 *= uvf.cgain2
                        else:
                            uvf.selfcal(type="amp", gnorm=self.dognorm)
                            cgain1 *= uvf.cgain1
                            cgain2 *= uvf.cgain2
                    else:
                        if self.dophscal:
                            uvf.selfcal(type="phs", gnorm=self.dognorm)
                            cgain1 *= uvf.cgain1
                            cgain2 *= uvf.cgain2
                self.cgain1 = cgain1
                self.cgain2 = cgain2

                # print statistical values : reduced chi-square, Akaike information criterion, Bayesian information criterion
                uvcomb = (
                    uvf.clamp["clamp"], uvf.clphs["clphs"],
                    uvf.clamp["sigma_clamp"], uvf.clphs["sigma_clphs"],
                    clamp_uvcomb, clphs_uvcomb
                )

                fty, chi, aic, bic = gamvas.utils.print_stats(uvf, uvcomb, self.nmprms, logz_v, logz_d, ftype)

                self.print_prms(
                    ufreq=[np.round(freq,1)], fitset=self.runfit_set, spectrum=self.spectrum, model=self.model,
                    stats=(fty, chi, aic, bic, logz_v, logz_d), printmsg=True,
                    save_path=path_fig, save_name="model_result.txt"
                )

                uvf.ploter.bnom = uvf.beam_prms
                uvf.ploter.prms = prms

                uvf.ploter.clq_obs = (uvf.clamp, uvf.clphs)
                uvf.ploter.clq_mod = gamvas.utils.set_closure(
                    data["u"], data["v"], uvf.data["vism"],
                    np.zeros(uvf.data["vism"].shape[0]), data["ant_name1"], data["ant_name2"],
                    self.y[3], self.y[4]
                )

                # plot and save figures
                uvf.ploter.draw_cgains(
                    uvf, cgain1, cgain2, truth=self.cgain_truth, plotimg=False,
                    save_csv=True, save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.complxgain",
                    save_form="pdf"
                )

                uvf.ploter.draw_trplot(
                    result=self.results, nmod=nmod_,
                    ifsingle=self.ifsingle, set_spectrum=self.set_spectrum, model=self.model,
                    fontsize=20, save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.trplot",
                    save_form="pdf"
                )

                uvf.ploter.draw_cnplot(
                    result=self.results, nmod=nmod_,
                    ifsingle=self.ifsingle, set_spectrum=self.set_spectrum, model=self.model,
                    fontsize=20, save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.cnplot",
                    save_form="pdf"
                )

                uvf.ploter.draw_radplot(
                    uvf=uvf, plotimg=False, show_title=False, plotvism=True,
                    save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.radplot.model",
                    save_form="pdf"
                )

                if "clamp" in ftype:
                    uvf.ploter.draw_closure(
                        type="clamp", model=True, plotimg=False, save_img=True,
                        save_path=path_fig,
                        save_name=f"{self.source}.{self.date}.clamp",
                        save_form="pdf"
                    )
                if "clphs" in ftype:
                    uvf.ploter.draw_closure(
                        type="clphs", model=True, plotimg=False, save_img=True,
                        save_path=path_fig,
                        save_name=f"{self.source}.{self.date}.clphs",
                        save_form="pdf"
                    )

                uvf.ploter.draw_image(
                    uvf=uvf, plotimg=False,
                    npix=self.npix, mindr=self.mindr, plot_resi=True, addnoise=True,
                    freq_ref=uvf.freq, freq=uvf.freq, model=self.model,
                    ifsingle=self.ifsingle, set_spectrum=self.set_spectrum,
                    save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.img",
                    save_form="pdf"
                )

                # set beam parameters
                uvf.ploter.bnom = (uvf.beam_prms[0], uvf.beam_prms[0], 0)

                uvf.ploter.draw_image(
                    uvf=uvf, plotimg=False,
                    npix=self.npix, mindr=self.mindr, plot_resi=False, addnoise=True,
                    freq_ref=uvf.freq, freq=uvf.freq, model=self.model,
                    ifsingle=self.ifsingle, set_spectrum=self.set_spectrum,
                    save_path=path_fig,
                    save_name=f"{self.source}.{self.date}.img.restore",
                    save_form="pdf"
                )

                uvf.drop_visibility_model()
                self.uvfs[nband] = uvf

            if self.runfit_pol:
                self.pol.run_pol(
                    uvfs=[copy.deepcopy(uvf)],
                    runmf=False,
                    uvw=self.uvw,
                    iprms=self.mprms,
                    ierrors=self.errors,
                    ftype=["vis"],
                    fwght=[1 for i in range(len(ftype))],
                    bands=[self.bands[nband]],
                    sampler=self.sampler,
                    bound=self.bound,
                    spectrum=self.spectrum,
                    freq_ref=uvfs[0].freq,
                    npix=self.npix,
                    mindr=3,
                    beam_prms=uvall.beam_prms,
                    save_path=path_fig,
                    source=self.source,
                    date=self.date
                )


    def run_mf(self):
        """
        Run multi-frequency model-fit
        """
        uvfs = copy.deepcopy(self.uvfs)
        uvf = gamvas.utils.set_uvf(self.uvfs, type="mf")
        cgain1 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)
        cgain2 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)

        for niter_ in range(self.niter):
            # set fit weights
            if niter_ == 0:
                if self.runfit_sf and not self.re_wamp_mf is None:
                    mask_ftype = self.ftype == "amp"
                    self.fwght[mask_ftype] = self.re_wamp_mf
                ftype = self.ftype.copy()
                if self.fwght is None:
                    fwght = gamvas.utils.get_fwght(ftype, uvf.data, uvf.clamp["clamp"], uvf.clphs["clphs"])
                else:
                    fwght = self.fwght.copy()
                self.fdict = dict(zip(ftype, fwght))
            else:
                if self.re_ftype is None and self.re_fwght is None:
                    ftype = self.ftype.copy()
                    fwght = [1 for i in range(len(ftype))]
                    self.fdict = dict(zip(ftype, fwght))
                elif not self.re_ftype is None and self.re_fwght is None:
                    ftype = self.re_ftype.copy()[niter_ - 1]
                    if self.fwght is None:
                        fwght = gamvas.utils.get_fwght(ftype, uvf.data, uvf.clamp["clamp"], uvf.clphs["clphs"])
                    else:
                        fwght = self.fwght.copy()
                    self.fdict = dict(zip(ftype, fwght))
                elif not self.re_ftype is None and not self.re_fwght is None:
                    ftype = self.re_ftype.copy()[niter_ - 1]
                    fwght = self.re_fwght.copy()[niter_ - 1]
                    self.fdict = dict(zip(ftype, fwght))

            # set maximum number of model
            nmod = self.maxn

            uvfs = copy.deepcopy(self.uvfs)
            uvf = gamvas.utils.set_uvf(self.uvfs, type="mf")
            zblf = self.factor_zblf * np.max(np.abs(uvf.data["vis"]))
            print(f"\n# Maximum baseline flux : {zblf:.3f} Jy")

            if niter_ == 0:
                selfcal_ = "initial"
            else:
                selfcal_ = "selfcal"

            uvf.ploter.draw_tplot(
                uvf, plotimg=False, show_title=False,
                save_path=self.path_fig, save_name=f"{self.source}.{self.date}.mf.{selfcal_}.tplot", save_form="pdf"
            )

            uvf.ploter.draw_radplot(
                uvf, plotimg=False, show_title=False,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.mf.{selfcal_}.radplot",
                save_form="pdf"
            )

            uvf.ploter.draw_uvcover(
                uvf, plotimg=False, show_title=False,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.mf.{selfcal_}.uvcover",
                save_form="pdf"
            )

            uvf.ploter.draw_dirtymap(
                uvf, plotimg=False, show_title=False,
                npix=self.npix, uvw=self.uvw,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.mf.{selfcal_}.dirtmap",
                save_form="pdf"
            )

            path_fig = self.path_fig + "mf/"
            gamvas.utils.mkdir(path_fig)

            if not self.bnd_f is None:
                bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i =\
                gamvas.utils.set_boundary(
                        nmod=nmod, select=self.select, spectrum=self.spectrum, zblf=zblf,
                        width=self.width, mrng=self.mrng, bnd_l=self.bnd_l, bnd_m=self.bnd_m, bnd_f=self.bnd_f
                )
            else:
                bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i =\
                gamvas.utils.set_boundary(
                        nmod=nmod, select=self.select, spectrum=self.spectrum, zblf=zblf,
                        width=self.width, mrng=self.mrng, bnd_l=self.bnd_l, bnd_m=self.bnd_m
                )

            bnds = gamvas.utils.sarray(
                    data=(bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i),
                    field=fields_mf,
                    dtype=dtypes_mf
            )

            ftype = list(self.fdict.keys())
            fwhgt = list(self.fdict.values())

            # set frequency information on closure quantities
            if uvf.clamp_check:
                f_clamp = np.ma.getdata(uvf.tmpl_clamp["freq"])
            else:
                f_clamp = []

            if uvf.clphs_check:
                f_clphs = np.ma.getdata(uvf.tmpl_clphs["freq"])
            else:
                f_clphs = []

            clamp_uvcomb, clphs_uvcomb =\
                gamvas.utils.set_uvcombination(
                    uvf.data,
                    uvf.tmpl_clamp,
                    uvf.tmpl_clphs
                )

            self.x =\
            (
                uvfs[0].freq,
                np.ma.getdata(uvf.data["freq"]),
                np.ma.getdata(uvf.data["u"]),
                np.ma.getdata(uvf.data["v"])
            )

            self.y =\
            (
                np.ma.getdata(uvf.data["vis"]),
                np.ma.getdata(uvf.clamp["clamp"]),
                np.ma.getdata(uvf.clphs["clphs"]),
                clamp_uvcomb,
                clphs_uvcomb
            )

            self.yerr =\
            (
                np.ma.getdata(uvf.data["sigma"]),
                np.ma.getdata(uvf.clamp["sigma_clamp"]),
                np.ma.getdata(uvf.clphs["sigma_clphs"])
            )

            self.args =\
            (
                self.x, self.y, self.yerr,
                (
                    np.ma.getdata(uvf.data["ant_name1"]),
                    np.ma.getdata(uvf.data["ant_name2"])
                )
            )

            self.boundset = bnds

            # count the number of visibility data
            self.nvis = uvf.data["vis"].shape[0]
            if uvf.clamp_check:
                self.ncamp = uvf.clamp["clamp"].shape[0]
            if uvf.clphs_check:
                self.ncphs = uvf.clphs["clphs"].shape[0]

            # set the number of free parameters
            self.nmod = nmod
            self.set_ndim(nmod=nmod)

            # set sampler
            if self.sampler is None:
                if nmod < 3:
                    insample = "rwalk"
                elif 3 <= nmod < 11:
                    insample = "rslice"
                elif 11 <= nmod:
                    insample = "slice"
            else:
                insample = self.sampler

            # print running information
            runtxt = f"\n# Running... (Pol {uvf.select.upper()}.{self.select.upper()}, MaxN_model={nmod}, sampler='{insample}', bound='multi')"
            if self.relmod:
                runtxt += " // ! relative position"
            print(runtxt)
            print(f"# Fit-parameters : {self.fdict}")

            # run dynesty
            self.run_util(nmod=nmod, sample=insample, bound=self.bound, run_type="mf", save_path=path_fig, save_name="model_params.xlsx", save_xlsx=True)

            if self.runfit_set == "sf":
                if not "vis" in ftype and not "amp" in ftype:
                    self.rsc_amplitude([uvfs])

            # extract statistical values
            logz_v = float(self.results.logz[-1])
            logz_d = float(self.results.logzerr[-1])
            prms = self.mprms
            nmod_ = int(np.round(prms["nmod"]))

            # add model visibility
            for nuvf in range(len(uvfs)):
                uvfs[nuvf].drop_visibility_model()
                uvfs[nuvf].append_visibility_model(
                    freq_ref=uvfs[0].freq, freq=uvfs[nuvf].freq,
                    theta=prms, fitset=self.runfit_set, model=self.model,
                    spectrum=self.spectrum, set_spectrum=self.set_spectrum
                )

                if self.dogscale:
                    if self.dophscal:
                        uvfs[nuvf].selfcal(type="phs", gnorm=self.dognorm)
                        cgain1[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain1
                        cgain2[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain2
                        uvfs[nuvf].selfcal(type="gscale", gnorm=self.dognorm)
                        cgain1[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain1
                        cgain2[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain2
                    else:
                        uvfs[nuvf].selfcal(type="gscale", gnorm=self.dognorm)
                        cgain1[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain1
                        cgain2[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain2
                else:
                    if self.doampcal:
                        if self.dophscal:
                            uvfs[nuvf].selfcal(type="phs", gnorm=self.dognorm)
                            cgain1[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain1
                            cgain2[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain2
                            uvfs[nuvf].selfcal(type="a&p", gnorm=self.dognorm)
                            cgain1[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain1
                            cgain2[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain2
                        else:
                            uvfs[nuvf].selfcal(type="amp", gnorm=self.dognorm)
                            cgain1[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain1
                            cgain2[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain2
                    else:
                        if self.dophscal:
                            uvfs[nuvf].selfcal(type="phs", gnorm=self.dognorm)
                            cgain1[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain1
                            cgain2[uvf.data["freq"] == self.ufreq[nuvf]] *= uvfs[nuvf].cgain2

            # re-set uvf
            uvf = gamvas.utils.set_uvf(uvfs, type="mf")

            # print statistical values : reduced chi-square, Akaike information criterion, Bayesian information criterion
            uvcomb = (
                uvf.clamp["clamp"], uvf.clphs["clphs"],
                uvf.clamp["sigma_clamp"], uvf.clphs["sigma_clphs"],
                clamp_uvcomb, clphs_uvcomb
            )
            fty, chi, aic, bic = gamvas.utils.print_stats(uvf, uvcomb, self.nmprms, logz_v, logz_d, ftype)
            self.print_prms(
                ufreq=uvf.ufreq, fitset=self.runfit_set, spectrum=self.spectrum, model=self.model,
                stats=(fty, chi, aic, bic, logz_v, logz_d), printmsg=True,
                save_path=path_fig, save_name="model_result.txt"
            )

            uvf.fit_beam(uvw=self.uvw)
            uvf.ploter.bnom = uvf.beam_prms
            uvf.ploter.prms = self.mprms
            uvf.ploter.spectrum = self.spectrum

            # set clousre quantities
            uvf.ploter.clq_obs = (uvf.clamp, uvf.clphs)
            uvf.ploter.clq_mod = gamvas.utils.set_closure(
                uvf.data["u"], uvf.data["v"], uvf.data["vism"],
                np.zeros(uvf.data["vism"].shape[0]), uvf.data["ant_name1"], uvf.data["ant_name2"],
                self.y[3], self.y[4]
            )

            # plot and save figures
            uvf.ploter.draw_cgains(
                uvf, cgain1, cgain2, truth=self.cgain_truth, plotimg=False,
                save_csv=True, save_path=path_fig, save_name=f"{self.source}.{self.date}.complxgain", save_form="pdf"
            )

            uvf.ploter.draw_trplot(
                result=self.results, nmod=nmod_,
                ifsingle=self.ifsingle, set_spectrum=self.set_spectrum, model=self.model,
                fontsize=20, save_path=path_fig, save_name=f"{self.source}.{self.date}.trplot", save_form="pdf"
            )

            uvf.ploter.draw_cnplot(
                result=self.results, nmod=nmod_,
                ifsingle=self.ifsingle, set_spectrum=self.set_spectrum, model=self.model,
                fontsize=20, save_path=path_fig, save_name=f"{self.source}.{self.date}.cnplot", save_form="pdf"
            )

            uvf.ploter.draw_radplot(
                uvf, plotimg=False, plotvism=True, show_title=False,
                save_path=path_fig, save_name=f"{self.source}.{self.date}.radplot", save_form="pdf"
            )

            uvf.ploter.draw_dirtymap(
                uvf, plotimg=False, plot_resi=True, show_title=False,
                npix=self.npix, uvw=self.uvw,
                save_path=path_fig, save_name=f"{self.source}.{self.date}.resimap", save_form="pdf"
            )

            if "clamp" in ftype:
                uvf.ploter.draw_closure(
                    type="clamp", model=True, plotimg=False, save_img=True,
                    save_path=path_fig, save_name=f"{self.source}.{self.date}.clamp", save_form="pdf"
                )
            if "clphs" in ftype:
                uvf.ploter.draw_closure(
                    type="clphs", model=True, plotimg=False, save_img=True,
                    save_path=path_fig, save_name=f"{self.source}.{self.date}.clphs", save_form="pdf"
                )

            for i in range(len(uvfs)):
                uvfs[i].ploter.bnom = uvfs[i].beam_prms
                uvfs[i].ploter.prms = self.mprms
                uvfs[i].ploter.fitset = self.runfit_set
                uvfs[i].ploter.spectrum = self.spectrum
                uvfs[i].ploter.draw_image(
                    uvf=uvfs[i], plotimg=False,
                    npix=self.npix, mindr=self.mindr, plot_resi=False, addnoise=True,
                    freq_ref=self.ufreq[0], freq=self.ufreq[i], model=self.model,
                    ifsingle=self.ifsingle, set_spectrum=self.set_spectrum,
                    save_path=path_fig, save_name=f"{self.source}.{self.date}.img.sf.{self.bands[i]}", save_form="pdf"
                )
            for i in range(len(uvfs)):
                uvf.ploter.draw_image(
                    uvf=uvf, plotimg=False,
                    npix=self.npix, mindr=self.mindr, plot_resi=True, addnoise=True,
                    freq_ref=self.ufreq[0], freq=self.ufreq[i], model=self.model,
                    ifsingle=self.ifsingle, set_spectrum=self.set_spectrum,
                    save_path=path_fig, save_name=f"{self.source}.{self.date}.img.mf.{self.bands[i]}", save_form="pdf"
                )
                uvfs[i].drop_visibility_model()
            uvf.drop_visibility_model()

            self.uvfs = uvfs

            gc.collect()
        self.ftype = ftype
        self.fwght = fwght
        self.path_fig = path_fig


    def run(self):
        """
        Run the modeling utilies
        """
        runfit_set = self.runfit_set
        runfit_sf = self.runfit_sf
        runfit_mf = self.runfit_mf
        if not runfit_set.lower() in ["sf", "mf"]:
            raise Exception("Given 'runfit_set' option is not intended. (available options: 'sf', 'mf')")
        ftype = self.ftype.copy()

        if "clphs" in ftype:
            self.relmod = True
        else:
            self.relmod = False

        if runfit_set == "sf":
            self.ifsingle = True
            self.set_spectrum = False
            self.run_sf()
        if runfit_set == "mf":
            if runfit_sf:
                self.ifsingle = False
                self.set_spectrum = False
                self.run_sf()
            if runfit_mf:
                for i in range(len(self.uvfs)):
                    self.uvfs[i].add_error_fraction(self.gacalerr, set_vis=True, set_clq=False)
                self.ifsingle = False
                self.set_spectrum = True
                self.run_mf()
                if self.runfit_pol:
                    uvfs_ = gamvas.utils.set_uvf(self.uvfs, type="mf")
                    self.pol.run_pol(
                        uvfs=self.uvfs,
                        uvw=self.uvw,
                        runmf=True,
                        iprms=self.mprms,
                        ierrors=self.errors,
                        ftype=["vis"],
                        fwght=[1 for i in range(len(ftype))],
                        bands=self.bands,
                        sampler=self.sampler,
                        bound=self.bound,
                        spectrum=self.spectrum,
                        freq_ref=self.uvfs[0].freq,
                        npix=self.npix,
                        mindr=3,
                        beam_prms=uvfs_.beam_prms,
                        save_path=self.path_fig,
                        source=self.source,
                        date=self.date
                    )


    def rsc_amplitude(self,
        uvfs
    ):
        """
        Rescale the visibility to the observed visibility amplitudes
            Arguments:
                uvfs (list): list of uvf objects
        """
        # rescale visibility amplitude at zero-baseline ("amp" not in ftype)
        ftype = list(self.fdict.keys())
        data_obs = uvfs[0].data
        zblf_obs, zbl_obs = uvfs[0].get_zblf()
        mask_zbl = (data_obs["ant_name1"] == zbl_obs[0]) | (data_obs["ant_name2"] == zbl_obs[1])
        zblf_mod = copy.deepcopy(uvfs[0])
        zblf_mod.append_visibility_model(
            freq_ref=uvfs[0].freq, freq=uvfs[0].freq,
            theta=self.mprms, fitset=self.run_type, model=self.model,
            spectrum=self.spectrum, set_spectrum=self.set_spectrum
        )
        zblf_mod = np.median(np.abs(zblf_mod.data[mask_zbl]["vism"]))
        scale_factor = zblf_obs/zblf_mod
        for i in range(nmod):
            self.mprms[f"{i+1}_S"] *= scale_factor


    def rsc_sigma(self,
        uvfs, run_type
    ):
        """
        Rescale the uncertainties using systematics
            Arguments:
                uvfs (list): list of uvf objects
                run_type (str): The modeling frequency setting (availables: 'sf' for single-frequency; 'mf' for multi-frequency)
        """

        if run_type == "sf":
            N = len(uvfs[0].data)
            f = np.ones(N, dtype="f4") * self.mprms[f"f1"]
            uvfs[0].data["sigma"] += f * np.abs(uvfs[0].data["vis"])

        if run_type == "mf":
            for i in range(len(self.ufreq)):
                N = len(uvfs[i].data)
                f = np.ones(N, dtype="f4") * self.mprms[f"f{i+1}"]
                uvfs[i].data["sigma"] += f * np.abs(uvfs[i].data["vis"])
        return uvfs


    def print_prms(self,
        ufreq, model="gaussian", fitset="sf", spectrum="spl", stats=None, printmsg=False, save_path=False, save_name=False
    ):
        """
        Print the model parameters
            Arguments:
                ufreq (list, float): The unique frequency
                fitset (str): The modeling frequency setting (availables: 'sf' for single-frequency; 'mf' for multi-frequency)
                spectrum (str): The spectrum type (availables: 'spl' for simple power-law; 'cpl' for complex power-law; 'ssa' for SSA)
                stats (tuple): The statistical values (e.g., chi-square, AIC, BIC)
                printmsg (bool): Print the message
                save_path (str): The path to save the model parameters
                save_name (str): The name of the file to save the model parameters
        """
        if save_path:
            gamvas.utils.mkdir(save_path)
        mprms = self.mprms.copy()
        nmod = int(np.round(mprms["nmod"]))
        if not isinstance(ufreq, list):
            if isinstance(ufreq, np.ndarray):
                ufreq = ufreq
            else:
                ufreq = np.array([ufreq])
        if isinstance(ufreq, list):
            ufreq = np.array(ufreq)

        if save_path and save_name:
            modelprms = open(save_path+save_name, "w")
            modelprms.close()

        for nfreq, freq in enumerate(ufreq):
            if model == "gaussian":
                for i in range(nmod):
                    if self.ifsingle:
                        if i == 0:
                            smax_, a_, l_, m_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], 0, 0
                        else:
                            smax_, a_, l_, m_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], mprms[f"{i+1}_l"], mprms[f"{i+1}_m"]
                        S_ = smax_
                    else:
                        if self.set_spectrum:
                            if spectrum in ["spl"]:
                                if i == 0:
                                    smax_, a_, l_, m_, alpha_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], 0, 0, mprms[f"{i+1}_alpha"]
                                else:
                                    smax_, a_, l_, m_, alpha_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], mprms[f"{i+1}_l"], mprms[f"{i+1}_m"], mprms[f"{i+1}_alpha"]
                                S_ = gamvas.functions.S_spl(ufreq[0], freq, smax_, alpha_)
                            if spectrum in ["cpl", "ssa"]:
                                if i == 0:
                                    smax_, a_, l_, m_, alpha_, tfreq_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], 0, 0, mprms[f"{i+1}_alpha"], mprms[f"{i+1}_freq"]
                                    if spectrum in ["cpl"]:
                                        S_ = gamvas.functions.S_cpl(freq, smax_, tfreq_, alpha_)
                                    elif spectrum in ["ssa"]:
                                        S_ = gamvas.functions.SSA(freq, smax_, tfreq_, alpha_)
                                else:
                                    smax_, a_, l_, m_, alpha_, tfreq_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], mprms[f"{i+1}_l"], mprms[f"{i+1}_m"], mprms[f"{i+1}_alpha"], mprms[f"{i+1}_freq"]
                                    if bool(np.round(mprms[f"{i+1}_thick"])):
                                        if spectrum in ["cpl"]:
                                            S_ = gamvas.functions.S_cpl(freq, smax_, tfreq_, alpha_)
                                        elif spectrum in ["ssa"]:
                                            S_ = gamvas.functions.SSA(freq, smax_, tfreq_, alpha_)
                                    else:
                                        S_ = gamvas.functions.S_spl(ufreq[0], freq, smax_, alpha_)
                        else:
                            if i == 0:
                                smax_, a_, l_, m_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], 0, 0
                            else:
                                smax_, a_, l_, m_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_a"], mprms[f"{i+1}_l"], mprms[f"{i+1}_m"]
                            S_ = smax_
                    r_, p_ = np.sqrt(l_**2+m_**2), np.arctan2(l_, m_)*u.rad.to(u.deg)
                    outprint = f"# ({freq:.1f} GHz) Model {i+1} : {S_:.3f}v {+r_:.3f}v {p_:.3f}v {a_:.3f}v"

                    if printmsg:
                        print(outprint)

                    if save_path and save_name:
                        modelprms = open(save_path+save_name, mode="a")
                        modelprms.write(f"{outprint}\n")
                        modelprms.close()
            elif model == "delta":
                for i in range(nmod):
                    if self.ifsingle:
                        if i == 0:
                            smax_, l_, m_ = mprms[f"{i+1}_S"], 0, 0
                        else:
                            smax_, l_, m_ = mprms[f"{i+1}_S"], mprms[f"{i+1}_l"], mprms[f"{i+1}_m"]
                        S_ = smax_

                    r_, p_ = np.sqrt(l_**2+m_**2), np.arctan2(l_, m_)*u.rad.to(u.deg)
                    outprint = f"# ({freq:.1f} GHz) Model {i+1} : {S_:.3f}v {+r_:.3f}v {p_:.3f}v 0.000"

                    if printmsg:
                        print(outprint)

                    if save_path and save_name:
                        modelprms = open(save_path+save_name, mode="a")
                        modelprms.write(f"{outprint}\n")
                        modelprms.close()

        if save_path and save_name:
            modelprms = open(save_path+save_name, mode="a")
            chi_tot = 0
            aic_tot = 0
            bic_tot = 0
            for i in range(len(stats[0])):
                outprint = f"Chi2_{stats[0][i]:9s} : {stats[1][i]:10.3f} | AIC_{stats[0][i]:9s} : {stats[2][i]:10.3f} | BIC_{stats[0][i]:9s} : {stats[3][i]:10.3f}\n"
                modelprms.write(outprint)
                if stats[0][i] in list(self.fdict.keys()):
                    chi_tot += stats[1][i]
                    aic_tot += stats[2][i]
                    bic_tot += stats[3][i]
            modelprms.write(f"Chi2_tot : {chi_tot:8.3f}\n")
            modelprms.write(f"AIC_tot  : {aic_tot:8.3f}\n")
            modelprms.write(f"BIC_tot  : {bic_tot:8.3f}\n")
            modelprms.write(f"logz : {stats[-2]:.3f} +/- {stats[-1]:.3f}\n")
            modelprms.close()
