
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

from uncertainties import ufloat
from uncertainties import unumpy as unp

import gamvas

class polarization:
    def __init__(self,
        uvfs=None, runmf=False, iprms=None, ierrors=None, ftype=None, fwght=None, bands=None,
        sampler=None, bound="multi", stokes=["q", "u"], spectrum=None, freq_ref=None,
        npix=128, mindr=3, bprms=None,
        save_path=None, source=None, date=None, ncpu=1
    ):
        self.uvfs = uvfs
        self.runmf = runmf
        self.iprms = iprms
        self.ierrors = ierrors
        self.ftype = ftype
        self.fwght = fwght
        self.bands = bands

        self.sampler = sampler
        self.bound = bound
        self.stokes = stokes
        self.spectrum = spectrum
        self.freq_ref = freq_ref

        self.npix = npix
        self.mindr = mindr
        self.bprms = bprms

        self.save_path = save_path
        self.source = source
        self.date = date
        self.ncpu = ncpu


    def objective_function(self, theta, x, y, yerr, args):
        N = len(x[1])
        model = np.zeros(N, dtype="c8")
        ufreq = np.unique(x[1])

        nmprm = 0

        nmod = len(theta)
        for i in range(nmod):
            model +=\
                gamvas.polarization.functions.gvis(
                    (x[0], x[1], x[2][i], x[3][i], x[4][i]),
                    theta[i]
                )
            nmprm += 1

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
                vis_sig2 = yerr[0]**2
                vis_res = np.abs(vis_mod - vis_obs)
                nobs = len(y[0])
                objective -= self.fdict["vis"] * compute_bic(vis_res, vis_sig2, "vis", nobs, nmprm)

            if "amp" in ftypes:
                nobs = len(y[0])
                amp_obs = np.abs(y[0])
                amp_mod = np.abs(model)
                amp_sig2 = yerr[0]**2
                amp_res = amp_mod - amp_obs
                objective -= self.fdict["amp"] * compute_bic(amp_res, amp_sig2, "amp", nobs, nmprm)

            if "phs" in ftypes:
                nobs = len(y[0])
                phs_obs = np.angle(y[0])
                phs_mod = np.angle(model)
                phs_sig2 = (yerr[0] / np.abs(y[0]))**2
                phs_res = np.abs(np.exp(1j * phs_mod) - np.exp(1j * phs_obs))
                objective -= self.fdict["phs"] * compute_bic(phs_res, phs_sig2, "phs", nobs, nmprm)

            if "clamp" in ftypes or "clphs" in ftypes:
                clqm = gamvas.utils.set_closure(x[0], x[1], model, np.zeros(model.shape[0]), args[0], args[1], y[3], y[4])

                if "clamp" in ftypes:
                    nobs = len(y[2])
                    clamp_obs = y[1]
                    clamp_mod = clqm[0]
                    clamp_sig2 = yerr[1]**2
                    clamp_res = np.abs( np.log(clamp_mod) - np.log(clamp_obs) )
                    objective -= self.fdict["clamp"] * compute_bic(clamp_res, clamp_sig2, "clamp", nobs, nmprm)

                if "clphs" in ftypes:
                    nobs = len(y[2])
                    clphs_obs = y[2]
                    clphs_mod = clqm[1]
                    clphs_sig2 = yerr[2]**2
                    clphs_res = np.abs( np.exp(1j * clphs_mod) - np.exp(1j * clphs_obs) )
                    objective -= self.fdict["clphs"] * compute_bic(clphs_res, clphs_sig2, "clphs", nobs, nmprm)
        else:
            objective = -np.inf
        return objective


    def prior_transform(self, theta):
        """
         : a*x+b
        This indicates the range of x : b to a+b

        In other words,
            a boundary between A to B : (B-A)*x+A
        """
        bounds = self.boundset
        results = []
        ndim = 0
        for i in range(self.nmod):
            self.set_field()
            for nfield, field in enumerate(self.fields):
                field = f"{field}_{i + 1}"
                results.append((bounds[field][1] - bounds[field][0]) * theta[ndim + nfield] + bounds[field][0])
            ndim += self.dims
        return results


    def set_field(self):
        self.dims = 1
        self.fields = [f"S"]


    def set_ndim(self, nmod=1):
        ndim = 0
        self.ufreq = np.unique(self.x[1])
        for i in range(nmod):
            nmod_ = i+1
            self.set_field()
            ndim += self.dims
        self.ndim = ndim


    def set_index(self):
        index_ = []
        for i in range(self.nmod):
            self.set_field()
            nums = np.full(self.dims, i + 1)
            fields = self.fields
            index_list = ["_".join([str(x), y]) for x, y in zip(nums, fields)]
            index_ = index_ + index_list

        self.index = index_


    def get_results(self, qs=(0.025, 0.500, 0.975), save_path=False, save_name=False, save_xlsx=False):
        samples = self.samples
        weights = self.weights
        nprms = samples.shape[1]
        qls = np.array([])
        qms = np.array([])
        qhs = np.array([])
        for i in range(nprms):
            ql, qm, qh = dyquan(samples[:, i], qs, weights=weights)
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
            out_xlsx = pd.DataFrame(self.prms, index=["lolim", "value", "uplim"]).T
            out_xlsx["idx"] = self.index
            out_xlsx.to_excel(f"{save_path}{save_name}")


    def run_util(self,
        nmod=1, sample="rwalk", bound="multi", dlogz=False, boundset=False,
        save_path=False, save_name=False, save_xlsx=False
    ):
        self.set_index()
        args = self.args
        ndim = self.ndim

        if not isinstance(self.boundset, bool):
            boundset = self.boundset
        elif not boundset:
            raise Exception("Boundary for parameters are not given.")

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
        dtypes = ["f8" for i in range(len(fields))]
        self.pprms = gamvas.utils.sarray(self.prms[1], field=fields, dtype=dtypes)
        self.errors = (self.prms[0] + self.prms[2]) / 2
        self.uflux = np.sum(unp.uarray(self.pprms.tolist(), self.errors.tolist()))

        if self.stoke.upper() == "Q":
            self.pprms_q = np.array(self.pprms.tolist())
        if self.stoke.upper() == "U":
            self.pprms_u = np.array(self.pprms.tolist())
        if self.stoke.upper() == "V":
            self.pprms_v = np.array(self.pprms.tolist())


    def run_pol(self,
        uvfs=None, uvw=None, runmf=False, iprms=None, ierrors=None, ftype=None, fwght=None, bands=None,
        sampler=None, bound=None, stokes=["q", "u"], spectrum=None, freq_ref=None,
        npix=128, mindr=3, bprms=None,
        save_path=None, source=None, date=None
    ):
        self.source = source
        self.date = date
        self.npix = npix
        self.mindr = mindr
        self.bprms = bprms

        nfreq = len(uvfs)
        ufreq = [uvfs[i].freq for i in range(nfreq)]

        nmod = int(np.round(iprms["nmod"]))

        print(f"\n# Modeling to Stokes parameters ({', '.join(stokes).upper()})")

        # set saving path
        if save_path is None:
            if self.save_path is None:
                raise Exception("Saving path is not assigned. (save_path={...})")
            save_path = self.save_path

        # set Stokes I model parameters
        if iprms is None:
            if self.iprms is None:
                raise Exception("Model parameters are not assigned. (irpms={...})")
            iprms = self.iprms

        # modeling
        for nfreq_ in range(nfreq):
            uvf_ = gamvas.utils.set_uvf([uvfs[nfreq_]], type="sf")

            # set total, model flux density
            sblf = []
            if runmf:
                if spectrum == "spl":
                    for nmod_ in range(nmod):
                        mask_s = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_S")[0][0]
                        mask_a = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_alpha")[0][0]
                        usblf =\
                            gamvas.functions.S_spl(
                                freq_ref,
                                uvf_.freq,
                                ufloat(np.ma.getdata(iprms).tolist()[mask_s], ierrors[mask_s]),
                                ufloat(np.ma.getdata(iprms).tolist()[mask_a], ierrors[mask_a])
                            )
                        sblf.append(usblf)
                elif spectrum == "cpl":
                    for nmod_ in range(nmod):
                        if nmod_ == 0:
                            set_thick = True
                        else:
                            set_thick = bool(np.round(iprms[f"{nmod_ + 1}_thick"]))
                        if set_thick:
                            mask_s = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_S")[0][0]
                            mask_f = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_freq")[0][0]
                            mask_a = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_alpha")[0][0]
                            usblf =\
                                gamvas.functions.S_cpl(
                                    uvf_.freq,
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_s], ierrors[mask_s]),
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_f], ierrors[mask_f]),
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_a], ierrors[mask_a])
                                )
                            sblf.append(usblf)
                        else:
                            mask_s = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_S")[0][0]
                            mask_a = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_alpha")[0][0]
                            usblf =\
                                gamvas.functions.S_spl(
                                    freq_ref,
                                    uvf_.freq,
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_s], ierrors[mask_s]),
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_a], ierrors[mask_a])
                                )
                            sblf.append(usblf)
                elif spectrum == "ssa":
                    for nmod_ in range(nmod):
                        if nmod_ == 0:
                            set_thick = True
                        else:
                            set_thick = bool(np.round(iprms[f"{nmod_ + 1}_thick"]))
                        if set_thick:
                            mask_s = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_S")[0][0]
                            mask_f = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_freq")[0][0]
                            mask_a = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_alpha")[0][0]
                            usblf =\
                                gamvas.functions.SSA(
                                    uvf_.freq,
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_s], ierrors[mask_s]),
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_f], ierrors[mask_f]),
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_a], ierrors[mask_a])
                                )
                            sblf.append(usblf)
                        else:
                            mask_s = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_S")[0][0]
                            mask_a = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_alpha")[0][0]
                            usblf =\
                                gamvas.functions.S_spl(
                                    freq_ref,
                                    uvf_.freq,
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_s], ierrors[mask_s]),
                                    ufloat(np.ma.getdata(iprms).tolist()[mask_a], ierrors[mask_a])
                                )
                            sblf.append(usblf)
            else:
                for nmod_ in range(nmod):
                    mask_s = np.where(np.array(iprms.dtype.names) == f"{nmod_ + 1}_S")[0][0]
                    usblf = ufloat(np.ma.getdata(iprms).tolist()[mask_s], ierrors[mask_s])
                    sblf.append(usblf)

            uiflux = np.sum(sblf)
            sblf = unp.nominal_values(sblf) / np.sqrt(2)

            for nstoke, stoke in enumerate(stokes):
                # set ata
                uvf = gamvas.utils.set_uvf([uvfs[nfreq_]], type="sf")
                iselect = uvf.select
                uvf.ploter.prms = iprms

                iprms_a = [iprms[f"{i + 1}_a"] for i in range(nmod)]
                iprms_l = [0] + [iprms[f"{i + 2}_l"] for i in range(nmod - 1)]
                iprms_m = [0] + [iprms[f"{i + 2}_m"] for i in range(nmod - 1)]

                x =\
                (
                    np.ma.getdata(uvf.data["u"]),
                    np.ma.getdata(uvf.data["v"]),
                    iprms_a,
                    iprms_l,
                    iprms_m
                )

                if runmf:
                    fitset = "mf"
                    set_spectrum_ = True
                    ifsingle = False
                else:
                    fitset = "sf"
                    set_spectrum_ = True
                    ifsingle = True

                # append polarization model visibility
                uvf.append_visibility_model(
                    freq_ref=freq_ref,
                    freq=uvf.freq,
                    theta=iprms,
                    fitset=fitset,
                    spectrum=spectrum,
                    set_spectrum=True,
                    args=x
                )

                # extract total emission map (nominal beam)
                uvf.ploter.bprms = uvf.bprms
                uvf.ploter.prms = iprms
                uvf.ploter.fitset = fitset
                uvf.ploter.spectrum = spectrum
                returned =\
                    uvf.ploter.draw_image(
                        uvf=uvf,
                        returned=True,
                        plotimg=False,
                        npix=self.npix,
                        mindr=self.mindr,
                        plot_resi=True,
                        addnoise=True,
                        freq_ref=freq_ref,
                        freq=uvf.freq,
                        ifsingle=ifsingle,
                        set_spectrum=set_spectrum_,
                    )
                uvf_.fits_image_vi = returned[0]
                uvf_.fits_image_rms_i = gamvas.utils.cal_rms(returned[1])
                uvf_.fits_imgcntr_i = gamvas.utils.make_cntr(returned[0], rms=uvf_.fits_image_rms_i)
                uvf_.fits_clean_vflux_i = unp.nominal_values(uiflux)
                uvf_.fits_clean_dflux_i = unp.std_devs(uiflux)

                # extract total emission map (restored beam)
                bnom_ = uvf.ploter.bprms
                if fitset == "mf":
                    uvf.ploter.bprms = self.bprms
                else:
                    uvf.ploter.bprms = (bnom_[0], bnom_[0], 0)
                returned =\
                    uvf.ploter.draw_image(
                        uvf=uvf,
                        returned=True,
                        plotimg=False,
                        npix=self.npix,
                        mindr=self.mindr,
                        plot_resi=True,
                        addnoise=True,
                        freq_ref=uvf.freq,
                        freq=uvf.freq,
                        ifsingle=True,
                        set_spectrum=set_spectrum_,
                    )
                uvf.ploter.bprms = bnom_
                uvf_.fits_image_vi_res = returned[0]
                uvf_.fits_image_rms_i_res = gamvas.utils.cal_rms(returned[1])
                uvf_.fits_imgcntr_i_res = gamvas.utils.make_cntr(returned[0], rms=uvf_.fits_image_rms_i)

                # load polarization uvf
                uvf.load_uvf(select=stoke, uvw=uvf.uvw, prt=False)
                data = uvf.data
                self.freq = uvf.freq
                self.stoke = stoke
                self.iprms = iprms

                # set save_path
                if runmf:
                    save_path_ =\
                        save_path.replace(
                            f"Pol_{iselect.upper()}",
                            f"Pol_{stoke.upper()}"
                        )
                    gamvas.utils.mkdir(save_path_)
                else:
                    save_path_ =\
                        save_path.replace(
                            f"Pol_{iselect.upper()}/{self.freq:.1f}",
                            f"Pol_{stoke.upper()}"
                        )
                    gamvas.utils.mkdir(save_path_)
                save_path_ = f"{save_path_}/{self.freq:.1f}/"
                gamvas.utils.mkdir(save_path_)

                iprms_a = [iprms[f"{i + 1}_a"] for i in range(nmod)]
                iprms_l = [0] + [iprms[f"{i + 2}_l"] for i in range(nmod - 1)]
                iprms_m = [0] + [iprms[f"{i + 2}_m"] for i in range(nmod - 1)]
                self.nmod = nmod

                # set boundary
                bnd_S =\
                    gamvas.utils.set_boundary(
                        nmod=nmod, select=stoke, spectrum="single", sblf=sblf
                    )

                bnds =\
                    gamvas.utils.sarray(
                        bnd_S,
                        [f"S_{i + 1}" for i in range(nmod)],
                        ["f8" for i in range(nmod)]
                    )

                # set fit weights
                if ftype is None:
                    if self.ftype is None:
                        raise Exception("Fit types are not assigned. (ftype={...})")
                    ftype = copy.deepcopy(self.ftype)
                if fwght is None:
                    if self.fwght is None:
                        fwght =\
                            gamvas.utils.get_fwght(
                                ftype, data, uvf.clamp["clamp"], uvf.clphs["clphs"]
                            )
                    else:
                        fwght = copy.deepcopy(self.fwght)
                self.fdict = dict(zip(ftype, fwght))

                # set uv-combinations
                clamp_uvcomb, clphs_uvcomb =\
                    gamvas.utils.set_uvcombination(
                        uvf.data, uvf.tmpl_clamp, uvf.tmpl_clphs
                    )

                # set x parameters
                self.x =\
                (
                    np.ma.getdata(uvf.data["u"]),
                    np.ma.getdata(uvf.data["v"]),
                    iprms_a,
                    iprms_l,
                    iprms_m
                )

                # set y parameters
                self.y =\
                (
                    np.ma.getdata(uvf.data[f"vis_{stoke}"]),
                    np.ma.getdata(uvf.clamp[f"clamp_{stoke}"]),
                    np.ma.getdata(uvf.clphs[f"clphs_{stoke}"]),
                    clamp_uvcomb,
                    clphs_uvcomb
                )

                # set yerr parameters
                self.yerr =\
                (
                    np.ma.getdata(uvf.data[f"sigma_{stoke}"]),
                    np.ma.getdata(uvf.clamp[f"sigma_clamp_{stoke}"]),
                    np.ma.getdata(uvf.clphs[f"sigma_clphs_{stoke}"])
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
                if sampler is None:
                    if self.sampler is None:
                        if nmod < 3:
                            insample = "rwalk"
                        elif 3 <= nmod < 11:
                            insample = "rslice"
                        elif 11 <= nmod:
                            insample = "slice"
                    else:
                        insample = self.sampler
                else:
                    insample = sampler

                # set bound
                if bound is None:
                    bound = self.bound
                else:
                    bound = bound

                # print running information
                runtxt = f"\n# Running {uvf.freq:.1f} GHz ... "
                runtxt += f"(Pol {stoke.upper()})"
                print(runtxt)

                # run dynesty
                self.run_util(
                    nmod=nmod,
                    sample=insample,
                    bound=bound,
                    dlogz=False,
                    save_path=save_path_,
                    save_name="model_params.xlsx",
                    save_xlsx=True
                )

                # extract statistical values
                logz_v = float(self.results.logz[-1])
                logz_d = float(self.results.logzerr[-1])
                pprms = self.pprms

                # add model visibility
                uvf.append_visibility_model(
                    freq_ref=uvf.freq,
                    freq=uvf.freq,
                    theta=pprms,
                    pol=True,
                    fitset="sf",
                    spectrum="single",
                    set_spectrum=False,
                    args=self.x
                )

                # print statistical values : reduced chi-square, Akaike information criterion, Bayesian information criterion
                uvcomb = (
                    uvf.clamp["clamp"], uvf.clphs["clphs"],
                    uvf.clamp["sigma_clamp"], uvf.clphs["sigma_clphs"],
                    clamp_uvcomb, clphs_uvcomb
                )
                fty, chi, aic, bic =\
                    gamvas.utils.print_stats(
                        uvf,
                        uvcomb,
                        self.nmod,
                        logz_v,
                        logz_d,
                        ftype
                    )
                self.print_prms(
                    ufreq=[np.round(self.freq,1)],
                    fitset="sf",
                    spectrum="single",
                    stats=(fty, chi, aic, bic, logz_v, logz_d),
                    printmsg=True,
                    save_path=save_path_,
                    save_name="model_result.txt"
                )

                uvf.ploter.bprms = uvf.bprms
                uvf.ploter.prms = iprms
                uvf.ploter.pprms = pprms

                uvf.ploter.clq_obs =\
                    (
                        copy.deepcopy(uvf.clamp),
                        copy.deepcopy(uvf.clphs)
                    )
                uvf.ploter.clq_mod =\
                    gamvas.utils.set_closure(
                        data["u"], data["v"], uvf.data["vism"],
                        np.zeros(uvf.data["vism"].shape[0]), data["ant_name1"], data["ant_name2"],
                        self.y[3], self.y[4]
                    )

                # plot and save figures
                uvf.ploter.draw_trplot(
                    result=self.results,
                    pol=True,
                    nmod=self.nmod,
                    ifsingle=True,
                    set_spectrum=False,
                    fontsize=20,
                    save_path=save_path_,
                    save_name=f"{self.source}.{self.date}.trplot.{fitset}",
                    save_form="pdf"
                )

                uvf.ploter.draw_cnplot(
                    result=self.results,
                    pol=True,
                    nmod=self.nmod,
                    ifsingle=True,
                    set_spectrum=False,
                    fontsize=20,
                    save_path=save_path_,
                    save_name=f"{self.source}.{self.date}.cnplot.{fitset}",
                    save_form="pdf"
                )

                uvf.ploter.draw_radplot(
                    uvf=uvf,
                    select=self.stoke,
                    plotimg=False,
                    show_title=False,
                    plotvism=True,
                    save_path=save_path_,
                    save_name=f"{self.source}.{self.date}.radplot.{fitset}.model",
                    save_form="pdf"
                )

                if "clamp" in ftype:
                    uvf.ploter.draw_closure(
                        type="clamp",
                        model=True,
                        plotimg=False,
                        save_img=True,
                        save_path=save_path_,
                        save_name=f"{self.source}.{self.date}.clphs.{fitset}",
                        save_form="pdf"
                    )

                if "clphs" in ftype:
                    uvf.ploter.draw_closure(
                        type="clphs",
                        model=True,
                        plotimg=False,
                        save_img=True,
                        save_path=save_path_,
                        save_name=f"{self.source}.{self.date}.clphs.{fitset}",
                        save_form="pdf"
                    )

                returned =\
                    uvf.ploter.draw_image(
                        uvf=uvf,
                        pol=True,
                        returned=True,
                        plotimg=False,
                        npix=self.npix,
                        mindr=self.mindr,
                        plot_resi=True,
                        addnoise=True,
                        freq_ref=uvf.freq,
                        freq=uvf.freq,
                        ifsingle=True,
                        set_spectrum="sf",
                        save_path=save_path_,
                        save_name=f"{self.source}.{self.date}.img.{fitset}",
                        save_form="pdf"
                    )

                if self.stoke.upper() == "Q":
                    uvf_.fits_image_vq = returned[0]
                    uvf_.fits_image_rms_q = gamvas.utils.cal_rms(returned[1])
                    uvf_.fits_imgcntr_q = gamvas.utils.make_cntr(returned[0], rms=uvf_.fits_image_rms_q)
                    uvf_.fits_clean_vflux_q = unp.nominal_values(self.uflux)
                    uvf_.fits_clean_dflux_q = unp.std_devs(self.uflux)
                if self.stoke.upper() == "U":
                    uvf_.fits_image_vu = returned[0]
                    uvf_.fits_image_rms_u = gamvas.utils.cal_rms(returned[1])
                    uvf_.fits_imgcntr_u = gamvas.utils.make_cntr(returned[0], rms=uvf_.fits_image_rms_u)
                    uvf_.fits_clean_vflux_u = unp.nominal_values(self.uflux)
                    uvf_.fits_clean_dflux_u = unp.std_devs(self.uflux)
                if self.stoke.upper() == "V":
                    uvf_.fits_image_vv = returned[0]
                    uvf_.fits_image_rms_v = gamvas.utils.cal_rms(returned[1])
                    uvf_.fits_imgcntr_v = gamvas.utils.make_cntr(returned[0], rms=uvf_.fits_image_rms_v)
                    uvf_.fits_clean_vflux_v = unp.nominal_values(self.uflux)
                    uvf_.fits_clean_dflux_v = unp.std_devs(self.uflux)

                # set restoring beam parameters
                if fitset == "mf":
                    uvf.ploter.bprms = self.bprms
                else:
                    uvf.ploter.bprms = (bnom_[0], bnom_[0], 0)
                returned_res =\
                    uvf.ploter.draw_image(
                        uvf=uvf,
                        pol=True,
                        returned=True,
                        plotimg=False,
                        npix=self.npix,
                        mindr=self.mindr,
                        plot_resi=True,
                        addnoise=True,
                        freq_ref=uvf.freq,
                        freq=uvf.freq,
                        ifsingle=True,
                        set_spectrum="sf",
                        save_path=save_path_,
                        save_name=f"{self.source}.{self.date}.img.{fitset}.restore",
                        save_form="pdf"
                    )
                if self.stoke.upper() == "Q":
                    uvf_.fits_image_vq_res = returned_res[0]
                    uvf_.fits_image_rms_q_res = gamvas.utils.cal_rms(returned_res[1])
                    uvf_.fits_imgcntr_q_res = gamvas.utils.make_cntr(returned_res[0], rms=uvf_.fits_image_rms_q_res)
                    uvf_.fits_clean_vflux_q = unp.nominal_values(self.uflux)
                    uvf_.fits_clean_dflux_q = unp.std_devs(self.uflux)
                if self.stoke.upper() == "U":
                    uvf_.fits_image_vu_res = returned_res[0]
                    uvf_.fits_image_rms_u_res = gamvas.utils.cal_rms(returned_res[1])
                    uvf_.fits_imgcntr_u_res = gamvas.utils.make_cntr(returned_res[0], rms=uvf_.fits_image_rms_u_res)
                    uvf_.fits_clean_vflux_u = unp.nominal_values(self.uflux)
                    uvf_.fits_clean_dflux_u = unp.std_devs(self.uflux)
                if self.stoke.upper() == "V":
                    uvf_.fits_image_vv_res = returned_res[0]
                    uvf_.fits_image_rms_v_res = gamvas.utils.cal_rms(returned_res[1])
                    uvf_.fits_imgcntr_v_res = gamvas.utils.make_cntr(returned_res[0], rms=uvf_.fits_image_rms_v_res)
                    uvf_.fits_clean_vflux_v = unp.nominal_values(self.uflux)
                    uvf_.fits_clean_dflux_v = unp.std_devs(self.uflux)

            # set pol image saving path
            if runmf:
                save_path_p = f"{save_path}/mf/"
            else:
                save_path_p = f"{save_path}/"


            # draw nominal-beam image
            ## set fits restoring beam size
            uvf_.fits_bmin = uvf_.bprms[0]
            uvf_.fits_bmaj = uvf_.bprms[1]
            uvf_.fits_bpa = uvf_.bprms[2]

            ## set image parameters
            fnpix = int(np.round(self.npix / 256))
            if fnpix == 0:
                fnpix = 1
            uvf_.fits_npix = self.npix
            uvf_.fits_psize = 24 * u.mas.to(u.deg) / self.npix
            uvf_.fits_grid_ra = uvf.xgrid/u.deg.to(u.mas)
            uvf_.fits_grid_dec = uvf.ygrid/u.deg.to(u.mas)
            uvf_.cal_polarization(
                snr_i=3, snr_p=3,
                evpalength=0.3 * fnpix, evpawidth=0.7 * fnpix
            )

            ## draw image
            uvf_.ploter.draw_fits_image(
                uvf_,
                select="p",
                rms=uvf_.fits_image_rms_i,
                xlim=False,
                ylim=False,
                cmap_snr_i=3,
                cmap_snr_p=3,
                fsize=6,
                contourw=0.5,
                pagap=5 * fnpix,
                plotimg=False,
                show_title=False,
                save_path=save_path_p,
                save_name=f"{self.source}.{self.date}.imgp.{fitset}.nominal.{bands[nfreq_]}",
                save_form="pdf"
            )

            ## draw restored-beam image
            ## set fits restoring beam size
            if fitset == "mf":
                uvf_.fits_bmin = self.bprms[0]
                uvf_.fits_bmaj = self.bprms[1]
                uvf_.fits_bpa = self.bprms[2]
            else:
                uvf_.fits_bmin = uvf_.bprms[0]
                uvf_.fits_bmaj = uvf_.bprms[0]
                uvf_.fits_bpa = 0

            ## set restored images
            uvf_.fits_image_vi = uvf_.fits_image_vi_res
            uvf_.fits_image_vq = uvf_.fits_image_vq_res
            uvf_.fits_image_vu = uvf_.fits_image_vu_res
            uvf_.fits_image_rms_i = uvf_.fits_image_rms_i_res
            uvf_.fits_image_rms_q = uvf_.fits_image_rms_q_res
            uvf_.fits_image_rms_u = uvf_.fits_image_rms_u_res
            uvf_.fits_imgcntr_i = uvf_.fits_imgcntr_i_res
            uvf_.fits_imgcntr_q = uvf_.fits_imgcntr_q_res
            uvf_.fits_imgcntr_u = uvf_.fits_imgcntr_u_res

            ## set image parameters
            uvf_.fits_npix = self.npix
            uvf_.fits_psize = 24 * u.mas.to(u.deg) / self.npix
            uvf_.fits_grid_ra = uvf.xgrid/u.deg.to(u.mas)
            uvf_.fits_grid_dec = uvf.ygrid/u.deg.to(u.mas)
            uvf_.cal_polarization(
                snr_i=3, snr_p=3,
                evpalength=0.3 * fnpix, evpawidth=0.7 * fnpix
            )

            ## draw image
            uvf_.ploter.draw_fits_image(
                uvf_,
                select="p",
                rms=uvf_.fits_image_rms_i,
                xlim=False,
                ylim=False,
                cmap_snr_i=3,
                cmap_snr_p=3,
                fsize=6,
                contourw=0.5,
                pagap=5 * fnpix,
                plotimg=False,
                show_title=False,
                save_path=save_path_p,
                save_name=f"{self.source}.{self.date}.imgp.{fitset}.restore.{bands[nfreq_]}",
                save_form="pdf"
            )

            pprms_q = self.pprms_q
            pprms_u = self.pprms_u
            lp = np.sqrt(pprms_q**2 + pprms_u**2)
            fp = lp / np.ma.getdata(sblf) * 100
            pa = 0.5 * np.arctan2(pprms_u, pprms_q) * u.rad.to(u.deg)

            modelprms = open(save_path_p + "model_result.txt", mode="a")
            for i in range(nmod):
                outprint = f"# ({self.freq:.1f} GHz, pol) Model {i+1} : "
                outprint += f"{np.round(lp[i] * 1e3, 2)} [mJy], "
                outprint += f"{np.round(fp[i], 2)} [%], "
                outprint += f"{np.round(pa[i], 2)} [deg]"
                modelprms.write(f"{outprint}\n")
            modelprms.close()


    def print_prms(self,
        ufreq, fitset="sf", spectrum="spl", stats=None, printmsg=False, save_path=False, save_name=False
    ):
        if save_path:
            gamvas.utils.mkdir(save_path)
        iprms = self.iprms.copy()
        pprms = self.pprms.copy()
        nmod = int(np.round(iprms["nmod"]))
        if not isinstance(ufreq, list):
            if isinstance(ufreq, np.ndarray):
                ufreq = ufreq
            else:
                ufreq = np.array([ufreq])
        if isinstance(ufreq, list):
            ufreq = np.array(ufreq)

        if save_path and save_name:
            modelprms = open(save_path + save_name, "w")
            modelprms.close()

        for nfreq, freq in enumerate(ufreq):
            for i in range(nmod):
                if i == 0:
                    smax_, a_, l_, m_ = pprms[f"{i+1}_S"], iprms[f"{i+1}_a"], 0, 0
                else:
                    smax_, a_, l_, m_ = pprms[f"{i+1}_S"], iprms[f"{i+1}_a"], iprms[f"{i+1}_l"], iprms[f"{i+1}_m"]
                S_ = smax_

                r_, p_ = np.sqrt(l_**2 + m_**2), np.arctan2(l_, m_) * u.rad.to(u.deg)
                outprint = f"# ({freq:.1f} GHz, Stokes {self.stoke.upper()}) Model {i+1} : {S_:.3f}v {+r_:.3f}v {p_:.3f}v {a_:.3f}v"

                if printmsg:
                    print(outprint)

                if save_path and save_name:
                    modelprms = open(save_path + save_name, mode="a")
                    modelprms.write(f"{outprint}\n")
                    modelprms.close()

        if save_path and save_name:
            modelprms = open(save_path + save_name, mode="a")
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
