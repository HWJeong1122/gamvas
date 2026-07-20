
import os
import sys
import gc
import copy
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
from numba import njit, jit
import itertools as it
from astropy import units as au
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import dynesty
from dynesty import NestedSampler
from dynesty.pool import Pool
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty.utils import quantile as dyquan
from uncertainties import ufloat

import gamvas as gv

r2m = au.rad.to(au.mas)
d2m = au.deg.to(au.mas)

fields_sf = ["Sq", "Su", "a", "l", "m"]
dtypes_sf = ["f8", "f8", "f8", "f8", "f8"]

fields_mf = ["Sq", "Su", "a", "l", "m", "freq", "alpha", "rm"]
dtypes_mf = ["f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8"]

class polarization:
    def __init__(
        self,
        uvfs=None, uvw="u", maxn=10, fixnmod=False, factor_sblf=1.0,
        model="gaussian", spectrum="flat", sampler="slice", bound="multi",
        mapfov=20, npix=1024, bnds=None, bnd_a=5, bnd_l=[-10, +10],
        bnd_m=[-10, +10], bnd_f=(13.5, 140), bnd_pa=(None, None),
        freq_ref=None, bands=None, freqtype="sf", bprms=None, source=None,
        date=None, mindr=3, save_path=None, ncpu=1
    ):
        self.uvfs = uvfs
        self.uvw = uvw
        self.maxn = maxn
        self.fixnmod = fixnmod
        self.factor_sblf = factor_sblf
        self.model = model
        self.spectrum = spectrum
        self.sampler = sampler
        self.bound = bound
        self.mapfov = mapfov
        self.npix = npix
        self.boundset = bnds
        self.bnd_a = bnd_a
        self.bnd_l = bnd_l
        self.bnd_m = bnd_m
        self.bnd_f = bnd_f
        self.bnd_pa = bnd_pa
        self.freq_ref = freq_ref
        self.bands = bands
        self.freqtype = freqtype
        self.bprms = bprms
        self.source = source
        self.date = date
        self.mindr = mindr
        self.save_path = save_path
        self.ncpu = ncpu


    def get_nmprms(self):
        """
        Get the number of parameters
        """
        mprms = self.mprms.copy()
        vals = rfn.structured_to_unstructured(mprms)
        nmod = round(float(mprms["nmod"]))

        mask_thick = np.array(
            list(map(
                lambda x: "thick" in x,
                mprms.dtype.names
            ))
        )

        mask_thick = np.round(
            vals[mask_thick]
        ).astype(int)

        nmprms = len(vals) - 2 * (len(mask_thick) - mask_thick.sum()) - 1

        self.nmprms = nmprms


    def prior_transform(self, theta):
        """
        Transform priori boundary conditions
        (a boundary between A to B: [B - A] * x + A)
        Args:
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

        for i in range(self.nmod):
            thick_offset = 0
            if self.spectrum in ["cpl", "ssa"]:
                results.append(1.98 * theta[1 + ndim] - 0.49)
                thick_offset = 1

            self.set_field(nmod=i + 1)
            for nfield, field in enumerate(self.fields):
                results.append(
                    (bounds[field][i][1] - bounds[field][i][0])
                    * theta[1 + thick_offset + ndim + nfield]
                    + bounds[field][i][0]
                )
            ndim += self.dims

        return results


    def set_field(self, nmod=1):
        """
        Set field names and dimensions
        Args:
            nmod (int): The number of models
        """
        if self.spectrum == "flat":
            self.dims = 5
            self.fields = ["Sq", "Su", "a", "l", "m"]
        elif self.spectrum == "spl":
            self.dims = 7
            self.fields = ["Sq", "Su", "a", "l", "m", "alpha", "rm"]
        elif self.spectrum in ["cpl", "ssa"]:
            self.dims = 9
            self.fields = ["Sq", "Su", "a", "l", "m", "alpha", "freq", "rm"]
        else:
            raise NotImplementedError("To be updated.")


    def set_ndim(self, nmod=1):
        """
        Set the number of dimensions
        Args:
            nmod (int): The number of models
        """
        ndim = 1
        self.ufreq = np.unique(self.x[1])
        for i in range(nmod):
            _nmod = i + 1
            self.set_field(nmod=_nmod)
            ndim += self.dims
        self.ndim = ndim


    def set_index(self):
        """
        Set field index
        """
        _index = ["nmod"]
        for i in range(self.nmod):
            if self.spectrum in ["cpl", "ssa"]:
                _index = _index + [f"{i + 1}_thick"]

            self.set_field(nmod=i + 1)
            nums = np.full(self.dims, i + 1)
            fields = self.fields
            index_list = [
                "_".join([str(x), y])
                for x, y in zip(nums, fields)
            ]

            _index = _index + index_list
        self.index = _index


    def get_results(
        self,
        qs=(0.025, 0.500, 0.975), save_path=False, save_name=False,
        save_xlsx=False
    ):
        """
        Get modeling results (parameters)
        Args:
            qs (tuple, flaot): quantile values
            save_path (str): path to save the results
            save_name (str): name of the file to save the results
            save_xlsx (bool): if True, save the results in xlsx format
        """
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
            nmod = round(float(qms[0]))

            idxn = np.array(
                list(map(
                    lambda x: x.split("_")[0],
                    self.index[1:]
                )),
                dtype=int
            )

            mask_nmod = idxn <= nmod
            mask_nmod = np.append(np.array([True]), mask_nmod)

            qls_ = qls[mask_nmod]
            qms_ = qms[mask_nmod]
            qhs_ = qhs[mask_nmod]
            idx_ = np.array(self.index)[mask_nmod]
            prms_ = np.array([qls_, qms_, qhs_])

            out_xlsx = pd.DataFrame(
                prms_, index=["lolim", "value", "uplim"]
            ).T

            out_xlsx["idx"] = idx_
            out_xlsx.to_excel(f"{save_path}{save_name}")
            self.out_xlsx = out_xlsx


    def run_modeling(
        self,
        uvfs=None, uvw="u", maxn=10, fixnmod=False, factor_sblf=1.0,
        model="gaussian", spectrum="flat", sampler="slice", bound="multi",
        mapfov=20, npix=1024, bnds=None, bnd_a=5, bnd_l=[-10, +10],
        bnd_m=[-10, +10], bnd_f=(13.5, 140), bnd_pa=(None, None),
        freq_ref=None, bands=None, freqtype="sf", bprms=None, source=None,
        date=None, mindr=3, save_path=None, ncpu=1
    ):
        self.uvfs = uvfs
        self.uvw = uvw
        self.maxn = maxn
        self.fixnmod = fixnmod
        self.factor_sblf = factor_sblf
        self.model = model
        self.spectrum = spectrum
        self.sampler = sampler
        self.bound = bound
        self.mapfov = mapfov
        self.npix = npix
        self.boundset = bnds
        self.bnd_a = bnd_a
        self.bnd_l = bnd_l
        self.bnd_m = bnd_m
        self.bnd_f = bnd_f
        self.bnd_pa = bnd_pa
        self.freq_ref = freq_ref
        self.bands = bands
        self.freqtype = freqtype
        self.bprms = bprms
        self.source = source
        self.date = date
        self.mindr = mindr
        self.ncpu = ncpu

        split_path = save_path.split("/")
        save_path = "/".join(split_path[:-2]) + "/Pol_P/"
        self.save_path = save_path

        uvfs = copy.deepcopy(self.uvfs)
        uvf = gv.utils.set_uvf(uvfs, dotype=freqtype)

        select_pol0 = []
        for nuvf in range(len(uvfs)):
            select_pol0.append(uvfs[nuvf].select_pol)

        if spectrum == "flat":
            freqtype = "sf"
            _freq = uvfs[0].freq_mean
            self.path_fig_init = f"{save_path}/{self.model}.{_freq:.1f}/"
            self.path_fig = self.path_fig_init
        else:
            freqtype = "mf"
            self.path_fig_init = save_path
            self.path_fig = f"{save_path}/mf.{self.model}.{self.spectrum}/"
        gv.utils.mkdir(self.path_fig)

        nfreq = len(uvfs)
        ufreq = [uvfs[i].freq for i in range(nfreq)]

        nmod = maxn

        # raise Exceptions
        if nfreq == 1 and spectrum != "flat":
            raise ValueError(
                "Single-frequency modeling "
                "is only supported for flat spectrum."
            )

        if nfreq == 2 and spectrum in ["cpl", "ssa", "poly"]:
            raise ValueError(
                "Two-frequency modeling "
                "is only supported for flat spectrum."
            )

        # short-baseline flux: for boundary conditions
        u = uvf.get_data(dotype="u") / 1e6
        v = uvf.get_data(dotype="v") / 1e6
        uvr = np.sqrt(u**2 + v**2)

        vis_q = uvf.get_data(dotype="vis_q")
        vis_u = uvf.get_data(dotype="vis_u")
        vis_p = vis_q + 1j * vis_u
        # plt.scatter(uvr, np.abs(vis_p), c="k", marker=".")
        # plt.show()
        # sys.exit()

        bslf_max = np.nanmax(np.abs(vis_p))
        sblf = self.factor_sblf * bslf_max

        # set boundary conditions
        for i in range(nfreq):
            nstokes = uvfs[i].nstokes
            if nstokes != 4:
                raise ValueError(
                    "The provided UVF file "
                    f"has non-4 Stokes parameters ({nstokes}) "
                    f"at {uvfs[i].freq_mean:.3f} GHz."
                )

        if self.boundset is None:
            if self.bnd_f is not None:
                _bnds = gv.utils.set_boundary(
                        nmod=nmod, select_pol="p",
                        spectrum=self.spectrum, sblf=sblf, bnd_a=self.bnd_a,
                        bnd_l=self.bnd_l, bnd_m=self.bnd_m, bnd_f=self.bnd_f
                )
            else:
                _bnds = gv.utils.set_boundary(
                        nmod=nmod, select_pol="p",
                        spectrum=self.spectrum, sblf=sblf, bnd_a=self.bnd_a,
                        bnd_l=self.bnd_l, bnd_m=self.bnd_m
                )
            bnds = gv.utils.structured_array(
                data=_bnds,
                field=fields_mf,
                dtype=dtypes_mf
            )

        else:
            bnds = gv.utils.structured_array(
                data=self.boundset,
                field=fields_mf,
                dtype=dtypes_mf
            )
        self.boundset = bnds

        # set arguments: x
        self.x = (
            uvfs[0].freq0,
            uvf.get_data(dotype="frequency").flatten(),
            uvf.get_data(dotype="u").flatten(),
            uvf.get_data(dotype="v").flatten()
        )

        # set arguments: y
        self.y = (
            uvf.get_data(dotype="vis_p").astype(np.complex64).flatten(),
            uvf.get_data(dotype="vis_q").astype(np.complex64).flatten(),
            uvf.get_data(dotype="vis_u").astype(np.complex64).flatten()
        )

        # set arguments: y error
        self.yerr = (
            uvf.get_data(dotype="sig_p").flatten(),
            uvf.get_data(dotype="sig_q").flatten(),
            uvf.get_data(dotype="sig_u").flatten(),
        )

        # set spectrum for jit operations
        spectrum = jit_spectrum(self.spectrum)

        # set model type for jit operations
        modeltype = jit_model(self.model)

        # set the number of free parameters
        self.nmod = nmod
        self.set_ndim(nmod=nmod)
        self.set_index()

        # map parameter index to number
        prmidx2num = {
            "thick":0, "Sq":1, "Su":2, "a":3, "l":4, "m":5, "alpha":6,
            "freq":7, "rm":8
        }

        _index_num = self.index.copy()[1:]
        _index_num = [0] + list(map(
            lambda x: 10 * int(x.split("_")[0]) + prmidx2num[x.split("_")[1]],
            _index_num
        ))

        # set arguments
        self.args = (
            self.x, self.y, self.yerr,
            (
                spectrum, modeltype,
                np.array([self.bnd_pa[0]], dtype=np.float32),
                np.array([self.bnd_pa[1]], dtype=np.float32),
                np.array(_index_num, dtype=np.int64)
            )
        )

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
        print(
            "\n# Running... (linear polarization, "
            f"maxn_model={nmod}, "
            f"sampler={insample!r})"
        )

        # run dynesty
        self.run_util(
            nmod=nmod, sample=insample,
            save_path=self.path_fig,
            save_name="model_params.xlsx",
            save_xlsx=True
        )

        # extract statistical values
        logz_v = float(self.results.logz[-1])
        logz_d = float(self.results.logzerr[-1])
        prms = self.mprms
        nmod_ = round(float(prms["nmod"]))

        # add model visibility
        for nuvf in range(len(uvfs)):
            uvfs[nuvf].select_pol = "p"
            uvfs[nuvf].model_visibility_drop()
            uvfs[nuvf].model_visibility_append(
                freq_ref=uvfs[0].freq0, theta=prms,
                model=self.model, spectrum=self.spectrum, closure=False
            )

        # re-set uvf
        uvf = gv.utils.set_uvf(uvfs, dotype=freqtype)

        # print statistics
        fty, chi, aic, bic = gv.utils.print_stats(
            uvf=uvf, uvcomb=None, k=self.nmprms, logz=logz_v, dlogz=logz_d,
            dotype=["vis"], pol=True
        )
        # uvf=None, uvcomb=None, k=None, logz=None, dlogz=None, dotype=None,
        # pol=False

        # print model parameters
        self.print_prms(
            ufreq=uvf.ufreq, spectrum=self.spectrum, model=self.model,
            stats=(fty, chi, aic, bic, logz_v, logz_d),
            prt=True,
            save_path=self.path_fig,
            save_name="model_result.txt"
        )

        uvf.bprms = gv.utils.fit_beam([uvf.u.flatten(), uvf.v.flatten()], sig=None, uvw=self.uvw)
        uvf.prms = self.mprms
        uvf.ploter.bprms = uvf.bprms
        uvf.ploter.prms = self.mprms
        uvf.ploter.spectrum = self.spectrum

        # plot and save figures
        # trace plot
        uvf.ploter.draw_trplot(
            result=self.results, nmod=nmod_, relmod=False,
            model=self.model,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.trplot",
            save_form="pdf"
        )

        # corner plot
        uvf.ploter.draw_cnplot(
            result=self.results, nmod=nmod_, relmod=False,
            model=self.model,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.cnplot",
            save_form="pdf"
        )

        # radial plot
        _ = uvf.select_pol
        select_pols = ["q", "u", "p"]
        for nsp, _select_pol in enumerate(select_pols):
            uvf.select_pol = _select_pol

            if _select_pol == "q":
                uvf.vism = uvf.vism_q
            else:
                uvf.vism = uvf.vism_u
            uvf.set_data(prt=False)

            uvf.ploter.draw_radplot(
                uvf, plotimg=False, plotmodel=True,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.radplot.{_select_pol}",
                save_form="pdf"
            )

            # dirty map
            uvf.ploter.draw_dirtymap(
                uvf, plotimg=False, plot_resi=True,
                npix=self.npix, uvw=self.uvw,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.resimap.{_select_pol}",
                save_form="pdf"
            )

            # reconstructed image (single-frequency beam)
            for nuvf in range(len(uvfs)):
                uvfs[nuvf].select_pol = _select_pol

                if _select_pol == "q":
                    uvfs[nuvf].vism = uvfs[nuvf].vism_q
                else:
                    uvfs[nuvf].vism = uvfs[nuvf].vism_u
                uvfs[nuvf].set_data(prt=False)

                uvfs[nuvf].ploter.bprms = uvfs[nuvf].bprms
                uvfs[nuvf].ploter.prms = self.mprms
                uvfs[nuvf].ploter.spectrum = self.spectrum
                uvfs[nuvf].ploter.draw_image(
                    uvf=uvfs[nuvf], pol=True, plotimg=False,
                    npix=self.npix, mindr=self.mindr, plot_resi=False,
                    addnoise=True,
                    freq_ref=uvfs[0].freq0,
                    freq=uvfs[nuvf].freq_mean,
                    model=self.model,
                    save_path=self.path_fig,
                    save_name=(
                        f"{self.source}.{self.date}.img.sf.{self.bands[nuvf]}"
                        f"{_select_pol}"
                    ),
                    save_form="pdf"
                )

                # reconstructed image (single-frequency minor beam)
                if freqtype == "sf":
                    # set circular beam
                    bprms = (uvfs[nuvf].bprms[1], uvfs[nuvf].bprms[1], 0)

                    uvfs[nuvf].ploter.draw_image(
                        uvf=uvfs[nuvf], pol=True, plotimg=False, bprms=bprms,
                        npix=self.npix, mindr=self.mindr, plot_resi=False,
                        addnoise=True,
                        freq_ref=uvfs[0].freq0,
                        freq=uvfs[nuvf].freq_mean,
                        model=self.model,
                        save_path=self.path_fig,
                        save_name=(
                            f"{self.source}.{self.date}.img.sf."
                            f"{self.bands[nuvf]}.restore.{_select_pol}"
                        ),
                        save_form="pdf"
                    )

            # reconstructed image (multi-frequency beam)
            if freqtype == "mf":
                for i in range(len(uvfs)):
                    uvf.ploter.draw_image(
                        uvf=uvf, pol=True, plotimg=False,
                        npix=self.npix, mindr=self.mindr, plot_resi=True,
                        addnoise=True,
                        freq_ref=self.ufreq[0],
                        freq=self.ufreq[i],
                        model=self.model,
                        save_path=self.path_fig,
                        save_name=(
                            f"{self.source}.{self.date}.img.mf.{self.bands[i]}"
                            f"{_select_pol}"
                        ),
                        save_form="pdf"
                    )

        uvf.select_pol = _
        uvf.set_data(prt=False)

        gc.collect()

    def run_util(
        self,
        nmod=1, sample="rwalk", dlogz=False, save_path=False, save_name=False,
        save_xlsx=False
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
            self.ncpu, loglike=objective_function,
            prior_transform=self.prior_transform, logl_args=args
        ) as pool:
            sampler = dynesty.DynamicNestedSampler(
                pool.loglike,
                pool.prior_transform,
                ndim,
                sample=sample,
                bound=self.bound,
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
        dtypes = ["f8"] + ["f8" for i in range(len(fields) - 1)]

        mprms = gv.utils.structured_array(
                data=self.prms[1].copy(),
                field=fields,
                dtype=dtypes
            )

        self.errors = (self.prms[0] + self.prms[2]) / 2
        self.mprms = mprms
        self.get_nmprms()


    def print_prms(
        self,
        ufreq, model="gaussian", spectrum="spl", stats=None,
        prt=False, save_path=False, save_name=False
    ):
        """
        Print the model parameters
        Args:
            ufreq (list, float): unique frequency
            spectrum (str): spectrum type
                - 'flat': flat spectrum (no spectrum model)
                - 'spl': simple power-law
                - 'cpl': curved power-law
                - 'ssa': self-absorbed spectrum
            stats (tuple): statistical values (e.g., chi-square, AIC, BIC)
            prt (bool): Print the message
            save_path (str): path to save the model parameters
            save_name (str): name of the file to save the model parameters
        """
        if save_path:
            gv.utils.mkdir(save_path)

        mprms = self.mprms.copy()
        nmod = round(float(mprms["nmod"]))

        ufreq = np.atleast_1d(np.asarray(ufreq, dtype=float))

        if save_path and save_name:
            modelprms = open(f"{save_path}/{save_name}", "w")
            modelprms.close()

        out_txt_init = (
            f"# FLUX_LP    FLUX_Q    FLUX_U    RADIUS    POS.A    FWHM    "
            "|    EVPA"
        )
        modelprms = open(f"{save_path}/{save_name}", mode="a")
        modelprms.write(f"{out_txt_init}\n")
        modelprms.close()
        print(out_txt_init)

        for nfreq, freq in enumerate(ufreq):
            if model == "gaussian":
                for i in range(nmod):
                    idx_q = f"{i + 1}_Sq"
                    idx_u = f"{i + 1}_Su"
                    idx_a = f"{i + 1}_a"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"

                    idx_i = f"{i + 1}_alpha"
                    idx_f = f"{i + 1}_freq"
                    idx_r = f"{i + 1}_rm"
                    idx_t = f"{i + 1}_thick"

                    _dtype = mprms.dtype.names

                    prm_q_ = mprms[idx_q]
                    prm_u_ = mprms[idx_u]
                    prm_a_ = mprms[idx_a]
                    prm_l_ = mprms[idx_l]
                    prm_m_ = mprms[idx_m]

                    has_i = idx_i in _dtype
                    has_f = idx_f in _dtype
                    has_r = idx_r in _dtype
                    has_t = idx_t in _dtype

                    if has_i:
                        prm_i_ = mprms[idx_i]

                    if has_f:
                        prm_f_ = mprms[idx_f]

                    if has_r:
                        prm_r_ = mprms[idx_r]

                    if spectrum == "flat":
                        sq = prm_q_
                        su = prm_u_
                    elif spectrum == "spl":
                        raise NotImplementedError("To be updated.")
                    elif spectrum in ["cpl", "ssa"]:
                        raise NotImplementedError("To be updated.")

                    prm_r_, prm_p_ = (
                        np.sqrt(prm_l_**2 + prm_m_**2),
                        np.arctan2(prm_l_, prm_m_) * au.rad.to(au.deg)
                    )

                    sp = np.sqrt(sq**2 + su**2)
                    evpa = 0.5 * np.arctan2(su, sq)

                    out_txt = (
                        f"# ({freq:.1f} GHz) Model {i + 1}: "
                        f"{sp:7.3f}v {sq:7.3f}v {su:7.3f}v {+prm_r_:6.3f}v "
                        f"{prm_p_:8.3f}v {prm_a_:6.3f}v | "
                        f"EVPA: {evpa * 180 / np.pi:7.3f} (deg)"
                    )

                    if prt:
                        print(out_txt)

                    if save_path and save_name:
                        modelprms = open(f"{save_path}/{save_name}", mode="a")
                        modelprms.write(f"{out_txt}\n")
                        modelprms.close()

            elif model == "delta":
                for i in range(nmod):
                    idx_q = f"{i + 1}_Sq"
                    idx_u = f"{i + 1}_Su"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"

                    idx_i = f"{i + 1}_alpha"
                    idx_f = f"{i + 1}_freq"
                    idx_r = f"{i + 1}_rm"
                    idx_t = f"{i + 1}_thick"

                    _dtype = mprms.dtype.names

                    prm_q_ = mprms[idx_q]
                    prm_u_ = mprms[idx_u]
                    prm_l_ = mprms[idx_l]
                    prm_m_ = mprms[idx_m]

                    has_i = idx_i in _dtype
                    has_f = idx_f in _dtype
                    has_r = idx_r in _dtype
                    has_t = idx_t in _dtype

                    if has_i:
                        prm_i_ = mprms[idx_i]

                    if has_f:
                        prm_f_ = mprms[idx_f]

                    if has_r:
                        prm_r_ = mprms[idx_r]

                    if spectrum == "flat":
                        sq = prm_q_
                        su = prm_u_
                    elif spectrum == "spl":
                        raise NotImplementedError("To be updated.")
                    elif spectrum in ["cpl", "ssa"]:
                        raise NotImplementedError("To be updated.")

                    prm_r_, prm_p_ = (
                        np.sqrt(prm_l_**2 + prm_m_**2),
                        np.arctan2(prm_l_, prm_m_) * au.rad.to(au.deg)
                    )

                    sp = np.sqrt(sq**2 + su**2)
                    evpa = 0.5 * np.arctan2(su, sq)

                    out_txt = (
                        f"# ({freq:.1f} GHz) Model {i + 1}: "
                        f"{sq:.3f}v {sq:.3f}v {+prm_r_:.3f}v {prm_p_:.3f}v | "
                        f"EVPA: {evpa * 180 / np.pi:.3f}"
                    )

                    if prt:
                        print(out_txt)

                    if save_path and save_name:
                        modelprms = open(f"{save_path}/{save_name}", mode="a")
                        modelprms.write(f"{out_txt}\n")
                        modelprms.close()

        if save_path and save_name:
            modelprms = open(f"{save_path}/{save_name}", mode="a")
            chi_tot = 0
            aic_tot = 0
            bic_tot = 0
            for i in range(len(stats[0])):
                out_txt = (
                    f"Chi2_{stats[0][i]:9s}: {stats[1][i]:10.3f} | "
                    f"AIC_{stats[0][i]:9s} : {stats[2][i]:10.3f} | "
                    f"BIC_{stats[0][i]:9s} : {stats[3][i]:10.3f}\n"
                )

                modelprms.write(out_txt)

                chi_tot = stats[1][i]
                aic_tot = stats[2][i]
                bic_tot = stats[3][i]

            modelprms.write(f"Chi2_tot : {chi_tot:8.3f}\n")
            modelprms.write(f"AIC_tot  : {aic_tot:8.3f}\n")
            modelprms.write(f"BIC_tot  : {bic_tot:8.3f}\n")
            modelprms.write(f"logz : {stats[-2]:.3f} +/- {stats[-1]:.3f}\n")
            modelprms.close()

@jit(nopython=True)
def dvis(args, sq, su, l, m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        sq (float): flux density for Stokes Q
        su (float): flux density for Stokes U
        l (float): right ascension position
        m (float): declination position
    Returns:
        complex visibility of delta-function model
    """
    u = args[0] / r2m
    v = args[1] / r2m
    sp = (sq**2 + su**2)**0.5
    phi = 0.5 * np.arctan2(su, sq)
    out = (
        sp
        * np.exp(2j * np.pi * (u * l + v * m))
    )
    out_q = out * np.cos(2 * phi)
    out_u = out * np.sin(2 * phi)

    return out_q, out_u


@jit(nopython=True)
def gvis(args, sq, su, fwhm, l, m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        sq (float): flux density for Stokes Q
        su (float): flux density for Stokes U
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
    Returns:
        complex visibility of Gaussian model
    """
    u = args[0] / r2m
    v = args[1] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    sp = (sq**2 + su**2)**0.5
    phi = 0.5 * np.arctan2(su, sq)
    out = (
        sp
        * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2))
        * np.exp(2j * np.pi * (u * l + v * m))
    )
    out_q = out * np.cos(2 * phi)
    out_u = out * np.sin(2 * phi)

    return out_q, out_u


def jit_model(model):
    if model == "delta":
        modeltype = 0
    elif model == "gaussian":
        modeltype = 1
    else:
        modeltype = 1
        print(
            f"Unexpected model type: {model!r}. Assume 'gaussian'."
        )

    return modeltype


def jit_spectrum(spectrum):
    if spectrum == "flat":
        spectrum = 0
    elif spectrum == "spl":
        spectrum = 1
    elif spectrum == "cpl":
        spectrum = 2
    elif spectrum == "ssa":
        spectrum = 3
    else:
        availables = ["flat", "spl", "cpl", "ssa"]
        raise ValueError(
            f"Invalid spectrum type: {spectrum!r}.\n"
            f"Available spectrum types are: {availables}."
        )

    return spectrum


@jit(nopython=True)
def objective_function(theta, x, y, yerr, args):
    """
    Compute objective function (Bayesian Information Criterion)
    Args:
        theta (list): A list of parameters
        x (tuple): A tuple of x-arguments
        y (tuple): A tuple of y-arguments
        yerr (tuple): A tuple of y-error-arguments
        args (tuple): Arguments set
            args[0] (int): spectrum ('flat', 'spl', 'cpl', 'ssa')
            args[1] (int): modeltype (0:'delta', 1: 'gaussian')
            args[2] (float): lower boundary for jet position angle
            args[3] (float): upper boundary for jet position angle
            args[4] (list, int): parameter index
    Returns:
        Bayesian Information Criterion value (float)
    """

    dshape = x[1].shape
    dtypes = theta.dtype.names

    model_p = np.zeros(dshape, dtype=np.complex64)
    model_q = np.zeros(dshape, dtype=np.complex64)
    model_u = np.zeros(dshape, dtype=np.complex64)
    ufreq = np.unique(x[1])

    nmprm = 0

    pa1 = np.exp(1j * np.deg2rad(args[2][0]))
    pa2 = np.exp(1j * np.deg2rad(args[3][0]))
    span = np.angle(pa2 / pa1) % (2 * np.pi)

    nmod = round(float(theta[0]))
    prms = theta[1:]
    mask_pa = 0

    # compute model visibility
    if args[1] == 1:    # model == "gaussian"
        for i in range(nmod):
            idxval_q = (i + 1) * 10 + 1
            idxval_u = (i + 1) * 10 + 2
            idxval_a = (i + 1) * 10 + 3
            idxval_l = (i + 1) * 10 + 4
            idxval_m = (i + 1) * 10 + 5

            mask_q = args[4] == idxval_q
            mask_u = args[4] == idxval_u
            mask_a = args[4] == idxval_a
            mask_l = args[4] == idxval_l
            mask_m = args[4] == idxval_m

            _q = theta[mask_q][0]
            _u = theta[mask_u][0]
            _a = theta[mask_a][0]
            _l = theta[mask_l][0]
            _m = theta[mask_m][0]

            pa = np.exp(1j * np.angle(_m + 1j * _l))
            offset = np.angle(pa / pa1) % (2 * np.pi)
            if i != 0 and offset > span:
                mask_pa = 1

            if args[0] == 0: # flat spectrum
                _args = (x[2], x[3])
                nmprm += 5
                model = gvis(_args, _q, _u, _a, _l, _m)

            # else:  # spl | cpl | ssa | quad
            #     idxval_t = (i + 1) * 10 + 0
            #     idxval_i = (i + 1) * 10 + 6
            #     idxval_f = (i + 1) * 10 + 7
            #     idxval_r = (i + 1) * 10 + 8

            #     mask_t = args[4] == idxval_t
            #     mask_i = args[4] == idxval_i
            #     mask_f = args[4] == idxval_f
            #     mask_r = args[4] == idxval_r

            #     has_t = mask_t.sum()

            #     _i = theta[mask_i][0]

            #     if args[0] == 1:    # simple power-law
            #         _args = (x[0], x[1], x[2], x[3])
            #         model = gvis_spl(_args, _s, _a, _l, _m, _i)
            #         nmprm += 5 if has_lm else 3
            #     else:   # curved power-law or synchrotron self-absorption
            #         _f = theta[mask_f][0]
            #         if i == 0:
            #             _args = (x[1], x[2], x[3])
            #             if args[0] == 2:
            #                 model = gvis_cpl(
            #                     _args, _s, _a, _l, _m, _i, _f
            #                 )

            #             elif args[0] == 3:
            #                 model = gvis_ssa(
            #                     _args, _s, _a, _l, _m, _i, _f
            #                 )
            #         else:
            #             mask_t = round(float(theta[mask_t][0])) == 0

            #             if mask_t:
            #                 _args = (x[0], x[1], x[2], x[3])
            #                 model = gvis_spl(_args, _s, _a, _l, _m, _i)
            #                 nmprm += 5 if has_lm else 3
            #             else:
            #                 _args = (x[1], x[2], x[3])
            #                 if args[0] == 2:
            #                     model = gvis_cpl(
            #                         _args, _s, _a, _l, _m, _i, _f
            #                     )

            #                 elif args[0] == 3:
            #                     model = gvis_ssa(
            #                         _args, _s, _a, _l, _m, _i, _f
            #                     )

            #                 nmprm += 6 if has_lm else 4

            model_p += model[0] + 1j * model[1]
            model_q += model[0]
            model_u += model[1]

    elif args[1] == 0:  # model == "delta"
        for i in range(nmod):
            idxval_q = (i + 1) * 10 + 1
            idxval_u = (i + 1) * 10 + 2
            idxval_l = (i + 1) * 10 + 4
            idxval_m = (i + 1) * 10 + 5

            mask_q = args[4] == idxval_q
            mask_u = args[4] == idxval_u
            mask_l = args[4] == idxval_l
            mask_m = args[4] == idxval_m

            _q = theta[mask_q][0]
            _u = theta[mask_u][0]
            _l = theta[mask_l][0]
            _m = theta[mask_m][0]
            _r = np.sqrt(_l**2 + _m**2)

            pa = np.exp(1j * np.angle(_m + 1j * _l))
            offset = np.angle(pa / pa1) % (2 * np.pi)
            if offset > span and _r > 2:
                mask_pa = 1

            if args[0] == 0: # flat spectrum
                _args = (x[2], x[3])

                nmprm += 3

                model = dvis(_args, _q, _u, _l, _m)

            # elif args[0] in [1, 2, 3]:  # spl | cpl | ssa | quad
            #     idxval_t = (i + 1) * 10 + 0
            #     idxval_i = (i + 1) * 10 + 5
            #     idxval_f = (i + 1) * 10 + 6

            #     mask_t = args[4] == idxval_t
            #     mask_i = args[4] == idxval_i
            #     mask_f = args[4] == idxval_f

            #     has_t = mask_t.sum()

            #     _i = theta[mask_i][0]

            #     if args[0] == 1:    # simple power-law
            #         _args = (x[0], x[1], x[2], x[3])
            #         model = dvis_spl(_args, _s, _l, _m, _i)
            #         nmprm += 4 if has_lm else 2

            #     else:   # curved power-law or synchrotron self-absorption
            #         _f = theta[mask_f][0]
            #         if i == 0:
            #             _args = (x[1], x[2], x[3])
            #             if args[0] == 2:
            #                 model = dvis_cpl(
            #                     _args, _s, _l, _m, _i, _f
            #                 )

            #             elif args[0] == 3:
            #                 model = dvis_ssa(
            #                     _args, _s, _l, _m, _i, _f
            #                 )

            #         else:
            #             mask_t = round(float(theta[mask_t][0])) == 0

            #             if mask_t:
            #                 _args = (x[0], x[1], x[2], x[3])
            #                 model = dvis_spl(_args, _s, _l, _m, _i)
            #                 nmprm += 4 if has_lm else 2
            #             else:
            #                 _args = (x[1], x[2], x[3])
            #                 if args[0] == 2:
            #                     model = dvis_cpl(
            #                         _args, _s, _l, _m, _i, _f
            #                     )

            #                 elif args[0] == 3:
            #                     model = dvis_ssa(
            #                         _args, _s, _l, _m, _i, _f
            #                     )

            #                 nmprm += 5 if has_lm else 3

            model_p += model[0] + 1j * model[1]
            model_q += model[0]
            model_u += model[1]

    nasum_q = np.nansum(np.abs(model_q))
    nasum_u = np.nansum(np.abs(model_u))

    mask_model_q = not np.isnan(nasum_q) and not nasum_q == 0
    mask_model_u = not np.isnan(nasum_u) and not nasum_u == 0

    # compute objective functions
    if mask_model_q and mask_model_u:
        objective = 0

        def compute_bic(in_res, in_sig2, in_type, in_nobs, in_nmprm):
            penalty = in_nmprm * np.log(in_nobs)

            nll = 0.5 * (
                np.nansum(
                    0.5 * (in_res**2 / in_sig2)
                    + np.log(2 * np.pi * in_sig2)
                )
            )

            return 2 * nll + penalty

        vis_p_obs = y[0]
        vis_p_mod = model_p
        vis_p_sig2 = yerr[0]**2
        vis_p_res = np.abs(vis_p_obs - vis_p_mod)
        nobs_p = y[0].size

        vis_q_obs = y[1]
        vis_q_mod = model_q
        vis_q_sig2 = yerr[1]**2
        vis_q_res = np.abs(vis_q_obs - vis_q_mod)
        nobs_q = y[1].size

        vis_u_obs = y[2]
        vis_u_mod = model_u
        vis_u_sig2 = yerr[2]**2
        vis_u_res = np.abs(vis_u_mod - vis_u_obs)
        nobs_u = y[2].size

        objective -= compute_bic(vis_p_res, vis_p_sig2, 0, nobs_p, nmprm)
        objective -= compute_bic(vis_q_res, vis_q_sig2, 0, nobs_q, nmprm)
        objective -= compute_bic(vis_u_res, vis_u_sig2, 0, nobs_u, nmprm)

        if mask_pa == 1:
            objective = -np.inf
    else:
        objective = -np.inf

    return objective
