
import os
import sys
import gc
import copy
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
from numba import njit, jit
import itertools as it
from astropy import units as u
from scipy import optimize
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
    NOTE: This modeling is based on 'dynesty'
          which is implementing Bayesian nested sampling
    (Web site: https://dynesty.readthedocs.io/en/stable/api.html#api)
    (NASA/ADS: https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract)

    Attributes:
        uvfs (list): list of uvf objects
        select (str): Stokes parameters (I, Q, U, and V)
                      or parallel/cross-hand polarization (RR, LL, RL, LR)
        x (tuple): tuple of x-arguments
        y (tuple): tuple of y-arguments
        yerr (tuple): tuple of y-error-arguments
        args (tuple): arguments set
        factor_sblf (float): factor on shortest-baseline flux density
        sampler (str): sampling method in 'dynesty'
                       (availables: 'rwalk', 'rslice', 'slice')
        bound (str): bounding condition in 'dynesty'
        runfit_set (str): modeling frequency setting
                          - 'sf' single-frequency
                          - 'mf' multi-frequency
        runfit_sf (bool): if True, run single-frequency modeling
        runfit_mf (bool): if True, run multi-frequency modeling
        runfit_pol (bool): if True, run polarization modeling
        ftype (list, str): fitting data terms
                           (availables: 'vis', 'amp' 'phs', 'clamp', 'clphs')
        fwght (list, float): fitting weights for the given data terms
        boundset (2D-list): list of boundary conditions for priors
        bnd_l (list): priori boundary condition in Right Ascension (RA)
        bnd_m (list): priori boundary condition in Declination (DEC)
        bnd_f (list): priori boundary condition of turnover frequency
                      (when 'spectrum' == 'cpl' | 'ssa')
        ufreq (list, float): unique frequency of 'uvfs'
        bands (list, str): frquency-band names to save the results
        spectrum (str): modeling spectrum
                        (availables: 'single', 'spl', 'cpl', 'ssa')
        uvw (str): uv-weighting option
                   - 'n': natural-weighting
                   - 'u': uniform-weighting
        shift ((float, float)): amount of shift in (RA, DEC)-direction
        fixnmod (bool): if True, fix the number of models to the 'maxn'
        maxn (int): maximum number of models to be allowed
        npix (int): number of pixels (on the image)
        mindr (float): minimum dynamic range to plot a contour in images
        mrng (float): map range
        dogscale (bool): if True, run a antenna gain-scaling
        doampcal (bool): if True, run visibility amplitude self-calibration
        dophscal (bool): if True, run visibility phase self-calibration
        path_fig (str): path to save the resultant figures
        source (str): source name
        date (str): observation date
        cgain_truth (DataFrame): truth complex antenna gain values
                                 (this option is for simulation)
        ncpu (int): number of CPU to run the modeling
    """
    def __init__(self,
        uvfs=None, select="i", x=None, y=None, yerr=None, args=None,
        factor_sblf=1.0, sampler="slice", bound="multi", runfit_set="mf",
        runfit_sf=True, runfit_mf=True, runfit_pol=False, ftype=None,
        fwght=None, re_wamp_mf=None, re_ftype=None, re_fwght=None,
        boundset=False, width=5, bnds=None, bnd_l=None, bnd_m=None, bnd_f=None,
        bnd_pa=[None, None], nflux=False, ufreq=None, bands=None, spectrum=None,
        model="gaussian", uvw=None, shift=None, fixnmod=False, maxn=None,
        npix=None, mindr=3, mrng=None, gacalerr=0, dognorm=True, selfflag=True,
        dogscale=False, doampcal=True, dophscal=True, path_fig=None,
        source=None, date=None, cgain_truth=None, save_uvfits=False, ncpu=1
    ):
        self.uvfs = uvfs
        self.select = select
        self.x = x
        self.y = y
        self.yerr = yerr
        self.args = args

        self.factor_sblf = factor_sblf
        self.sampler = sampler
        self.bound = bound

        self.runfit_set = runfit_set
        self.runfit_sf = runfit_sf
        self.runfit_mf = runfit_mf
        self.runfit_pol = runfit_pol

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
        self.bnds = bnds
        self.bnd_l = bnd_l
        self.bnd_m = bnd_m
        self.bnd_f = bnd_f
        self.bnd_pa = bnd_pa
        self.nflux = nflux

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
        self.selfflag = selfflag

        self.path_fig = path_fig
        self.source = source
        self.date = date
        self.cgain_truth = cgain_truth
        self.save_uvfits = save_uvfits
        self.ncpu = ncpu

        self.pol = gamvas.polarization.modeling.polarization(ncpu=self.ncpu)


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
                        (bounds[field][i][1] - bounds[field][i][0])
                        * theta[1 + ndim + nfield]
                        + bounds[field][i][0]
                    )
                ndim += self.dims
        else:
            if self.set_spectrum:
                for i in range(self.nmod):

                    if self.spectrum in ["cpl", "ssa"] and i != 0:
                        results.append(+1.98 * theta[1 + ndim] - 0.49)
                        self.set_field(nmod=i+1)
                        for nfield, field in enumerate(self.fields):
                            results.append(
                                (bounds[field][i][1] - bounds[field][i][0])
                                * theta[2 + ndim + nfield]
                                + bounds[field][i][0]
                            )
                    else:
                        self.set_field(nmod=i+1)
                        for nfield, field in enumerate(self.fields):
                            results.append(
                                (bounds[field][i][1] - bounds[field][i][0])
                                * theta[1 + ndim + nfield]
                                + bounds[field][i][0]
                            )

                    ndim += self.dims
            else:
                for i in range(self.nmod):
                    self.set_field(nmod=i+1)
                    for nfield, field in enumerate(self.fields):
                        results.append(
                            (bounds[field][i][1] - bounds[field][i][0])
                            * theta[1 + ndim + nfield]
                            + bounds[field][i][0]
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
                    if not self.relmod:
                        self.dims = 4
                        self.fields =\
                            ["S", "a", "l", "m"]
                else:
                    self.dims = 4
                    self.fields =\
                        ["S", "a", "l", "m"]
            elif self.model == "delta":
                if nmod == 1:
                    self.dims = 1
                    self.fields =\
                        ["S"]
                    if not self.relmod:
                        self.dims = 3
                        self.fields =\
                            ["S", "l", "m"]
                else:
                    self.dims = 3
                    self.fields =\
                        ["S", "l", "m"]
        else:
            if self.set_spectrum:
                if self.model == "gaussian":
                    if self.spectrum in ["spl"]:
                        if nmod == 1:
                            self.dims = 3
                            self.fields =\
                                ["S", "a", "alpha"]
                            if not self.relmod:
                                self.dims = 5
                                self.fields =\
                                    ["S", "a", "l", "m", "alpha"]
                        else:
                            self.dims = 5
                            self.fields =\
                                ["S", "a", "l", "m", "alpha"]
                    elif self.spectrum in ["cpl", "ssa"]:
                        if nmod == 1:
                            self.dims = 4
                            self.fields =\
                                ["S", "a", "alpha", "freq"]
                            if not self.relmod:
                                self.dims = 6
                                self.fields =\
                                    ["S", "a", "l", "m", "alpha", "freq"]
                        else:
                            self.dims = 7
                            self.fields =\
                                ["S", "a", "l", "m", "alpha", "freq"]
                elif self.model == "delta":
                    if nmod == 1:
                        self.dims = 3
                        self.fields =\
                            ["S", "alpha", "freq"]
                        if not self.relmod:
                            self.dims = 5
                            self.fields =\
                                ["S", "l", "m", "alpha", "freq"]

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
                        if not self.relmod:
                            self.dims = 4
                            self.fields =\
                                ["S", "a", "l", "m"]
                    else:
                        self.dims = 4
                        self.fields =\
                            ["S", "a", "l", "m"]
                elif self.model == "delta":
                    if nmod == 1:
                        self.dims = 1
                        self.fields =\
                            ["S"]
                        if not self.relmod:
                            self.dims = 3
                            self.fields =\
                                ["S", "l", "m"]
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
            if i != 0 and \
            self.set_spectrum and \
            self.spectrum in ["cpl", "ssa"]:
                index_ = index_ + [f"{i + 1}_thick"]

            self.set_field(nmod=i+1)
            nums = np.full(self.dims, i + 1)
            fields = self.fields
            index_list = ["_".join([str(x), y]) for x, y in zip(nums, fields)]
            index_ = index_ + index_list
        self.index = index_

    def get_results(self,
        qs=(0.025, 0.500, 0.975), save_path=False, save_name=False,
        save_xlsx=False):
        """
        Get modeling results (parameters)
            Arguments:
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

            idxn = np.array(list(
                map(
                    lambda x: x.split("_")[0],
                    self.index[1:])
            ), dtype=int)

            mask_nmod = idxn <= nmod
            mask_nmod = np.append(np.array([True]), mask_nmod)

            qls_ = qls[mask_nmod]
            qms_ = qms[mask_nmod]
            qhs_ = qhs[mask_nmod]
            idx_ = np.array(self.index)[mask_nmod]
            prms_ = np.array([qls_, qms_, qhs_])

            out_xlsx =\
                pd.DataFrame(prms_, index=["lolim", "value", "uplim"]).T
            out_xlsx["idx"] = idx_
            out_xlsx.to_excel(f"{save_path}{save_name}")


    def get_nmprms(self):
        """
        Get the number of parameters
        """
        mprms = self.mprms.copy()
        vals = rfn.structured_to_unstructured(mprms)
        nmod = int(np.round(mprms["nmod"]))
        mask_thick =\
            np.array(list(map(lambda x: "thick" in x, mprms.dtype.names)))
        mask_thick =\
            np.round(vals[mask_thick]).astype(int)
        nmprms = len(vals) - 2 * (len(mask_thick) - np.sum(mask_thick)) - 1

        self.nmprms = nmprms


    def run_util(self,
        nmod=1, sample="rwalk", bound="multi", boundset=None, run_type=None,
        save_path=None, save_name=None, save_xlsx=False
    ):
        """
        Run 'dynesty' utilies
            Arguments:
                sample (str): sampling method in 'dynesty'
                              (availables: 'rwalk', 'rslice', 'slice')
                bound (str): bounding condition in 'dynesty'
                boundset (2D-list): list of boundary conditions for priors
                run_type (str): modeling frequency setting
                                - 'sf': single-frequency
                                - 'mf': multi-frequency
                save_path (str): path to save the results
                save_name (str): name of the file to save the results
                save_xlsx (bool): if True, save the results in xlsx format

        """
        self.set_index()
        args = self.args
        ndim = self.ndim

        if self.ndim != len(self.index):
            out_txt =\
                f"Number of dimensions ({self.ndim})" \
                f"is not matched" \
                f"with the number of index ({len(self.index)})."
            raise Exception(out_txt)

        if not self.boundset is None:
            boundset=self.boundset
        else:
            out_txt = "Boundary conditions for priors are not given."
            raise Exception(out_txt)

        self.sampler = sample
        self.bound = bound

        # run dynesty
        with Pool(
            self.ncpu,
            loglike=objective_function,
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
            out_txt = "The number of uvf files and bands are not matched."
            raise Exception(out_txt)
        else:
            nfreq = len(self.uvfs)

        for nband in range(nfreq):
            uvfs = copy.deepcopy(self.uvfs)
            uvf =\
                gamvas.utils.set_uvf(
                    [copy.deepcopy(uvfs[nband])],
                    type="sf"
                )
            cgain1 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)
            cgain2 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)

            uvfs = copy.deepcopy(self.uvfs)
            uvall = gamvas.utils.set_uvf(uvfs, type="mf")

            # set fit weights
            ftype = self.ftype.copy()
            if self.fwght is None:
                fwght =\
                    gamvas.utils.get_fwght(
                        ftype,
                        copy.deepcopy(uvf.data),
                        copy.deepcopy(uvf.clamp["clamp"]),
                        copy.deepcopy(uvf.clphs["clphs"])
                    )
            else:
                fwght = self.fwght.copy()
            self.fdict = dict(zip(ftype, fwght))

            band = self.bands[nband]
            uvf =\
                gamvas.utils.set_uvf(
                    [copy.deepcopy(uvfs[nband])],
                    type="sf"
                )
            sblf = self.factor_sblf * uvf.get_sblf()[0]

            freq = uvf.freq
            nmod = self.maxn

            path_fig = self.path_fig + f"{freq:.1f}/"
            gamvas.utils.mkdir(path_fig)

            uvf.ploter.draw_tplot(
                uvf, plotimg=False, show_title=False,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.initial.tplot",
                save_form="pdf"
            )

            uvf.ploter.draw_radplot(
                uvf, plotimg=False, show_title=False,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.initial.radplot",
                save_form="pdf"
            )

            uvf.ploter.draw_uvcover(
                uvf, plotimg=False, show_title=False,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.initial.uvcover",
                save_form="pdf"
            )

            uvf.ploter.draw_dirtymap(
                uvf, plotimg=False, show_title=False,
                npix=self.npix, uvw=self.uvw,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.initial.dirtmap",
                save_form="pdf"
            )

            data = uvf.data

            if self.bnds is None:
                bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i =\
                    gamvas.utils.set_boundary(
                        nmod=nmod, select=self.select, spectrum="single",
                        sblf=sblf, width=self.width, mrng=self.mrng,
                        bnd_l=self.bnd_l, bnd_m=self.bnd_m, nflux=self.nflux,
                        relmod=self.relmod
                    )

                bnds =\
                    gamvas.utils.sarray(
                        (bnd_S, bnd_a, bnd_l, bnd_m),
                        field=fields_sf,
                        dtype=dtypes_sf
                    )
            else:
                bnds = gamvas.utils.sarray(
                    data=self.bnds,
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
            self.clamp_uvcomb = clamp_uvcomb
            self.clphs_uvcomb = clphs_uvcomb

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
                np.ma.getdata(uvf.data["vis"]).astype(np.complex64),
                np.ma.getdata(uvf.clamp["clamp"]),
                np.ma.getdata(uvf.clphs["clphs"]),
                copy.deepcopy(clamp_uvcomb),
                copy.deepcopy(clphs_uvcomb)
            )

            # set yerr parameters
            self.yerr =\
            (
                np.ma.getdata(uvf.data["sigma"]),
                np.ma.getdata(uvf.clamp["sigma_clamp"]),
                np.ma.getdata(uvf.clphs["sigma_clphs"])
            )

            Nant =\
                len(np.unique(np.append(
                    np.ma.getdata(uvf.data["ant_name1"]),
                    np.ma.getdata(uvf.data["ant_name2"]))
                ))
            uv_coord = np.column_stack((self.x[2], self.x[3]))
            uv_coord = uv_coord.reshape(len(uv_coord), -1)
            if Nant >= 4:
                mask_amp12 =\
                    self.clamp_uvcomb[1].reshape(len(self.clamp_uvcomb[1]), -1)
                mask_amp34 =\
                    self.clamp_uvcomb[2].reshape(len(self.clamp_uvcomb[2]), -1)
                mask_amp13 =\
                    self.clamp_uvcomb[3].reshape(len(self.clamp_uvcomb[3]), -1)
                mask_amp24 =\
                    self.clamp_uvcomb[4].reshape(len(self.clamp_uvcomb[4]), -1)
                mask_amp12 =\
                    np.array(np.argmax(
                        (mask_amp12[:, None, :] \
                            == uv_coord[None, :, :]).all(axis=2), axis=1),
                    dtype=np.int64
                    )
                mask_amp34 =\
                    np.array(np.argmax(
                        (mask_amp34[:, None, :] \
                            == uv_coord[None, :, :]).all(axis=2), axis=1),
                    dtype=np.int64
                    )
                mask_amp13 =\
                    np.array(np.argmax(
                        (mask_amp13[:, None, :] \
                            == uv_coord[None, :, :]).all(axis=2), axis=1),
                    dtype=np.int64
                    )
                mask_amp24 =\
                    np.array(np.argmax(
                        (mask_amp24[:, None, :] \
                            == uv_coord[None, :, :]).all(axis=2), axis=1),
                    dtype=np.int64
                    )
            if Nant >= 3:
                mask_phs12 =\
                    self.clphs_uvcomb[1].reshape(len(self.clphs_uvcomb[1]), -1)
                mask_phs23 =\
                    self.clphs_uvcomb[2].reshape(len(self.clphs_uvcomb[2]), -1)
                mask_phs31 =\
                    self.clphs_uvcomb[3].reshape(len(self.clphs_uvcomb[3]), -1)
                mask_phs12 =\
                    np.array(
                        np.argmax(
                            (
                                mask_phs12[:, None, :]
                                == uv_coord[None, :, :]
                            ).all(axis=2),
                            axis=1
                        ),
                        dtype=np.int64
                    )
                mask_phs23 =\
                    np.array(
                        np.argmax(
                            (
                                mask_phs23[:, None, :]
                                == uv_coord[None, :, :]
                            ).all(axis=2),
                            axis=1
                        ),
                        dtype=np.int64
                    )
                mask_phs31 =\
                    np.array(
                        np.argmax(
                            (
                                mask_phs31[:, None, :]
                                == uv_coord[None, :, :]
                            ).all(axis=2),
                            axis=1
                        ),
                        dtype=np.int64
                    )
            if Nant == 3:
                mask_amp12 = np.array([0], dtype=np.int64)
                mask_amp34 = np.array([0], dtype=np.int64)
                mask_amp13 = np.array([0], dtype=np.int64)
                mask_amp24 = np.array([0], dtype=np.int64)

            if self.set_spectrum:
                set_spectrum = 1
            else:
                set_spectrum = 0

            if self.spectrum == "single":
                spectrum = 0
            elif self.spectrum == "spl":
                spectrum = 1
            elif self.spectrum == "cpl":
                spectrum = 2
            elif self.spectrum == "ssa":
                spectrum = 3
            else:
                out_txt =\
                    f"Unexpected spectrum type is given: " \
                    f"{self.spectrum}." \
                    f"Choose 'single', 'spl', 'cpl', or 'ssa'."
                raise Exception(out_txt)

            if self.model == "delta":
                modeltype = 0
            elif self.model == "gaussian":
                modeltype = 1
            else:
                modeltype = 1
                out_txt =\
                    f"Unexpected model type is given: {self.model}" \
                    f" Assume 'gaussian'."
                print(out_txt)

            if self.ifsingle:
                ifsingle = 1
            else:
                ifsingle = 0

            ftype_ = []
            fwght_ = [0, 0, 0, 0, 0]
            filoc_ = []
            if "vis"   in list(self.fdict.keys()):
                ftype_.append(0)
                fwght_[0] =\
                    np.array(
                        list(self.fdict.values())
                    )[np.isin(list(self.fdict.keys()), "vis")][0]
            if "amp"   in list(self.fdict.keys()):
                ftype_.append(1)
                fwght_[1] =\
                    np.array(
                        list(self.fdict.values())
                    )[np.isin(list(self.fdict.keys()), "amp")][0]
            if "phs"   in list(self.fdict.keys()):
                ftype_.append(2)
                fwght_[2] =\
                    np.array(
                        list(self.fdict.values())
                    )[np.isin(list(self.fdict.keys()), "phs")][0]
            if "clamp" in list(self.fdict.keys()):
                if Nant >= 4:
                    ftype_.append(3)
                    fwght_[3] =\
                        np.array(
                            list(self.fdict.values())
                        )[np.isin(list(self.fdict.keys()), "clamp")][0]
            if "clphs" in list(self.fdict.keys()):
                ftype_.append(4)
                fwght_[4] =\
                    np.array(
                        list(self.fdict.values())
                    )[np.isin(list(self.fdict.keys()), "clphs")][0]

            ftype_ = np.array(ftype_, dtype=np.int64)
            fwght_ = np.array(fwght_, dtype=np.float64)
            self.args =\
            (
                self.x, self.y, self.yerr,
                (
                    np.ma.getdata(uvf.data["ant_name1"]),
                    np.ma.getdata(uvf.data["ant_name2"]),
                    set_spectrum, spectrum, modeltype, ifsingle,
                    np.array([self.bnd_pa[0]], dtype=np.float64),
                    np.array([self.bnd_pa[1]], dtype=np.float64),
                    ftype_, fwght_,
                    mask_amp12, mask_amp34, mask_amp13, mask_amp24,
                    mask_phs12, mask_phs23, mask_phs31,
                    int(self.relmod)
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
            runtxt =\
                f"\n# Running {uvf.freq:.1f} GHz ... " \
                f"(Pol {uvf.select.upper()}, " \
                f"MaxN_model={nmod}, " \
                f"sampler='{insample}')"
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
                    uvf.selfcal(type="gscale",
                        gnorm=self.dognorm, selfflag=self.selfflag)
                    cgain1 *= uvf.cgain1
                    cgain2 *= uvf.cgain2
                else:
                    uvf.selfcal(type="gscale",
                        gnorm=self.dognorm, selfflag=self.selfflag)
                    cgain1 *= uvf.cgain1
                    cgain2 *= uvf.cgain2
            else:
                if self.doampcal:
                    if self.dophscal:
                        uvf.selfcal(type="phs")
                        cgain1 *= uvf.cgain1
                        cgain2 *= uvf.cgain2
                        uvf.selfcal(type="a&p", selfflag=self.selfflag)
                        cgain1 *= uvf.cgain1
                        cgain2 *= uvf.cgain2
                    else:
                        uvf.selfcal(type="amp", selfflag=self.selfflag)
                        cgain1 *= uvf.cgain1
                        cgain2 *= uvf.cgain2
                else:
                    if self.dophscal:
                        uvf.selfcal(type="phs")
                        cgain1 *= uvf.cgain1
                        cgain2 *= uvf.cgain2
            self.cgain1 = cgain1
            self.cgain2 = cgain2

            # print statistical values
            #   - reduced chi-square
            #   - Akaike information criterion
            #   - Bayesian information criterion
            uvcomb =\
                (
                    copy.deepcopy(uvf.clamp["clamp"]),
                    copy.deepcopy(uvf.clphs["clphs"]),
                    copy.deepcopy(uvf.clamp["sigma_clamp"]),
                    copy.deepcopy(uvf.clphs["sigma_clphs"]),
                    copy.deepcopy(clamp_uvcomb),
                    copy.deepcopy(clphs_uvcomb)
                )

            fty, chi, aic, bic =\
                gamvas.utils.print_stats(
                    uvf, uvcomb, self.nmprms, logz_v, logz_d, ftype
                )

            self.print_prms(
                ufreq=[np.round(freq,1)], fitset=self.runfit_set,
                relmod=self.relmod, spectrum=self.spectrum, model=self.model,
                stats=(fty, chi, aic, bic, logz_v, logz_d), printmsg=True,
                save_path=path_fig, save_name="model_result.txt"
            )

            uvf.ploter.bprms = uvf.bprms
            uvf.ploter.prms = prms

            # set clousre quantities
            uvf.ploter.clq_mod =\
                gamvas.utils.set_closure(
                    data["u"], data["v"], uvf.data["vism"],
                    np.zeros(uvf.data["vism"].shape[0]),
                    data["ant_name1"], data["ant_name2"],
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
                result=self.results, nmod=nmod_, ifsingle=self.ifsingle,
                relmod=self.relmod, set_spectrum=self.set_spectrum,
                model=self.model, fontsize=20,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.trplot",
                save_form="pdf"
            )

            uvf.ploter.draw_cnplot(
                result=self.results, nmod=nmod_, ifsingle=self.ifsingle,
                relmod=self.relmod, set_spectrum=self.set_spectrum,
                model=self.model, fontsize=20,
                save_path=path_fig,
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
                npix=self.npix, mindr=self.mindr, plot_resi=True,
                addnoise=True, freq_ref=uvf.freq, freq=uvf.freq,
                model=self.model, ifsingle=self.ifsingle,
                set_spectrum=self.set_spectrum,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.img",
                save_form="pdf"
            )

            # set beam parameters
            uvf.ploter.bprms = (uvf.bprms[0], uvf.bprms[0], 0)

            uvf.ploter.draw_image(
                uvf=uvf, plotimg=False,
                npix=self.npix, mindr=self.mindr, plot_resi=False,
                addnoise=True, freq_ref=uvf.freq, freq=uvf.freq,
                model=self.model, ifsingle=self.ifsingle,
                set_spectrum=self.set_spectrum,
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
                    bprms=uvall.bprms,
                    save_path=path_fig,
                    source=self.source,
                    date=self.date
                )

            if self.save_uvfits:
                outpath = f"{self.path_fig}/uvfs/"
                outname =\
                    f"gamvas." \
                    f"sf." \
                    f"{uvf.freq:.0f}." \
                    f"{uvf.source}." \
                    f"{uvf.date}.uvf"
                gamvas.utils.mkdir(outpath)
                uvf.save_uvfits(
                    save_path=outpath,
                    save_name=outname
                )


    def run_mf(self):
        """
        Run multi-frequency model-fit
        """
        uvfs = copy.deepcopy(self.uvfs)
        uvf = gamvas.utils.set_uvf(uvfs, type="mf")

        cgain1 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)
        cgain2 = np.ones(uvf.data.shape[0]) * np.exp(1j * 0)

        # set fit weights
        if self.runfit_sf and not self.re_wamp_mf is None:
            mask_ftype = self.ftype == "amp"
            self.fwght[mask_ftype] = self.re_wamp_mf
        ftype = self.ftype.copy()
        if self.fwght is None:
            fwght =\
                gamvas.utils.get_fwght(
                    ftype,
                    copy.deepcopy(uvf.data),
                    copy.deepcopy(uvf.clamp["clamp"]),
                    copy.deepcopy(uvf.clphs["clphs"])
                )
        else:
            fwght = self.fwght.copy()
        self.fdict = dict(zip(ftype, fwght))

        # set maximum number of model
        nmod = self.maxn

        sblf = self.factor_sblf * np.max(np.abs(uvf.data["vis"]))
        print(f"\n# Maximum baseline flux : {sblf / self.factor_sblf:.3f} Jy")

        uvf.ploter.draw_tplot(
            uvf, plotimg=False, show_title=False,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.mf.initial.tplot",
            save_form="pdf"
        )

        uvf.ploter.draw_radplot(
            uvf, plotimg=False, show_title=False,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.mf.initial.radplot",
            save_form="pdf"
        )

        uvf.ploter.draw_uvcover(
            uvf, plotimg=False, show_title=False,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.mf.initial.uvcover",
            save_form="pdf"
        )

        uvf.ploter.draw_dirtymap(
            uvf, plotimg=False, show_title=False,
            npix=self.npix, uvw=self.uvw,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.mf.initial.dirtmap",
            save_form="pdf"
        )

        path_fig = self.path_fig + "mf/"
        gamvas.utils.mkdir(path_fig)

        if self.bnds is None:
            if self.bnd_f is not None:
                bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i =\
                gamvas.utils.set_boundary(
                        nmod=nmod, select=self.select, spectrum=self.spectrum,
                        sblf=sblf, width=self.width, mrng=self.mrng,
                        bnd_l=self.bnd_l, bnd_m=self.bnd_m, bnd_f=self.bnd_f,
                        relmod=self.relmod
                )
            else:
                bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i =\
                gamvas.utils.set_boundary(
                        nmod=nmod, select=self.select, spectrum=self.spectrum,
                        sblf=sblf, width=self.width, mrng=self.mrng,
                        bnd_l=self.bnd_l, bnd_m=self.bnd_m,
                        relmod=self.relmod
                )

            bnds = gamvas.utils.sarray(
                    data=(bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i),
                    field=fields_mf,
                    dtype=dtypes_mf
            )
        else:
            bnds = gamvas.utils.sarray(
                data=self.bnds,
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
        self.clamp_uvcomb = copy.deepcopy(clamp_uvcomb)
        self.clphs_uvcomb = copy.deepcopy(clphs_uvcomb)

        self.x =\
        (
            uvfs[0].freq,
            np.ma.getdata(uvf.data["freq"]),
            np.ma.getdata(uvf.data["u"]),
            np.ma.getdata(uvf.data["v"])
        )

        self.y =\
        (
            np.ma.getdata(uvf.data["vis"]).astype(np.complex64),
            np.ma.getdata(uvf.clamp["clamp"]),
            np.ma.getdata(uvf.clphs["clphs"]),
            copy.deepcopy(clamp_uvcomb),
            copy.deepcopy(clphs_uvcomb)
        )

        self.yerr =\
        (
            np.ma.getdata(uvf.data["sigma"]),
            np.ma.getdata(uvf.clamp["sigma_clamp"]),
            np.ma.getdata(uvf.clphs["sigma_clphs"])
        )

        Nant =\
            len(np.unique(
                np.append(
                    np.ma.getdata(uvf.data["ant_name1"]),
                    np.ma.getdata(uvf.data["ant_name2"])
                ))
            )
        uv_coord = np.column_stack((self.x[2], self.x[3]))
        uv_coord = uv_coord.reshape(len(uv_coord), -1)
        if Nant >= 4:
            mask_amp12 =\
                self.clamp_uvcomb[1].reshape(len(self.clamp_uvcomb[1]), -1)
            mask_amp34 =\
                self.clamp_uvcomb[2].reshape(len(self.clamp_uvcomb[2]), -1)
            mask_amp13 =\
                self.clamp_uvcomb[3].reshape(len(self.clamp_uvcomb[3]), -1)
            mask_amp24 =\
                self.clamp_uvcomb[4].reshape(len(self.clamp_uvcomb[4]), -1)
            mask_amp12 =\
                np.array(
                    np.argmax(
                        (
                            mask_amp12[:, None, :]
                            == uv_coord[None, :, :]
                        ).all(axis=2),
                        axis=1
                    ),
                    dtype=np.int64
                )
            mask_amp34 =\
                np.array(
                    np.argmax(
                        (
                            mask_amp34[:, None, :]
                            == uv_coord[None, :, :]
                        ).all(axis=2),
                        axis=1
                    ),
                    dtype=np.int64
                )
            mask_amp13 =\
                np.array(
                    np.argmax(
                        (
                            mask_amp13[:, None, :]
                            == uv_coord[None, :, :]
                        ).all(axis=2),
                        axis=1
                    ),
                    dtype=np.int64
                )
            mask_amp24 =\
                np.array(
                    np.argmax(
                        (
                            mask_amp24[:, None, :]
                            == uv_coord[None, :, :]
                        ).all(axis=2),
                        axis=1
                    ),
                    dtype=np.int64
                )
        if Nant >= 3:
            mask_phs12 =\
                self.clphs_uvcomb[1].reshape(len(self.clphs_uvcomb[1]), -1)
            mask_phs23 =\
                self.clphs_uvcomb[2].reshape(len(self.clphs_uvcomb[2]), -1)
            mask_phs31 =\
                self.clphs_uvcomb[3].reshape(len(self.clphs_uvcomb[3]), -1)
            mask_phs12 =\
                np.array(
                    np.argmax(
                        (
                            mask_phs12[:, None, :]
                            == uv_coord[None, :, :]
                        ).all(axis=2),
                        axis=1
                    ),
                    dtype=np.int64
                )
            mask_phs23 =\
                np.array(
                    np.argmax(
                        (
                            mask_phs23[:, None, :]
                            == uv_coord[None, :, :]
                        ).all(axis=2),
                        axis=1
                    ),
                    dtype=np.int64
                )
            mask_phs31 =\
                np.array(
                    np.argmax(
                        (
                            mask_phs31[:, None, :]
                            == uv_coord[None, :, :]
                        ).all(axis=2),
                        axis=1
                    ),
                    dtype=np.int64
                )
        if Nant == 3:
            mask_amp12 = np.array([0], dtype=np.int64)
            mask_amp34 = np.array([0], dtype=np.int64)
            mask_amp13 = np.array([0], dtype=np.int64)
            mask_amp24 = np.array([0], dtype=np.int64)

        if self.set_spectrum:
            set_spectrum = 1
        else:
            set_spectrum = 0

        if self.spectrum == "single":
            spectrum = 0
        elif self.spectrum == "spl":
            spectrum = 1
        elif self.spectrum == "cpl":
            spectrum = 2
        elif self.spectrum == "ssa":
            spectrum = 3
        else:
            out_txt =\
                f"Unexpected spectrum type is given: {self.spectrum}" \
                f" Choose 'single', 'spl', 'cpl', or 'ssa'."
            raise Exception(out_txt)

        if self.model == "delta":
            modeltype = 0
        elif self.model == "gaussian":
            modeltype = 1
        else:
            modeltype = 1
            out_txt =\
                f"Unexpected model type is given: {self.model}" \
                f" Assume 'gaussian'."
            print(out_txt)

        if self.ifsingle:
            ifsingle = 1
        else:
            ifsingle = 0

        ftype_ = []
        fwght_ = [0, 0, 0, 0, 0]
        filoc_ = []
        if "vis"   in list(self.fdict.keys()):
            ftype_.append(0)
            fwght_[0] =\
                np.array(
                    list(self.fdict.values())
                )[np.isin(list(self.fdict.keys()), "vis")][0]
        if "amp"   in list(self.fdict.keys()):
            ftype_.append(1)
            fwght_[1] =\
                np.array(
                    list(self.fdict.values())
                )[np.isin(list(self.fdict.keys()), "amp")][0]
        if "phs"   in list(self.fdict.keys()):
            ftype_.append(2)
            fwght_[2] =\
                np.array(
                    list(self.fdict.values())
                )[np.isin(list(self.fdict.keys()), "phs")][0]
        if "clamp" in list(self.fdict.keys()):
            if Nant >= 4:
                ftype_.append(3)
                fwght_[3] =\
                    np.array(
                        list(self.fdict.values())
                    )[np.isin(list(self.fdict.keys()), "clamp")][0]
        if "clphs" in list(self.fdict.keys()):
            ftype_.append(4)
            fwght_[4] =\
                np.array(
                    list(self.fdict.values())
                )[np.isin(list(self.fdict.keys()), "clphs")][0]
        ftype_ = np.array(ftype_, dtype=np.int64)
        fwght_ = np.array(fwght_, dtype=np.float64)
        self.args =\
        (
            self.x, self.y, self.yerr,
            (
                np.ma.getdata(uvf.data["ant_name1"]),
                np.ma.getdata(uvf.data["ant_name2"]),
                set_spectrum, spectrum, modeltype, ifsingle,
                np.array([self.bnd_pa[0]], dtype=np.float64),
                np.array([self.bnd_pa[1]], dtype=np.float64),
                ftype_, fwght_,
                mask_amp12, mask_amp34, mask_amp13, mask_amp24,
                mask_phs12, mask_phs23, mask_phs31,
                int(self.relmod)
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
        runtxt =\
            f"\n# Running... " \
            f"(Pol {uvf.select.upper()}.{self.select.upper()}, " \
            f"MaxN_model={nmod}, " \
            f"sampler='{insample}')"
        if self.relmod:
            runtxt += " // ! relative position"
        print(runtxt)
        print(f"# Fit-parameters : {self.fdict}")

        # run dynesty
        self.run_util(
            nmod=nmod, sample=insample, bound=self.bound, run_type="mf",
            save_path=path_fig, save_name="model_params.xlsx", save_xlsx=True
        )

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
                    cgain1[uvf.data["freq"] == self.ufreq[nuvf]] \
                        *= uvfs[nuvf].cgain1
                    cgain2[uvf.data["freq"] == self.ufreq[nuvf]] \
                        *= uvfs[nuvf].cgain2
                    uvfs[nuvf].selfcal(type="gscale",
                        gnorm=self.dognorm, selfflag=self.selfflag)
                    cgain1[uvf.data["freq"] == self.ufreq[nuvf]] \
                        *= uvfs[nuvf].cgain1
                    cgain2[uvf.data["freq"] == self.ufreq[nuvf]] \
                        *= uvfs[nuvf].cgain2
                else:
                    uvfs[nuvf].selfcal(type="gscale",
                        gnorm=self.dognorm, selfflag=self.selfflag)
                    cgain1[uvf.data["freq"] == self.ufreq[nuvf]] \
                        *= uvfs[nuvf].cgain1
                    cgain2[uvf.data["freq"] == self.ufreq[nuvf]] \
                        *= uvfs[nuvf].cgain2
            else:
                if self.doampcal:
                    if self.dophscal:
                        uvfs[nuvf].selfcal(type="phs")
                        cgain1[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain1
                        cgain2[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain2
                        uvfs[nuvf].selfcal(type="a&p",
                            selfflag=self.selfflag)
                        cgain1[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain1
                        cgain2[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain2
                    else:
                        uvfs[nuvf].selfcal(type="amp",
                            selfflag=self.selfflag)
                        cgain1[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain1
                        cgain2[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain2
                else:
                    if self.dophscal:
                        uvfs[nuvf].selfcal(type="phs")
                        cgain1[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain1
                        cgain2[uvf.data["freq"] == self.ufreq[nuvf]] \
                            *= uvfs[nuvf].cgain2

        # re-set uvf
        uvf = gamvas.utils.set_uvf(uvfs, type="mf")

        # print statistical values
        #   - reduced chi-square
        #   - Akaike information criterion
        #   - Bayesian information criterion
        uvcomb =\
            (
                copy.deepcopy(uvf.clamp["clamp"]),
                copy.deepcopy(uvf.clphs["clphs"]),
                copy.deepcopy(uvf.clamp["sigma_clamp"]),
                copy.deepcopy(uvf.clphs["sigma_clphs"]),
                copy.deepcopy(clamp_uvcomb),
                copy.deepcopy(clphs_uvcomb)
            )
        fty, chi, aic, bic =\
            gamvas.utils.print_stats(
                uvf, uvcomb, self.nmprms, logz_v, logz_d, ftype
            )
        self.print_prms(
            ufreq=uvf.ufreq, fitset=self.runfit_set, spectrum=self.spectrum,
            relmod=self.relmod,  model=self.model,
            stats=(fty, chi, aic, bic, logz_v, logz_d),
            printmsg=True, save_path=path_fig, save_name="model_result.txt"
        )

        uvf.fit_beam(uvw=self.uvw)
        uvf.ploter.bprms = uvf.bprms
        uvf.ploter.prms = self.mprms
        uvf.ploter.spectrum = self.spectrum

        # set clousre quantities
        uvf.ploter.clq_mod =\
            gamvas.utils.set_closure(
                uvf.data["u"], uvf.data["v"], uvf.data["vism"],
                np.zeros(uvf.data["vism"].shape[0]),
                uvf.data["ant_name1"], uvf.data["ant_name2"],
                self.y[3], self.y[4]
            )

        # plot and save figures
        uvf.ploter.draw_cgains(
            uvf, cgain1, cgain2, truth=self.cgain_truth, plotimg=False,
            save_csv=True, save_path=path_fig,
            save_name=f"{self.source}.{self.date}.complxgain", save_form="pdf"
        )

        uvf.ploter.draw_trplot(
            result=self.results, nmod=nmod_, ifsingle=self.ifsingle,
            relmod=self.relmod, set_spectrum=self.set_spectrum,
            model=self.model, fontsize=20, save_path=path_fig,
            save_name=f"{self.source}.{self.date}.trplot", save_form="pdf"
        )

        uvf.ploter.draw_cnplot(
            result=self.results, nmod=nmod_, ifsingle=self.ifsingle,
            relmod=self.relmod, set_spectrum=self.set_spectrum,
            model=self.model, fontsize=20, save_path=path_fig,
            save_name=f"{self.source}.{self.date}.cnplot", save_form="pdf"
        )

        uvf.ploter.draw_radplot(
            uvf, plotimg=False, plotvism=True, show_title=False,
            save_path=path_fig, save_name=f"{self.source}.{self.date}.radplot",
            save_form="pdf"
        )

        uvf.ploter.draw_dirtymap(
            uvf, plotimg=False, plot_resi=True, show_title=False,
            npix=self.npix, uvw=self.uvw,
            save_path=path_fig, save_name=f"{self.source}.{self.date}.resimap",
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

        for i in range(len(uvfs)):
            uvfs[i].ploter.bprms = uvfs[i].bprms
            uvfs[i].ploter.prms = self.mprms
            uvfs[i].ploter.fitset = self.runfit_set
            uvfs[i].ploter.spectrum = self.spectrum
            uvfs[i].ploter.draw_image(
                uvf=uvfs[i], plotimg=False,
                npix=self.npix, mindr=self.mindr, plot_resi=False,
                addnoise=True, freq_ref=self.ufreq[0], freq=self.ufreq[i],
                model=self.model, ifsingle=self.ifsingle,
                set_spectrum=self.set_spectrum,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.img.sf.{self.bands[i]}",
                save_form="pdf"
            )
        for i in range(len(uvfs)):
            uvf.ploter.draw_image(
                uvf=uvf, plotimg=False,
                npix=self.npix, mindr=self.mindr, plot_resi=True,
                addnoise=True, freq_ref=self.ufreq[0], freq=self.ufreq[i],
                model=self.model, ifsingle=self.ifsingle,
                set_spectrum=self.set_spectrum,
                save_path=path_fig,
                save_name=f"{self.source}.{self.date}.img.mf.{self.bands[i]}",
                save_form="pdf"
            )
            uvfs[i].drop_visibility_model()
        uvf.drop_visibility_model()

        self.uvfs = copy.deepcopy(uvfs)

        if self.save_uvfits:
            for i in range(len(uvfs)):
                outpath = f"{self.path_fig}/uvfs/"
                outname =\
                    f"gamvas." \
                    f"mf." \
                    f"{uvfs[i].freq:.0f}." \
                    f"{uvfs[i].source}." \
                    f"{uvfs[i].date}.uvf"
                gamvas.utils.mkdir(outpath)
                uvfs[i].save_uvfits(
                    save_path=outpath,
                    save_name=outname
                )

        gc.collect()


    def run(self, uvave=None):
        """
        Run the modeling utilies
        """

        uvfs = copy.deepcopy(self.uvfs)
        uvall = gamvas.utils.set_uvf(uvfs, type="mf")
        uant =\
            np.unique(
                np.append(
                    uvall.data["ant_name1"],
                    uvall.data["ant_name2"]
                )
            )

        nvis  = uvall.data.shape[0]
        if len(uant) >= 4:
            ncamp = uvall.tmpl_clamp.shape[0]
        else:
            ncamp = 0
        ncphs = uvall.tmpl_clphs.shape[0]

        mrng = self.mrng
        spectrum = self.spectrum
        ftype = self.ftype.copy()
        fwght = self.fwght.copy()
        model = self.model
        select = self.select.upper()
        ncpu = self.ncpu
        if uvave is None:
            uvave = uvall.avgtime
            if isinstance(uvave, (float, int)):
                uvave_unit = "(sec)"
            else:
                uvave_unit = ""
        else:
            uvave_unit = ""

        out_txt =\
            f"\n### Running parameters ###\n" \
            f"# Map range: {mrng:.1f} (mas)\n" \
            f"# B_min: {uvall.bprms[0]:.3f} (mas)\n" \
            f"# B_maj: {uvall.bprms[1]:.3f} (mas)\n" \
            f"# B_pa: {uvall.bprms[2]:.3f} (deg)\n" \
            f"# Fit-spec: '{spectrum}'\n" \
            f"# Fit-type: {ftype}\n" \
            f"# Fit-wght: {fwght}\n" \
            f"# G.model: '{model}'\n" \
            f"# Number of complex visibility: {nvis}\n" \
            f"# Number of closure amplitude: {ncamp}\n" \
            f"# Number of closure phase: {ncphs}\n" \
            f"# uv-average time: {uvave} {uvave_unit}\n" \
            f"# Selected polarization: '{select}'\n" \
            f"# Number of active CPU cores: {ncpu}/{os.cpu_count()}\n"
        print(out_txt)

        runfit_set = self.runfit_set
        runfit_sf = self.runfit_sf
        runfit_mf = self.runfit_mf
        if not runfit_set.lower() in ["sf", "mf"]:
            out_txt =\
                "Given 'runfit_set' option is not intended." \
                "(available options: 'sf', 'mf')"
            raise Exception(out_txt)

        if "clphs" in ftype:
            if "phs" not in ftype and "vis" not in ftype:
                self.relmod = True
            else:
                self.relmod = False
        else:
            self.relmod = False

        if runfit_set == "sf":
            self.check_dof(uvfs=uvfs, runfit_set="sf", maxn=self.maxn)
            self.ifsingle = True
            self.set_spectrum = False
            self.run_sf()
        if runfit_set == "mf":
            if runfit_sf:
                self.check_dof(uvfs=uvfs, runfit_set="sf", maxn=self.maxn)
                self.ifsingle = False
                self.set_spectrum = False
                self.run_sf()
            if runfit_mf:
                self.check_dof(uvfs=uvfs, runfit_set="mf", maxn=self.maxn)
                for i in range(len(uvfs)):
                    uvfs[i].add_error_fraction(self.gacalerr,
                        set_vis=True, set_clq=False
                    )
                self.ifsingle = False
                self.set_spectrum = True
                self.run_mf()
                if self.runfit_pol:
                    uvfs_ = gamvas.utils.set_uvf(uvfs, type="mf")
                    self.pol.run_pol(
                        uvfs=copy.deepcopy(self.uvfs),
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
                        freq_ref=uvfs[0].freq,
                        npix=self.npix,
                        mindr=3,
                        bprms=uvfs_.bprms,
                        save_path=self.path_fig,
                        source=self.source,
                        date=self.date
                    )


    def check_dof(self, uvfs=None, runfit_set=None, maxn=None):
        if np.any([uvfs is None, runfit_set is None, maxn is None]):
            out_txt = "'uvfs', 'runfit_set', and 'maxn' must be provided."
            raise Exception(out_txt)

        if runfit_set == "sf":
            dof = 2 + 4 * (maxn - 1)
            for uvf in uvfs:
                nvis = uvf.data .size
                ncla = uvf.clamp.size
                nclp = uvf.clphs.size
                if ncla == 1:
                    ncla = np.nan
                if nclp == 1:
                    nclp = np.nan
                ndat = np.nanmin([nvis, ncla, nclp]).astype(int)
                if ndat <= 2 * dof:
                    if ndat <= 1 * dof:
                        out_txt =\
                            f"Insufficient data ({uvf.freq:.1f} GHz): " \
                            f"{ndat} (data) & {dof} (allowed dof). " \
                            f"Please check the number of visibility " \
                            f"or reduce either 'uvave' or 'snrflag'."
                        raise Exception(out_txt)
                    else:
                        out_txt =\
                            f"WARNING! The number of data ({ndat}) " \
                            f"is recommended " \
                            f"to be twice the allowed dof ({dof})."
                        print(out_txt)

        if runfit_set == "mf":
            dof = 4 + 6 * (maxn - 1)
            uvf = gamvas.utils.set_uvf(uvfs, type="mf")
            nvis = uvf.data .size
            ncla = uvf.clamp.size
            nclp = uvf.clphs.size
            if ncla == 1:
                ncla = np.nan
            if nclp == 1:
                nclp = np.nan
            ndat = np.nanmin([nvis, ncla, nclp]).astype(int)
            if ndat <= 2 * dof:
                if ndat <= 1 * dof:
                    out_txt =\
                        f"Insufficient data: " \
                        f"{ndat} (data) & " \
                        f"{dof} (allowed dof). " \
                        "Please check the number of visibility " \
                        f"or reduce either 'uvave' or 'snrflag'."
                    raise Exception(out_txt)
                else:
                    out_txt =\
                        f"WARNING! The number of data ({ndat}) " \
                        f"is recommended " \
                        f"to be twice the allowed dof ({dof})."
                    print(out_txt)


    def rsc_amplitude(self,
        uvfs
    ):
        """
        Rescale the visibility to the observed visibility amplitudes
            Arguments:
                uvfs (list): list of uvf objects
        """
        def cal_nll(rscf, inobs, inmod, inmprms, inmodel, inmask_sbl):
            nmod = int(np.round(inmprms["nmod"]))
            inmprms_ = copy.deepcopy(inmprms)
            for i in range(nmod):
                inmprms_[f"{i+1}_S"] *= rscf[0]

            inmod.append_visibility_model(
                freq_ref=inmod.freq, freq=inmod.freq,
                theta=inmprms_, fitset="sf", model=inmodel,
                spectrum="single", set_spectrum=False
            )

            amp_obs = np.abs(inobs.data[inmask_sbl]["vis"])
            sig_obs = np.abs(inobs.data[inmask_sbl]["sigma"])
            amp_mod = np.abs(inmod.data[inmask_sbl]["vism"])

            inmod.drop_visibility_model()

            out =\
                0.5 * np.sum(
                    np.abs(amp_mod - amp_obs)**2 / sig_obs**2
                    + np.log(2 * np.pi * sig_obs**2)
                )
            return out

        obs = copy.deepcopy(uvfs[0])
        mod = copy.deepcopy(uvfs[0])
        sblf_obs, sbl_obs = obs.get_sblf()

        # use shortest baseline for rescaling
        mask_sbl =\
            (obs.data["ant_name1"] == sbl_obs[0]) \
            & (obs.data["ant_name2"] == sbl_obs[1])

        # use all baselines for rescaling
        # mask_sbl = np.ones(len(obs.data), dtype=bool)

        def nll(*args):
            return cal_nll(*args)

        soln = optimize.minimize(
            nll,
            [1],
            args=(obs, mod, copy.deepcopy(self.mprms), self.model, mask_sbl),
            bounds=[[0, 2]], method="Powell"
        )

        rscf = soln.x[0]

        nmod = int(np.round(self.mprms["nmod"]))
        for i in range(nmod):
            self.mprms[f"{i+1}_S"] *= rscf

    def print_prms(self,
        ufreq, model="gaussian", fitset="sf", spectrum="spl", stats=None,
        relmod=True, printmsg=False, save_path=False, save_name=False
    ):
        """
        Print the model parameters
            Arguments:
                ufreq (list, float): unique frequency
                fitset (str): modeling frequency setting
                    - 'sf': single-frequency
                    - 'mf': multi-frequency
                spectrum (str): spectrum type
                    - 'spl': simple power-law
                    - 'cpl': curved power-law
                    - 'ssa': self-absorbed spectrum
                stats (tuple): statistical values (e.g., chi-square, AIC, BIC)
                printmsg (bool): Print the message
                save_path (str): path to save the model parameters
                save_name (str): name of the file to save the model parameters
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
                            smax_, a_, l_, m_ =\
                                (
                                    mprms[f"{i+1}_S"],
                                    mprms[f"{i+1}_a"],
                                    0, 0
                                )
                            if not relmod:
                                smax_, a_, l_, m_ =\
                                    (
                                        mprms[f"{i+1}_S"],
                                        mprms[f"{i+1}_a"],
                                        mprms[f"{i+1}_l"],
                                        mprms[f"{i+1}_m"]
                                    )
                        else:
                            smax_, a_, l_, m_ =\
                                (
                                    mprms[f"{i+1}_S"],
                                    mprms[f"{i+1}_a"],
                                    mprms[f"{i+1}_l"],
                                    mprms[f"{i+1}_m"]
                                )
                        S_ = smax_
                    else:
                        if self.set_spectrum:
                            if spectrum in ["spl"]:
                                if i == 0:
                                    smax_, a_, l_, m_, alpha_ =\
                                    (
                                        mprms[f"{i+1}_S"],
                                        mprms[f"{i+1}_a"],
                                        0, 0,
                                        mprms[f"{i+1}_alpha"]
                                    )
                                    if not reldmod:
                                        smax_, a_, l_, m_, alpha_ =\
                                        (
                                            mprms[f"{i+1}_S"],
                                            mprms[f"{i+1}_a"],
                                            mprms[f"{i+1}_l"],
                                            mprms[f"{i+1}_m"],
                                            mprms[f"{i+1}_alpha"]
                                        )
                                else:
                                    smax_, a_, l_, m_, alpha_ =\
                                    (
                                        mprms[f"{i+1}_S"],
                                        mprms[f"{i+1}_a"],
                                        mprms[f"{i+1}_l"],
                                        mprms[f"{i+1}_m"],
                                        mprms[f"{i+1}_alpha"]
                                    )
                                S_ = S_spl(ufreq[0], freq, smax_, alpha_)
                            if spectrum in ["cpl", "ssa"]:
                                if i == 0:
                                    smax_, a_, l_, m_, alpha_, tfreq_ =\
                                        (
                                            mprms[f"{i+1}_S"],
                                            mprms[f"{i+1}_a"],
                                            0, 0,
                                            mprms[f"{i+1}_alpha"],
                                            mprms[f"{i+1}_freq"]
                                        )
                                    if not relmod:
                                        smax_, a_, l_, m_, alpha_, tfreq_ =\
                                            (
                                                mprms[f"{i+1}_S"],
                                                mprms[f"{i+1}_a"],
                                                mprms[f"{i+1}_l"],
                                                mprms[f"{i+1}_m"],
                                                mprms[f"{i+1}_alpha"],
                                                mprms[f"{i+1}_freq"]
                                            )
                                    if spectrum in ["cpl"]:
                                        S_ = S_cpl(freq, smax_, tfreq_, alpha_)
                                    elif spectrum in ["ssa"]:
                                        S_ = SSA(freq, smax_, tfreq_, alpha_)
                                else:
                                    smax_, a_, l_, m_, alpha_, tfreq_ =\
                                        (
                                            mprms[f"{i+1}_S"],
                                            mprms[f"{i+1}_a"],
                                            mprms[f"{i+1}_l"],
                                            mprms[f"{i+1}_m"],
                                            mprms[f"{i+1}_alpha"],
                                            mprms[f"{i+1}_freq"]
                                        )
                                    if bool(np.round(mprms[f"{i+1}_thick"])):
                                        if spectrum in ["cpl"]:
                                            S_ =\
                                                S_cpl(
                                                    freq,
                                                    smax_,
                                                    tfreq_,
                                                    alpha_
                                                )
                                        elif spectrum in ["ssa"]:
                                            S_ =\
                                                SSA(
                                                    freq,
                                                    smax_,
                                                    tfreq_,
                                                    alpha_
                                                )
                                    else:
                                        S_ =\
                                            S_spl(
                                                ufreq[0],
                                                freq,
                                                smax_,
                                                alpha_
                                            )
                        else:
                            if i == 0:
                                smax_, a_, l_, m_ =\
                                    (
                                        mprms[f"{i+1}_S"],
                                        mprms[f"{i+1}_a"],
                                        0, 0
                                    )
                                if not relmod:
                                    smax_, a_, l_, m_ =\
                                        (
                                            mprms[f"{i+1}_S"],
                                            mprms[f"{i+1}_a"],
                                            mprms[f"{i+1}_l"],
                                            mprms[f"{i+1}_m"]
                                        )
                            else:
                                smax_, a_, l_, m_ =\
                                    (
                                        mprms[f"{i+1}_S"],
                                        mprms[f"{i+1}_a"],
                                        mprms[f"{i+1}_l"],
                                        mprms[f"{i+1}_m"]
                                    )
                            S_ = smax_
                    r_, p_ =\
                        (
                            np.sqrt(l_**2+m_**2),
                            np.arctan2(l_, m_)*u.rad.to(u.deg)
                        )
                    out_txt =\
                        f"# ({freq:.1f} GHz) Model {i+1}: " \
                        f"{S_:.3f}v {+r_:.3f}v {p_:.3f}v {a_:.3f}v"

                    if printmsg:
                        print(out_txt)

                    if save_path and save_name:
                        modelprms = open(save_path+save_name, mode="a")
                        modelprms.write(f"{out_txt}\n")
                        modelprms.close()
            elif model == "delta":
                for i in range(nmod):
                    if self.ifsingle:
                        if i == 0:
                            smax_, l_, m_ =\
                                (
                                    [f"{i+1}_S"],
                                    0, 0
                                )
                            if not relmod:
                                smax_, l_, m_ =\
                                    (
                                        [f"{i+1}_S"],
                                        [f"{i+1}_l"],
                                        [f"{i+1}_m"]
                                    )
                        else:
                            smax_, l_, m_ =\
                                (
                                    [f"{i+1}_S"],
                                    [f"{i+1}_l"],
                                    [f"{i+1}_m"]
                                )
                        S_ = smax_

                    r_, p_ =\
                        (
                            np.sqrt(l_**2+m_**2),
                            np.arctan2(l_, m_)*u.rad.to(u.deg)
                        )
                    out_txt =\
                        f"# ({freq:.1f} GHz) Model {i+1}: " \
                        f"{S_:.3f}v {+r_:.3f}v {p_:.3f}v 0.000"

                    if printmsg:
                        print(out_txt)

                    if save_path and save_name:
                        modelprms = open(save_path+save_name, mode="a")
                        modelprms.write(f"{out_txt}\n")
                        modelprms.close()

        if save_path and save_name:
            modelprms = open(save_path+save_name, mode="a")
            chi_tot = 0
            aic_tot = 0
            bic_tot = 0
            for i in range(len(stats[0])):
                out_txt =\
                    f"Chi2_{stats[0][i]:9s}: {stats[1][i]:10.3f} | " \
                    f"AIC_{stats[0][i]:9s} : {stats[2][i]:10.3f} | " \
                    f"BIC_{stats[0][i]:9s} : {stats[3][i]:10.3f}\n"
                modelprms.write(out_txt)
                if stats[0][i] in list(self.fdict.keys()):
                    chi_tot += stats[1][i]
                    aic_tot += stats[2][i]
                    bic_tot += stats[3][i]
            modelprms.write(f"Chi2_tot : {chi_tot:8.3f}\n")
            modelprms.write(f"AIC_tot  : {aic_tot:8.3f}\n")
            modelprms.write(f"BIC_tot  : {bic_tot:8.3f}\n")
            modelprms.write(f"logz : {stats[-2]:.3f} +/- {stats[-1]:.3f}\n")
            modelprms.close()


# @staticmethod
@jit(nopython=True)
def objective_function(theta, x, y, yerr, args):
    """
    Compute objective function (Bayesian Information Criterion)
        Arguments:
            theta (list): A list of parameters
            x (tuple): A tuple of x-arguments
            y (tuple): A tuple of y-arguments
            yerr (tuple): A tuple of y-error-arguments
            args (tuple): Arguments set
                args[0] (array, str): antenna name 1
                args[1] (array, str): antenna name 2
                args[2] (int): set_spectrum // 1: True, 0: False
                args[3] (int): spectrum ('single', 'spl', 'cpl', 'ssa')
                args[4] (int): modeltype (0:'delta', 1: 'gaussian')
                args[5] (int): ifsingle // 1: True, 0: False
                args[6] (float): lower boundary of bnd_pa
                args[7] (float): upper boundary of bnd_pa
                args[8] (list, str): 'fdict' keys
                args[9] (list, str): 'fdict' values
                args[10 - 13] (array): mask for closure amplitudes
                args[14 - 16] (array): mask for closure phases
                args[17] (int): boolean option for relative position
        Returns:
            Bayesian Information Criterion value (float)
    """
    N = len(x[1])
    model = np.zeros(N, dtype=np.complex64)
    ufreq = np.unique(x[1])

    nidx = 0
    nmprm = 0

    pa_bnd1 = args[6][0]
    pa_bnd2 = args[7][0]

    nmod = int(np.round(theta[0]))
    prms = theta[1:]
    mask_pa = 0

    if args[4] == 1: # model == "gaussian"
        for i in range(nmod):
            if args[5] == 1: # ifsingle == True (single-frequency)
                if i == 0:
                    if args[17] == 1:
                        model +=\
                            gvis0(
                                (x[2], x[3]),
                                prms[nidx+0],
                                prms[nidx+1]
                            )
                        nidx += 2
                        nmprm += 2
                    else:
                        model +=\
                            gvis(
                                (x[2], x[3]),
                                prms[nidx+0],
                                prms[nidx+1],
                                prms[nidx+2],
                                prms[nidx+3]
                            )
                        nidx += 4
                        nmprm += 4
                else:
                    if not np.isnan(pa_bnd1):
                        pa =\
                            180 / np.pi \
                            * np.angle(
                                prms[nidx+3] + 1j * prms[nidx+2]
                            )
                        if pa < 0 :
                            pa += 360
                        if pa_bnd1 < 0:
                            pa_bnd1 += 360
                        if pa_bnd2 < 0:
                            pa_bnd2 += 360
                        if pa_bnd1 > pa or pa > pa_bnd2:
                            mask_pa = 1

                    model +=\
                        gvis(
                            (x[2], x[3]),
                            prms[nidx+0],
                            prms[nidx+1],
                            prms[nidx+2],
                            prms[nidx+3]
                        )
                    nidx += 4
                    nmprm += 4
            else: # ifsingle == False (multi-frequency)
                if args[2] == 1: # set_spectrum == True
                    if i == 0:
                        if args[3] == 1: # spectrum == "spl"
                            if args[17] == 1:
                                model +=\
                                    gvis_spl0(
                                        (x[0], x[1], x[2], x[3]),
                                        prms[nidx+0],
                                        prms[nidx+1],
                                        prms[nidx+2]
                                    )
                                nidx += 3
                                nmprm += 3
                            else:
                                model +=\
                                    gvis_spl(
                                        (x[0], x[1], x[2], x[3]),
                                        prms[nidx+0],
                                        prms[nidx+1],
                                        prms[nidx+2],
                                        prms[nidx+3],
                                        prms[nidx+4]
                                    )
                                nidx += 5
                                nmprm += 5
                        elif args[3] == 2: # spectrum == "cpl"
                            if args[17] == 1:
                                model +=\
                                    gvis_cpl0(
                                        (x[1], x[2], x[3]),
                                        prms[nidx+0],
                                        prms[nidx+1],
                                        prms[nidx+2],
                                        prms[nidx+3]
                                    )
                                nidx += 4
                                nmprm += 4
                            else:
                                model +=\
                                    gvis_cpl(
                                        (x[1], x[2], x[3]),
                                        prms[nidx+0],
                                        prms[nidx+1],
                                        prms[nidx+2],
                                        prms[nidx+3],
                                        prms[nidx+4],
                                        prms[nidx+5]
                                    )
                                nidx += 6
                                nmprm += 6
                        elif args[3] == 3: # spectrum == "ssa"
                            if args[17] == 1:
                                model +=\
                                    gvis_ssa0(
                                        (x[1], x[2], x[3]),
                                        prms[nidx+0],
                                        prms[nidx+1],
                                        prms[nidx+2],
                                        prms[nidx+3]
                                    )
                                nidx += 4
                                nmprm += 4
                            else:
                                model +=\
                                    gvis_ssa(
                                        (x[1], x[2], x[3]),
                                        prms[nidx+0],
                                        prms[nidx+1],
                                        prms[nidx+2],
                                        prms[nidx+3],
                                        prms[nidx+4],
                                        prms[nidx+5]
                                    )
                                nidx += 6
                                nmprm += 6
                    else:
                        if args[3] == 1: # spectrum == "spl"
                            if not np.isnan(pa_bnd1):
                                pa =\
                                    180 / np.pi \
                                    * np.angle(
                                        prms[nidx+3] + 1j * prms[nidx+2]
                                    )
                                if pa < 0 :
                                    pa += 360
                                if pa_bnd1 < 0:
                                    pa_bnd1 += 360
                                if pa_bnd2 < 0:
                                    pa_bnd2 += 360
                                if pa_bnd1 > pa or pa > pa_bnd2:
                                    mask_pa = 1

                            model +=\
                                gvis_spl(
                                    (x[0], x[1], x[2], x[3]),
                                    prms[nidx+0],
                                    prms[nidx+1],
                                    prms[nidx+2],
                                    prms[nidx+3],
                                    prms[nidx+4]
                                )
                            nidx += 5
                            nmprm += 5
                        else: # spectrum in ["cpl", "ssa"]
                            if not np.isnan(pa_bnd1):
                                pa =\
                                    180 / np.pi \
                                    * np.angle(
                                        prms[nidx+4] + 1j * prms[nidx+3]
                                    )
                                if pa < 0 :
                                    pa += 360
                                if pa_bnd1 < 0:
                                    pa_bnd1 += 360
                                if pa_bnd2 < 0:
                                    pa_bnd2 += 360
                                if pa_bnd1 > pa or pa > pa_bnd2:
                                    mask_pa = 1

                            # optically thin spectrum
                            if int(np.round(prms[nidx])) == 0:
                                model +=\
                                    gvis_spl(
                                        (x[0], x[1], x[2], x[3]),
                                        prms[nidx+1],
                                        prms[nidx+2],
                                        prms[nidx+3],
                                        prms[nidx+4],
                                        prms[nidx+5]
                                    )
                                nidx += 7
                                nmprm += 5
                            # optically thick spectrum
                            else:
                                if args[3] == 2: # spectrum == "cpl"
                                    model +=\
                                        gvis_cpl(
                                            (x[1], x[2], x[3]),
                                            prms[nidx+1],
                                            prms[nidx+2],
                                            prms[nidx+3],
                                            prms[nidx+4],
                                            prms[nidx+5],
                                            prms[nidx+6]
                                        )
                                    nidx += 7
                                    nmprm += 6
                                elif args[3] == 3: # spectrum == "ssa"
                                    model +=\
                                        gvis_ssa(
                                            (x[1], x[2], x[3]),
                                            prms[nidx+1],
                                            prms[nidx+2],
                                            prms[nidx+3],
                                            prms[nidx+4],
                                            prms[nidx+5],
                                            prms[nidx+6]
                                        )
                                    nidx += 7
                                    nmprm += 6
                else: # set_spectrum == False
                    if i == 0:
                        if args[17] == 1:
                            model +=\
                                gvis0(
                                    (x[2], x[3]),
                                    prms[nidx+0],
                                    prms[nidx+1]
                                )
                            nidx += 2
                            nmprm += 2
                        else:
                            model +=\
                                gvis(
                                    (x[2], x[3]),
                                    prms[nidx+0],
                                    prms[nidx+1],
                                    prms[nidx+2],
                                    prms[nidx+3]
                                )
                            nidx += 4
                            nmprm += 4
                    else:
                        if not np.isnan(pa_bnd1):
                            pa =\
                                180 / np.pi \
                                * np.angle(
                                    prms[nidx+3] + 1j * prms[nidx+2]
                                )
                            if pa < 0 :
                                pa += 360
                            if pa_bnd1 < 0:
                                pa_bnd1 += 360
                            if pa_bnd2 < 0:
                                pa_bnd2 += 360
                            if pa_bnd1 > pa or pa > pa_bnd2:
                                mask_pa = 1

                        model +=\
                            gvis(
                                (x[2], x[3]),
                                prms[nidx+0],
                                prms[nidx+1],
                                prms[nidx+2],
                                prms[nidx+3]
                            )
                        nidx += 4
                        nmprm += 4
    elif args[4] == 0: # model == "delta"
        for i in range(nmod):
            if args[5] == 1: # ifsingle == True (single-frequency)
                if i == 0:
                    if args[17] == 1:
                        model += prms[nidx] * np.ones(N, dtype=np.complex64)
                        nidx += 1
                        nmprm += 1
                    else:
                        model +=\
                            dvis(
                                (x[2], x[3]),
                                theta[nidx+0],
                                theta[nidx+1],
                                theta[nidx+2]
                            )
                        nidx += 3
                        nmprm += 3
                else:
                    if not np.isnan(pa_bnd1):
                        pa =\
                            180 / np.pi \
                            * np.angle(
                                prms[nidx+2] + 1j * prms[nidx+1]
                            )
                        if pa < 0 :
                            pa += 360
                        if pa_bnd1 < 0:
                            pa_bnd1 += 360
                        if pa_bnd2 < 0:
                            pa_bnd2 += 360
                        if pa_bnd1 > pa or pa > pa_bnd2:
                            mask_pa = 1

                    model +=\
                        dvis(
                            (x[2], x[3]),
                            theta[nidx+0],
                            theta[nidx+1],
                            theta[nidx+2]
                        )
                    nidx += 3
                    nmprm += 3

    nasum = np.nansum(np.abs(model))

    # compute objective functions
    if not np.isnan(nasum) and not nasum == 0:
        objective = 0

        def compute_bic(in_res, in_sig2, in_type, in_nobs, in_nmprm):
            penalty = in_nmprm * np.log(in_nobs)
            if in_type == 0:
                nll =\
                    0.5 \
                    * (
                        np.nansum(
                            0.5 * (in_res**2 / in_sig2)
                            + np.log(2 * np.pi * in_sig2)
                        )
                    )
            else:
                nll =\
                    0.5 \
                    * (
                        np.nansum(
                            1.0 * (in_res**2 / in_sig2)
                            + np.log(2 * np.pi * in_sig2)
                        )
                    )
            return 2 * nll + penalty

        if 0 in args[8]:
            vis_obs = y[0]
            vis_mod = model
            vis_res = np.abs(vis_mod - vis_obs)
            nobs = len(y[0])
            vis_sig2 = yerr[0]**2
            objective -=\
                args[9][0] * compute_bic(vis_res, vis_sig2, 0, nobs, nmprm)

        if 1 in args[8]:
            nobs = len(y[0])
            amp_obs = np.abs(y[0])
            amp_obs =\
                np.where(
                    amp_obs <= yerr[0],
                    0,
                    np.sqrt(amp_obs**2 - yerr[0]**2)
                )
            amp_mod = np.abs(model)
            amp_sig2 = yerr[0]**2
            amp_res = amp_mod - amp_obs
            objective -=\
                args[9][1] * compute_bic(amp_res, amp_sig2, 1, nobs, nmprm)

        if 2 in args[8]:
            nobs = len(y[0])
            phs_obs = np.angle(y[0])
            phs_mod = np.angle(model)
            phs_sig2 = (yerr[0] / np.abs(y[0]))**2
            phs_res = np.abs(np.exp(1j * phs_mod) - np.exp(1j * phs_obs))
            objective -=\
                args[9][2] * compute_bic(phs_res, phs_sig2, 2, nobs, nmprm)

        if 3 in args[8]:
            amp12 = np.abs(model[args[10]])
            amp34 = np.abs(model[args[11]])
            amp13 = np.abs(model[args[12]])
            amp24 = np.abs(model[args[13]])
            clamp_mod = (amp12 * amp34) / (amp13 * amp24)
            nobs = len(y[2])
            clamp_obs = y[1]
            clamp_sig2 = yerr[1]**2
            clamp_res = np.abs( np.log(clamp_mod) - np.log(clamp_obs) )
            objective -=\
                args[9][3] * compute_bic(clamp_res, clamp_sig2, 3, nobs, nmprm)

        if 4 in args[8]:
            phs12 = np.angle(model[args[14]])
            phs23 = np.angle(model[args[15]])
            phs31 = np.angle(model[args[16]].conjugate())
            clphs_mod = phs12 + phs23 + phs31
            nobs = len(y[2])
            clphs_obs = y[2]
            clphs_sig2 = yerr[2]**2
            clphs_res = np.abs(np.exp(1j * clphs_mod) - np.exp(1j * clphs_obs))
            objective -=\
                args[9][4] * compute_bic(clphs_res, clphs_sig2, 4, nobs, nmprm)
        if mask_pa == 1:
            objective = -np.inf
    else:
        objective = -np.inf
    return objective

r2m = u.rad.to(u.mas)
d2m = u.rad.to(u.mas)

@jit(nopython=True)
def linear(x, m, a):
    """
        Arguments:
            x (array): input x-axis data points
            m (float): slope of the linear function
            a (float): offset of the linear function (constant)
        Returns:
            A linear function
    """
    out = m * x + a
    return out


@jit(nopython=True)
def gaussian_1d(x, peak, a, mx):
    """
        Arguments:
            x (array): input x-axis data points
            peak (float): peak-value of the Gaussian
            a (float): standard deviation of the Gaussian
            mx (float): offset of the peak in x-axis from the zero-position
        Returns:
            A 1-D Guassian function
    """
    out = peak * np.e**(-((x - mx) / a)**2 / 2)
    return out


@jit(nopython=True)
def gaussian_2d(xy, peak, ax, ay, mx, my, theta):
    """
        Arguments:
            xy (2D-array, tuple): input x/y-axis data points
            peak (float): peak-value of the Gaussian
            ax/y (float): standard deviation of the Gaussian in x/y-axis
            mx/y (float): offset of the peak in x/y-axis from the zero-position
            theta (float): position angle of the Gaussian (elliptical Guassian)
        Returns:
            A 2-D Guassian function
    """
    # x, y = xy
    out =\
        peak \
        * np.e**(
            -(xy[0] - mx)**2 / (2 * ax**2) -(xy[1] - my)**2 / (2 * ay**2)
        )
    return out


@jit(nopython=True)
def SSA(nu, Smax, tf, alpha):
    """
    NOTE: This function assumes optically thick spectral index as 2.5
    (Turler+1999, A&A, 349, 45T)
        Arguments:
            nu (array or float): input frequency
            Smax (float): flux density at 'tf'
            tf (float): turnover frequency of the SSA spectrum
            alpha (float): optically thin spectral index
        Returns:
            SSA flux density at nu
    """
    out =\
        Smax \
        * ((nu / tf)**2.5) \
        * (
            (1
                - np.e**(
                    -(1.5 * ((1 - (8 * alpha) / 7.5)**0.5 - 1))
                    * (nu / tf)**(alpha - 2.5)
                )
            )
            / (1 - np.e**(-(1.5 * ((1 - (8 * alpha) / 7.5)**0.5 - 1))))
        )
    return out


@jit(nopython=True)
def S_spl(nu_ref, nu, Smax, alpha):
    """
        Arguments:
            nu_ref (float): reference frequency
                            (recommended to set at the lowest one)
            nu (array or float): input frequency
            Smax (float): flux density at 'nu_ref'
            alpha (float): optically thin spectral index
        Returns:
            Simple power-law flux density at nu
    """
    out = 10**(alpha * (np.log10(nu) - np.log10(nu_ref)) + np.log10(Smax))
    return out


@jit(nopython=True)
def S_cpl(nu, Smax, tf, alpha):
    """
        Arguments:
            nu (array or float): input frequency
            Smax (float): flux density at 'tf'
            alpha (float): optically thin spectral index
        Returns:
            Curved power-law flux density at nu
    """
    out = Smax * (nu / tf)**(alpha * np.log10(nu / tf))
    return out


@jit(nopython=True)
def dvis0(args, S):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of Gaussian model
        Returns:
            complex visibility of a delta function model
    """
    out = S * np.ones(len(S), dtype=np.complex64)
    return out


@jit(nopython=True)
def gvis0(args, S, fwhm):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of Gaussian model
            fwhm (float): full-width at half maximum of Gaussian model
        Returns:
            complex visibility of a Gaussian model
    """
    out =\
        S \
        * np.e**(
            -2*np.pi**2 * (fwhm / (2 * (2 * np.log(2))**0.5))**2 \
            * ((args[1] / r2m)**2 + (args[0] / r2m)**2)
        )
    return out


@jit(nopython=True)
def gvis_spl0(args, Smax, fwhm, alpha):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (float): reference frequency
                                 (recommended to set at the lowest one)
                args[1] (array or float): input frequency
                args[2] (1D-array): u-axis data points
                args[3] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'args[0]'
            fwhm (float): full-width at half maximum of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model
                (spl-based)
    """
    out =\
        S_spl(args[0], args[1], Smax, alpha) \
        * np.e**(
            -2 * (np.pi * (fwhm / (2 * (2 * np.log(2))**0.5)))**2 \
            * ((args[2] / r2m)**2 + (args[3] / r2m)**2)
        )
    return out


@jit(nopython=True)
def gvis_cpl0(args, Smax, fwhm, alpha, nu_m):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (array or float): input frequency
                args[1] (1D-array): u-axis data points
                args[2] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model
                (cpl-based)
    """
    out =\
        S_cpl(args[0], Smax, nu_m, alpha) \
        * np.e**(
            -2 * (np.pi * (fwhm / (2 * (2 * np.log(2))**0.5)))**2
            * ((args[1] / r2m)**2 + (args[2] / r2m)**2)
        )
    return out


@jit(nopython=True)
def gvis_ssa0(args, Smax, fwhm, alpha, nu_m):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (array or float): input frequency
                args[1] (1D-array): u-axis data points
                args[2] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
            nu_m (float): turnover frequency
        Returns:
            complex visibility of Gaussian model
                (SSA-based; Turler+1999, A&A, 349, 45T)
    """
    out =\
        SSA(args[0], Smax, nu_m, alpha) \
        * np.e**(
            -2 * (np.pi * (fwhm / (2 * (2 * np.log(2))**0.5)))**2
            * ((args[1] / r2m)**2 + (args[2] / r2m)**2)
        )
    return out


@jit(nopython=True)
def dvis(args, S, l, m):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of delta function model
            l (float): right ascension position of delta function model
            m (float): declination position of delta function model
        Returns:
            complex visibility of Gaussian model
    """
    out =\
        S \
        * np.e**(
            2j * np.pi * ((args[0] / r2m) * l + (args[1] / r2m) * m)
        )
    return out


@jit(nopython=True)
def gvis(args, S, fwhm, l, m):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of Gaussian model
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
        Returns:
            complex visibility of Gaussian model
    """
    out =\
        S \
        * np.e**(
            -2 * np.pi**2 * (fwhm / (2 * (2 * np.log(2))**0.5))**2
            * ((args[0] / r2m)**2 + (args[1] / r2m)**2)
            + 2j * np.pi * ((args[0] / r2m) * l + (args[1] / r2m) * m)
        )
    return out


@jit(nopython=True)
def gvis_spl(args, Smax, fwhm, l, m, alpha):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (float): reference frequency, recommended to set at the lowest one
                args[1] (array or float): input frequency
                args[2] (1D-array): u-axis data points
                args[3] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'args[0]'
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model
                (spl-based)
    """
    out =\
        S_spl(args[0], args[1], Smax, alpha) \
        * np.e**(
            -2 * (np.pi * (fwhm / (2 * (2 * np.log(2))**0.5)))**2
            * ((args[2] / r2m)**2 + (args[3] / r2m)**2)
            + 2j * np.pi * ((args[2] / r2m) * l + (args[3] / r2m) * m)
        )
    return out


@jit(nopython=True)
def gvis_cpl(args, Smax, fwhm, l, m, alpha, nu_m):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (float): reference frequency, recommended to set at the lowest one
                args[1] (array or float): input frequency
                args[2] (1D-array): u-axis data points
                args[3] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model
                (cpl-based)
    """
    out =\
        S_cpl(args[0], Smax, nu_m, alpha) \
        * np.e**(
            -2 * (np.pi * (fwhm / (2 * (2 * np.log(2))**0.5)))**2
            * ((args[1] / r2m)**2 + (args[2] / r2m)**2)
            + 2j * np.pi * ((args[1] / r2m) * l + (args[2] / r2m) * m)
        )
    return out


@jit(nopython=True)
def gvis_ssa(args, Smax, fwhm, l, m, alpha, nu_m):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (array or float): input frequency
                args[1] (1D-array): u-axis data points
                args[2] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
            nu_m (float): turnover frequency
        Returns:
            complex visibility of Gaussian model
                (SSA-based; Turler+1999, A&A, 349, 45T)
    """
    out =\
        SSA(args[0], Smax, nu_m, alpha) \
        * np.e**(
            -2
            * (np.pi * (fwhm / (2 * (2 * np.log(2))**0.5)))**2
            * ((args[1] / r2m)**2 + (args[2] / r2m)**2)
            + 2j * np.pi * ((args[1] / r2m) * l + (args[2] / r2m) * m)
        )
    return out


@jit(nopython=True)
def set_closure(
    data_vis, data_ant1, data_ant2,
    mask_amp12, mask_amp34, mask_amp13, mask_amp24,
    mask_phs12, mask_phs23, mask_phs31
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
            tuple: (closure_amplitude, closure_phase)
    """

    clamp = np.array([])
    clphs = np.array([])

    Nant = len(np.unique(np.append(data_ant1, data_ant2)))

    if Nant >= 4:
        amp12 = np.abs(data_vis[mask_amp12])
        amp34 = np.abs(data_vis[mask_amp34])
        amp13 = np.abs(data_vis[mask_amp13])
        amp24 = np.abs(data_vis[mask_amp24])
        clamp = (amp12 * amp34) / (amp13 * amp24)

    if Nant >= 3:
        phs12 = np.angle(data_vis[mask_phs12])
        phs23 = np.angle(data_vis[mask_phs23])
        phs31 = np.angle(data_vis[mask_phs31].conj())
        clphs = phs12 + phs23 + phs31

    if Nant >= 3:
        return clamp, clphs
    else:
        out_txt = "There are no valid closure quantities"
        raise Exception(out_txt)
