
import os
import sys
import gc
import copy
import ctypes
import warnings
import functools
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
from numba import njit, jit
import itertools as it
from astropy import units as au
from scipy import optimize
import matplotlib.pyplot as plt
import dynesty
from dynesty import NestedSampler
from dynesty.pool import Pool
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
from dynesty.utils import quantile as dyquan

import gamvas as gv

r2m = au.rad.to(au.mas)
d2m = au.deg.to(au.mas)

fields1 = ["S", "a", "l", "m", "freq", "alpha"]
dtypes1 = ["f8", "f8", "f8", "f8", "f8", "f8"]

fields2 = ["S", "a", "l", "m", "alpha", "beta"]
dtypes2 = ["f8", "f8", "f8", "f8", "f8", "f8"]

class modeling:
    """
    NOTE: This modeling is based on 'dynesty'
          which is implementing Bayesian nested sampling
    (Web site: https://dynesty.readthedocs.io/en/stable/api.html#api)
    (NASA/ADS: https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract)

    Attributes:
        uvfs (list): list of uvf objects
        select_pol (str): polarization type (e.g., i, q, u ,v, etc..)
            or parallel/cross-hand polarization (RR, LL, RL, LR)
        x (tuple): tuple of x-arguments
        y (tuple): tuple of y-arguments
        yerr (tuple): tuple of y-error-arguments
        args (tuple): arguments set
        factor_sblf (float): factor on shortest-baseline flux density
        sampler (str): sampling method in 'dynesty'
            (availables: 'rwalk', 'rslice', 'slice')
        bound (str): bounding condition in 'dynesty'
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
            (availables: 'flat', 'spl', 'cpl', 'ssa')
        uvw (str): uv-weighting option
            - 'w': weighting by visibility weight
            - 'u': unity weighting
        shift ((float, float)): amount of shift in (RA, DEC)-direction
        fixnmod (bool): if True, fix the number of models to the 'maxn'
        maxn (int): maximum number of models to be allowed
        npix (int): number of pixels (on the image)
        mindr (float): minimum dynamic range to plot a contour in images
        mapfov (float): field of view
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
    def __init__(
        self,
        uvfs=None, select_pol="i", x=None, y=None, yerr=None, args=None,
        factor_sblf=1.0, sampler="slice", bound="multi", runfit_pol=False,
        ftype=None, fwght=None, boundset=None, bnd_a=5, bnd_l=None,
        bnd_m=None, bnd_f=None, bnd_pa=(None, None), nflux=False, ufreq=None,
        bands=None, spectrum="flat", model="gaussian", uvw="u", shift=None,
        fixnmod=False, maxn=None, npix=1024, mindr=3, mapfov=None, gacalerr=0,
        dognorm=True, dogscale=False, doampcal=True, zero_cp=False,
        selfflag=True, rscsbl=True, dophscal=True, path_fig=None, source=None,
        date=None, cgain_truth=None, save_uvfits=True, save_imgfits=True,
        ncpu=1
    ):
        if isinstance(uvfs, list):
            self.uvfs = uvfs
        else:
            self.uvfs = [uvfs]

        self.select_pol = select_pol
        self.x = x
        self.y = y
        self.yerr = yerr
        self.args = args

        self.factor_sblf = factor_sblf
        self.sampler = sampler
        self.bound = bound

        self.runfit_pol = runfit_pol

        self.ftype = ftype
        if ftype is not None and fwght is not None:
            self.fwght = fwght
            self.fdict = dict(zip(self.ftype, self.fwght))
        else:
            self.fwght = fwght
            self.fdict = None

        self.boundset = boundset
        self.bnd_a = bnd_a
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
        self.mapfov = mapfov

        self.gacalerr = gacalerr
        self.dognorm  = dognorm
        self.dogscale = dogscale
        self.doampcal = doampcal
        self.dophscal = dophscal
        self.zero_cp = zero_cp
        self.selfflag = selfflag
        self.rscsbl = rscsbl

        self.path_fig_base = path_fig
        self.path_fig = path_fig
        self.source = source
        self.date = date
        self.cgain_truth = cgain_truth
        self.save_uvfits = save_uvfits
        self.save_imgfits = save_imgfits
        self.ncpu = ncpu

        self.nrun = 0

        self.pol = gv.polarization.modeling.polarization(ncpu=self.ncpu)

    def check_dof(self, uvfs=None, spectrum=None, maxn=None):
        if uvfs is None:
            raise ValueError("'uvfs' must be provided.")
        if spectrum is None:
            raise ValueError("'spectrum' must be provided.")
        if maxn is None:
            raise ValueError("'maxn' must be provided.")

        if spectrum == "flat":
            dof = 2 + 4 * (maxn - 1)
            for uvf in uvfs:
                nvis = uvf.data.size
                ncla = uvf.clamp.size
                nclp = uvf.clphs.size
                if ncla == 1:
                    ncla = np.nan
                if nclp == 1:
                    nclp = np.nan
                counts = {"vis": nvis, "clamp": ncla, "clphs": nclp}
                ndat = int(np.nanmin(list(counts.values())))
                _types_dof = [k for k, v in counts.items() if v == ndat][0]
                if ndat <= 2 * dof:
                    if ndat <= 1 * dof:
                        raise ValueError(
                            f"Insufficient data ({uvf.freq0:.1f} GHz): "
                            f"{ndat} (data, {_types_dof!r}) "
                            f"& {dof} (allowed dof).\n"
                            f"Please check the number of visibility "
                            f"or consider to reduce either 'avgtime' or 'snr'."
                        )
                    else:
                        warnings.warn(
                            f"The number of data ({ndat}, {_types_dof!r}, "
                            f"{uvf.freq0:.1f} GHz) is recommended "
                            f"to be twice the allowed dof ({dof}).",
                            UserWarning
                        )

        else:
            dof = 4 + 6 * (maxn - 1)
            uvf = gv.utils.set_uvf(uvfs, dotype="mf")
            nvis = uvf.data.size
            ncla = uvf.clamp.size
            nclp = uvf.clphs.size
            if ncla == 1:
                ncla = np.nan
            if nclp == 1:
                nclp = np.nan
            ndat = int(np.nanmin([nvis, ncla, nclp]))
            if ndat <= 2 * dof:
                if ndat <= 1 * dof:
                    raise Exception(
                        f"Insufficient data: "
                        f"{ndat} (data) & "
                        f"{dof} (allowed dof). "
                        "Please check the number of visibility "
                        f"or reduce either 'uvave' or 'snrflag'."
                    )
                else:
                    warnings.warn(
                        f"The number of data ({ndat}) is recommended "
                        f"to be twice the allowed dof ({dof}).",
                        UserWarning
                    )

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
        ntheta = samples.shape[1]

        qls = np.array([])
        qms = np.array([])
        qhs = np.array([])

        for i in range(ntheta):
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
            theta_ = np.array([qls_, qms_, qhs_])

            out_xlsx = pd.DataFrame(
                theta_, index=["lolim", "value", "uplim"]
            ).T

            out_xlsx["idx"] = idx_
            out_xlsx.to_excel(f"{save_path}{save_name}")
            self.out_xlsx = out_xlsx

    def get_ntheta(self):
        """
        Get the number of parameters
        """
        theta = self.theta.copy()
        vals = rfn.structured_to_unstructured(theta)
        nmod = round(float(theta["nmod"]))

        mask_thick = np.array(
            list(map(
                lambda x: "thick" in x,
                theta.dtype.names
            ))
        )

        mask_thick = np.round(
            vals[mask_thick]
        ).astype(int)

        ntheta = len(vals) - 2 * (len(mask_thick) - mask_thick.sum()) - 1

        self.ntheta = ntheta

    def initialize_memory(self):
        self.results = None
        self.samples = None
        self.weights = None

        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass

    def print_theta(
        self,
        ufreq, model="gaussian", spectrum="spl", stats=None, relmod=True,
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

        theta = self.theta.copy()
        nmod = round(float(theta["nmod"]))

        ufreq = np.atleast_1d(np.asarray(ufreq, dtype=float))

        if save_path and save_name:
            out_theta = open(f"{save_path}/{save_name}", "w")
            out_theta.close()

        out_txt_init = f"# FLUX    RADIUS    POS.A    FWHM"
        out_theta = open(f"{save_path}/{save_name}", mode="a")
        out_theta.write(f"{out_txt_init}\n")
        out_theta.close()
        print(out_txt_init)

        for nfreq, freq in enumerate(ufreq):
            if model == "gaussian":
                for i in range(nmod):
                    idx_s = f"{i + 1}_S"
                    idx_a = f"{i + 1}_a"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"
                    idx_i = f"{i + 1}_alpha"
                    idx_f = f"{i + 1}_freq"
                    idx_b = f"{i + 1}_beta"
                    idx_t = f"{i + 1}_thick"

                    _dtype = theta.dtype.names

                    has_l = idx_l in _dtype
                    has_m = idx_m in _dtype
                    has_i = idx_i in _dtype
                    has_f = idx_f in _dtype
                    has_b = idx_b in _dtype
                    has_t = idx_t in _dtype

                    _s = theta[idx_s]
                    _a = theta[idx_a]

                    if has_l and has_m:
                        _l = theta[idx_l]
                        _m = theta[idx_m]
                    else:
                        _l = 0
                        _m = 0

                    if has_i:
                        _i = theta[idx_i]

                    if has_f:
                        _f = theta[idx_f]

                    if spectrum == "flat":
                        S = _s
                    elif spectrum == "spl":
                        S = gv.functions.spl(ufreq[0], freq, _s, _i)
                    elif spectrum in ["cpl", "ssa"]:
                        if has_t:
                            mask_thick = round(float(theta[idx_t])) == 0
                            if mask_thick:
                                S = gv.functions.spl(
                                    ufreq[0], freq,
                                    _s, _i
                                )
                            else:
                                if spectrum == "cpl":
                                    S = gv.functions.cpl(
                                        freq, _s, _f, _i
                                    )
                                else:
                                    S = gv.functions.ssa(
                                        freq, _s, _f, _i
                                    )
                        else:
                            if spectrum == "cpl":
                                S = gv.functions.cpl(
                                    freq, _s, _f, _i
                                )
                            else:
                                S = gv.functions.ssa(
                                    freq, _s, _f, _i
                                )
                    elif spectrum == "poly":
                        _b = theta[idx_b]
                        S = gv.functions.poly(ufreq[0], freq, _s, _i, _b)

                    _r, _p = (
                        np.sqrt(_l**2 + _m**2),
                        np.arctan2(_l, _m) * au.rad.to(au.deg)
                    )

                    out_txt = (
                        f"# ({freq:.1f} GHz) Model {i + 1}: "
                        f"{S:6.3f}v {+_r:6.3f}v {_p:8.3f}v "
                        f"{_a:6.3f}v"
                    )

                    if prt:
                        print(out_txt)

                    if save_path and save_name:
                        out_theta = open(f"{save_path}/{save_name}", mode="a")
                        out_theta.write(f"{out_txt}\n")
                        out_theta.close()

            elif model == "delta":
                for i in range(nmod):
                    idx_s = f"{i + 1}_S"
                    idx_l = f"{i + 1}_l"
                    idx_m = f"{i + 1}_m"
                    idx_i = f"{i + 1}_alpha"
                    idx_f = f"{i + 1}_freq"
                    idx_b = f"{i + 1}_beta"
                    idx_t = f"{i + 1}_thick"

                    _dtype = theta.dtype.names

                    has_l = idx_l in _dtype
                    has_m = idx_m in _dtype
                    has_i = idx_i in _dtype
                    has_f = idx_f in _dtype
                    has_b = idx_b in _dtype
                    has_t = idx_t in _dtype

                    _s = theta[idx_s]

                    if has_l and has_m:
                        _l = theta[idx_l]
                        _m = theta[idx_m]
                    else:
                        _l = 0
                        _m = 0

                    if has_i:
                        _i = theta[idx_i]

                    if has_f:
                        _f = theta[idx_f]

                    if spectrum == "flat":
                        S = _s
                    elif spectrum == "spl":
                        S = gv.functions.spl(ufreq[0], freq, _s, _i)
                    elif spectrum in ["cpl", "ssa"]:
                        if has_t:
                            mask_thick = round(float(theta[idx_t])) == 0
                            if mask_thick:
                                S = gv.functions.spl(
                                    ufreq[0], freq,
                                    _s, _i
                                )
                            else:
                                if spectrum == "cpl":
                                    S = gv.functions.cpl(
                                        freq, _s, _f, _i
                                    )
                                else:
                                    S = gv.functions.ssa(
                                        freq, _s, _f, _i
                                    )
                        else:
                            if spectrum == "cpl":
                                S = gv.functions.cpl(
                                    freq, _s, _f, _i
                                )
                            else:
                                S = gv.functions.ssa(
                                    freq, _s, _f, _i
                                )
                    elif spectrum == "poly":
                        _b = theta[idx_b]
                        S = gv.functions.poly(ufreq[0], freq, _s, _i, _b)

                    _r, _p = (
                        np.sqrt(_l**2 + _m**2),
                        np.arctan2(_l, _m) * au.rad.to(au.deg)
                    )

                    out_txt = (
                        f"# ({freq:.1f} GHz) Model {i + 1}: "
                        f"{S:6.3f}v {+_r:6.3f}v {_p:8.3f}v "
                        f"{0.0:6.3f}"
                    )

                    if prt:
                        print(out_txt)

                    if save_path and save_name:
                        out_theta = open(f"{save_path}/{save_name}", mode="a")
                        out_theta.write(f"{out_txt}\n")
                        out_theta.close()

        if save_path and save_name:
            out_theta = open(f"{save_path}/{save_name}", mode="a")
            chi_tot = 0
            aic_tot = 0
            bic_tot = 0
            for i in range(len(stats[0])):
                out_txt = (
                    f"Chi2_{stats[0][i]:9s}: {stats[1][i]:10.3f} | "
                    f"AIC_{stats[0][i]:9s} : {stats[2][i]:10.3f} | "
                    f"BIC_{stats[0][i]:9s} : {stats[3][i]:10.3f}\n"
                )

                out_theta.write(out_txt)

                if stats[0][i] in list(self.fdict.keys()):
                    chi_tot += stats[1][i]
                    aic_tot += stats[2][i]
                    bic_tot += stats[3][i]

            out_theta.write(f"Chi2_tot : {chi_tot:8.3f}\n")
            out_theta.write(f"AIC_tot  : {aic_tot:8.3f}\n")
            out_theta.write(f"BIC_tot  : {bic_tot:8.3f}\n")
            out_theta.write(f"logz : {stats[-2]:.3f} +/- {stats[-1]:.3f}\n")
            out_theta.close()

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
            if self.spectrum in ["cpl", "ssa"] and i != 0:
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

    def rsc_amplitude(
        self,
        uvfs, rscsbl=True, save_path=None, save_name=None, save_xlsx=False
    ):
        """
        Rescale the visibility to the observed visibility amplitudes
        Args:
            uvfs (list): list of uvf objects
        """
        def cal_nll(rscf, inobs, inmod, intheta, inmodel, sbl_mask):
            nmod = round(float(intheta["nmod"]))
            intheta_ = copy.deepcopy(intheta)
            for i in range(nmod):
                intheta_[f"{i + 1}_S"] *= rscf

            inmod.model_visibility_append(
                freq_ref=inmod.freq0, theta=intheta_, model=inmodel,
                spectrum="flat"
            )
            inmod.set_data(prt=False)

            amp_obs = inobs.get_data(dotype="amp", flatten=True)[sbl_mask]
            sig_obs = inobs.get_data(dotype="sig0", flatten=True)[sbl_mask]
            amp_mod = np.abs(
                inmod.get_data(dotype="vism", flatten=True)[sbl_mask]
            )
            res = amp_mod - amp_obs
            sig2 = sig_obs**2

            inmod.model_visibility_drop()

            out = compute_nll(in_res=res, in_sig2=sig2, in_type=1)

            return out

        obs = copy.deepcopy(uvfs[0])
        mod = copy.deepcopy(uvfs[0])

        if rscsbl:
            sblf, sbl, sbl_mask = uvfs[0].get_sblf()
        else:
            sbl_mask = np.ones_like(uvfs[0].data, dtype=bool)

        soln = optimize.minimize_scalar(
            cal_nll, bounds=(0, 100), method="bounded",
            args=(obs, mod, copy.deepcopy(self.theta), self.model, sbl_mask)
        )

        rscf = soln.x

        nmod = round(float(self.theta["nmod"]))
        for i in range(nmod):
            self.theta[f"{i + 1}_S"] *= rscf

            mask_xlsx = self.out_xlsx["idx"] == f"{i + 1}_S"
            self.out_xlsx.loc[mask_xlsx, ["lolim", "value", "uplim"]] *= rscf

            if save_xlsx:
                self.out_xlsx.to_excel(f"{save_path}{save_name}")

    def _validate_closure_fit_types(self, uvf):
        """Reject requested closure terms unavailable after flagging."""

        requested = [] if self.ftype is None else self.ftype
        missing = []
        if "clamp" in requested and not uvf.clamp_check:
            missing.append("clamp")
        if "clphs" in requested and not uvf.clphs_check:
            missing.append("clphs")
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(
                "Requested closure fit type(s) are unavailable after "
                f"flagging: {missing_text}. Remove them from 'ftype' or "
                "relax the flags and rebuild closures."
            )

    def run(self, uvave=None):
        """
        Run the modeling utilies
        """
        availables = ["flat", "spl", "cpl", "ssa", "poly"]
        if self.spectrum not in availables:
            raise ValueError(
                f"Invalid spectrum: {self.spectrum!r}.\n"
                f"Availables: {availables}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            uvfs = copy.deepcopy(self.uvfs)

        uvall = gv.utils.set_uvf(uvfs, dotype="mf")

        freq = uvall.get_data(dotype="frequency").astype("f4")
        ant1 = uvall.get_data(dotype="ant1")
        ant2 = uvall.get_data(dotype="ant2")

        ufreq = np.unique(freq)
        uant = np.unique(np.append(ant1, ant2))

        if len(ufreq) == 1 and self.spectrum != "flat":
            raise ValueError(
                f"Single frequency ({ufreq[0]:.3f} GHz) "
                f"does not support {self.spectrum!r} spectrum."
            )

        if len(ufreq) == 2 and self.spectrum in ["cpl", "ssa"]:
            raise ValueError(
                f"Dual frequency ({ufreq[0]:.3f} and {ufreq[1]:.3f} GHz) "
                f"does not support{self.spectrum!r} spectrum."
            )

        narrow_frequency = (ufreq.max() - ufreq.min()) / ufreq.mean() <= 0.1
        if narrow_frequency and self.spectrum != "flat":
            warnings.warn(
                f"Narrow frequency range "
                f"({ufreq.min():.3f} - {ufreq.max():.3f} GHz) "
                f"may not be appropriate for {self.spectrum!r} spectrum.",
                UserWarning
            )

        nvis  = uvall.data.shape[0]
        mask_clamp = uvall.clamp_check
        mask_clphs = uvall.clphs_check
        self._validate_closure_fit_types(uvall)
        if mask_clamp:
            ncamp = uvall.tmpl_clamp.shape[0]
        else:
            ncamp = 0

        if mask_clphs:
            ncphs = uvall.tmpl_clphs.shape[0]
        else:
            ncphs = 0

        if not mask_clamp and not mask_clphs:
            raise ValueError("No closure amplitudes and phases are available.")

        if uvave is None:
            uvave = uvall.avg_timebin
            if isinstance(uvave, (float, int)):
                uvave_unit = "(sec)"
            else:
                uvave_unit = ""
        else:
            uvave_unit = ""

        print(
            f"\n### Running parameters ###\n"
            f"# Field of view: {self.mapfov:.1f} ({uvall.mapunit.name})\n"
            f"# Synthesized beam: "
            f"{uvall.bprms[0] * au.mas.to(uvall.mapunit):.2f}"
            f" \u00D7 "
            f"{uvall.bprms[1] * au.mas.to(uvall.mapunit):.2f} "
            f"{uvall.mapunit.name}, "
            f"{uvall.bprms[2]:.1f} deg\n"
            f"# Fit-spec: {self.spectrum!r}\n"
            f"# Fit-type: {self.ftype}\n"
            f"# Fit-wght: {self.fwght}\n"
            f"# Fit-mode: {self.model!r}\n"
            f"# Selected polarization: {self.select_pol.upper()!r}\n"
            f"# uv-average time: {uvave} {uvave_unit}\n"
            f"# Number of complex visibility: {nvis}\n"
            f"# Number of closure amplitude: {ncamp}\n"
            f"# Number of closure phase: {ncphs}\n"
            f"# Number of active CPU cores: {self.ncpu}/{os.cpu_count()}\n"
        )

        if "clphs" in self.ftype:
            if "phs" not in self.ftype and "vis" not in self.ftype:
                self.relmod = True
            else:
                self.relmod = False
        else:
            self.relmod = False

        self.check_dof(uvfs=uvfs, spectrum=self.spectrum, maxn=self.maxn)
        self.run_modeling()

    def run_modeling(self):
        """
        Run model-fit
        """
        self.nrun += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            uvfs = copy.deepcopy(self.uvfs)

        if self.spectrum == "flat":
            freqtype = "sf"
            plotalif = True
        else:
            freqtype = "mf"
            plotalif = False

        uvf = gv.utils.set_uvf(uvfs, dotype=freqtype)
        self._validate_closure_fit_types(uvf)

        cg_pol1_ant1 = np.ones(uvf.data.shape[0], dtype=np.complex64)
        cg_pol1_ant2 = np.ones(uvf.data.shape[0], dtype=np.complex64)
        cg_pol2_ant1 = np.ones(uvf.data.shape[0], dtype=np.complex64)
        cg_pol2_ant2 = np.ones(uvf.data.shape[0], dtype=np.complex64)

        # set fit types & weights
        ftype = self.ftype.copy()

        if self.fwght is None:
            fwght = gv.utils.get_fwght(
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

        # short-baseline flux: for boundary conditions
        _vis = uvf.get_data("vis")
        bslf_max = np.nanmax(np.abs(_vis))
        sblf = self.factor_sblf * bslf_max

        print(f"\n# Maximum baseline flux density: {bslf_max:.3f} Jy")

        # plot & save initial figures
        if self.nrun == 1:
            # set figure path & make directory
            if self.spectrum == "flat":
                _freq = uvfs[0].freq_mean
                self.path_fig_init = (
                    f"{self.path_fig}/{self.model}.{_freq:.1f}/"
                )
                self.path_fig = (
                    f"{self.path_fig}/{self.model}.{_freq:.1f}/"
                )
            else:
                self.path_fig_init = self.path_fig
                self.path_fig = (
                    f"{self.path_fig}/mf.{self.model}.{self.spectrum}/"
                )
            gv.utils.mkdir(self.path_fig)

            for _uvf in uvfs:
                _uvf.ploter.draw_tplot(
                    _uvf, plotimg=False,
                    save_path=self.path_fig_init,
                    save_name=(
                        f"{self.source}.{self.date}.{freqtype}."
                        f"initial.tplot.{round(_uvf.freq_mean)}"
                    ),
                    save_form="pdf"
                )

            uvf.ploter.draw_radplot(
                uvf, plotimg=False,
                save_path=self.path_fig_init,
                save_name=(
                    f"{self.source}.{self.date}.{freqtype}.initial.radplot"
                ),
                save_form="pdf"
            )

            uvf.ploter.draw_uvcover(
                uvf, plotimg=False,
                save_path=self.path_fig_init,
                save_name=(
                    f"{self.source}.{self.date}.{freqtype}.initial.uvcover"
                ),
                save_form="pdf"
            )

            uvf.ploter.draw_dirtymap(
                uvf, plotimg=False, plot_resi=False,
                npix=self.npix, uvw=self.uvw,
                save_path=self.path_fig_init,
                save_name=(
                    f"{self.source}.{self.date}.{freqtype}.initial.dirtmap"
                ),
                save_form="pdf"
            )

        # set boundary conditions
        if self.spectrum in ["flat", "spl", "cpl", "ssa"]:
            fields_bnds = fields1
            dtypes_bnds = dtypes1
        elif self.spectrum == "poly":
            fields_bnds = fields2
            dtypes_bnds = dtypes2

        if self.boundset is None:
            if self.bnd_f is not None:
                _bnds_f = self.bnd_f
            else:
                _bnds_f = [uvf.ufreq[0], uvf.ufreq[-1]]

            _bnds = gv.utils.set_boundary(
                    nmod=nmod, select_pol=self.select_pol,
                    spectrum=self.spectrum, sblf=sblf, bnd_a=self.bnd_a,
                    bnd_l=self.bnd_l, bnd_m=self.bnd_m, bnd_f=_bnds_f,
                    relmod=self.relmod
            )

            bnds = gv.utils.structured_array(
                data=_bnds,
                field=fields_bnds,
                dtype=dtypes_bnds
            )

        else:
            if isinstance(self.boundset, (list, tuple)):
                bnds = gv.utils.structured_array(
                    data=self.boundset,
                    field=fields_bnds,
                    dtype=dtypes_bnds
                )
            else:
                bnds = self.boundset

        self.boundset = bnds

        ftype = list(self.fdict.keys())
        fwght = list(self.fdict.values())

        # set frequency information on closure quantities
        if uvf.clamp_check:
            f_clamp = np.ma.getdata(uvf.tmpl_clamp["freq"])
        else:
            f_clamp = []

        if uvf.clphs_check:
            f_clphs = np.ma.getdata(uvf.tmpl_clphs["freq"])
        else:
            f_clphs = []

        clamp_uvcomb, clphs_uvcomb = gv.utils.set_uvcombination(uvf)

        self.clamp_uvcomb = copy.deepcopy(clamp_uvcomb)
        self.clphs_uvcomb = copy.deepcopy(clphs_uvcomb)

        # set arguments: x
        self.x = (
            uvfs[0].freq0,
            uvf.get_data(dotype="frequency", flatten=True),
            uvf.get_data(dotype="u", flatten=True),
            uvf.get_data(dotype="v", flatten=True)
        )

        # set arguments: y
        self.y = (
            uvf.get_data(dotype="vis", flatten=True),
            np.ma.getdata(uvf.clamp["clamp"]),
            np.ma.getdata(uvf.clphs["clphs"]),
            copy.deepcopy(clamp_uvcomb),
            copy.deepcopy(clphs_uvcomb)
        )

        # set arguments: y error
        self.yerr = (
            uvf.get_data(dotype="sig", flatten=True),
            np.ma.getdata(uvf.clamp["sig_logclamp"]),
            np.ma.getdata(uvf.clphs["sig_clphs"])
        )

        Nant = len(np.unique(
            np.append(
                uvf.get_data(dotype="ant1", flatten=True),
                uvf.get_data(dotype="ant2", flatten=True)
            )
        ))

        def _uv_idx(mask, closure_freq):
            return gv.utils._uv_lookup_indices(
                self.x[1], self.x[2], self.x[3], closure_freq, mask
            )

        # index mask for closure quantities by uv coordinate
        mask_clamp = uvf.clamp_check
        mask_clphs = uvf.clphs_check
        if mask_clamp:
            mask_amp12 = self.clamp_uvcomb[1].reshape(
                len(self.clamp_uvcomb[1]), -1
            )

            mask_amp34 = self.clamp_uvcomb[2].reshape(
                len(self.clamp_uvcomb[2]), -1
            )

            mask_amp13 = self.clamp_uvcomb[3].reshape(
                len(self.clamp_uvcomb[3]), -1
            )

            mask_amp24 = self.clamp_uvcomb[4].reshape(
                len(self.clamp_uvcomb[4]), -1
            )

            mask_amp12 = _uv_idx(mask_amp12, f_clamp)
            mask_amp34 = _uv_idx(mask_amp34, f_clamp)
            mask_amp13 = _uv_idx(mask_amp13, f_clamp)
            mask_amp24 = _uv_idx(mask_amp24, f_clamp)
        else:
            mask_amp12 = np.array([0], dtype=np.int64)
            mask_amp34 = np.array([0], dtype=np.int64)
            mask_amp13 = np.array([0], dtype=np.int64)
            mask_amp24 = np.array([0], dtype=np.int64)

        if mask_clphs:
            mask_phs12 = self.clphs_uvcomb[1].reshape(
                len(self.clphs_uvcomb[1]), -1
            )

            mask_phs23 = self.clphs_uvcomb[2].reshape(
                len(self.clphs_uvcomb[2]), -1
            )

            mask_phs31 = self.clphs_uvcomb[3].reshape(
                len(self.clphs_uvcomb[3]), -1
            )

            mask_phs12 = _uv_idx(mask_phs12, f_clphs)
            mask_phs23 = _uv_idx(mask_phs23, f_clphs)
            mask_phs31 = _uv_idx(mask_phs31, f_clphs)
        else:
            mask_phs12 = np.array([0], dtype=np.int64)
            mask_phs23 = np.array([0], dtype=np.int64)
            mask_phs31 = np.array([0], dtype=np.int64)

        # set spectrum for jit operations
        spectrum = jit_spectrum(self.spectrum)

        # set model type for jit operations
        modeltype = jit_model(self.model)

        # set fit type & weights for jit operations
        ftype_, fwght_ = jit_ftw(self.fdict, Nant)

        # set the number of free parameters
        self.nmod = nmod
        self.set_ndim(nmod=nmod)
        self.set_index()

        # map parameter index to number
        theta_idx2num = {
            "thick":0, "S":1, "a":2, "l":3, "m":4, "alpha":5, "freq":6,
            "beta":7
        }

        _index_num = self.index.copy()[1:]
        _index_num = [0] + list(map(
            lambda x: 10 * int(
                x.split("_")[0]
            ) + theta_idx2num[x.split("_")[1]],
            _index_num
        ))

        # set arguments
        self.args = (
            self.x, self.y, self.yerr,
            (
                uvf.get_data(dotype="ant1", flatten=True),
                uvf.get_data(dotype="ant2", flatten=True),
                spectrum, modeltype,
                np.array([self.bnd_pa[0]], dtype=np.float32),
                np.array([self.bnd_pa[1]], dtype=np.float32),
                ftype_, fwght_,
                mask_amp12, mask_amp34, mask_amp13, mask_amp24,
                mask_phs12, mask_phs23, mask_phs31,
                np.array(_index_num, dtype=np.int64),
                int(self.fixnmod),
                uvf.bprms
            )
        )

        # print running information
        runtxt = (
            f"\n# Running... (pol {uvf.select_pol}, "
            f"maxn_model={nmod}, "
            f"sampler={self.sampler!r})"
        )

        if self.relmod:
            runtxt += " // ! relative position"

        print(runtxt)

        print(f"# Fit-parameters : {self.fdict}")

        # run dynesty
        self.run_util(
            nmod=nmod, sample=self.sampler, bound=self.bound,
            save_path=self.path_fig,
            save_name="model_params.xlsx",
            save_xlsx=True
        )

        if not "vis" in ftype and not "amp" in ftype:
            self.rsc_amplitude(
                uvfs, rscsbl=self.rscsbl,
                save_path=self.path_fig,
                save_name="model_params.xlsx",
                save_xlsx=True
            )

        # extract statistical values
        logz_v = float(self.results.logz[-1])
        logz_d = float(self.results.logzerr[-1])
        theta = self.theta
        nmod_ = round(float(theta["nmod"]))

        # add model visibility
        for nuvf in range(len(uvfs)):
            uvfs[nuvf].model_visibility_drop()
            uvfs[nuvf].model_visibility_append(
                freq_ref=uvfs[0].freq0, theta=theta,
                model=self.model, spectrum=self.spectrum, closure=True
            )

            if self.dophscal:
                uvfs[nuvf].selfcal(
                    dotype="phs",
                    selfflag=self.selfflag,
                    zero_cp=self.zero_cp
                )
                if self.dogscale:
                    uvfs[nuvf].selfcal(
                        dotype="gscale",
                        selfflag=self.selfflag,
                        zero_cp=self.zero_cp
                    )
                if self.doampcal:
                    uvfs[nuvf].selfcal(
                        dotype="a&p",
                        selfflag=self.selfflag,
                        zero_cp=self.zero_cp
                    )
            else:
                if self.dogscale:
                    uvfs[nuvf].selfcal(
                        dotype="gscale",
                        selfflag=self.selfflag,
                        zero_cp=self.zero_cp
                    )
                if self.doampcal:
                    uvfs[nuvf].selfcal(
                        dotype="amp",
                        selfflag=self.selfflag,
                        zero_cp=self.zero_cp
                    )

            if self.save_uvfits:
                for i in range(len(uvfs)):
                    outpath = f"{self.path_fig_base}/uvfits/"
                    gv.utils.mkdir(outpath)

                    outname = (
                        f"gv."
                        f"{freqtype}."
                        f"{self.model}."
                        f"{self.spectrum}."
                        f"{uvfs[nuvf].freq_mean:.0f}."
                        f"{uvfs[nuvf].source}."
                        f"{uvfs[nuvf].date}."
                        f"uvf"
                    )

                    uvfs[nuvf].save_uvfits(
                        save_path=outpath,
                        save_name=outname
                    )

        # re-set uvf
        uvf = gv.utils.set_uvf(uvfs, dotype=freqtype)

        # print statistical values
        #   - reduced chi-square
        #   - Akaike information criterion
        #   - Bayesian information criterion
        uvcomb = (
            copy.deepcopy(np.ma.getdata(uvf.clamp["clamp"])),
            copy.deepcopy(np.ma.getdata(uvf.clphs["clphs"])),
            copy.deepcopy(np.ma.getdata(uvf.clamp["sig_logclamp"])),
            copy.deepcopy(np.ma.getdata(uvf.clphs["sig_clphs"])),
            copy.deepcopy(clamp_uvcomb),
            copy.deepcopy(clphs_uvcomb)
        )

        # print statistics results
        fty, chi, aic, bic = gv.utils.print_stats(
            uvf=uvf, uvcomb=uvcomb, k=self.ntheta, logz=logz_v, dlogz=logz_d,
            dotype=ftype
        )

        # print model parameters
        self.print_theta(
            ufreq=uvf.ufreq, spectrum=self.spectrum, model=self.model,
            relmod=self.relmod, stats=(fty, chi, aic, bic, logz_v, logz_d),
            prt=True,
            save_path=self.path_fig,
            save_name="model_result.txt"
        )

        uvc = uvf.set_uvcov(flatten=True, returned=True)
        uvf.bprms = gv.utils.fit_beam(uvc, sig=None, uvw=self.uvw)
        uvf.theta = self.theta
        uvf.ploter.bprms = uvf.bprms
        uvf.ploter.theta = self.theta
        uvf.ploter.spectrum = self.spectrum

        # plot and save figures
        # complex antenna gain
        for _uvf in uvfs:
            _uvf.ploter.draw_cgain(
                _uvf, truth=self.cgain_truth, plotimg=False,
                save_csv=True,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.complxgain",
                save_form="pdf"
            )

        # trace plot
        uvf.ploter.draw_trplot(
            result=self.results, nmod=nmod_, relmod=self.relmod,
            model=self.model,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.trplot",
            save_form="pdf"
        )

        # corner plot
        uvf.ploter.draw_cnplot(
            result=self.results, nmod=nmod_, relmod=self.relmod,
            model=self.model,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.cnplot",
            save_form="pdf"
        )

        # radial plot
        uvf.ploter.draw_radplot(
            uvf, plotimg=False, plotmodel=True,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.radplot",
            save_form="pdf"
        )

        # dirty map
        uvf.ploter.draw_dirtymap(
            uvf, plotimg=False, plot_resi=True,
            npix=self.npix, uvw=self.uvw,
            save_path=self.path_fig,
            save_name=f"{self.source}.{self.date}.resimap",
            save_form="pdf"
        )

        # logarithmic closure amplitude
        if "clamp" in ftype:
            uvf.ploter.draw_closure(
                dotype="clamp", plotmodel=True, plotimg=False,
                plotalif=plotalif,
                save_img=True,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.clamp",
                save_form="pdf"
            )

        # closure phase
        if "clphs" in ftype:
            uvf.ploter.draw_closure(
                dotype="clphs", plotmodel=True, plotimg=False,
                plotalif=plotalif,
                save_img=True,
                save_path=self.path_fig,
                save_name=f"{self.source}.{self.date}.clphs",
                save_form="pdf"
            )

        # reconstructed image (single-frequency beam)
        for nuvf in range(len(uvfs)):
            uvfs[nuvf].ploter.bprms = uvfs[nuvf].bprms
            uvfs[nuvf].ploter.theta = self.theta
            uvfs[nuvf].ploter.spectrum = self.spectrum

            outname = (
                f"{self.source}.{self.date}.img."
                f"{freqtype}.{self.bands[nuvf]}"
            )

            uvfs[nuvf].ploter.draw_image(
                uvf=uvfs[nuvf], plotimg=False,
                npix=self.npix, mindr=self.mindr, plot_resi=False,
                addnoise=True,
                freq_ref=uvfs[0].freq0,
                freq=uvfs[nuvf].freq_mean,
                model=self.model,
                save_path=self.path_fig,
                save_name=outname,
                save_form="pdf"
            )

            if self.save_imgfits:
                outpath = f"{self.path_fig_base}/imgfits/"
                gv.utils.mkdir(outpath)

                image = uvfs[nuvf].image
                component = uvfs[nuvf].component

                gv.utils.save_imgfits(
                    uvf=uvfs[nuvf],
                    save_path=outpath,
                    save_name=f"{outname}.fits",
                    image=image, component=component
                )

        # reconstructed image (single-frequency minor beam)
        if freqtype == "sf":
            # set circular beam
            bprms = (uvfs[nuvf].bprms[1], uvfs[nuvf].bprms[1], 0)

            outname = (
                f"{self.source}.{self.date}.img."
                f"{freqtype}.{self.bands[nuvf]}.restore"
            )

            uvfs[nuvf].ploter.draw_image(
                uvf=uvfs[nuvf], plotimg=False, bprms=bprms,
                npix=self.npix, mindr=self.mindr, plot_resi=False,
                addnoise=True,
                freq_ref=uvfs[0].freq0,
                freq=uvfs[nuvf].freq_mean,
                model=self.model,
                save_path=self.path_fig,
                save_name=outname,
                save_form="pdf"
            )

            if self.save_imgfits:
                outpath = f"{self.path_fig_base}/imgfits/"
                gv.utils.mkdir(outpath)

                image = uvfs[nuvf].image
                component = uvfs[nuvf].component

                gv.utils.save_imgfits(
                    uvf=uvfs[nuvf],
                    save_path=outpath,
                    save_name=f"{outname}.fits",
                    image=image, component=component,
                    bmaj=uvfs[nuvf].bprms[1], bmin=uvfs[nuvf].bprms[1],
                    bpa=0
                )

        # reconstructed image (multi-frequency beam)
        if freqtype == "mf":
            for i in range(len(uvfs)):
                outname = (
                    f"{self.source}.{self.date}.img."
                    f"{freqtype}.{self.bands[i]}.restore"
                )

                uvf.ploter.draw_image(
                    uvf=uvf, plotimg=False,
                    npix=self.npix, mindr=self.mindr, plot_resi=True,
                    addnoise=True,
                    freq_ref=self.ufreq[0],
                    freq=self.ufreq[i],
                    model=self.model,
                    save_path=self.path_fig,
                    save_name=outname,
                    save_form="pdf"
                )
                uvfs[i].model_visibility_drop()

                if self.save_imgfits:
                    outpath = f"{self.path_fig_base}/imgfits/"
                    gv.utils.mkdir(outpath)

                    image = uvf.image
                    component = uvf.component

                    gv.utils.save_imgfits(
                        uvf=uvf,
                        save_path=outpath,
                        save_name=f"{outname}.fits",
                        image=image, component=component,
                        freq=uvfs[i].freq_mean
                    )

        uvf.model_visibility_drop()

        self.uvfs = uvfs

        self.initialize_memory()
        del clamp_uvcomb, clphs_uvcomb

    def run_pol(self):
        _uvfs = copy.deepcopy(self.uvfs)
        _nuvf = len(_uvfs)

        if _nuvf == 1:
            freqtype = "sf"
        else:
            freqtype = "mf"

        _uvf = gv.utils.set_uvf(_uvfs, dotype=freqtype)

        self.pol.run_modeling(
            uvfs=_uvfs, uvw=self.uvw, maxn=self.maxn,
            factor_sblf=self.factor_sblf, model=self.model,
            spectrum=self.spectrum, sampler=self.sampler, bound=self.bound,
            mapfov=self.mapfov, npix=self.npix, bnd_a=self.bnd_a,
            bnd_l=self.bnd_l, bnd_m=self.bnd_m, bnd_f=self.bnd_f,
            bnd_pa=self.bnd_pa, freq_ref=self.uvfs[0].freq0, bands=self.bands,
            freqtype=freqtype, bprms=_uvf.bprms, source=self.source,
            date=self.date, mindr=3, save_path=self.path_fig, ncpu=self.ncpu
        )

    def run_util(
        self,
        nmod=1, sample="rwalk", bound="multi", boundset=None, save_path=None,
        save_name=None, save_xlsx=False
    ):
        """
        Run 'dynesty' utilies
        Args:
            sample (str): sampling method in 'dynesty'
                (availables: 'rwalk', 'rslice', 'slice')
            bound (str): bounding condition in 'dynesty'
            boundset (2D-list): list of boundary conditions for priors
            save_path (str): path to save the results
            save_name (str): name of the file to save the results
            save_xlsx (bool): if True, save the results in xlsx format

        """
        self.set_index()
        args = self.args
        ndim = self.ndim

        if self.ndim != len(self.index):
            raise ValueError(
                f"Invalid dimension is given: {self.ndim},"
                f"compared to the number of index ({len(self.index)})."
            )

        if not self.boundset is None:
            boundset = self.boundset
        else:
            raise ValueError(
                f"Invalid boundary conditions for priors: {self.boundset}."
            )

        self.sampler = sample
        self.bound = bound

        # Precompute per-model fields/dims so prior_transform does not
        # depend on 'self' (spawn requires picklable args; self.uvfs
        # carries open FITS file handles which cannot be pickled).
        fields_per_nmod = []
        dims_per_nmod = []
        for i in range(self.nmod):
            self.set_field(nmod=i + 1)
            fields_per_nmod.append(list(self.fields))
            dims_per_nmod.append(int(self.dims))

        pt_static = functools.partial(
            _prior_transform_static,
            fixnmod=bool(self.fixnmod),
            maxn=self.maxn,
            nmod=int(self.nmod),
            spectrum=self.spectrum,
            fields_per_nmod=fields_per_nmod,
            dims_per_nmod=dims_per_nmod,
            boundset=self.boundset,
        )

        # run dynesty
        with Pool(
            self.ncpu, loglike=objective_function,
            prior_transform=pt_static, logl_args=args
        ) as pool:
            sampler = dynesty.DynamicNestedSampler(
                pool.loglike,
                pool.prior_transform,
                ndim,
                sample=sample,
                bound=bound,
                pool=pool
            )

            sampler.run_nested(save_bounds=False)

        # extract dynesty results
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

        theta = gv.utils.structured_array(
                data=self.prms[1].copy(),
                field=fields,
                dtype=dtypes
            )

        self.errors = (self.prms[0] + self.prms[2]) / 2
        self.theta = theta
        self.get_ntheta()

    def set_field(self, nmod=1):
        """
        Set field names and dimensions
        Args:
            nmod (int): The number of models
        """
        # Gaussian
        if self.model == "gaussian":
            # flat spectrum
            if self.spectrum == "flat":
                if self.relmod and nmod == 1:
                    self.dims = 2
                    self.fields = ["S", "a"]
                else:
                    self.dims = 4
                    self.fields = ["S", "a", "l", "m"]

            # simple power-law
            elif self.spectrum == "spl":
                if self.relmod and nmod == 1:
                    self.dims = 3
                    self.fields = ["S", "a", "alpha"]
                else:
                    self.dims = 5
                    self.fields = ["S", "a", "l", "m", "alpha"]

            # curved power-law or synchrotron self-absorption
            elif self.spectrum in ["cpl", "ssa"]:
                if self.relmod:
                    if nmod == 1:
                        self.dims = 4
                        self.fields = ["S", "a", "alpha", "freq"]
                    else:
                        self.dims = 7
                        self.fields = ["S", "a", "l", "m", "alpha", "freq"]

                else:
                    self.fields = ["S", "a", "l", "m", "alpha", "freq"]
                    if nmod == 1:
                        self.dims = 6
                    else:
                        self.dims = 7

            # 2nd-order polynomial
            elif self.spectrum == "poly":
                if self.relmod and nmod == 1:
                    self.dims = 4
                    self.fields = ["S", "a", "alpha", "beta"]
                else:
                    self.dims = 6
                    self.fields = ["S", "a", "l", "m", "alpha", "beta"]

        # delta-function
        elif self.model == "delta":
            # flat spectrum
            if self.spectrum == "flat":
                if self.relmod and nmod == 1:
                    self.dims = 1
                    self.fields = ["S"]
                else:
                    self.dims = 3
                    self.fields = ["S", "l", "m"]

            # simple power-law
            elif self.spectrum == "spl":
                if self.relmod and nmod == 1:
                    self.dims = 2
                    self.fields = ["S", "alpha"]
                else:
                    self.dims = 4
                    self.fields = ["S", "l", "m", "alpha"]

            # curved power-law or synchrotron self-absorption
            elif self.spectrum in ["cpl", "ssa"]:
                if self.relmod:
                    if nmod == 1:
                        self.dims = 3
                        self.fields = ["S", "alpha", "freq"]
                    else:
                        self.dims = 6
                        self.fields = ["S", "l", "m", "alpha", "freq"]

                else:
                    self.fields = ["S", "l", "m", "alpha", "freq"]
                    if nmod == 1:
                        self.dims = 5
                    else:
                        self.dims = 6

            # 2nd-order polynomial
            elif self.spectrum == "poly":
                if self.relmod and nmod == 1:
                    self.dims = 3
                    self.fields = ["S", "alpha", "beta"]
                else:
                    self.dims = 5
                    self.fields = ["S", "l", "m", "alpha", "beta"]

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
            if (
                i != 0 and self.spectrum in ["cpl", "ssa"]
            ):
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

def _prior_transform_static(
    theta, fixnmod, maxn, nmod, spectrum,
    fields_per_nmod, dims_per_nmod, boundset
):
    """
    Module-level prior_transform so it is picklable under spawn.
    Receives only small, picklable state (no 'self', no uvfs/file handles).
    """
    results = []
    ndim = 0
    if fixnmod:
        results.append(1.0 * theta[0] + maxn - 0.5)
    else:
        results.append((maxn - 0.01) * theta[0] + 0.5)

    for i in range(nmod):
        thick_offset = 0
        if spectrum in ("cpl", "ssa") and i != 0:
            results.append(1.98 * theta[1 + ndim] - 0.49)
            thick_offset = 1

        for nfield, field in enumerate(fields_per_nmod[i]):
            lo = boundset[field][i][0]
            hi = boundset[field][i][1]
            results.append(
                (hi - lo) * theta[1 + thick_offset + ndim + nfield] + lo
            )
        ndim += dims_per_nmod[i]

    return results

@jit(nopython=True)
def compute_bic(in_res, in_sig2, in_type, in_nobs, in_ntheta):
    penalty = in_ntheta * np.log(in_nobs)
    nll = compute_nll(in_res, in_sig2, in_type)

    return 2 * nll + penalty

@jit(nopython=True)
def compute_nll(in_res, in_sig2, in_type):
    if in_type == 0:    # for complex visibility
        nll = (
            np.nansum(
                (in_res**2 / in_sig2) * 0.5
                + np.log(2 * np.pi * in_sig2)
            )
        )

    else:
        nll = (
            np.nansum(
                (in_res**2 / in_sig2)
                + np.log(2 * np.pi * in_sig2)
            )
        ) * 0.5

    return nll

@jit(nopython=True)
def compute_obj(in_res, in_sig2, in_type, in_nobs, in_ntheta, in_fixnmod):
    if in_fixnmod == 0:
        return compute_bic(in_res, in_sig2, in_type, in_nobs, in_ntheta)
    else:
        return compute_nll(in_res, in_sig2, in_type)

@jit(nopython=True)
def cpl(nu, smax, tf, alpha):
    """
    Args:
        nu (array or float): input frequency
        smax (float): flux density at 'tf'
        alpha (float): optically thin spectral index
    Returns:
        Curved power-law flux density at nu
    """
    out = smax * (nu / tf)**(alpha * np.log10(nu / tf))

    return out

@jit(nopython=True)
def dvis(args, S, l, m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        S (float): flux density of delta function model
        l (float): right ascension position of delta function model
        m (float): declination position of delta function model
    Returns:
        complex visibility of Gaussian model
    """
    out = S * np.exp(2j * np.pi * (args[0] * l + args[1] * m) / r2m)

    return out

@jit(nopython=True)
def dvis_cpl(args, smax, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (array or float): input frequency
            args[1] (1D-array): u-axis data points
            args[2] (1D-array): v-axis data points
        smax (float): flux density of Gaussian model at 'nu_m'
        l (float): right ascension position of Gaussian model
        m (float): declination position of Gaussian model
        alpha (float): optically thin spectral index of Gaussian model
    Returns:
        complex visibility of delta-function model (cpl-based)
    """
    out = cpl(args[0], smax, nu_m, alpha) * np.exp(
        2j * np.pi * (args[1] * l + args[2] * m) / r2m
    )

    return out

@jit(nopython=True)
def dvis_poly(args, s_ref, l, m, alpha, beta):
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    S = poly(nu_ref, nu, s_ref, alpha, beta)
    out = S * np.exp(2j * np.pi * (uu * l + vv * m))

    return out.astype("c8")

@jit(nopython=True)
def dvis_spl(args, smax, l, m, alpha):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (float): reference frequency (recommended to set at the
                lowest one)
            args[1] (array or float): input frequency
            args[2] (1D-array): u-axis data points
            args[3] (1D-array): v-axis data points
        smax (float): flux density of Gaussian model at 'args[0]'
        l (float): right ascension position of Gaussian model
        m (float): declination position of Gaussian model
        alpha (float): optically thin spectral index of Gaussian model
    Returns:
        complex visibility of delta-function model (spl-based)
    """
    out = spl(args[0], args[1], smax, alpha) * np.exp(
        2j * np.pi * (args[2] * l + args[3] * m) / r2m
    )

    return out

@jit(nopython=True)
def dvis_ssa(args, smax, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (array or float): input frequency
            args[1] (1D-array): u-axis data points
            args[2] (1D-array): v-axis data points
        smax (float): flux density of Gaussian model at 'nu_m'
        l (float): right ascension position of Gaussian model
        m (float): declination position of Gaussian model
        alpha (float): optically thin spectral index of Gaussian model
        nu_m (float): turnover frequency
    Returns:
        complex visibility of delta-function model (ssa-based)
    """
    out = ssa(args[0], smax, nu_m, alpha) * np.exp(
        2j * np.pi * (args[1] * l + args[2] * m) / r2m
    )

    return out

@jit(nopython=True)
def gaussian_1d(x, peak, a, mx):
    """
    Args:
        x (array): input x-axis data points
        peak (float): peak-value of the Gaussian
        a (float): standard deviation of the Gaussian
        mx (float): offset of the peak in x-axis from the zero-position
    Returns:
        A 1-D Guassian function
    """
    out = peak * np.exp(-((x - mx) / a)**2 / 2)

    return out

@jit(nopython=True)
def gaussian_2d(xy, peak, ax, ay, mx, my, theta):
    """
    Args:
        xy (2D-array, tuple): input x/y-axis data points
        peak (float): peak-value of the Gaussian
        ax/y (float): standard deviation of the Gaussian in x/y-axis
        mx/y (float): offset of the peak in x/y-axis from the zero-position
        theta (float): position angle of the Gaussian (elliptical Guassian)
    Returns:
        A 2-D Guassian function
    """
    out = (
        peak * np.exp(
            -(xy[0] - mx)**2 / (2 * ax**2)
            -(xy[1] - my)**2 / (2 * ay**2)
        )
    )

    return out

@jit(nopython=True)
def gvis(args, S, fwhm, l, m):
    """
    Args:
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
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    out = S * np.exp(
        -2 * np.pi**2 * sigma**2 * (args[0]**2 + args[1]**2) / r2m**2
        + 2j * np.pi * (args[0] * l + args[1] * m) / r2m
    )

    return out

@jit(nopython=True)
def gvis_cpl(args, smax, fwhm, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (array or float): input frequency
            args[1] (1D-array): u-axis data points
            args[2] (1D-array): v-axis data points
        smax (float): flux density of Gaussian model at 'nu_m'
        fwhm (float): full-width at half maximum of Gaussian model
        l (float): right ascension position of Gaussian model
        m (float): declination position of Gaussian model
        alpha (float): optically thin spectral index of Gaussian model
    Returns:
        complex visibility of Gaussian model (cpl-based)
    """
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    out = cpl(args[0], smax, nu_m, alpha) * np.exp(
        -2 * (np.pi * sigma)**2 * (args[1]**2 + args[2]**2) / r2m**2
        + 2j * np.pi * (args[1] * l + args[2] * m) / r2m
    )

    return out

@jit(nopython=True)
def gvis_poly(args, s_ref, fwhm, l, m, alpha, beta):
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    S = poly(nu_ref, nu, s_ref, alpha, beta)
    out = S * np.exp(
        -2 * (np.pi * sigma)**2 * (uu**2 + vv**2)
        + 2j * np.pi * (uu * l + vv * m)
    )

    return out.astype("c8")

@jit(nopython=True)
def gvis_spl(args, smax, fwhm, l, m, alpha):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (float): reference frequency (recommended to set at the
                lowest one)
            args[1] (array or float): input frequency
            args[2] (1D-array): u-axis data points
            args[3] (1D-array): v-axis data points
        smax (float): flux density of Gaussian model at 'args[0]'
        fwhm (float): full-width at half maximum of Gaussian model
        l (float): right ascension position of Gaussian model
        m (float): declination position of Gaussian model
        alpha (float): optically thin spectral index of Gaussian model
    Returns:
        complex visibility of Gaussian model (spl-based)
    """
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    out = spl(args[0], args[1], smax, alpha) * np.exp(
        -2 * (np.pi * sigma)**2 * (args[2]**2 + args[3]**2) / r2m**2
        + 2j * np.pi * (args[2] * l + args[3] * m) / r2m
    )

    return out

@jit(nopython=True)
def gvis_ssa(args, smax, fwhm, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (array or float): input frequency
            args[1] (1D-array): u-axis data points
            args[2] (1D-array): v-axis data points
        smax (float): flux density of Gaussian model at 'nu_m'
        fwhm (float): full-width at half maximum of Gaussian model
        l (float): right ascension position of Gaussian model
        m (float): declination position of Gaussian model
        alpha (float): optically thin spectral index of Gaussian model
        nu_m (float): turnover frequency
    Returns:
        complex visibility of Gaussian model (ssa-based)
    """
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    out = ssa(args[0], smax, nu_m, alpha) * np.exp(
        -2 * (np.pi * sigma)**2 * (args[1]**2 + args[2]**2) / r2m**2
        + 2j * np.pi * (args[1] * l + args[2] * m) / r2m
    )

    return out

def jit_ftw(fdict, nant):
    ftype_ = []
    fwght_ = [0, 0, 0, 0, 0]

    _fmap = [
        ("vis",   0, True),
        ("amp",   1, True),
        ("phs",   2, True),
        ("clamp", 3, nant >= 4),
        ("clphs", 4, True),
    ]
    for key, idx, cond in _fmap:
        if key in fdict and cond:
            ftype_.append(idx)
            fwght_[idx] = fdict[key]

    ftype_ = np.array(ftype_, dtype=np.int64)
    fwght_ = np.array(fwght_, dtype=np.float32)

    return ftype_, fwght_

def jit_model(model):
    if model == "delta":
        modeltype = 0
    elif model == "gaussian":
        modeltype = 1
    else:
        modeltype = 1
        print(
            f"Unexpected model type: {self.model!r}. Assume 'gaussian'."
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
    elif spectrum == "poly":
        spectrum = 4
    else:
        availables = ["flat", "spl", "cpl", "ssa", "poly"]
        raise ValueError(
            f"Invalid spectrum type: {spectrum!r}.\n"
            f"Available spectrum types are: {availables}."
        )

    return spectrum

@jit(nopython=True)
def linear(x, m, a):
    """
    Args:
        x (array): input x-axis data points
        m (float): slope of the linear function
        a (float): offset of the linear function (constant)
    Returns:
        A linear function
    """
    out = m * x + a

    return out

@jit(nopython=True)
def poly(nu_ref, nu, s_ref, alpha, beta):
    x = np.log(nu / nu_ref)
    out = s_ref * np.exp(alpha * x + beta * x**2)

    return out

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
            args[0] (array, str): antenna name 1
            args[1] (array, str): antenna name 2
            args[2] (int): spectrum ('flat', 'spl', 'cpl', 'ssa')
            args[3] (int): modeltype (0:'delta', 1: 'gaussian')
            args[4] (float): lower boundary for jet position angle
            args[5] (float): upper boundary for jet position angle
            args[6] (list, str): 'fdict' keys
            args[7] (list, str): 'fdict' values
            args[8 - 11] (array): mask for closure amplitudes
            args[12 - 14] (array): mask for closure phases
            args[15] (list, int): parameter index
            args[16] (int): boolean of 'fixnmod'
            args[17] (tuple): beam parameters
    Returns:
        Bayesian Information Criterion value (float)
    """

    dshape = x[1].shape

    model = np.zeros(dshape, dtype=np.complex64)
    ufreq = np.unique(x[1])

    ntheta = 0

    pa1 = np.exp(1j * np.deg2rad(args[4][0]))
    pa2 = np.exp(1j * np.deg2rad(args[5][0]))
    span = np.angle(pa2 / pa1) % (2 * np.pi)

    nmod = round(float(theta[0]))
    mask_pa = 0

    # compute model visibility
    # model == "gaussian"
    if args[3] == 1:
        for i in range(nmod):
            idxval_s = (i + 1) * 10 + 1
            idxval_a = (i + 1) * 10 + 2
            idxval_l = (i + 1) * 10 + 3
            idxval_m = (i + 1) * 10 + 4

            mask_s = args[15] == idxval_s
            mask_a = args[15] == idxval_a
            mask_l = args[15] == idxval_l
            mask_m = args[15] == idxval_m

            _s = theta[mask_s][0]
            _a = theta[mask_a][0]

            if (mask_l.sum() == 0 and mask_m.sum() == 0):
                _l = 0.0
                _m = 0.0
                has_lm = False
            else:
                _l = theta[mask_l][0]
                _m = theta[mask_m][0]
                _r = np.sqrt(_l**2 + _m**2)
                has_lm = True

                pa = np.exp(1j * np.angle(_m + 1j * _l))
                offset = np.angle(pa / pa1) % (2 * np.pi)
                if offset > span and _r > args[17][1]:
                    mask_pa = 1

            # Gaussian & flat spectrum
            if args[2] == 0:
                _args = (x[2], x[3])

                model += gvis(_args, _s, _a, _l, _m)
                ntheta += 4 if has_lm else 2

            # Gaussian & (spl | cpl | ssa)
            elif args[2] in [1, 2, 3]:
                idxval_t = (i + 1) * 10 + 0
                idxval_i = (i + 1) * 10 + 5
                idxval_f = (i + 1) * 10 + 6

                mask_t = args[15] == idxval_t
                mask_i = args[15] == idxval_i
                mask_f = args[15] == idxval_f

                has_t = mask_t.sum()

                _i = theta[mask_i][0]

                # Gaussian & simple power-law
                if args[2] == 1:
                    _args = (x[0], x[1], x[2], x[3])
                    model += gvis_spl(_args, _s, _a, _l, _m, _i)
                    ntheta += 5 if has_lm else 3

                # Gaussian & (curved power-law | synchrotron self-absorption)
                else:
                    _f = theta[mask_f][0]
                    if i == 0:
                        _args = (x[1], x[2], x[3])
                        if args[2] == 2:
                            model += gvis_cpl(
                                _args, _s, _a, _l, _m, _i, _f
                            )

                        elif args[2] == 3:
                            model += gvis_ssa(
                                _args, _s, _a, _l, _m, _i, _f
                            )
                        ntheta += 6 if has_lm else 4
                    else:
                        mask_t = round(float(theta[mask_t][0])) == 0

                        if mask_t:
                            _args = (x[0], x[1], x[2], x[3])
                            model += gvis_spl(_args, _s, _a, _l, _m, _i)
                            ntheta += 5 if has_lm else 3
                        else:
                            _args = (x[1], x[2], x[3])
                            if args[2] == 2:
                                model += gvis_cpl(
                                    _args, _s, _a, _l, _m, _i, _f
                                )

                            elif args[2] == 3:
                                model += gvis_ssa(
                                    _args, _s, _a, _l, _m, _i, _f
                                )

                            ntheta += 6 if has_lm else 4

            # Gaussian & 2nd-order polynomial
            elif args[2] == 4:
                idxval_i = (i + 1) * 10 + 5
                idxval_b = (i + 1) * 10 + 7

                mask_i = args[15] == idxval_i
                mask_b = args[15] == idxval_b

                _i = theta[mask_i][0]
                _b = theta[mask_b][0]

                _args = (x[0], x[1], x[2], x[3])
                model += gvis_poly(_args, _s, _a, _l, _m, _i, _b)
                ntheta += 6 if has_lm else 4

    # model == "delta"
    elif args[3] == 0:
        for i in range(nmod):
            idxval_s = (i + 1) * 10 + 1
            idxval_l = (i + 1) * 10 + 3
            idxval_m = (i + 1) * 10 + 4

            mask_s = args[15] == idxval_s
            mask_l = args[15] == idxval_l
            mask_m = args[15] == idxval_m

            _s = theta[mask_s][0]

            if (mask_l.sum() == 0 and mask_m.sum() == 0):
                _l = 0.0
                _m = 0.0
                has_lm = False
            else:
                _l = theta[mask_l][0]
                _m = theta[mask_m][0]
                has_lm = True

                pa = np.exp(1j * np.angle(_m + 1j * _l))
                offset = np.angle(pa / pa1) % (2 * np.pi)
                if offset > span:
                    mask_pa = 1

            # delta & flat spectrum
            if args[2] == 0:
                _args = (x[2], x[3])

                if has_lm:
                    ntheta += 3
                else:
                    ntheta += 1

                model += dvis(_args, _s, _l, _m)

            # delta & (spl | cpl | ssa)
            elif args[2] in [1, 2, 3]:
                idxval_t = (i + 1) * 10 + 0
                idxval_i = (i + 1) * 10 + 5
                idxval_f = (i + 1) * 10 + 6

                mask_t = args[15] == idxval_t
                mask_i = args[15] == idxval_i
                mask_f = args[15] == idxval_f

                has_t = mask_t.sum()

                _i = theta[mask_i][0]

                # delta & simple power-law
                if args[2] == 1:
                    _args = (x[0], x[1], x[2], x[3])
                    model += dvis_spl(_args, _s, _l, _m, _i)
                    ntheta += 4 if has_lm else 2

                # delta & (curved power-law | synchrotron self-absorption)
                else:
                    _f = theta[mask_f][0]
                    if i == 0:
                        _args = (x[1], x[2], x[3])
                        if args[2] == 2:
                            model += dvis_cpl(
                                _args, _s, _l, _m, _i, _f
                            )

                        elif args[2] == 3:
                            model += dvis_ssa(
                                _args, _s, _l, _m, _i, _f
                            )
                    else:
                        mask_t = round(float(theta[mask_t][0])) == 0

                        if mask_t:
                            _args = (x[0], x[1], x[2], x[3])
                            model += dvis_spl(_args, _s, _l, _m, _i)
                            ntheta += 4 if has_lm else 2
                        else:
                            _args = (x[1], x[2], x[3])
                            if args[2] == 2:
                                model += dvis_cpl(
                                    _args, _s, _l, _m, _i, _f
                                )

                            elif args[2] == 3:
                                model += dvis_ssa(
                                    _args, _s, _l, _m, _i, _f
                                )

                            ntheta += 5 if has_lm else 3

            # delta & 2nd-order polynomial
            elif args[2] == 4:
                idxval_i = (i + 1) * 10 + 5
                idxval_b = (i + 1) * 10 + 7

                mask_i = args[15] == idxval_i
                mask_b = args[15] == idxval_b

                _i = theta[mask_i][0]
                _b = theta[mask_b][0]

                _args = (x[0], x[1], x[2], x[3])
                model += dvis_poly(_args, _s, _l, _m, _i, _b)
                ntheta += 5 if has_lm else 3

    nansum_model = np.nansum(np.abs(model))

    # compute objective functions
    if not np.isnan(nansum_model) and not nansum_model == 0:
        objective = 0

        # complex visibility
        if 0 in args[6]:
            vis_obs = y[0]
            vis_mod = model
            vis_sig2 = yerr[0]**2
            vis_res = np.abs(vis_mod - vis_obs)
            nobs = y[0].size
            objective -= (
                args[7][0]
                * compute_bic(vis_res, vis_sig2, 0, nobs, ntheta)
            )

        # visibility amplitude
        if 1 in args[6]:
            amp_obs = np.abs(y[0])
            amp_mod = np.abs(model)
            amp_sig2 = yerr[0]**2
            amp_res = amp_mod - amp_obs
            nobs = y[0].size
            objective -= (
                args[7][1]
                * compute_bic(amp_res, amp_sig2, 1, nobs, ntheta)
            )

        # visibility phase
        if 2 in args[6]:
            phs_sig2 = (yerr[0] / np.abs(y[0]))**2
            phs_res = np.angle(model / y[0])
            nobs = y[0].size
            objective -= (
                args[7][2]
                * compute_bic(phs_res, phs_sig2, 2, nobs, ntheta)
            )

        # closure amplitude
        if 3 in args[6]:
            amp12 = np.abs(model[args[8]])
            amp34 = np.abs(model[args[9]])
            amp13 = np.abs(model[args[10]])
            amp24 = np.abs(model[args[11]])

            clamp_obs = y[1]
            clamp_mod = ((amp12 * amp34) / (amp13 * amp24))
            clamp_sig2 = yerr[1]**2
            clamp_res = np.abs(np.log(clamp_mod) - np.log(clamp_obs))
            nobs = y[1].size
            objective -= (
                args[7][3]
                * compute_bic(clamp_res, clamp_sig2, 3, nobs, ntheta)
            )

        # closure phase
        if 4 in args[6]:
            phs12 = np.angle(model[args[12]])
            phs23 = np.angle(model[args[13]])
            phs31 = np.angle(model[args[14]].conjugate())

            clphs_obs = y[2]
            clphs_mod = (phs12 + phs23 + phs31)
            clphs_sig2 = yerr[2]**2
            clphs_res = np.angle(np.exp(1j * (clphs_mod - clphs_obs)))
            nobs = y[2].size
            objective -= (
                args[7][4]
                * compute_bic(clphs_res, clphs_sig2, 4, nobs, ntheta)
            )

        if mask_pa == 1:
            objective = -np.inf
    else:
        objective = -np.inf

    return objective

@jit(nopython=True)
def spl(nu_ref, nu, smax, alpha):
    """
    Args:
        nu_ref (float): reference frequency
                        (recommended to set at the lowest one)
        nu (array or float): input frequency
        smax (float): flux density at 'nu_ref'
        alpha (float): optically thin spectral index
    Returns:
        Simple power-law flux density at nu
    """
    out = 10**(alpha * (np.log10(nu) - np.log10(nu_ref)) + np.log10(smax))

    return out

@jit(nopython=True)
def ssa(nu, smax, tf, alpha):
    """
    NOTE: This function assumes optically thick spectral index as 2.5
    (Turler+1999, A&A, 349, 45T)
    Args:
        nu (array or float): input frequency
        smax (float): flux density at 'tf'
        tf (float): turnover frequency of the SSA spectrum
        alpha (float): optically thin spectral index
    Returns:
        SSA flux density at nu
    """
    c = 1.5 * ((1 - (8 * alpha) / 7.5)**0.5 - 1)
    x = nu / tf
    out = smax * x**2.5 * (1 - np.exp(-c * x**(alpha - 2.5))) / (1 - np.exp(-c))

    return out

@jit(nopython=True)
def set_closure(
    data_vis, data_ant1, data_ant2,
    mask_amp12, mask_amp34, mask_amp13, mask_amp24,
    mask_phs12, mask_phs23, mask_phs31
):
    """
    Set the closure quantities for the input dataset
    Args:
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
        raise ValueError(
            "Invalid number of antennas to compute closure quantities."
        )
