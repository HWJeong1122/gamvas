
import os
import sys
import gc
import copy
import warnings
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import matplotlib.patches as patches
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.modeling.models import Gaussian2D
from astropy.time import Time as atime
from astropy.time import TimeDelta
from astropy import units as au
from dynesty import plotting as dyplot
from dynesty.utils import quantile as dyquan

import gamvas as gv

r2m = au.rad.to(au.mas)
d2m = au.deg.to(au.mas)
d2r = au.deg.to(au.rad)
m2d = au.mas.to(au.deg)
m2r = au.mas.to(au.rad)

class plotter:
    def __init__(self,
        mapfov=False, npix=1024, nmod=False, theta=None, freq_ref=False,
        freq=False, bmin=False, bmaj=False, bpa=False, bprms=None,
        source=False, date=False
    ):

        self.mapfov = mapfov
        self.npix = npix
        self.nmod = nmod
        self.theta = theta
        self.freq = freq
        self.freq_ref = freq_ref

        self.psize = self.mapfov / self.npix

        self.bprms = bprms

        axis_range = np.linspace(-mapfov / 2, +mapfov / 2, npix)
        xgrid, ygrid = np.meshgrid(-axis_range, +axis_range)
        self.xgrid = xgrid
        self.ygrid = ygrid

        self.source = source
        self.date = date

        self.fitset = None
        self.spectrum = None

    def convolve_image(self, uvf, npix, image=False, bprms=None):
        """
        Convolve the generated intensity image with restoring beam (Jy/beam)
        Args:
            uvf (python class): opened-fits file in uvf-class
            npix (int): the number of pixels in the map
            image (2D-array): generated image
            bprms (tuple): beam parameters (minor, major, position angle)
        Returns:
            conv_image (2D-array): convolved image (n by n)
        """
        if bprms is None:
            bprms = self.bprms

        bmin, bmaj, bpa = bprms
        bsize = np.pi * bmin * bmaj / 4 / np.log(2)

        mapfov, npix, psize = self.set_imgprms(
            uvf=uvf, npix=npix, mapfov=uvf.mapfov.value
        )

        kmin = bmin / np.sqrt(8 * np.log(2)) / psize
        kmaj = bmaj / np.sqrt(8 * np.log(2)) / psize

        gauss_kernel = Gaussian2DKernel(
            x_stddev=kmaj, y_stddev=kmin,
            theta=bpa * au.deg
        )

        conv_image = convolve_fft(image, gauss_kernel, normalize_kernel=True)
        conv_image = bsize * conv_image
        return conv_image

    def draw_cgain(
        self,
        uvf, truth=None, plotimg=True,
        save_csv=False, save_path=False, save_name=False, save_form="png"
    ):
        clrs = [
            "tab:blue", "tab:orange", "tab:green", "tab:purple",
            "tab:brown", "tab:pink", "tab:gray" , "tab:olive"
        ]

        r2d = au.rad.to(au.deg)

        cg_pol1_ant1 = uvf.cg_pol1_ant1
        cg_pol1_ant2 = uvf.cg_pol1_ant2
        cg_pol2_ant1 = uvf.cg_pol2_ant1
        cg_pol2_ant2 = uvf.cg_pol2_ant2

        mask_cgain = (
            (cg_pol1_ant1 is None)
            | (cg_pol1_ant2 is None)
            | (cg_pol2_ant1 is None)
            | (cg_pol2_ant2 is None)
        )

        if mask_cgain:
            _dshape = uvf.time.shape
            cg_pol1_ant1 = np.ones(_dshape, dtype="c8")
            cg_pol1_ant2 = np.ones(_dshape, dtype="c8")
            cg_pol2_ant1 = np.ones(_dshape, dtype="c8")
            cg_pol2_ant2 = np.ones(_dshape, dtype="c8")
            # raise ValueError(f"Invalid complex gain is given.")

        time = uvf.get_data(dotype="time").astype("f4")
        freq = uvf.get_data(dotype="frequency").astype("f4")
        ant1 = uvf.get_data(dotype="ant1_name")
        ant2 = uvf.get_data(dotype="ant2_name")

        ufreq = np.sort(np.unique(freq))
        uant = np.unique(np.append(ant1, ant2))

        time_min = int(time.min())
        time_max = int(time.max() + 1)

        nfig = len(uant) // 10
        nax = len(uant) % 10

        if nax != 0:
            nfig += 1

        if uvf.nstokes >= 2:
            npol = 2
            plot_pol2 = True
        else:
            npol = 1
            plot_pol2 = False

        figs = [None] * nfig

        for _nfig in range(nfig):
            pl_nax = nax if _nfig == nfig - 1 and nax != 0 else 10

            figs[_nfig] = plt.figure(figsize=(16, 9))
            figs[_nfig].supxlabel("Time (hour)", fontsize=12)

            axes = [[None] * (npol * 2) for _ in range(pl_nax)]

            gs_main = gridspec.GridSpec(
                pl_nax, 2, figure=figs[_nfig],
                hspace=0.20, wspace=0.15,
                top=0.96, bottom=0.10, left=0.07, right=0.98
            )

            for _nax in range(pl_nax):
                idx_ant = 10 * _nfig + _nax

                for _col in range(2):
                    if npol == 2:
                        gs_sub = gridspec.GridSpecFromSubplotSpec(
                            1, npol, subplot_spec=gs_main[_nax, _col],
                            hspace=0.2, wspace=0.15,
                        )

                        specs = [gs_sub[j] for j in range(npol)]

                        if _col == 0:
                            _title = [f"Amp (RCP)", f"Phs (RCP, deg)"]
                        else:
                            _title = [f"Amp (LCP)", f"Phs (LCP, deg)"]
                    else:
                        specs = [gs_main[_nax, _col]]

                        select_pol = uvf.select_pol.split(".")[-1].upper()
                        _title = [
                            f"Amp ({select_pol[0]}CP)",
                            f"Phs ({select_pol[0]}CP, deg)",
                        ]

                    for _i, spec in enumerate(specs):
                        idx = _col * npol + _i

                        sharex = (
                            None if (_nax == 0 and idx == 0)
                            else axes[0][0]
                        )

                        ax = figs[_nfig].add_subplot(spec, sharex=sharex)
                        ax.set_rasterized(True)

                        if _nax != pl_nax - 1:
                            ax.tick_params(labelbottom=False)

                        if _nax == 0:
                            if npol == 1:
                                ax.set_title(_title[_col])
                            else:
                                ax.set_title(_title[_i])

                        if (_col == 0 and _i == 0):
                            ax.set_ylabel(
                                f"{uant[idx_ant]}",
                                fontsize=12,
                                rotation="horizontal",
                                labelpad=15
                            )

                        ax.set_xlim(time_min, time_max)
                        ax.xaxis.set_major_locator(MultipleLocator(3.0))
                        ax.xaxis.set_minor_locator(MultipleLocator(1.0))

                        axes[_nax][idx] = ax

                for nf, _freq in enumerate(ufreq):
                    mask_freq = (freq == _freq)

                    mask_ant1 = (ant1 == uant[idx_ant])
                    mask_ant2 = (ant2 == uant[idx_ant])

                    _cg_time_ant1 = time[mask_ant1 & mask_freq]
                    _cg_time_ant2 = time[mask_ant2 & mask_freq]

                    _cg_pol1_ant1 = cg_pol1_ant1[mask_ant1 & mask_freq]
                    _cg_pol1_ant2 = cg_pol1_ant2[mask_ant2 & mask_freq]
                    _cg_pol2_ant1 = cg_pol2_ant1[mask_ant1 & mask_freq]
                    _cg_pol2_ant2 = cg_pol2_ant2[mask_ant2 & mask_freq]

                    _cg_time = [_cg_time_ant1, _cg_time_ant2]
                    _cg_pol1 = [_cg_pol1_ant1, _cg_pol1_ant2]
                    _cg_pol2 = [_cg_pol2_ant1, _cg_pol2_ant2]
                    _cg = [_cg_pol1, _cg_pol2]

                    for _npol in range(npol):
                        idx_ax1 = 2 * _npol + 0
                        idx_ax2 = 2 * _npol + 1
                        axes[_nax][idx_ax1].scatter(
                            _cg_time[0], np.abs(_cg[_npol][0]),
                            marker="o", s=15, fc=clrs[nf], ec="black"
                        )

                        axes[_nax][idx_ax1].scatter(
                            _cg_time[1], np.abs(_cg[_npol][1]),
                            marker="o", s=15, fc=clrs[nf], ec="black"
                        )

                        axes[_nax][idx_ax2].scatter(
                            _cg_time[0], np.angle(_cg[_npol][0]) * r2d,
                            marker="o", s=15, fc=clrs[nf], ec="black"
                        )

                        axes[_nax][idx_ax2].scatter(
                            _cg_time[1], np.angle(_cg[_npol][1]) * r2d,
                            marker="o", s=15, fc=clrs[nf], ec="black"
                        )

        if save_name:
            save_path_ = save_path + "plot_cgain/"
            gv.utils.mkdir(save_path_)

            for _nfig in range(nfig):
                save_name_ = save_name + f".{uvf.freq_mean:.0f}.v{_nfig + 1}"
                figs[_nfig].savefig(
                    f"{save_path_}" + f"{save_name_}.{save_form}",
                    format=save_form, dpi=200
                )

        if plotimg:
            plt.show()

        for _nfig in range(nfig):
            close_figure(figs[_nfig])

        flat_time = time.flatten()
        flat_freq = freq.flatten()
        flat_ant1 = ant1.flatten()
        flat_ant2 = ant2.flatten()
        flat_cg_pol1_ant1 = cg_pol1_ant1.flatten()
        flat_cg_pol1_ant2 = cg_pol1_ant2.flatten()
        flat_cg_pol2_ant1 = cg_pol2_ant1.flatten()
        flat_cg_pol2_ant2 = cg_pol2_ant2.flatten()

        if save_csv:
            save_path_ = save_path + "plot_cgain/"
            save_name_ = save_name + f".{uvf.freq_mean:.0f}.csv"

            gv.utils.mkdir(save_path_)

            gv.utils.save_cgain(
                uvf=uvf,
                save_path=save_path_,
                save_name=save_name_
            )

    def draw_closure(
        self,
        dotype="clphs", plotimg=False, plotmodel=False, plotalif=True,
        save_img=False, save_path=None, save_name=None, save_form="png"
    ):
        """
        draw closure quantities
        Args:
            dotype (str) : type of closure quantity ('amp', 'phs')
            plotimg (bool): toggle option if plot the result
        """
        availables = ["amp", "phs", "logclamp", "clamp", "clphs"]
        if dotype not in availables:
            raise ValueError(
                f"Invalid type is given: {dotype!r}.\n"
                f"Availables: {availables}"
            )

        if save_path is not None:
            save_path_ = save_path + f"/plot_{dotype}/"
            os.system(f"rm -rf {save_path_}")
            gv.utils.mkdir(save_path_)

        if dotype in ["amp", "clamp"]:
            obs = copy.deepcopy(self.clq_obs[0])

            idx_x = "time"
            idx_y = "clamp"
            idx_yerr = "sig_logclamp"
            idx_bsli = "quadra"

            label_y = r"$\ln(A_{\rm C})$"

            warn_message = "amplitude"

            if plotmodel:
                mod = self.clq_mod[0]

        elif dotype in ["phs", "clphs"]:
            obs = copy.deepcopy(self.clq_obs[1])

            idx_x = "time"
            idx_y = "clphs"
            idx_yerr = "sig_clphs"
            idx_bsli = "triangle"

            label_y = r"$\phi_{\rm C}~({\rm deg})$"

            warn_message = "phase"

            if plotmodel:
                mod = self.clq_mod[1]

        ufreq = np.unique(obs["freq"].astype("f4"))
        nfreq = len(ufreq)
        mfc = [
            "tab:blue", "tab:orange", "tab:green", "tab:purple",
            "tab:brown", "tab:pink", "tab:gray" , "tab:olive"
        ]

        if plotalif:
            ubsli = np.unique(obs[idx_bsli])
            nbsli = len(ubsli)

            nfig = nbsli // 10
            if nbsli % 10 != 0:
                nfig += 1

            fsize = 9
            figs = [None for i in range(nfig)]

            for _nfig in range(nfig):
                if _nfig != nfig - 1:
                    nax = 10
                else:
                    nax = nbsli % 10 if nbsli % 10 != 0 else 10

                margin_top = 0.3
                margin_bot = 0.5
                fig_h = fsize * nax / 10 + margin_top + margin_bot

                figs[_nfig] = plt.figure(
                    figsize=(fsize, fig_h)
                )
                figs[_nfig].supylabel(label_y, fontsize=12)
                axes = [None for i in range(nax)]

                gs_main = gridspec.GridSpec(
                    nax, 1, figure=figs[_nfig],
                    hspace=0.15, wspace=0.10,
                    top=1 - margin_top / fig_h,
                    bottom=margin_bot / fig_h,
                    left=0.10, right=0.98
                )

                for _nax in range(nax):
                    if _nax == 0:
                        axes[_nax] = figs[_nfig].add_subplot(
                            gs_main[_nax]
                        )
                    else:
                        axes[_nax] = figs[_nfig].add_subplot(
                            gs_main[_nax], sharex=axes[0]
                        )
                    axes[_nax].set_rasterized(True)

                    axes[_nax].xaxis.set_major_locator(MultipleLocator(2.0))
                    axes[_nax].xaxis.set_minor_locator(MultipleLocator(1.0))
                    if _nax != nax - 1:
                        axes[_nax].tick_params(labelbottom=False)
                    else:
                        axes[_nax].set_xlabel("Time (hour)", fontsize=12)

                    bsli_name = ubsli[10 * _nfig + _nax]

                    for nf, freq in enumerate(ufreq):
                        mask = (
                            (obs["freq"].astype("f4") == freq)
                            & (obs[idx_bsli] == bsli_name)
                        )
                        obs_sel = obs[mask]

                        if len(obs_sel) == 0:
                            continue

                        ax_label = f"{bsli_name} ({freq:.3f} GHz)"

                        if dotype in ["amp", "clamp"]:
                            x = obs_sel[idx_x]
                            yobs = np.log(obs_sel[idx_y])
                            yerr = obs_sel[idx_yerr]
                        elif dotype in ["phs", "clphs"]:
                            r2d = au.rad.to(au.deg)
                            x = obs_sel[idx_x]
                            yobs = (
                                r2d
                                * np.angle(np.exp(1j * obs_sel[idx_y]))
                            )
                            yerr = r2d * obs_sel[idx_yerr]

                        axes[_nax].errorbar(
                            x, yobs, yerr,
                            marker="o", markersize=6,
                            c=mfc[nf], ls="",
                            mfc=mfc[nf], mec="k",
                            capsize=6, capthick=1, zorder=1,
                            label=ax_label, alpha=0.5
                        )

                        if plotmodel:
                            mod_sel = mod[mask]
                            if dotype in ["amp", "clamp"]:
                                ymod = np.log(mod_sel).flatten()
                            elif dotype in ["phs", "clphs"]:
                                ymod = (
                                    r2d
                                    * np.angle(np.exp(1j * mod_sel))
                                ).flatten()

                            ymod = np.where(np.isfinite(yobs), ymod, np.nan)

                            order = np.argsort(np.asarray(x))
                            axes[_nax].plot(
                                np.asarray(x)[order], ymod[order],
                                marker="x", markersize=8,
                                c="red", ls=":",
                                zorder=2
                            )

                    if axes[_nax].get_legend_handles_labels()[1]:
                        axes[_nax].legend(fontsize=8, ncol=4)

                if save_img:
                    if save_path is not None and save_name is not None:
                        save_name_ = save_name + f".{_nfig + 1}"
                        figs[_nfig].savefig(
                            f"{save_path_}{save_name_}.{save_form}",
                            format=save_form,
                            dpi=300
                        )
                    else:
                        raise ValueError(
                            f"'save_path' and/or 'save_name' not given "
                            f"(closure {warn_message})."
                        )

            if plotimg:
                plt.show()

            for _nfig in range(nfig):
                close_figure(figs[_nfig])

        else:
            for nf, freq in enumerate(ufreq):
                mask_freq = obs["freq"] == freq
                obs_freq = copy.deepcopy(obs[mask_freq])

                if len(obs_freq) == 0:
                    warnings.warn(
                        f"Empty closure {warn_message} data "
                        f"at {freq:.3f} GHz. "
                        f"Skip drawing closure {warn_message} "
                        f"at this frequency.",
                        UserWarning
                    )
                    continue

                if plotmodel:
                    mod_freq = copy.deepcopy(mod[mask_freq])

                ubsli = np.unique(obs_freq[idx_bsli])
                nbsli = len(ubsli)

                nfig = nbsli // 10
                if nbsli % 10 != 0:
                    nfig += 1

                fsize = 9
                figs = [None for i in range(nfig)]

                for _nfig in range(nfig):
                    if _nfig != nfig - 1:
                        nax = 10
                    else:
                        nax = nbsli % 10 if nbsli % 10 != 0 else 10

                    margin_top = 0.3
                    margin_bot = 0.5
                    fig_h = fsize * nax / 10 + margin_top + margin_bot

                    figs[_nfig] = plt.figure(
                        figsize=(fsize, fig_h)
                    )
                    figs[_nfig].supylabel(label_y, fontsize=12)
                    axes = [None for i in range(nax)]

                    gs_main = gridspec.GridSpec(
                        nax, 1, figure=figs[_nfig],
                        hspace=0.15, wspace=0.10,
                        top=1 - margin_top / fig_h,
                        bottom=margin_bot / fig_h,
                        left=0.10, right=0.98
                    )

                    for _nax in range(nax):
                        if _nax == 0:
                            axes[_nax] = figs[_nfig].add_subplot(
                                gs_main[_nax]
                            )
                        else:
                            axes[_nax] = figs[_nfig].add_subplot(
                                gs_main[_nax], sharex=axes[0]
                            )
                        axes[_nax].set_rasterized(True)

                        axes[_nax].xaxis.set_major_locator(
                            MultipleLocator(2.0)
                        )
                        axes[_nax].xaxis.set_minor_locator(
                            MultipleLocator(1.0)
                        )
                        if _nax != nax - 1:
                            axes[_nax].tick_params(labelbottom=False)
                        else:
                            axes[_nax].set_xlabel(
                                "Time (hour)", fontsize=12
                            )

                        mask_bsli = (
                            obs_freq[idx_bsli]
                            == ubsli[10 * _nfig + _nax]
                        )

                        obs_freq_bsli = copy.deepcopy(
                            obs_freq[mask_bsli]
                        )

                        if plotmodel:
                            mod_freq_bsli = copy.deepcopy(
                                mod_freq[mask_bsli]
                            )

                        ax_label = (
                            f"{ubsli[10 * _nfig + _nax]} "
                            f"({ufreq[nf]:.3f} GHz)"
                        )

                        if dotype in ["amp", "clamp"]:
                            x = obs_freq_bsli[idx_x]
                            yobs = np.log(obs_freq_bsli[idx_y])
                            yerr = obs_freq_bsli[idx_yerr]
                            if plotmodel:
                                ymod = np.log(mod_freq_bsli).flatten()

                        elif dotype in ["phs", "clphs"]:
                            r2d = au.rad.to(au.deg)
                            x = obs_freq_bsli[idx_x]
                            yobs = (
                                r2d
                                * np.angle(
                                    np.exp(1j * obs_freq_bsli[idx_y])
                                )
                            )
                            yerr = r2d * obs_freq_bsli[idx_yerr]
                            if plotmodel:
                                ymod = (
                                    r2d
                                    * np.angle(np.exp(1j * mod_freq_bsli))
                                ).flatten()

                        axes[_nax].errorbar(
                            x, yobs, yerr,
                            marker="o", markersize=6, c="black",
                            ls="",
                            mfc="black", mec="dimgray",
                            capsize=6, capthick=1, zorder=1,
                            label=ax_label
                        )

                        if plotmodel:
                            ymod = np.where(np.isfinite(yobs), ymod, np.nan)
                            order = np.argsort(np.asarray(x))
                            axes[_nax].plot(
                                np.asarray(x)[order], ymod[order],
                                marker="x", markersize=8,
                                c="red", ls=":",
                                zorder=2
                            )
                        axes[_nax].legend(fontsize=8, ncol=4)

                    if save_img:
                        if (save_path is not None
                            and save_name is not None
                        ):
                            if nfreq == 1:
                                save_name_ = save_name + f".{_nfig + 1}"
                            else:
                                save_name_ = (
                                    save_name
                                    + f".{ufreq[nf]:.0f}"
                                    + f".{_nfig + 1}"
                                )

                            figs[_nfig].savefig(
                                f"{save_path_}"
                                f"{save_name_}.{save_form}",
                                format=save_form,
                                dpi=300
                            )
                        else:
                            raise ValueError(
                                f"'save_path' and/or 'save_name' not given "
                                f"(closure {warn_message})."
                            )

                if plotimg:
                    plt.show()

                for _nfig in range(nfig):
                    close_figure(figs[_nfig])

    def draw_cnplot(
        self,
        result=None, pol=False, nmod=None, relmod=True, spectrum="spl",
        model="gaussian", save_path=False, save_name=False, save_form="png"
    ):
        """
        draw corner plot
        Args:
            result (array): results of dynesty model-fit
            nmod (int) : the number of models
            save_path (str): path to saving directory
            save_name (str): name of figure to be saved
        """
        if not self.fitset is None:
            fitset = self.fitset
        if not self.spectrum is None:
            spectrum = self.spectrum

        save_path = save_path + "plot_cn/"
        if os.path.isdir(save_path):
            os.system(f"rm -rf {save_path}")
        gv.utils.mkdir(save_path)

        if pol:
            nidx = 0
        else:
            nidx = 1
        sidx = [0]

        for i in range(nmod):
            n = i + 1
            nidx_, field = get_trcnidx(pol, model, spectrum, relmod, n)

            if spectrum in ["cpl", "ssa"]:
                if n == 1:
                    add = 0
                else:
                    add = 1
                    sidx.append(nidx)

                    ql, qm, qh = dyquan(
                        result.samples[:,nidx],
                        (0.025, 0.500, 0.975),
                        weights=result.importance_weights()
                    )

                    mask_spectrum = round(float(qm)) == 0
                    if mask_spectrum:
                        field = field[:-1]
            else:
                add = 0

            # draw corner-plot
            fig_cn, axes_cn = dyplot.cornerplot(
                result,
                show_titles=True,
                truth_color="black",
                dims=list(range(
                    nidx + add, nidx + add + len(field)
                )),
                labels=field,
                label_kwargs={"fontsize":20}
            )

            for nax1 in np.arange(axes_cn.shape[0]):
                for nax2 in np.arange(axes_cn.shape[1]):
                    axes_cn[nax1,nax2].set_rasterized(True)
            fig_cn.tight_layout()
            fig_cn.savefig(
                f"{save_path}{save_name}.mod{n}.{save_form}",
                format=save_form,
                dpi=300
            )
            close_figure(fig_cn)
            nidx += nidx_

        if not pol:
            # draw corner-plot of the number of model and spectrum
            slabel = ["nmod"] + [
                f"{i+2}_spectrum" for i in range(len(sidx)-1)
            ]

            fig_cn, axes_cn = dyplot.cornerplot(
                result,
                show_titles=True,
                truth_color="black",
                dims=sidx,
                labels=slabel,
                label_kwargs={"fontsize":20}
            )
            for nax1 in np.arange(axes_cn.shape[0]):
                for nax2 in np.arange(axes_cn.shape[1]):
                    axes_cn[nax1,nax2].set_rasterized(True)
            fig_cn.tight_layout()
            fig_cn.savefig(
                f"{save_path}{save_name}.mod.spectrum.{save_form}",
                format=save_form,
                dpi=300
            )
            close_figure(fig_cn)

    def draw_dirtymap(
        self,
        uvf, select=None, npix=1024, uvw="u",
        plot_resi=False, plotimg=True,
        cmap="gist_heat", ccontour="lightgrey", cmodel="cyan",
        save_path=False, save_name=False, save_form="png",
        returned=False
    ):
        mapfov = uvf.mapfov.value
        mapunit = uvf.mapunit
        extent = [+mapfov/2, -mapfov/2, -mapfov/2, mapfov/2]

        if save_path:
            gv.utils.mkdir(save_path)

        beam, dirty = gv.utils.dft_vis(uvf, plot_resi=plot_resi, npix=npix)

        if plot_resi:
            uvf.resid = dirty
        else:
            uvf.dirty = dirty

        if plotimg or save_name:
            if plot_resi:
                dirtitle = (
                    rf"$\rm residual~$"
                    rf"$\rm (map~range:~$"
                    rf"${dirty.min():.3f} <-> {dirty.max():.3f})$"
                )
            else:
                dirtitle = (
                    r"$\rm dirty~map~$"
                    r"$(I_{\rm peak}=$" + f"{dirty.max():.3f} Jy)"
                )

            beamtitle = (
                rf"$\rm beam~pattern~(\sigma \approx {np.std(beam):.3f})$"
            )

            fsize = 15
            fig = plt.figure(figsize=(fsize, fsize * 8 / 16))
            gs_main = gridspec.GridSpec(
                1, 2, figure=fig,
                hspace=0.15, wspace=0.10,
                top=0.95, bottom=0.10, left=0.07, right=0.98
            )

            ax_bim = fig.add_subplot(gs_main[0])
            ax_dir = fig.add_subplot(gs_main[1], sharex=ax_bim)

            cb_bim = inset_axes(
                ax_bim, width="90%", height="5%", loc="upper center"
            )
            cb_dir = inset_axes(
                ax_dir, width="90%", height="5%", loc="upper center"
            )
            ax_bim.set_rasterized(True)
            ax_dir.set_rasterized(True)
            ax_bim.set_aspect("equal")
            ax_dir.set_aspect("equal")
            ax_bim.set_title(beamtitle, fontsize=12)
            ax_dir.set_title(dirtitle, fontsize=12)

            ax_bim.tick_params(labelsize=12)
            ax_dir.tick_params(labelsize=12)

            fig.supxlabel(f"Relative R.A ({mapunit})", fontsize=15)
            fig.supylabel(f"Relative Dec ({mapunit})", fontsize=15)

            cb_bim.tick_params(left=False, labelleft=False)
            cb_dir.tick_params(left=False, labelleft=False)

            cb_bim.tick_params(
                axis="both", which="both",
                colors=ccontour, labelcolor=ccontour, labelsize=8
            )
            cb_dir.tick_params(
                axis="both", which="both",
                colors=ccontour, labelcolor=ccontour, labelsize=8
            )

            cmap_bim = ax_bim.imshow(beam,
                cmap=cmap, interpolation="gaussian", extent=extent,
                vmin=beam.min(), vmax=beam.max()
            )
            cmap_dir = ax_dir.imshow(dirty,
                cmap=cmap, interpolation="gaussian", extent=extent,
                vmin=dirty.min(), vmax=dirty.max()
            )

            cbar_bim = fig.colorbar(
                cmap_bim, cax=cb_bim, ax=ax_bim, orientation="horizontal"
            )
            cbar_dir = fig.colorbar(
                cmap_dir, cax=cb_dir, ax=ax_dir, orientation="horizontal"
            )

            for spine in cb_bim.spines.values():
                spine.set_edgecolor(ccontour)
                spine.set_linewidth(1)
            for spine in cb_dir.spines.values():
                spine.set_edgecolor(ccontour)
                spine.set_linewidth(1)

            ax_bim = set_mapticks(ax_bim, mapfov)
            ax_dir = set_mapticks(ax_dir, mapfov)

            if plotimg:
                plt.show()

            if save_name:
                fig.savefig(
                    f"{save_path}{save_name}.{save_form}",
                    format=save_form,
                    dpi=300
                )

            close_figure(fig)

        if returned:
            return (beam, dirty, extent)

    def draw_crossvis(
        self,
        uvf, plotimg=True, save_path=False, save_name=False, save_form="png"
    ):
        if uvf.nstokes != 4:
            raise ValueError(
                "Provided uvf file does not have full Stokes parameters."
            )

        rr = uvf.get_data("vis_rr").flatten()
        ll = uvf.get_data("vis_ll").flatten()
        rl = uvf.get_data("vis_rl").flatten()
        lr = uvf.get_data("vis_lr").flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rlrr = rl / rr
            lrrr = lr / rr
            rlll = rl / ll
            lrll = lr / ll

        axlim = 1.2 * max(
            np.nanmax(np.abs(rlrr)),
            np.nanmax(np.abs(lrrr)),
            np.nanmax(np.abs(rlll)),
            np.nanmax(np.abs(lrll))
        )

        fig_dterm, axes_dterm = plt.subplots(2, 2, figsize=(8, 8))
        ax_rlrr = axes_dterm[0, 0]
        ax_lrrr = axes_dterm[0, 1]
        ax_rlll = axes_dterm[1, 0]
        ax_lrll = axes_dterm[1, 1]
        ax_rlrr.set_aspect("equal")
        ax_lrrr.set_aspect("equal")
        ax_rlll.set_aspect("equal")
        ax_lrll.set_aspect("equal")
        ax_rlrr.tick_params(labelbottom=False, labelleft=True )
        ax_lrrr.tick_params(labelbottom=False, labelleft=False)
        ax_rlll.tick_params(labelbottom=True, labelleft=True )
        ax_lrll.tick_params(labelbottom=True, labelleft=False)
        ax_rlrr.scatter(rlrr.real, rlrr.imag, c="black", marker="x", s=10)
        ax_lrrr.scatter(lrrr.real, lrrr.imag, c="black", marker="x", s=10)
        ax_rlll.scatter(rlll.real, rlll.imag, c="black", marker="x", s=10)
        ax_lrll.scatter(lrll.real, lrll.imag, c="black", marker="x", s=10)
        ax_rlrr.set_xlim(-axlim, +axlim)
        ax_lrrr.set_xlim(-axlim, +axlim)
        ax_rlll.set_xlim(-axlim, +axlim)
        ax_lrll.set_xlim(-axlim, +axlim)
        ax_rlrr.set_ylim(-axlim, +axlim)
        ax_lrrr.set_ylim(-axlim, +axlim)
        ax_rlll.set_ylim(-axlim, +axlim)
        ax_lrll.set_ylim(-axlim, +axlim)

        if plotimg:
            plt.show()

        close_figure(fig_dterm)

    def draw_image(
        self,
        uvf, pol=False, returned=False, bprms=None, freq_ref=None, freq=None,
        genlevels=False, npix=1024, mindr=3, minlev=0.01, maxlev=0.99, step=2,
        contourw=0.3, mintick_map=0.5, majtick_map=2.5, mintick_cb=0.2,
        majtick_cb=1.0, model="gaussian", xlim=False, ylim=False,
        cmap="gist_heat", ccontour="lightgrey", cmodel="cyan", plotimg=True,
        plot_resi=False, addnoise=False, outfig=False, evpa_gap=None,
        evpa_length=0.5, evpa_width=2.0, save_img=False, save_path=False,
        save_name=False, save_form="png"
    ):
        """
        draw final image
        Args:
            uvf (python class): opened-fits file in uvf-class
            bprms (tuple): beam parameters (minor, major, position angle)
            freq_ref (float): reference frequency
            freq (float): frequency to plot by considering estimated spectrum
            levels (list): contour levels to draw
            minlev (float): starting contour level in fraction
            maxlev (float): final contour level in fraction
            step (int): step size of the contour
            mintick_map (flaot): size of minor tick label in the intensity map
            majtick_map (flaot): size of major tick label in the intensity map
            mintick_cb (flaot): size of minor tick label in the color bar
            majtick_cb (flaot): size of major tick label in the color bar
            save_img (bool): toggole option if save the final image
            save_path (str or bool): (if set) path of the saving image
            save_name (str or bool): (if set) name of saving image
            plotimg (bool): (if True) plot final image
            npix (int): the number of pixels in the map
            addnoise (bool): (if True) add the noise in the residual map
        """
        if pol:
            contourw = 2 * contourw

        mapfov = uvf.mapfov.value
        mapunit = uvf.mapunit
        extent = [+mapfov/2, -mapfov/2, -mapfov/2, mapfov/2]
        psize = mapfov / npix

        if save_path:
            gv.utils.mkdir(save_path)

        if freq_ref is None:
            freq_ref = self.freq_ref

        if freq is None:
            freq = self.freq

        if bprms is None:
            self.set_beamprms(uvf)
            bprms = self.bprms

        if self.theta is None:
            theta = uvf.theta
        else:
            theta = self.theta

        bmin, bmaj, bpa = bprms

        mapfov = uvf.mapfov.value
        npix = npix

        nmod = round(float(theta["nmod"]))

        if plot_resi and save_name:
            save_name_ = save_name.replace("img", "resimap")

            # reconstruct residual maps
            if "mf" in save_name_:
                self.img_count = 0
                if "_C" in save_name_:
                    save_name_ = save_name_.replace("_C", "")
                    self.img_count += 1
                if "_X" in save_name_:
                    save_name_ = save_name_.replace("_X", "")
                    self.img_count += 1
                if "_U" in save_name_:
                    save_name_ = save_name_.replace("_U", "")
                    self.img_count += 1
                if "_K" in save_name_:
                    save_name_ = save_name_.replace("_K", "")
                    self.img_count += 1
                if "_Q" in save_name_:
                    save_name_ = save_name_.replace("_Q", "")
                    self.img_count += 1
                if "_W" in save_name_:
                    save_name_ = save_name_.replace("_W", "")
                    self.img_count += 1
                if "_D" in save_name_:
                    save_name_ = save_name_.replace("_D", "")
                    self.img_count += 1
            else:
                self.img_count = 1
            # reconstruct residual-only map
            if self.img_count == 1:
                self.draw_dirtymap(
                    uvf=uvf, plotimg=False, plot_resi=True,
                    npix=npix, uvw=uvf.uvw,
                    save_path=save_path,
                    save_name=save_name_,
                    save_form=save_form
                )

        # reconstruct model+residual map
        self.draw_dirtymap(
            uvf=uvf, plot_resi=addnoise, plotimg=False,
            npix=npix, uvw=uvf.uvw
        )
        self.generate_image(
            uvf=uvf, npix=npix, pol=pol, freq_ref=freq_ref, freq=freq,
            theta=theta, model=model, spectrum=self.spectrum
        )

        image = self.image.copy()
        image = self.convolve_image(
            uvf=uvf, npix=npix, image=image, bprms=bprms
        )

        # add residual noise map
        if addnoise:
            resim = uvf.resid
            image += np.flip(resim, axis=0)

            rms = gv.utils.cal_rms(resim)
            if rms == 0:
                rms = 0.01 * image.max()

            levels = [mindr * rms]
            if mindr * rms < image.max() * np.sqrt(2):
                while levels[-1] < image.max():
                    levels.append(levels[-1] * np.sqrt(2))

            if pol and uvf.select_pol != "p":
                levels_n = [-mindr * rms]
                if mindr * rms < np.abs(image.min()) * np.sqrt(2):
                    while np.abs(levels_n[-1]) < np.abs(image.min()):
                        levels_n.append(levels_n[-1] * np.sqrt(2))
        else:
            levels = np.array([1, 2, 4, 8, 16, 32, 64]) / 100 * image.max()

        # set contour levels
        if genlevels:
            levels = [maxlev]
            while levels[-1] > minlev:
                levels.append(levels[-1] / step)
            levels = np.sort(image.max() * np.array(levels))

        uvf.image = np.flip(image, axis=0)

        # assign Stokes Q/U maps
        if uvf.select_pol == "q":
            uvf.image_q = image
        elif uvf.select_pol == "u":
            uvf.image_u = image

        # plot figure
        fsize = 9
        fig = plt.figure(figsize=(fsize, fsize))
        gs_main = gridspec.GridSpec(
            1, 1, figure=fig,
            hspace=0.15, wspace=0.10,
            top=0.95, bottom=0.10, left=0.07, right=0.98
        )

        ax_map = fig.add_subplot(gs_main[0])
        cb_map = inset_axes(
            ax_map, width="90%", height="5%", loc="upper center"
        )

        ax_map.set_rasterized(True)
        ax_map.set_aspect("equal")

        ax_map.tick_params(labelsize=12)

        ax_map.set_xlabel(f"Relative R.A ({mapunit})", fontsize=15)
        ax_map.set_ylabel(f"Relative Dec ({mapunit})", fontsize=15)

        cb_map.tick_params(left=False, labelleft=False)

        if pol:
            cb_map.tick_params(
                axis="both", which="both",
                colors="black", labelcolor="black", labelsize=8
            )
        else:
            cb_map.tick_params(
                axis="both", which="both",
                colors=ccontour, labelcolor=ccontour, labelsize=8
            )

        if uvf.select_pol == "p":
            ra_1d = np.linspace(mapfov/2, -mapfov/2, npix)
            dec_1d = np.linspace(-mapfov/2, mapfov/2, npix)
            ra_2d, dec_2d = np.meshgrid(ra_1d, dec_1d)

            if evpa_gap is None:
                evpa_gap = npix // 75

            evpa, evpa_x, evpa_y, evpa_set = gv.utils.cal_evpa(
                p=image, q=uvf.image_q, u=uvf.image_u, rms=rms, snr=3,
                mapfov=mapfov, npix=npix, evpa_length=evpa_length,
                evpa_width=evpa_width
            )

            cmap_map = ax_map.imshow(image,
                cmap=polcba, interpolation="gaussian", extent=extent,
                vmin=image.min(), vmax=image.max(), origin="lower"
            )
            ax_map.contour(image,
                levels=levels, colors="k", linewidths=contourw,
                extent=extent
            )
            ax_map.quiver(
                ra_2d[::evpa_gap, ::evpa_gap], dec_2d[::evpa_gap, ::evpa_gap],
                evpa_x[::evpa_gap, ::evpa_gap], evpa_y[::evpa_gap, ::evpa_gap],
                **evpa_set, zorder=2
            )
        else:
            cmap_map = ax_map.imshow(image,
                cmap=cmap, interpolation="gaussian", extent=extent,
                vmin=image.min(), vmax=image.max(), origin="lower"
            )
            ax_map.contour(image,
                levels=levels, colors=ccontour, linewidths=contourw,
                extent=extent
            )

        if pol and uvf.select_pol != "p":
            ax_map.contour(np.abs(image),
                levels=np.abs(levels_n), colors="royalblue",
                linewidths=contourw, linestyles="--", extent=extent
            )

        cbar_map = fig.colorbar(
            cmap_map, cax=cb_map, ax=ax_map, orientation="horizontal"
        )

        if pol:
            cb_map.set_xlabel("Intensity (Jy/beam)", fontsize=12, c="k")
            for spine in cb_map.spines.values():
                spine.set_edgecolor("k")
                spine.set_linewidth(1)
        else:
            cb_map.set_xlabel("Intensity (Jy/beam)", fontsize=12, c=ccontour)
            for spine in cb_map.spines.values():
                spine.set_edgecolor(ccontour)
                spine.set_linewidth(1)

        ax_map = set_mapticks(ax_map, mapfov)

        ang = np.deg2rad(bpa)
        beam_hw = np.hypot(bmaj / 2 * np.cos(ang), bmin / 2 * np.sin(ang))
        beam_hh = np.hypot(bmaj / 2 * np.sin(ang), bmin / 2 * np.cos(ang))

        beam_margin = 0.05 * mapfov
        beam = patches.Ellipse(
            (
                mapfov / 2 - beam_hw - beam_margin,
                -mapfov / 2 + beam_hh + beam_margin
            ),
            bmaj, bmin, angle=-bpa,
            fc="grey", ec="yellow", lw=1.0
        )
        ax_map.add_patch(beam)

        _dtype = theta.dtype.names

        for i in range(nmod):
            has_l = f"{i + 1}_l" in _dtype
            has_m = f"{i + 1}_m" in _dtype
            if has_l and has_m:
                ra, dec = theta[f"{i + 1}_l"], theta[f"{i + 1}_m"]
            else:
                ra, dec = 0, 0

            _r = np.sqrt(ra**2 + dec**2)
            _p = np.arctan2(ra, dec) * 180 / np.pi
            if model == "gaussian":
                a = theta[f"{i + 1}_a"]
                if a >= psize:
                    Gmodel = patches.Ellipse(
                        (ra, dec),
                        a, a, angle=0,
                        fc="none", ec=cmodel, lw=1.0
                    )

                    stick1 = patches.ConnectionPatch(
                        xyA=(ra - a / 2, dec),
                        xyB=(ra + a / 2, dec),
                        coordsA="data", color=cmodel, lw=1.0
                    )

                    stick2 = patches.ConnectionPatch(
                        xyA=(ra, dec - a / 2),
                        xyB=(ra, dec + a / 2),
                        coordsA="data", color=cmodel, lw=1.0
                    )

                    ax_map.add_patch(Gmodel)
                    ax_map.add_patch(stick1)
                    ax_map.add_patch(stick2)
                else:
                    ax_map.scatter(ra, dec, color=cmodel, marker="+", s=50)
            elif model == "delta":
                ax_map.scatter(ra, dec, color=cmodel, marker="+", s=50)

        if save_path and save_name:
            fig.savefig(
                f"{save_path}{save_name}.{save_form}",
                format=save_form,
                dpi=300
            )
        if plotimg:
            plt.show()

        if returned:
            close_figure(fig)
            if addnoise:
                return (image, resim)
            else:
                return (image)
        if outfig:
            return (fig, ax_map)
            close_figure(fig)
        close_figure(fig)

    def draw_radplot(
        self,
        uvf, select_pol=None, plotmodel=False, plotimg=True,
        dotype=["amp", "phs"],
        save_path=False, save_name=False, save_form="png"
    ):
        availables = ["amp", "phs", "sigma", "snr"]
        for _type in dotype:
            if _type not in availables:
                raise ValueError(
                    f"Invalid plot type is included: {_type!r}.\n"
                    f"Availables: {availables}."
                )
        ntype = len(dotype)

        if save_path:
            gv.utils.mkdir(save_path)

        freq = uvf.get_data(dotype="frequency").flatten().astype("f4")

        ufreq = np.unique(freq)

        cs = 6          # cap size
        ct = 1          # cap thick
        mse = 6         # marker size (errorbar)
        mss = 35        # marker size (scatter)
        mec = "black"   # marker edge color
        alpha = 0.3
        if len(ufreq) == 1:
            mfc = ["none"]
        else:
            mfc = [
                "tab:blue", "tab:orange", "tab:green", "tab:purple",
                "tab:brown", "tab:pink", "tab:gray" , "tab:olive"
            ]

        u = uvf.get_data(dotype="u", flatten=True) / 1e6
        v = uvf.get_data(dotype="v", flatten=True) / 1e6
        uvr = np.sqrt(u**2 + v**2)

        if select_pol is None:
            if uvf.select_pol is None:
                vis = uvf.get_data(dotype="vis", flatten=True)
                sig = uvf.get_data(dotype="sig", flatten=True)
            else:
                select_pol = uvf.select_pol.split(".")[-1]
                vis = uvf.get_data(dotype=f"vis_{select_pol}", flatten=True)
                sig = uvf.get_data(dotype=f"sig_{select_pol}", flatten=True)
        else:
            vis = uvf.get_data(dotype=f"vis_{select_pol}", flatten=True)
            sig = uvf.get_data(dotype=f"sig_{select_pol}", flatten=True)

        vis = np.where(u < 0, vis.conj(), vis)
        amp = np.abs(vis)
        phs = np.angle(vis) * au.rad.to(au.deg)
        snr = amp / sig

        if plotmodel:
            if uvf.vism is None:
                raise ValueError(f"Model visibility is not presented.")

            vism = uvf.get_data(dotype="vism", flatten=True)
            vism = np.where(u < 0, vism.conj(), vism)

        sig_amp = sig
        sig_phs = sig / amp * au.rad.to(au.deg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            maxphs = 1.2 * np.nanmax(np.abs(phs) + sig_phs)

        if np.isnan(maxphs):
            maxphs = 200

        if maxphs > 200:
            maxphs = 200

        date = atime(uvf.mjd.min(), format="mjd").iso[:10]

        label_amp = r"$S~({\rm Jy})$"
        label_phs = r"$\phi~({\rm deg})$"
        label_sig = r"$\sigma$"
        label_snr = r"$SNR$"

        fsize = 16
        fig = plt.figure(figsize=(fsize, fsize * 8 / 16))
        axes = [None for i in range(ntype)]

        gs_main = gridspec.GridSpec(
            ntype, 1, figure=fig,
            hspace=0.0, wspace=0.1,
            top=0.98, bottom=0.10, left=0.07, right=0.98
        )

        label_uvr = r"$\rm UV~radius~(10^{6}~\lambda)$"
        for nt, _type in enumerate(dotype):
            if nt == 0:
                axes[nt] = fig.add_subplot(gs_main[nt])
            else:
                axes[nt] = fig.add_subplot(gs_main[nt], sharex=axes[0])
            axes[nt].set_rasterized(True)

            axes[nt].tick_params(labelsize=13)
            if nt != ntype - 1:
                axes[nt].tick_params(labelbottom=False, bottom=False)
            else:
                axes[nt].set_xlabel(label_uvr, fontsize=17)

            x = uvr
            if _type == "amp":
                _label = label_amp
                y = amp
                yerr = sig_amp
                if plotmodel:
                    model = np.abs(vism)

            elif _type == "phs":
                _label = label_phs
                y = phs
                yerr = sig_phs
                if plotmodel:
                    model = np.angle(vism) * au.rad.to(au.deg)
                axes[nt].set_ylim(-maxphs, +maxphs)

            elif _type == "sigma":
                _label = label_sig
                y = sig

            elif _type == "snr":
                _label = label_snr
                y = snr

            axes[nt].set_ylabel(_label, fontsize=15)

            for nf, _freq in enumerate(ufreq):
                if nt == 0:
                    label_freq = f"{_freq:.3f} GHz"
                else:
                    label_freq = None

                mask_freq = freq == _freq
                _x = uvr[mask_freq]
                _y = y[mask_freq]

                mask_nan = np.isnan(_y)

                if _type in ["amp", "phs"]:
                    _yerr = yerr[mask_freq]
                    axes[nt].errorbar(
                        _x, _y, _yerr,
                        ls="", capsize=cs, capthick=ct,
                        marker="o", markersize=mse, mfc=mfc[nf], mec=mec,
                        c=mec, alpha=alpha, zorder=1, label=label_freq
                    )

                    if plotmodel:
                        if nf == 0 and nt == 0:
                            label_model = "model"
                        else:
                            label_model = None

                        _x_model = _x[~mask_nan]
                        _y_model = model[mask_freq][~mask_nan]

                        axes[nt].scatter(
                            _x_model, _y_model,
                            marker="s", s=7, c="red", zorder=2,
                            label=label_model
                        )
                else:
                    axes[nt].scatter(
                        _x, _y,
                        marker="o", s=mss, fc=mfc[nf], ec=mec,
                        alpha=alpha, zorder=1,
                        label=label_freq
                    )
            axes[0].legend(fontsize=10)

        if save_name:
            fig.savefig(
                f"{save_path}{save_name}.{save_form}",
                format=save_form,
                dpi=300
            )

        if plotimg:
            plt.show()

        close_figure(fig)

    def draw_tplot(
        self,
        uvf, select_pol=None, plotimg=True, dotype="utc",
        save_path=False, save_name=False, save_form="png"
    ):
        dict_ants = dict(
            zip(
                uvf.tarr["name"],
                np.arange(len(uvf.tarr["name"]))
            )
        )

        if save_path:
            gv.utils.mkdir(save_path)

        data = uvf.get_data(dotype="vis")
        time = uvf.get_data(dotype="time")

        mask_nan = ~np.isnan(data)

        if dotype == "gst":
            t0 = atime(uvf.date, scale="utc")
            utc_time = t0 + TimeDelta(time * au.hour)
            gst_time = utc_time.sidereal_time("apparent", "greenwich").value
            tdiff = time - gst_time
            time = gst_time.copy()
            time = np.where(tdiff > +12, time + 24, time)
            time = np.where(tdiff < -12, time - 24, time)

        freqs = uvf.freq0
        ufreq = np.unique(freqs)

        ant1_name = uvf.get_data(dotype="ant1_name")
        ant2_name = uvf.get_data(dotype="ant2_name")
        uants = np.unique(np.append(ant1_name, ant2_name))

        fsize = 10
        for nfreq, freq in enumerate(ufreq):
            plabel = False
            yticks = []
            ytick_valid = []

            fig_tplot, ax_tplot = plt.subplots(
                1, 1, figsize=(fsize, fsize * 0.5)
            )

            ax_tplot.set_rasterized(True)
            for nant, ant in enumerate(uants):
                mask_freq = freqs == freq
                mask_ants = (ant1_name == ant) | (ant2_name == ant)
                mask = mask_freq & mask_ants & mask_nan

                if mask.sum() == 0:
                    continue

                if not plabel:
                    label_freq = uvf.freq_mean
                    if isinstance(label_freq, str):
                        label = label_freq
                    else:
                        label = f"{uvf.freq_mean:.3f} GHz"
                else:
                    label = None

                _data = data[mask]
                _time = time[mask]
                _ndat = len(_data)
                addidx = 1 + dict_ants[ant] + len(ufreq) * nfreq
                ax_tplot.scatter(
                    _time,
                    np.zeros(_ndat) + addidx,
                    c="black", marker="+", s=200,
                    label=label
                )

                yticks.append(ant)
                ytick_valid.append(addidx)

                if not plabel:
                    plabel = True

            ax_tplot.set_ylim(
                min(ytick_valid) - 1, max(ytick_valid) + 1
            )

            ax_tplot.set_xlabel(f"Time (hour, {dotype.upper()})", fontsize=20)
            ax_tplot.set_ylabel("Antenna", fontsize=20)
            ax_tplot.xaxis.set_major_locator(MultipleLocator(2.0))
            ax_tplot.xaxis.set_minor_locator(MultipleLocator(1.0))
            ax_tplot.tick_params("both", labelsize=15)
            ax_tplot.set_yticks(ytick_valid)
            ax_tplot.set_yticklabels(yticks)
            ax_tplot.grid(True, ls="--", axis="y")

            handles, labels = ax_tplot.get_legend_handles_labels()
            ax_tplot.legend(
                handles[::-1], labels[::-1],
                ncol=len(ufreq),
                loc="upper right",
                fontsize=13
            )

            fig_tplot.tight_layout()

            if save_name:
                if "mf" in uvf.select_pol:
                    fig_tplot.savefig(
                        f"{save_path}{save_name}.{freq:.0f}.{save_form}",
                        format=save_form,
                        dpi=300
                    )
                else:
                    fig_tplot.savefig(
                        f"{save_path}{save_name}.{save_form}",
                        format=save_form,
                        dpi=300
                    )

            if plotimg:
                plt.show()

        close_figure(fig_tplot)

    def draw_trplot(
        self,
        result=None, pol=False, nmod=None, relmod=True, spectrum="spl",
        model="gaussian", save_path=False, save_name=False, save_form="png"
    ):
        """
        draw trace plot
        Args:
            result (array): results of dynesty model-fit
            nmod  (int): the number of models
            save_path (str): path to saving directory
            save_name (str): name of figure to be saved
        """
        if not self.spectrum is None:
            spectrum = self.spectrum

        save_path = save_path + "plot_tr/"
        if os.path.isdir(save_path):
            os.system(f"rm -rf {save_path}")
        gv.utils.mkdir(save_path)

        if pol:
            nidx = 0
        else:
            nidx = 1
        sidx = [0]

        for i in range(nmod):
            n = i + 1
            nidx_, field = get_trcnidx(pol, model, spectrum, relmod, n)

            if spectrum in ["cpl", "ssa"]:
                if n == 1:
                    add = 0
                else:
                    add = 1
                    sidx.append(nidx)

                    ql, qm, qh = dyquan(
                        result.samples[:,nidx],
                        (0.025, 0.500, 0.975),
                        weights=result.importance_weights()
                    )

                    mask_spectrum = round(float(qm)) == 0
                    if mask_spectrum:
                        field = field[:-1]
            else:
                add= 0

            # draw trace-plot
            fig_tr, axes_tr = dyplot.traceplot(
                result,
                show_titles=True,
                truth_color="black",
                trace_cmap="viridis",
                dims=list(range(
                    nidx + add, nidx + add + len(field)
                )),
                labels=field,
                label_kwargs={"fontsize":20}
            )

            for nax1 in np.arange(axes_tr.shape[0]):
                for nax2 in np.arange(axes_tr.shape[1]):
                    axes_tr[nax1,nax2].set_rasterized(True)

            fig_tr.tight_layout()
            fig_tr.savefig(
                f"{save_path}{save_name}.mod{n}.{save_form}",
                format=save_form,
                dpi=300
            )
            close_figure(fig_tr)
            nidx += nidx_

        if not pol:
            # draw trace-plot of the number of model and spectrum
            slabel = ["nmod"] + [
                f"{i + 2}_spectrum" for i in range(len(sidx) - 1)
            ]
            fig_tr, axes_tr = dyplot.traceplot(
                result,
                show_titles=True,
                truth_color="black",
                trace_cmap="viridis",
                dims=sidx,
                labels=slabel,
                label_kwargs={"fontsize":20}
            )

            for nax1 in np.arange(axes_tr.shape[0]):
                for nax2 in np.arange(axes_tr.shape[1]):
                    axes_tr[nax1,nax2].set_rasterized(True)

            fig_tr.tight_layout()
            fig_tr.savefig(
                f"{save_path}{save_name}.mod.spectrum.{save_form}",
                format=save_form,
                dpi=300
            )
            close_figure(fig_tr)

    def draw_uvcover(
        self,
        uvf, plotimg=True, dotype="a&p", highlight=None,
        save_path=False, save_name=False, save_form="png"
    ):
        if isinstance(highlight, str):
            highlight = [highlight]

        if save_path:
            gv.utils.mkdir(save_path)
        select_pol = uvf.select_pol.upper()

        vis = uvf.get_data("vis").flatten()
        amp = np.abs(vis)
        phs = np.angle(vis) * au.rad.to(au.deg)

        mjd = uvf.get_data("mjd").flatten()
        u = uvf.get_data("u").flatten() / 1e6
        v = uvf.get_data("v").flatten() / 1e6

        date = atime(mjd.min(), format="mjd").iso[:10]

        rng_a = 1.0 * np.nanmax(np.abs(amp))
        rng_p = 1.0 * np.nanmax(np.abs(phs))
        limv = 1.1 * max(np.nanmax(np.abs(u)), np.nanmax(np.abs(v)))

        fsize = 14

        def _setup_ax(ax, ylabel=True):
            ax.set_rasterized(True)
            ax.set_aspect("equal")
            ax.set_xlim(+limv, -limv)
            ax.set_ylim(-limv, +limv)
            ax.tick_params("both", labelsize=15)
            ax.grid(True)
            ax.set_xlabel(r"$\rm U~(10^{6}~\lambda)$", fontsize=20)
            if ylabel:
                ax.set_ylabel(r"$\rm V~(10^{6}~\lambda)$", fontsize=20)

        if dotype == "a&p":
            fig_uvc, ax_uvc = plt.subplots(
                1, 2, figsize=(fsize, fsize * 9 / 16)
            )
            ax_amp, ax_phs = ax_uvc[0], ax_uvc[1]
            ax_amp.get_shared_x_axes().joined(ax_amp, ax_phs)
            ax_amp.get_shared_y_axes().joined(ax_amp, ax_phs)
            ax_amp.set_facecolor("lightgray")
            ax_phs.set_facecolor("lightgray")
            for s in (+1, -1):
                cmap1 = ax_amp.scatter(
                    s * u, s * v, c=amp, s=30, ec="black", cmap="jet",
                    vmin=0, vmax=+rng_a, zorder=2
                )
                cmap2 = ax_phs.scatter(
                    s * u, s * v, c=s * phs, s=30, ec="black", cmap="bwr",
                    vmin=-rng_p, vmax=+rng_p, zorder=2
                )
            cbar1 = fig_uvc.colorbar(
                cmap1, ax=ax_amp, orientation="horizontal"
            )
            cbar2 = fig_uvc.colorbar(
                cmap2, ax=ax_phs, orientation="horizontal"
            )
            cbar1.set_label("Amplitude (Jy)", fontsize=15)
            cbar2.set_label("Phase (deg)", fontsize=15)
            _setup_ax(ax_amp, ylabel=True)
            _setup_ax(ax_phs, ylabel=False)

        elif dotype == "amp":
            fig_uvc, ax_amp = plt.subplots(
                1, 1, figsize=(fsize * 9 / 16, fsize * 9 / 16)
            )
            ax_amp.set_facecolor("lightgray")
            for s in (+1, -1):
                cmap1 = ax_amp.scatter(
                    s * u, s * v, c=amp, s=30, ec="black", cmap="jet",
                    vmin=0, vmax=+rng_a, zorder=2
                )
            cbar1 = fig_uvc.colorbar(
                cmap1, ax=ax_amp, orientation="horizontal"
            )
            cbar1.set_label("Amplitude (Jy)", fontsize=15)
            _setup_ax(ax_amp)

        elif dotype == "phs":
            fig_uvc, ax_phs = plt.subplots(
                1, 1, figsize=(fsize * 9 / 16, fsize * 9 / 16)
            )
            ax_phs.set_facecolor("lightgray")
            for s in (+1, -1):
                cmap2 = ax_phs.scatter(
                    s * u, s * v, c=s * phs, s=30, ec="black", cmap="bwr",
                    vmin=-rng_p, vmax=+rng_p, zorder=2
                )
            cbar2 = fig_uvc.colorbar(
                cmap2, ax=ax_phs, orientation="horizontal"
            )
            cbar2.set_label("Phase (deg)", fontsize=15)
            _setup_ax(ax_phs)

        else:   # dotype is None: plain uv-coverage (black), highlight in red
            ant1 = uvf.get_data("ant1").flatten()
            ant2 = uvf.get_data("ant2").flatten()
            if highlight is not None:
                name2num = dict(zip(
                    [str(n).strip() for n in uvf.tarr["name"]],
                    uvf.tarr["number"]
                ))
                hl_nums = [name2num[h] for h in highlight if h in name2num]
                hl = np.isin(ant1, hl_nums) | np.isin(ant2, hl_nums)
            else:
                hl = np.zeros(u.shape, dtype=bool)

            fig_uvc, ax_uv = plt.subplots(
                1, 1, figsize=(fsize * 9 / 16, fsize * 9 / 16)
            )
            ax_uv.set_facecolor("lightgray")
            for s in (+1, -1):
                ax_uv.scatter(
                    s * u[~hl], s * v[~hl], c="black", s=20, zorder=2
                )
                if hl.any():
                    ax_uv.scatter(
                        s * u[hl], s * v[hl], c="red", s=28, zorder=3
                    )
            _setup_ax(ax_uv)

        fig_uvc.tight_layout()

        if save_name:
            fig_uvc.savefig(
                f"{save_path}{save_name}.{save_form}",
                format=save_form,
                dpi=300
            )

        if plotimg:
            plt.show()

        close_figure(fig_uvc)

    def generate_image(
        self,
        uvf, pol=False, npix=1024, freq_ref=False, freq=False,
        theta=None, model="gaussian", spectrum="spl"
    ):
        """
        generate intensity image (Jy/beam)
        Args:
            uvf (python class): opened-fits file in uvf-class
            freq_ref (float): reference frequency
            freq (float): frequency to plot by considering estimated spectrum
            theta (array): set of model parameters
        Returns:
            image (2D-array): image (n by n) // not convolved
        """
        if theta is None:
            if self.theta is None:
                theta = uvf.theta
            else:
                theta = self.theta

        if isinstance(freq_ref, bool) and not freq_ref:
            freq_ref = self.freq_ref

        if isinstance(freq, bool) and not freq:
            freq = self.freq

        nmod = round(float(theta["nmod"]))
        xgrid, ygrid = gv.utils.get_xygrid(uvf, npix)

        image = np.zeros(xgrid.shape)
        mapfov = uvf.mapfov.value
        psize = mapfov / xgrid.shape[0]

        list_s = []
        list_a = []
        list_l = []
        list_m = []

        if pol:
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

                    _dtype = theta.dtype.names

                    _q = theta[idx_q]
                    _u = theta[idx_u]
                    _a = theta[idx_a]
                    _l = theta[idx_l]
                    _m = theta[idx_m]

                    has_i = idx_i in _dtype
                    has_f = idx_f in _dtype
                    has_r = idx_r in _dtype
                    has_t = idx_t in _dtype

                    if has_i:
                        _i = theta[idx_i]

                    if has_f:
                        _f = theta[idx_f]

                    if has_r:
                        _r = theta[idx_r]

                    if spectrum == "flat":
                        sq = _q
                        su = _u
                    elif spectrum == "spl":
                        raise NotImplementedError("To be updated.")
                    elif spectrum in ["cpl", "ssa"]:
                        raise NotImplementedError("To be updated.")

                    if uvf.select_pol == "p":
                        s = np.sqrt(sq**2 + su**2)
                    elif uvf.select_pol == "q":
                        s = sq
                    elif uvf.select_pol == "u":
                        s = su

                    ax = _a / np.sqrt(8 * np.log(2))
                    ay = _a / np.sqrt(8 * np.log(2))
                    if ax > psize:
                        I = s / (2 * np.pi * ax * ay)
                        gaussian_model = Gaussian2D(
                            amplitude=I,
                            x_mean=_l, y_mean=_m,
                            x_stddev=ax, y_stddev=ay,
                            theta=0
                        )

                        addimg = gaussian_model(xgrid, ygrid)

                    elif ax <= psize:
                        I = s / psize**2
                        loc = [
                            round(float((-_l + mapfov / 2) / psize)),
                            round(float((-_m + mapfov / 2) / psize))
                        ]

                        addimg = np.zeros(xgrid.shape)
                        addimg[loc[0], loc[1]] = I
                    image += addimg

            elif model == "delta":
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

                    _dtype = theta.dtype.names

                    _q = theta[idx_q]
                    _u = theta[idx_u]
                    _a = theta[idx_a]
                    _l = theta[idx_l]
                    _m = theta[idx_m]

                    has_i = idx_i in _dtype
                    has_f = idx_f in _dtype
                    has_r = idx_r in _dtype
                    has_t = idx_t in _dtype

                    if has_i:
                        _i = theta[idx_i]

                    if has_f:
                        _f = theta[idx_f]

                    if has_r:
                        _r = theta[idx_r]

                    if spectrum == "flat":
                        sq = _q
                        su = _u
                    elif spectrum == "spl":
                        raise NotImplementedError("To be updated.")
                    elif spectrum in ["cpl", "ssa"]:
                        raise NotImplementedError("To be updated.")

                    if uvf.select_pol == "p":
                        s = np.sqrt(sq**2 + su**2)
                    elif uvf.select_pol == "q":
                        s = sq
                    elif uvf.select_pol == "u":
                        s = su

                    ax = _a / np.sqrt(8 * np.log(2))
                    ay = _a / np.sqrt(8 * np.log(2))
                    if ax > psize:
                        I = s / (2 * np.pi * ax * ay)
                        gaussian_model = Gaussian2D(
                            amplitude=I,
                            x_mean=_l, y_mean=_m,
                            x_stddev=ax, y_stddev=ay,
                            theta=0
                        )

                        addimg = gaussian_model(xgrid, ygrid)

                    elif ax <= psize:
                        I = s / psize**2
                        loc = [
                            round(float((-_l + mapfov / 2) / psize)),
                            round(float((-_m + mapfov / 2) / psize))
                        ]

                        addimg = np.zeros(xgrid.shape)
                        addimg[loc[0], loc[1]] = I
                    image += addimg

        else:
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
                        s = _s
                    elif spectrum == "spl":
                        s = gv.functions.spl(freq_ref, freq, _s, _i)
                    elif spectrum in ["cpl", "ssa"]:
                        if i == 0:
                            if spectrum == "cpl":
                                s = gv.functions.cpl(
                                    freq, _s, _f, _i
                                )
                            else:
                                s = gv.functions.ssa(
                                    freq, _s, _f, _i
                                )
                        else:
                            mask_thick = round(float(theta[idx_t])) == 0

                            if mask_thick:
                                s = gv.functions.spl(
                                    freq_ref, freq,
                                    _s, _i
                                )
                            else:
                                if spectrum == "cpl":
                                    s = gv.functions.cpl(
                                        freq, _s, _f, _i
                                    )
                                else:
                                    s = gv.functions.ssa(
                                        freq, _s, _f, _i
                                    )
                    elif spectrum == "poly":
                        _b = theta[idx_b]
                        s = gv.functions.poly(freq_ref, freq, _s, _i, _b)

                    list_s.append(s)
                    list_a.append(_a)
                    list_l.append(_l)
                    list_m.append(_m)

                    ax = _a / np.sqrt(8 * np.log(2))
                    ay = _a / np.sqrt(8 * np.log(2))
                    if ax > psize:
                        I = s / (2 * np.pi * ax * ay)
                        gaussian_model = Gaussian2D(
                            amplitude=I,
                            x_mean=_l, y_mean=_m,
                            x_stddev=ax, y_stddev=ay,
                            theta=0
                        )

                        addimg = gaussian_model(xgrid, ygrid)

                    elif ax <= psize:
                        I = s / psize**2
                        loc = [
                            round(float((-_l + mapfov / 2) / psize)),
                            round(float((-_m + mapfov / 2) / psize))
                        ]

                        addimg = np.zeros(xgrid.shape)
                        addimg[loc[0], loc[1]] = I
                    image += addimg

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
                        s = _s
                    elif spectrum == "spl":
                        s = gv.functions.spl(freq_ref, freq, _s, _i)
                    elif spectrum in ["cpl", "ssa"]:
                        if i == 0:
                            if spectrum == "cpl":
                                s = gv.functions.cpl(
                                    freq, _s, _f, _i
                                )
                            else:
                                s = gv.functions.ssa(
                                    freq, _s, _f, _i
                                )
                        else:
                            mask_thick = round(float(theta[idx_t])) == 0

                            if mask_thick:
                                s = gv.functions.spl(
                                    freq_ref, freq,
                                    _s, _i
                                )
                            else:
                                if spectrum == "cpl":
                                    s = gv.functions.cpl(
                                        freq, _s, _f, _i
                                    )
                                else:
                                    s = gv.functions.ssa(
                                        freq, _s, _f, _i
                                    )
                    elif spectrum == "poly":
                        _b = theta[idx_b]
                        s = gv.functions.poly(freq_ref, freq, _s, _i, _b)

                    list_s.append(s)
                    list_a.append(0.0)
                    list_l.append(_l)
                    list_m.append(_m)

                    I = s / psize**2
                    loc = [
                        round(float((-_l + mapfov / 2) / psize)),
                        round(float((-_m + mapfov / 2) / psize))
                    ]

                    addimg = np.zeros(xgrid.shape)
                    addimg[loc[0], loc[1]] = I
                    image += addimg

        comp_prms = [
            list_s, list_l, list_m, list_a, list_a,
            [0 for i in range(nmod)], [0 for i in range(nmod)]
        ]
        comp_keys = [
            "FLUX", "DELTAX", "DELTAY", "MAJOR AX", "MINOR AX",
            "POSANGLE", "TYPE OBJ"
        ]
        uvf.component = gv.utils.structured_array(
            data=comp_prms,
            field=comp_keys,
            dtype=["f8" for i in range(len(comp_keys))]
        )

        self.image = np.flip(image.T, axis=0)

    def set_beamprms(self, uvf):
        uvc = uvf.set_uvcov(flatten=True, returned=True)
        uvw = uvf.uvw
        bprms = gv.utils.fit_beam(uvc=uvc, sig=None, uvw=uvw)
        self.bprms = bprms

    def set_imgprms(self, uvf, npix, mapfov):
        """
        set imaging parameters
        Args:
            uvf (python class): opened-fits file in uvf-class
            npix (int): the number of pixels in the map
            mapfov (float): map range // (-mapfov / 2, +mapfov / 2)
        Returns:
            imgprms (tuple): tuple of imaging parameters
        """
        psize = mapfov/npix
        self.mapfov = mapfov
        self.npix = npix
        self.psize = psize
        self.imgprms = (mapfov, npix, psize)
        return self.imgprms

def close_figure(fig):
    """
    close figure
    """
    plt.close(fig)
    plt.close("all")
    gc.collect()

def get_trcnidx(pol, model, spectrum, relmod, n):
    if pol:
        field = [rf"$S_{n}$"]
        nidx_ = 1
    else:
        if model == "gaussian":
            if spectrum == "flat":
                if relmod and n == 1:
                    nidx_ = 2
                    field = [rf"$S_{n}$", rf"$a_{n}$"]
                else:
                    nidx_ = 4
                    field = [
                        rf"$S_{n}$", rf"$a_{n}$", rf"$l_{n}$", rf"$m_{n}$"
                    ]

            elif spectrum == "spl":
                if relmod and n == 1:
                    nidx_ = 3
                    field = [rf"$S_{n}$", rf"$a_{n}$", rf"$\alpha_{n}$"]
                else:
                    nidx_ = 5
                    field = [
                        rf"$S_{n}$", rf"$a_{n}$", rf"$l_{n}$", rf"$m_{n}$",
                        rf"$\alpha_{n}$"
                    ]

            elif spectrum in ["cpl", "ssa"]:
                if relmod:
                    if n == 1:
                        nidx_ = 4
                        field = [
                            rf"$S_{n}$", rf"$a_{n}$",
                            rf"$\alpha_{n}$", rf"$\nu_{{\rm m,{n}}}$"
                        ]
                    else:
                        nidx_ = 7
                        field = [
                            rf"$S_{n}$", rf"$a_{n}$", rf"$l_{n}$", rf"$m_{n}$",
                            rf"$\alpha_{n}$", rf"$\nu_{{\rm m,{n}}}$"
                        ]
                else:
                    field = [
                        rf"$S_{n}$", rf"$a_{n}$", rf"$l_{n}$", rf"$m_{n}$",
                        rf"$\alpha_{n}$", rf"$\nu_{{\rm m,{n}}}$"
                    ]
                    if n == 1:
                        nidx_ = 6
                    else:
                        nidx_ = 7

            elif spectrum == "poly":
                if relmod and n == 1:
                    nidx_ = 4
                    field = [
                        rf"$S_{n}$", rf"$a_{n}$",
                        rf"$\alpha_{n}$", rf"$\beta_{n}$"
                    ]
                else:
                    nidx_ = 6
                    field = [
                        rf"$S_{n}$", rf"$a_{n}$", rf"$l_{n}$", rf"$m_{n}$",
                        rf"$\alpha_{n}$", rf"$\beta_{n}$"
                    ]

        elif model == "delta":
            if spectrum == "flat":
                if relmod and n == 1:
                    nidx_ = 1
                    field = [rf"$S_{n}$"]
                else:
                    nidx_ = 3
                    field = [rf"$S_{n}$", rf"$l_{n}$", rf"$m_{n}$"]

            elif spectrum == "spl":
                if relmod and n == 1:
                    nidx_ = 2
                    field = [rf"$S_{n}$", rf"$\alpha_{n}$"]
                else:
                    nidx_ = 4
                    field = [
                        rf"$S_{n}$", rf"$l_{n}$", rf"$m_{n}$",
                        rf"$\alpha_{n}$"
                    ]

            elif spectrum in ["cpl", "ssa"]:
                if relmod:
                    if n == 1:
                        nidx_ = 4
                        field = [
                            rf"$S_{n}$",
                            rf"$\alpha_{n}$", rf"$\nu_{{\rm m,{n}}}$"
                        ]
                    else:
                        nidx_ = 6
                        field = [
                            rf"$S_{n}$", rf"$l_{n}$", rf"$m_{n}$",
                            rf"$\alpha_{n}$", rf"$\nu_{{\rm m,{n}}}$"
                        ]
                else:
                    field = [
                        rf"$S_{n}$",
                        rf"$l_{n}$", rf"$m_{n}$",
                        rf"$\alpha_{n}$", rf"$\nu_{{\rm m,{n}}}$"
                    ]
                    if n == 1:
                        nidx_ = 5
                    else:
                        nidx_ = 6

            elif spectrum == "poly":
                if relmod and n == 1:
                    nidx_ = 3
                    field = [rf"$S_{n}$", rf"$\alpha_{n}$", rf"$\beta_{n}$"]
                else:
                    nidx_ = 5
                    field = [
                        rf"$S_{n}$", rf"$l_{n}$", rf"$m_{n}$",
                        rf"$\alpha_{n}$", rf"$\beta_{n}$"
                    ]
    return nidx_, field

def set_mapticks(ax, fov):
    if fov >= 2000:
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.yaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(20))
    elif 500 <= fov < 2000:
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.yaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_minor_locator(MultipleLocator(50))
    elif 80 <= fov < 500:
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(4))
        ax.yaxis.set_minor_locator(MultipleLocator(4))
    elif 40 <= fov < 80:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(2))
    elif 12 <= fov < 40:
        ax.xaxis.set_major_locator(MultipleLocator(5.0))
        ax.yaxis.set_major_locator(MultipleLocator(5.0))
        ax.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))
    elif 2 <= fov < 12:
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    elif 0.5 <= fov < 2.0:
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(0.10))
        ax.yaxis.set_major_locator(MultipleLocator(0.10))
        ax.xaxis.set_minor_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.02))
    return ax


colors = [
    cls.hsv_to_rgb((240/360,0.05,1.00)),    # whiteblue
    cls.hsv_to_rgb((240/360,0.91,0.86)),    # blue
    cls.hsv_to_rgb((266/360,0.88,0.90)),    # purple
    cls.hsv_to_rgb((355/360,0.95,0.89)),    # pink
    cls.hsv_to_rgb((0/360,0.79,0.90)),      # red
    cls.hsv_to_rgb((27/360,0.91,0.88)),     # orange
    cls.hsv_to_rgb((60/360,1.00,1.00))      # yellow
]

polcba = cls.LinearSegmentedColormap.from_list("fpmap", colors, gamma=2)
