
import os
import sys
import gc
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cls
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling.models import Gaussian2D
from astropy.time import Time as Ati
from astropy import units as u
from dynesty import plotting as dyplot
from dynesty.utils import quantile as dyquan

import gamvas

r2m = u.rad.to(u.mas)
d2m = u.deg.to(u.mas)
d2r = u.deg.to(u.rad)
m2d = u.mas.to(u.deg)
m2r = u.mas.to(u.rad)

class plotter:
    def __init__(self,
        mrng=False, npix=128, nmod=False, prms=False, freq_ref=False, freq=False,
        bmin=False, bmaj=False, bpa=False, bnom=False, source=False, date=False
    ):

        self.mrng = mrng
        self.npix = npix
        self.nmod = nmod
        self.prms = prms
        self.freq = freq
        self.freq_ref = freq_ref

        self.bmin = bmin
        self.bmaj = bmaj
        self.bpa = bpa
        self.psize = self.mrng / self.npix

        self.bnom = bnom

        axis_range = np.linspace(-mrng, +mrng, npix)
        xgrid, ygrid = np.meshgrid(-axis_range, +axis_range)
        self.xgrid = xgrid
        self.ygrid = ygrid

        self.source = source
        self.date = date

        self.fitset = None
        self.spectrum = None


    def set_beamprms(self):
        if not isinstance(self.bmin, bool):
            bmin = self.bmin
        else:
            bmin = 0.5

        if not isinstance(self.bmaj, bool):
            bmaj = self.bmaj
        else:
            bmaj = 0.5

        if not isinstance(self.bpa, bool):
            bpa = self.bpa
        else:
            bpa = 0.0
        self.bmin = bmin
        self.bmaj = bmaj
        self.bpa = bpa
        self.bnom = (bmin, bmaj, bpa)
        return self.bnom


    def draw_cgains(self,
        uvf, cgain1, cgain2, truth=None, plotimg=True,
        save_csv=False, save_path=False, save_name=False, save_form="png"
    ):
        data = uvf.data
        uant = np.unique(np.append(data["ant_name1"], data["ant_name2"]))
        freqs = uvf.data["freq"]
        ufreq = np.sort(np.unique(freqs))
        t1 = int(np.min(data["time"]))
        t2 = int(np.max(data["time"])+1)
        nfig = len(uant)//10
        nax = len(uant)%10
        if nax != 0:
            nfig += 1
        fsize = 16

        if save_csv:
            save_path_ = save_path + "plot_cgain/"
            gamvas.utils.mkdir(save_path_)
            if truth is not None:
                out_gain = pd.DataFrame([])
                for ngain, gain in enumerate(truth):
                    if ngain == 0:
                        out_gain["antenna"] = list(gain.keys())
                    out_gain[f"gain_{ufreq[ngain]:.1f}"] = list(gain.values())
                save_name_truth = save_name + "_truth.csv"
                out_gain.to_csv(save_path_ + save_name_truth)
            for nfreq, freq in enumerate(ufreq):
                time_ = data["time"][freqs == freq]
                ant1_ = data["ant_name1"][freqs == freq]
                ant2_ = data["ant_name2"][freqs == freq]
                cgain1_ = cgain1[freqs == freq]
                cgain2_ = cgain2[freqs == freq]
                out_csv = gamvas.utils.sarray(
                    data =[time_, ant1_, ant2_, cgain1_, cgain2_],
                    field=["time", "ant_name1", "ant_name2", "gain1", "gain2"],
                    dtype=["f8", "U32", "U32", "c8", "c8"]
                )
                out_csv = pd.DataFrame(out_csv)
                save_name_csv = save_name + f".{freq:.0f}.csv"
                out_csv.to_csv(save_path_ + save_name_csv)

        for nfreq, freq in enumerate(ufreq):
            data_ = data[freqs == freq]
            cgain1_ = cgain1[freqs == freq]
            cgain2_ = cgain2[freqs == freq]
            amplim =\
                [
                    0.9 * min(np.nanmin(np.abs(cgain1_)), np.nanmin(np.abs(cgain2_))),
                    1.1 * max(np.nanmax(np.abs(cgain1_)), np.nanmax(np.abs(cgain2_)))
                ]
            newdat = gamvas.utils.sarray(
                data =[data_["time"], data_["ant_name1"], data_["ant_name2"], cgain1_, cgain2_],
                field=["time", "ant_name1", "ant_name2", "cgain1", "cgain2"],
                dtype=["f8", "U32", "U32", "c8", "c8"])
            for i in range(nfig):
                if i != nfig-1:
                    nax = 10
                else:
                    if len(uant) == 10:
                        nax = 10
                    else:
                        nax = len(uant)%10

                fig_cgain, axes_cgain = plt.subplots(nax, 2, figsize=(fsize, fsize*8/16))
                for k in range(nax):
                    ax_cgamp = axes_cgain[k, 0]
                    ax_cgphs = axes_cgain[k, 1]
                    ax_cgamp.set_xlim(t1, t2)
                    ax_cgphs.set_xlim(t1, t2)
                    ax_cgamp.set_rasterized(True)
                    ax_cgphs.set_rasterized(True)
                    ax_cgamp.xaxis.set_major_locator(MultipleLocator(3.0))
                    ax_cgphs.xaxis.set_major_locator(MultipleLocator(3.0))
                    ax_cgamp.xaxis.set_minor_locator(MultipleLocator(1.0))
                    ax_cgphs.xaxis.set_minor_locator(MultipleLocator(1.0))
                    if k != nax-1:
                        ax_cgamp.tick_params(labelbottom=False)
                        ax_cgphs.tick_params(labelbottom=False)
                    if k == 0:
                        ax_cgamp.set_title("Gain Amplitude", fontsize=13, fontweight="bold")
                        ax_cgphs.set_title("Gain Phase (deg)", fontsize=13, fontweight="bold")
                    ant = uant[k+10*i]
                    mask1 = newdat["ant_name1"] == ant
                    mask2 = newdat["ant_name2"] == ant
                    newdat_ = newdat[mask1 | mask2]
                    mask1_ = newdat_["ant_name1"] == ant
                    mask2_ = newdat_["ant_name2"] == ant
                    ax_cgamp.scatter(newdat_["time"][mask1_], np.abs(newdat_["cgain1"][mask1_]), c="black", marker="o", s=12, label=f"{ant} ({freq:.1f} GHz)")
                    ax_cgamp.scatter(newdat_["time"][mask2_], np.abs(newdat_["cgain2"][mask2_]), c="black", marker="o", s=12)
                    ax_cgphs.scatter(newdat_["time"][mask1_], np.angle(newdat_["cgain1"][mask1_], deg=True), c="black", marker="o", s=12)
                    ax_cgphs.scatter(newdat_["time"][mask2_], np.angle(newdat_["cgain2"][mask2_].conj(), deg=True), c="black", marker="o", s=12)
                    if truth is not None:
                        ax_cgamp.axhline(y=truth[nfreq][ant], c="red", ls="--")
                    ax_cgamp.set_ylim(amplim)
                    ax_cgamp.legend()
                fig_cgain.supxlabel(r"$\rm Time~(hour)$", fontsize=15, fontweight="bold")
                fig_cgain.supylabel(r"$G_{\rm ant}$", fontsize=15, fontweight="bold")
                fig_cgain.tight_layout()
                if save_name:
                    save_path_ = save_path + "plot_cgain/"
                    gamvas.utils.mkdir(save_path_)
                    save_name_ = save_name + f".{freq:.0f}.v{i + 1}"
                    fig_cgain.savefig(f"{save_path_}" + f"{save_name_}.{save_form}", format=save_form, dpi=200)
                if plotimg:
                    plt.show()
                close_figure(fig_cgain)


    def draw_tplot(self,
        uvf, select=None, plotimg=True, show_title=False,
        save_path=False, save_name=False, save_form="png"
    ):
        dict_ants = dict(zip(uvf.tarr["name"], np.arange(len(uvf.tarr["name"]))))

        if save_path:
            gamvas.utils.mkdir(save_path)

        data = uvf.data

        if uvf.select == "mf":
            freqs = uvf.data["freq"]
            ufreq = np.unique(freqs)
        else:
            freqs = np.full(len(data), uvf.freq)
            ufreq = np.unique(freqs)

        uants = np.unique(np.append(data["ant_name1"], data["ant_name2"]))

        fsize = 10
        for nfreq, freq in enumerate(ufreq):
            yticks = []
            ytick_valid = []
            fig_tplot, ax_tplot = plt.subplots(1,1, figsize=(fsize, fsize * 8 / 16))
            ax_tplot.set_rasterized(True)
            for nant, ant in enumerate(uants):
                if nant == 0:
                    label = f"{freq:.1f} GHz"
                else:
                    label = None
                mask_freq = freqs == freq
                mask_ants = (data["ant_name1"] == ant) | (data["ant_name2"] == ant)
                data_ = data[mask_freq & mask_ants]
                ndat_ = len(data_)
                addidx = 1 + dict_ants[ant] + len(ufreq) * nfreq
                ax_tplot.scatter(data_["time"], np.zeros(ndat_) + addidx, c="black", marker="+", s=200, label=label)

                yticks.append(ant)
                ytick_valid.append(addidx)
            ax_tplot.scatter(data["time"], np.zeros(len(data)) + addidx + 1, c="white")

            ax_tplot.set_xlabel("Time (hour)", fontsize=20, fontweight="bold")
            ax_tplot.set_ylabel("Antenna", fontsize=20, fontweight="bold")
            ax_tplot.xaxis.set_major_locator(MultipleLocator(2.0))
            ax_tplot.xaxis.set_minor_locator(MultipleLocator(1.0))
            ax_tplot.tick_params("both", labelsize=15)
            ax_tplot.set_yticks(ytick_valid)
            ax_tplot.set_yticklabels(yticks)
            ax_tplot.grid(True, ls="--", axis="y")

            handles, labels = ax_tplot.get_legend_handles_labels()
            ax_tplot.legend(handles[::-1], labels[::-1], ncol=len(ufreq), loc="upper right", fontsize=13)

            fig_tplot.tight_layout()

            if show_title:
                if uvf.select == "mf":
                    title = f"{uvf.source} ({uvf.date}, Select={uvf.select.upper()})"
                else:
                    title = f"{uvf.source} ({uvf.date}, Select={uvf.select.upper()}, {uvf.freq:.2f} GHz)"
                fig_tplot.suptitle(title, fontsize=20, fontweight="bold")
            fig_tplot.tight_layout()

            if save_name:
                if uvf.select == "mf":
                    fig_tplot.savefig(f"{save_path}" + f"{save_name}.{freq:.0f}.{save_form}", format=save_form, dpi=300)
                else:
                    fig_tplot.savefig(f"{save_path}" + f"{save_name}.{save_form}", format=save_form, dpi=300)

            if plotimg:
                plt.show()

        close_figure(fig_tplot)


    def draw_radplot(self,
        uvf, select=None, plotvism=False, plotimg=True, plotsnr=False, show_title=False,
        save_path=False, save_name=False, save_form="png"
    ):
        if save_path:
            gamvas.utils.mkdir(save_path)
        data = uvf.data
        if uvf.select == "mf":
            clrs = ["tab:blue", "tab:green", "tab:olive", "tab:purple", "tab:orange", "tab:gray"]
        uu = data["u"]
        vv = data["v"]
        uvd = np.sqrt(uu**2 + vv**2)

        if select is None:
            vis = data["vis"]
            sig = data["sigma"]
        else:
            vis = data[f"vis_{select}"]
            sig = data[f"sigma_{select}"]

        vis = np.where(data["u"] < 0, vis.conj(), vis)
        amp = np.abs(vis)
        phs = np.angle(vis) * u.rad.to(u.deg)

        sig_a = sig
        sig_p = sig/np.abs(vis) * u.rad.to(u.deg)
        date = Ati(data["mjd"], format="mjd").iso[0][:10]

        fsize=16
        maxphs = 1.2*np.nanmax([np.abs(phs-sig_p), np.abs(phs+sig_p)])
        if maxphs >= 200:
            maxphs=200

        if plotsnr:
            fig_tplot, ax_radplot = plt.subplots(4, 1, figsize=(fsize, fsize * 10 / 16), sharex=True)
            ax_radplot_s = ax_radplot[3]
            ax_radplot_s.set_rasterized(True)
        else:
            fig_tplot, ax_radplot = plt.subplots(3, 1, figsize=(fsize, fsize*8/16), sharex=True)
        ax_radplot_a = ax_radplot[0]
        ax_radplot_p = ax_radplot[1]
        ax_radplot_e = ax_radplot[2]
        ax_radplot_a.set_rasterized(True)
        ax_radplot_p.set_rasterized(True)
        ax_radplot_e.set_rasterized(True)
        if uvf.select == "mf":
            ufreq = np.unique(uvf.data["freq"])
            for nfreq, freq in enumerate(ufreq):
                mfc = clrs[nfreq]
                mec = "black"
                cs = 6
                ct = 1
                mse = 6
                mss = 35

                uvd_ = uvd[uvf.data["freq"] == freq]
                amp_ = amp[uvf.data["freq"] == freq]
                phs_ = phs[uvf.data["freq"] == freq]
                sig_ = sig[uvf.data["freq"] == freq]
                sig_a_ = sig_a[uvf.data["freq"] == freq]
                sig_p_ = sig_p[uvf.data["freq"] == freq]
                ax_radplot_a.errorbar(uvd_/1e6, amp_, sig_a_,
                    marker="o", markersize=mse, ls="", mfc=mfc, mec=mec, c=mec,
                    capsize=cs, capthick=ct, zorder=1,
                    label=f"{freq:.1f}"
                )
                ax_radplot_p.errorbar(uvd_/1e6, phs_, sig_p_,
                    marker="o", markersize=mse, ls="", mfc=mfc, mec=mec, c=mec,
                    capsize=cs, capthick=ct, zorder=1
                )
                ax_radplot_e.scatter(uvd_/1e6, sig_,
                    marker="o", s=mss, fc=mfc, ec=mec
                )
                if plotsnr:
                    ax_radplot_s.scatter(uvd_/1e6, np.log10(amp_/sig_a_),
                        marker="o", s=mss, fc=mfc, ec=mec
                    )
        else:
            mfc = "black"
            mec = "dimgray"
            cs = 6
            ct = 1
            mse = 6
            mss = 35

            ax_radplot_a.errorbar(uvd/1e6, amp, sig_a,
                marker="o", markersize=mse, ls="", mfc=mfc, mec=mec, c=mec,
                capsize=cs, capthick=ct, zorder=1,
                label="obs"
            )
            ax_radplot_p.errorbar(uvd/1e6, phs, sig_p,
                marker="o", markersize=mse, ls="", mfc=mfc, mec=mec, c=mec,
                capsize=cs, capthick=ct, zorder=1
            )
            ax_radplot_e.scatter (uvd/1e6, sig,
                marker="o", s=mss, fc=mfc, ec=mec
            )
            if plotsnr:
                ax_radplot_s.scatter (uvd/1e6, np.log10(amp/sig_a),
                    marker="o", s=mss, fc=mfc, ec=mec
                )

        if plotvism:
            vism = np.where(data["u"] < 0, data["vism"].conj(), data["vism"])
            ampm = np.abs(vism)
            phsm = np.angle(vism) * u.rad.to(u.deg)
            ax_radplot_a.scatter(uvd/1e6, ampm, marker="s", s=7, c="red", zorder=2, label="model")
            ax_radplot_p.scatter(uvd/1e6, phsm, marker="s", s=7, c="red", zorder=2)
        ax_radplot_a.legend(ncol=10)

        ax_radplot_a.tick_params("both", labelsize=13)
        ax_radplot_p.tick_params("both", labelsize=13)
        ax_radplot_e.tick_params("both", labelsize=13)
        if plotsnr:
            ax_radplot_s.tick_params("both", labelsize=13)
        ax_radplot_p.set_ylim(-maxphs, +maxphs)

        ax_radplot_a.set_ylabel(r"$\rm Amplitude~(Jy)$", fontsize=17, fontweight="bold")
        ax_radplot_p.set_ylabel(r"$\rm Phase~(deg)$", fontsize=17, fontweight="bold")
        ax_radplot_e.set_ylabel(r"$\rm Sigma$", fontsize=17, fontweight="bold")

        if plotsnr:
            ax_radplot_s.set_ylabel(r"$\rm log_{10}\,(SNR)$", fontsize=17, fontweight="bold")
        fig_tplot.supxlabel(r"$\rm UV~radius~(10^{6}~\lambda)$", fontsize=20, fontweight="bold")

        if show_title:
            if uvf.select == "mf":
                title = f"{uvf.source} ({date}, Select={uvf.select.upper()})"
            else:
                title = f"{uvf.source} ({date}, Select={uvf.select.upper()}, {uvf.freq:.2f} GHz)"
            fig_tplot.suptitle(title, fontsize=20, fontweight="bold")
        fig_tplot.tight_layout()

        if save_name:
            fig_tplot.savefig(f"{save_path}" + f"{save_name}.{save_form}", format=save_form, dpi=300)

        if plotimg:
            plt.show()

        close_figure(fig_tplot)


    def draw_uvcover(self,
        uvf, plotimg=True, show_title=False,
        save_path=False, save_name=False, save_form="png"
    ):
        if save_path:
            gamvas.utils.mkdir(save_path)
        select = uvf.select.upper()
        data = uvf.data
        vis = data["vis"]

        amp1 = np.abs(vis)
        phs1 = np.angle(vis) * u.rad.to(u.deg)
        amp2 = np.abs(vis)
        phs2 = np.angle(vis) * u.rad.to(u.deg)
        uu = data["u"] / 1e6
        vv = data["v"] / 1e6
        date= Ati(data["mjd"], format="mjd").iso[0][:10]


        rng_a = 1.0 * np.max(np.abs(amp1))
        rng_p = 1.0 * np.max(np.abs(phs1))
        limv = 1.1 * max(np.max(np.abs(uu)), np.max(np.abs(vv)))

        fsize = 14
        fig_uvc, ax_uvc = plt.subplots(1, 2, figsize=(fsize,fsize*11/16))
        ax_amp, ax_phs = ax_uvc[0], ax_uvc[1]
        ax_amp.set_rasterized(True)
        ax_phs.set_rasterized(True)
        ax_amp.get_shared_x_axes().joined(ax_amp, ax_phs)
        ax_amp.get_shared_y_axes().joined(ax_amp, ax_phs)
        ax_amp.set_aspect("equal")
        ax_phs.set_aspect("equal")
        ax_amp.set_facecolor("gray")
        ax_phs.set_facecolor("gray")
        cmap1 = ax_amp.scatter(+uu, +vv, c=+amp1, s=30, ec="black", cmap="jet", vmin=0, vmax=+rng_a, zorder=2)
        cmap2 = ax_phs.scatter(+uu, +vv, c=+phs1, s=30, ec="black", cmap="bwr", vmin=-rng_p, vmax=+rng_p, zorder=2)
        cmap1 = ax_amp.scatter(-uu, -vv, c=+amp2, s=30, ec="black", cmap="jet", vmin=0, vmax=+rng_a, zorder=2)
        cmap2 = ax_phs.scatter(-uu, -vv, c=-phs2, s=30, ec="black", cmap="bwr", vmin=-rng_p, vmax=+rng_p, zorder=2)
        cbar1 = fig_uvc.colorbar(cmap1, ax=ax_amp, orientation="horizontal")
        cbar2 = fig_uvc.colorbar(cmap2, ax=ax_phs, orientation="horizontal")
        cbar1.set_label("Amplitude (Jy)", fontsize=15, fontweight="bold")
        cbar2.set_label("Phase (deg)", fontsize=15, fontweight="bold")
        ax_amp.set_xlim(+limv, -limv)
        ax_amp.set_ylim(-limv, +limv)
        ax_phs.set_xlim(+limv, -limv)
        ax_phs.set_ylim(-limv, +limv)
        ax_amp.tick_params("both", labelsize=15)
        ax_phs.tick_params("both", labelsize=15)
        ax_amp.grid(True)
        ax_phs.grid(True)
        ax_amp.set_ylabel(r"$\rm V~(10^{6}~\lambda)$", fontsize=20, fontweight="bold")
        ax_amp.set_xlabel(r"$\rm U~(10^{6}~\lambda)$", fontsize=20, fontweight="bold")
        ax_phs.set_xlabel(r"$\rm U~(10^{6}~\lambda)$", fontsize=20, fontweight="bold")

        if show_title:
            if uvf.select == "mf":
                title = f"{uvf.source} ({date}, Select={uvf.select.upper()})"
            else:
                title = f"{uvf.source} ({date}, Select={uvf.select.upper()}, {uvf.freq:.2f} GHz)"
            fig_uvc.suptitle (title, fontsize=20, fontweight="bold")
        fig_uvc.tight_layout()

        if save_name:
            fig_uvc.savefig(f"{save_path}" + f"{save_name}.{save_form}", format=save_form, dpi=300)

        if plotimg:
            plt.show()

        close_figure(fig_uvc)


    def draw_dirtymap(self,
        uvf, mrng=10, npix=128, uvw="natural",
        plot_resi=False, plotimg=True, show_title=False,
        save_path=False, save_name=False, save_form="png",
        returned_beam=False, returned_dirim=False
    ):
        if save_path:
            gamvas.utils.mkdir(save_path)
        data = uvf.data
        uu = data["u"]
        vv = data["v"]
        if plot_resi:
            vis = data["vis"] - data["vism"]
        else:
            vis = data["vis"]
        sig = data["sigma"]

        if uvw in ["n", "natural"]:
            wfn = 1 / sig**2
            weight = "n"
        else:
            wfn = np.ones(vis.shape)
            weight = "u"

        uut = np.tile(uu, (npix,1)).astype(float)
        vvt = np.tile(vv, (npix,1)).astype(float)
        fn = np.tile(wfn, (npix,1)).astype(float)

        uvdir = np.zeros((npix, npix))
        uvbim = np.zeros((npix, npix))

        xlist = -np.linspace(-uvf.mrng.value, uvf.mrng.value, npix) * u.mas.to(u.rad)
        ygrid, xgrid = np.meshgrid(xlist, xlist)
        for i in range(npix):
            xset = xgrid[i, :].reshape(-1, 1)
            yset = ygrid[i, :].reshape(-1, 1)
            uvbim[i, :] = np.mean(fn * np.cos(-2*np.pi*(xset*uut + yset*vvt)), axis=1)

        uvd = np.sqrt(uu**2 + vv**2)
        for i in range(npix):
            xset = xgrid[i, :].reshape(-1, 1)
            yset = ygrid[i, :].reshape(-1, 1)
            expo = np.exp(-2j*np.pi * ((xset*uut + yset*vvt)))
            uvdir[i, :] = np.mean(fn * (vis.real*expo.real - vis.imag*expo.imag), axis=1)

        if abs(np.max(uvdir)) < abs(np.min(uvdir)):
            nloc = np.where(uvdir < 0)
            ploc = np.where(uvdir > 0)
            uvdir[nloc] = -uvdir[nloc]
            uvdir[ploc] = -uvdir[ploc]

        scale = np.max(uvbim)/uu.size
        uvbim /= scale*uu.size
        uvdir /= scale*uu.size

        max_bim = np.max(uvbim)
        max_dir = np.max(uvdir)

        uvf.xgrid = xgrid
        uvf.ygrid = ygrid
        uvf.uvbim = uvbim
        if plot_resi:
            uvf.resid = uvdir
        else:
            uvf.dirty = uvdir

        bimtitle = rf"$\rm beam~pattern~(\sigma \approx {np.std(uvbim):.3f})$"
        if plot_resi:
            dirtitle = rf"$\rm residual~ (map~range : {np.min(uvdir):.3f} <-> {np.max(uvdir):.3f})$"
        else:
            dirtitle = r"$\rm dirty~map~(I_{\rm peak}$="+f"{max_dir:.2f} Jy)"

        xgrid *= u.rad.to(u.mas)
        ygrid *= u.rad.to(u.mas)
        fsize = 15
        fig_dirmap, ax_map = plt.subplots(1, 2, figsize=(fsize, fsize * 10 / 16))
        ax_bim = ax_map[0]
        ax_dir = ax_map[1]
        ax_bim.set_rasterized(True)
        ax_dir.set_rasterized(True)
        ax_bim.set_aspect("equal")
        ax_dir.set_aspect("equal")
        cmap_bim = ax_bim.contourf(xgrid, ygrid, uvbim, cmap="gist_heat", levels=101, vmin=np.min(uvbim), vmax=np.max(uvbim))
        cmap_dir = ax_dir.contourf(xgrid, ygrid, uvdir, cmap="gist_heat", levels=101, vmin=np.min(uvdir), vmax=np.max(uvdir))
        cbar_bim = fig_dirmap.colorbar(cmap_bim, ax=ax_bim, orientation="horizontal")
        cbar_dir = fig_dirmap.colorbar(cmap_dir, ax=ax_dir, orientation="horizontal")
        cbar_bim.set_label("Beam Response", fontsize=15, fontweight="bold")
        cbar_dir.set_label("Amplitude (Jy/beam)", fontsize=15, fontweight="bold")
        ax_bim.set_xlabel(f"Relative R.A ({uvf.mrng.unit})", fontsize=17, fontweight="bold")
        ax_dir.set_xlabel(f"Relative R.A ({uvf.mrng.unit})", fontsize=17, fontweight="bold")
        ax_bim.set_ylabel(f"Relative DeC ({uvf.mrng.unit})", fontsize=17, fontweight="bold")
        ax_dir.set_ylabel(f"Relative DeC ({uvf.mrng.unit})", fontsize=17, fontweight="bold")
        ax_bim.set_title(bimtitle, fontsize=17, fontweight="bold")
        ax_dir.set_title(dirtitle, fontsize=17, fontweight="bold")
        ax_bim.tick_params("both", labelsize=15)
        ax_dir.tick_params("both", labelsize=15)
        ax_dir.invert_xaxis()
        ax_bim.invert_xaxis()
        ax_dir.get_shared_x_axes().joined(ax_dir, ax_bim)
        ax_dir.get_shared_y_axes().joined(ax_dir, ax_bim)
        cbar_bim.ax.xaxis.set_major_locator(MultipleLocator(0.2))
        cbar_bim.ax.xaxis.set_minor_locator(MultipleLocator(0.1))

        if show_title:
            if uvf.select == "mf":
                title = f"{uvf.source} ({uvf.date}, Select={uvf.select.upper()})"
            else:
                title = f"{uvf.source} ({uvf.date}, Select={uvf.select.upper()}, {uvf.freq:.2f} GHz)"
            fig_dirmap.suptitle(title, fontsize=20, fontweight="bold")
        fig_dirmap.tight_layout()

        if save_name:
            fig_dirmap.savefig(f"{save_path}" + f"{save_name}.{save_form}", format=save_form, dpi=300)

        if plotimg:
            plt.show()

        if returned_beam:
            close_figure(fig_dirmap)
            return (xgrid, ygrid, uvbim)

        if returned_dirim:
            close_figure(fig_dirmap)
            return (xgrid, ygrid, uvdir)

        close_figure(fig_dirmap)


    def draw_trplot(self,
        result=None, pol=False, weight=None, nmod=None, ifsingle=True, spectrum="spl", set_spectrum=True,
        fontsize=15, fontweight="bold", save_path=False, save_name=False, save_form="png"
    ):
        """
        draw trace plot
        Arguments:
            result (array): results of dynesty model-fit
            nmod  (int): the number of models
            save_path (str): path to saving directory
            save_name (str): name of figure to be saved
        """
        if not self.spectrum is None:
            spectrum = self.spectrum

        save_path = save_path + "plot_tr/"
        gamvas.utils.mkdir(save_path)
        if pol:
            nidx = 0
        else:
            nidx = 1
        sidx = [0]
        for i in range(nmod):
            n = i + 1
            if pol:
                field = [r"$S_{%s}$"%(n)]
                nidx_ = 1
            else:
                if ifsingle:
                    if n == 1:
                        field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n)]
                        nidx_ = 2
                    else:
                        field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n)]
                        nidx_ = 4

                else:
                    if set_spectrum:
                        if spectrum in ["spl"]:
                            if n == 1:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$\alpha_{%s}$"%(n)]
                                nidx_ = 4
                            else:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n), r"$\alpha_{%s}$"%(n)]
                                nidx_ = 7

                        elif spectrum in ["cpl", "ssa"]:
                            if n == 1:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$\alpha_{%s}$"%(n), r"$\nu_{\rm m,%s}$"%(n)]
                                nidx_ = 4
                            else:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n), r"$\alpha_{%s}$"%(n), r"$\nu_{\rm m,%s}$"%(n)]
                                nidx_ = 7
                    else:
                        if n == 1:
                            field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n)]
                            nidx_ = 2
                        else:
                            field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n)]
                            nidx_ = 4

            if set_spectrum:
                if i == 0:
                    add = 0
                else:
                    add = 1
                    sidx.append(nidx)
                # mask_spectrum = int(np.round(np.median(result.samples[:,nidx]))) == 0
                ql, qm, qh = dyquan(result.samples[:,nidx], (0.025, 0.500, 0.975), weights=result.importance_weights())
                mask_spectrum = int(np.round(qm)) == 0
                if mask_spectrum:
                    field = field[:-1]
            else:
                add= 0

            # draw trace-plot
            fig_tr, axes_tr = dyplot.traceplot(
                result, truth_color='black', show_titles=True, trace_cmap='viridis',
                dims=list(range(nidx+add, nidx+add+len(field))),
                labels=field, label_kwargs={"fontsize":fontsize, "fontweight":fontweight}
            )

            for nax1 in np.arange(axes_tr.shape[0]):
                for nax2 in np.arange(axes_tr.shape[1]):
                    axes_tr[nax1,nax2].set_rasterized(True)

            fig_tr.tight_layout()
            fig_tr.savefig(f"{save_path}" + f"{save_name}.mod{n}.{save_form}", format=save_form, dpi=300)
            close_figure(fig_tr)
            nidx += nidx_

        if not pol:
            # draw trace-plot of the number of model and spectrum
            slabel = ["nmod"] + [f"{i+2}_spectrum" for i in range(len(sidx)-1)]
            fig_tr, axes_tr = dyplot.traceplot(
                result, truth_color='black', show_titles=True, trace_cmap='viridis',
                dims=sidx,
                labels=slabel, label_kwargs={"fontsize":fontsize, "fontweight":fontweight}
            )

            for nax1 in np.arange(axes_tr.shape[0]):
                for nax2 in np.arange(axes_tr.shape[1]):
                    axes_tr[nax1,nax2].set_rasterized(True)

            fig_tr.tight_layout()
            fig_tr.savefig(f"{save_path}" + f"{save_name}.mod.spectrum.{save_form}", format=save_form, dpi=300)
            close_figure(fig_tr)


    def draw_cnplot(self,
        result=None, pol=False, nmod=None, ifsingle=True, spectrum="spl", set_spectrum=True,
        fontsize=15, fontweight="bold", save_path=False, save_name=False, save_form="png"
    ):
        """
        draw corner plot
        Arguments:
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
        gamvas.utils.mkdir(save_path)
        if pol:
            nidx = 0
        else:
            nidx = 1
        sidx = [0]
        for i in range(nmod):
            n = i + 1
            if pol:
                field = [r"$S_{%s}$"%(n)]
                nidx_ = 1
            else:
                if ifsingle:
                    if n == 1:
                        field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n)]
                        nidx_ = 2
                    else:
                        field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n)]
                        nidx_ = 4

                else:
                    if set_spectrum:
                        if spectrum in ["spl"]:
                            if n == 1:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$\alpha_{%s}$"%(n)]
                                nidx_ = 4
                            else:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n), r"$\alpha_{%s}$"%(n)]
                                nidx_ = 7

                        elif spectrum in ["cpl", "ssa"]:
                            if n == 1:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$\alpha_{%s}$"%(n), r"$\nu_{\rm m,%s}$"%(n)]
                                nidx_ = 4
                            else:
                                field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n), r"$\alpha_{%s}$"%(n), r"$\nu_{\rm m,%s}$"%(n)]
                                nidx_ = 7
                    else:
                        if n == 1:
                            field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n)]
                            nidx_ = 2
                        else:
                            field = [r"$S_{%s}$"%(n), r"$a_{%s}$"%(n), r"$l_{%s}$"%(n), r"$m_{%s}$"%(n)]
                            nidx_ = 4

            if set_spectrum:
                if i == 0:
                    add = 0
                else:
                    add = 1
                    sidx.append(nidx)
                    # mask_spectrum = int(np.round(np.median(result.samples[:,nidx]))) == 0
                    ql, qm, qh = dyquan(result.samples[:,nidx], (0.025, 0.500, 0.975), weights=result.importance_weights())
                    mask_spectrum = int(np.round(qm)) == 0
                    if mask_spectrum:
                        field = field[:-1]
            else:
                add = 0

            # draw corner-plot
            fig_cn, axes_cn = dyplot.cornerplot(
                result, truth_color='black', show_titles=True,
                dims=list(range(nidx+add, nidx+add+len(field))),
                labels=field, label_kwargs={"fontsize":fontsize, "fontweight":fontweight}
            )

            for nax1 in np.arange(axes_cn.shape[0]):
                for nax2 in np.arange(axes_cn.shape[1]):
                    axes_cn[nax1,nax2].set_rasterized(True)
            fig_cn.tight_layout()
            fig_cn.savefig(f"{save_path}" + f"{save_name}.mod{n}.{save_form}", format=save_form, dpi=300)
            close_figure(fig_cn)
            nidx += nidx_

        if not pol:
            # draw corner-plot of the number of model and spectrum
            slabel = ["nmod"] + [f"{i+2}_spectrum" for i in range(len(sidx)-1)]
            fig_cn, axes_cn = dyplot.cornerplot(
                result, truth_color='black', show_titles=True,
                dims=sidx, labels=slabel, label_kwargs={"fontsize":fontsize, "fontweight":fontweight}
            )
            for nax1 in np.arange(axes_cn.shape[0]):
                for nax2 in np.arange(axes_cn.shape[1]):
                    axes_cn[nax1,nax2].set_rasterized(True)
            fig_cn.tight_layout()
            fig_cn.savefig(f"{save_path}" + f"{save_name}.mod.spectrum.{save_form}", format=save_form, dpi=300)
            close_figure(fig_cn)


    def draw_closure(self,
        type="clphs", fsize=9, model=False, plotimg=False,
        save_img=False, save_path=None, save_name=None, save_form="png"
    ):
        """
        draw closure quantities
        Arguments:
            type (str) : type of closure quantity ('amp', 'phs')
            fsize (float): figure size
            plotimg (bool): toggle option if plot the result
        """
        if save_path is not None:
            save_path_ = save_path + f"/plot_{type}/"
            gamvas.utils.mkdir(save_path_)

        if type in ["amp", "clamp"]:
            clq_obs = self.clq_obs[0]

            if model:
                clq_mod = self.clq_mod[0]
            ufreq = np.unique(clq_obs["freq"])
            nfreq = len(ufreq)

            for i in range(nfreq):
                clq_obs_1 = clq_obs[clq_obs["freq"] == ufreq[i]]
                if len(clq_obs_1) == 0:
                    print("WARNING: Empty closure amplitude data! Skip drawing closure amplitude.")
                    continue

                if model:
                    clq_mod_1 = clq_mod[clq_obs["freq"] == ufreq[i]]

                uquad = np.unique(clq_obs_1["quadra"])
                nquad = len(uquad)
                nfig = nquad // 10
                nax = nquad % 10

                if nax != 0:
                    nfig += 1

                for j in range(nfig):
                    if j != nfig - 1:
                        nax = 10
                    else:
                        if int(nfig*10) == nquad:
                            nax = 10
                        else:
                            nax = nquad % 10

                    fig_clamp, ax_clamp = plt.subplots(nax, 1, figsize=(fsize, fsize * (nax + 1)/10), sharex=True)

                    for k in range(nax):
                        clq_obs_2 = clq_obs_1[clq_obs_1["quadra"] == uquad[10 * j + k]]
                        if model:
                            clq_mod_2 = clq_mod_1[clq_obs_1["quadra"] == uquad[10 * j + k]]
                        if nax > 1:
                            plot_ax = ax_clamp[k]
                        else:
                            plot_ax = ax_clamp
                        plot_ax.set_rasterized(True)
                        plot_ax.errorbar(
                            clq_obs_2["time"], np.log(clq_obs_2["clamp"]), clq_obs_2["sigma_clamp"],
                            marker="o", markersize=6, c="black", ls="", mfc="black", mec="dimgray",
                            capsize=6, capthick=1, zorder=1,
                            label=f"{uquad[10*j+k]} ({ufreq[i]:.1f} GHz)"
                        )
                        if model:
                            plot_ax.plot(
                                clq_obs_2["time"], np.log(clq_mod_2),
                                marker="o", markersize=4, c="red", ls=":", zorder=2
                            )
                        plot_ax.xaxis.set_major_locator(MultipleLocator(2.0))
                        plot_ax.xaxis.set_minor_locator(MultipleLocator(1.0))
                        plot_ax.legend(fontsize=10)
                        if k != nax-1:
                            plot_ax.tick_params("both", labelsize=10, labelbottom=False)
                        else:
                            plot_ax.tick_params("both", labelsize=10)
                    fig_clamp.supylabel(r"${\rm ln}(A_{\rm C})$", fontsize=12, fontweight="bold")
                    fig_clamp.supxlabel("Time (hour)", fontsize=12, fontweight="bold")
                    fig_clamp.tight_layout()
                    if save_img:
                        if save_path is not None and save_name is not None:
                            if nfreq == 1:
                                save_name_ = save_name + f".{j + 1}"
                            else:
                                save_name_ = save_name + f".{ufreq[i]:.0f}.{j + 1}"
                            fig_clamp.savefig(f"{save_path_}" + f"{save_name_}.{save_form}", format=save_form, dpi=200)
                        else:
                            raise Exception("'save_path' and/or 'save_name' not given (closure amplitude).")
                    if plotimg:
                        plt.show()
                    close_figure(fig_clamp)

        if type in ["phs", "clphs"]:
            r2d = u.rad.to(u.deg)
            clq_obs = self.clq_obs[1]
            if model:
                clq_mod = self.clq_mod[1]
            ufreq = np.unique(clq_obs["freq"])
            nfreq = len(ufreq)
            for i in range(nfreq):
                clq_obs_1 = clq_obs[clq_obs["freq"] == ufreq[i]]
                if len(clq_obs_1) == 0:
                    print("WARNING: Empty closure phase data! Skip drawing closure phase.")
                    continue

                if model:
                    clq_mod_1 = clq_mod[clq_obs["freq"] == ufreq[i]]
                utria = np.unique(clq_obs_1["triangle"])
                ntria = len(utria)
                nfig = ntria // 10
                nax = ntria % 10
                if nax != 0:
                    nfig += 1
                for j in range(nfig):
                    if j != nfig - 1:
                        nax = 10
                    else:
                        if int(nfig*10) == ntria:
                            nax = 10
                        else:
                            nax = ntria % 10
                    fig_clphs, ax_clphs = plt.subplots(nax, 1, figsize=(fsize, fsize * (nax + 1) / 10), sharex=True)
                    for k in range(nax):
                        clq_obs_2 = clq_obs_1[clq_obs_1["triangle"] == utria[10 * j + k]]
                        if model:
                            clq_mod_2 = clq_mod_1[clq_obs_1["triangle"] == utria[10 * j + k]]
                        if nax > 1:
                            plot_ax = ax_clphs[k]
                        else:
                            plot_ax = ax_clphs
                        plot_ax.set_rasterized(True)
                        plot_ax.errorbar(
                            clq_obs_2["time"], r2d * np.angle(np.exp(1j * clq_obs_2["clphs"])), r2d * clq_obs_2["sigma_clphs"],
                            marker="o", markersize=6, c="black", ls="", mfc="black", mec="dimgray",
                            capsize=6, capthick=1, zorder=1,
                            label=f"{utria[10 * j + k]} ({ufreq[i]:.1f} GHz)")
                        if model:
                            plot_ax.plot(
                                clq_obs_2["time"], r2d * np.angle(np.exp(1j * clq_mod_2)),
                                marker="o", markersize=4, c="red", ls=":", zorder=2
                            )
                        plot_ax.xaxis.set_major_locator(MultipleLocator(2.0))
                        plot_ax.xaxis.set_minor_locator(MultipleLocator(1.0))
                        plot_ax.legend(fontsize=10)
                        if k != nax-1:
                            plot_ax.tick_params("both", labelsize=10, labelbottom=False)
                        else:
                            plot_ax.tick_params("both", labelsize=10)
                    fig_clphs.supylabel(r"$\phi_{\rm C}~{\rm (deg)}$", fontsize=12, fontweight="bold")
                    fig_clphs.supxlabel("Time (hour)", fontsize=12, fontweight="bold")
                    fig_clphs.tight_layout()
                    if save_img:
                        if save_path is not None and save_name is not None:
                            save_name_ = save_name + f".{ufreq[i]:.0f}.{j + 1}"
                            fig_clphs.savefig(f"{save_path_}" + f"{save_name_}.{save_form}", format=save_form, dpi=200)
                        else:
                            raise Exception("'save_path' and/or 'save_name' not given (closure phase).")
                    if plotimg:
                        plt.show()
                    close_figure(fig_clphs)


    def draw_dterm(self,
        uvf, uvw="natural", plotimg=True, show_title=False,
        save_path=False, save_name=False, save_form="png"
    ):
        data = uvf.data

        rr = data["vis_rr"]
        ll = data["vis_ll"]
        rl = data["vis_rl"]
        lr = data["vis_lr"]

        rlrr = rl / rr
        lrrr = lr / rr
        rlll = rl / ll
        lrll = lr / ll
        axlim = 1.2 * max(
            np.max(np.abs(rlrr)),
            np.max(np.abs(lrrr)),
            np.max(np.abs(rlll)),
            np.max(np.abs(lrll))
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


    def set_imgprms(self,
        uvf, npix, mrng
    ):
        """
        set imaging parameters
        Arguments:
            uvf (python class): opened-fits file in uvf-class
            npix (int): the number of pixels in the map
            mrng (float): map range // (-map_ragne, +mrng)
        Returns:
            imgprms (tuple): tuple of imaging parameters
        """
        psize = 2*mrng/npix
        self.mrng = mrng
        self.npix = npix
        self.psize = psize
        self.imprms = (mrng, npix, psize)
        return self.imprms


    def generate_image(self,
        uvf, pol=False, freq_ref=False, freq=False, prms=False, pprms=False, ifsingle=True, set_spectrum=False, spectrum="spl"
    ):
        """
        generate intensity image (Jy/beam)
        Arguments:
            uvf (python class): opened-fits file in uvf-class
            freq_ref (float): reference frequency
            freq (float): frequency to plot by considering estimated spectrum
            prms (array): set of model parameters
        Returns:
            image (2D-array): image (n by n) // not convolved
        """
        if isinstance(prms, bool) and not prms : prms=self.prms

        if isinstance(freq_ref, bool) and not freq_ref:
            freq_ref = self.freq_ref
        if isinstance(freq, bool) and not freq:
            freq = self.freq

        nmod = int(np.round(prms["nmod"]))
        xgrid = uvf.xgrid
        ygrid = uvf.ygrid
        image = np.zeros(xgrid.shape)
        mrng = np.max(xgrid)
        psize = mrng / xgrid.shape[0] * 2

        for i in range(nmod):
            if pol:
                S = pprms[f"{i + 1}_S"]
                prm_a_ = prms[f"{i + 1}_a"]
                if i == 0:
                    prm_l_ = 0
                    prm_m_ = 0
                else:
                    prm_l_ = prms[f"{i + 1}_l"]
                    prm_m_ = prms[f"{i + 1}_m"]
            else:
                prm_S_ = prms[f"{i + 1}_S"]
                prm_a_ = prms[f"{i + 1}_a"]
                if ifsingle:
                    if i == 0:
                        prm_l_ = 0
                        prm_m_ = 0
                    else:
                        prm_l_ = prms[f"{i + 1}_l"]
                        prm_m_ = prms[f"{i + 1}_m"]
                    S = prm_S_
                else:
                    if set_spectrum:
                        prm_i_ = prms[f"{i + 1}_alpha"]
                        if spectrum in ["spl"]:
                            if i == 0:
                                prm_l_ = 0
                                prm_m_ = 0
                            else:
                                prm_l_ = prms[f"{i + 1}_l"]
                                prm_m_ = prms[f"{i + 1}_m"]
                            S = gamvas.functions.S_spl(freq_ref, freq, prm_S_, prm_i_)
                        elif spectrum in ["cpl", "ssa"]:
                            prm_f_ = prms[f"{i + 1}_freq"]
                            if i == 0:
                                prm_l_ = 0
                                prm_m_ = 0
                            else:
                                mask_sindex = int(np.round(prms[f"{i + 1}_thick"])) == 0
                                prm_l_ = prms[f"{i + 1}_l"]
                                prm_m_ = prms[f"{i + 1}_m"]
                            if spectrum in ["cpl"]:
                                if i == 0:
                                    S = gamvas.functions.S_cpl(freq, prm_S_, prm_f_, prm_i_)
                                else:
                                    if mask_sindex:
                                        S = gamvas.functions.S_spl(freq_ref, freq, prm_S_, prm_i_)
                                    else:
                                        S = gamvas.functions.S_cpl(freq, prm_S_, prm_f_, prm_i_)
                            elif spectrum in ["ssa"]:
                                if i == 0:
                                    S = gamvas.functions.SSA(freq, prm_S_, prm_f_, prm_i_)
                                else:
                                    if mask_sindex:
                                        S = gamvas.functions.S_spl(freq_ref, freq, prm_S_, prm_i_)
                                    else:
                                        S = gamvas.functions.SSA(freq, prm_S_, prm_f_, prm_i_)
                    else:
                        if i == 0:
                            prm_l_ = 0
                            prm_m_ = 0
                        else:
                            prm_l_ = prms[f"{i + 1}_l"]
                            prm_m_ = prms[f"{i + 1}_m"]
                        S = prm_S_

            ax = prm_a_ / np.sqrt(8 * np.log(2))
            ay = prm_a_ / np.sqrt(8 * np.log(2))
            if ax > psize:
                I = S / (2 * np.pi * ax * ay)
                gaussian_model = Gaussian2D(amplitude=I, x_mean=prm_l_, y_mean=prm_m_, x_stddev=ax, y_stddev=ay, theta=0)
                addimg = gaussian_model(xgrid, ygrid)
            elif ax <= psize:
                I = S/psize**2
                loc = [int(np.round((-prm_l_ + mrng) / psize)), int(np.round((-prm_m_ + mrng) / psize))]
                addimg = np.zeros(xgrid.shape)
                addimg[loc[0], loc[1]] = I
            image += addimg
        self.image = image


    def convolve_image(self,
        uvf, npix, image=False, bnom=False
    ):
        """
        convolve the generated intensity image with restoring beam (Jy/beam)
        Arguments:
            uvf (python class): opened-fits file in uvf-class
            npix (int): the number of pixels in the map
            image (2D-array): generated image
            bnom (tuple): beam parameters (beam minor, beam major, beam position angle)
        Returns:
            conv_image (2D-array): convolved image (n by n)
        """
        if isinstance(bnom, bool) and not bnom:
            bnom = self.bnom
        bmin, bmaj, bpa = bnom
        bsize = np.pi*bmin*bmaj/4/np.log(2)

        mrng, npix, psize = self.set_imgprms(uvf=uvf, npix=npix, mrng=uvf.mrng.value)

        kmin = bmin/np.sqrt(8*np.log(2))/psize
        kmaj = bmaj/np.sqrt(8*np.log(2))/psize

        gauss_kernel = Gaussian2DKernel(x_stddev=kmin, y_stddev=kmaj, theta=(bpa+90)*u.deg)
        conv_image = convolve(image, gauss_kernel, normalize_kernel=True)
        conv_image = bsize * conv_image
        return conv_image


    def draw_image(self,
        uvf, pol=False, returned=False, bnom=None, freq_ref=None, freq=None, genlevels=False, npix=128, mindr=3, minlev=0.01, maxlev=0.99, step=2, fsize=8,
        contourw=0.3, mintick_map=0.5, majtick_map=2.5, mintick_cb=0.2, majtick_cb=1.0, ifsingle=True, set_spectrum=True, xlim=False, ylim=False,
        save_img=False, save_path=False, save_name=False, save_form="png",
        plotimg=True, plot_resi=False, addnoise=False, outfig=False, title=None, show_title=False
    ):
        """
        draw final image
        Arguments:
            uvf (python class): opened-fits file in uvf-class
            bnom (tuple): beam parameters (beam minor, beam major, beam position angle)
            freq_ref (float): reference frequency
            freq (float): frequency to plot by considering estimated spectrum
            levels (list): contour levels to draw
            minlev (float): starting contour level in fraction (0.01 == starting at 1% of the peak)
            maxlev (float): final contour level in fraction (0.99 == starting at 99% of the peak)
            step (int): step size of the contour
            fsize (flaot): figure size
            mintick_map (flaot): size of minor tick label in the intensity map
            majtick_map (flaot): size of major tick label in the intensity map
            mintick_cb (flaot): size of minor tick label in the color bar
            majtick_cb (flaot): size of major tick label in the color bar
            save_img (bool): toggole option if save the final image
            save_path (str or bool): if set, this will be the path of the saving image
            save_name (str or bool): if set, this will be the name of saving image
            plotimg (bool): if True, the final image will be plotted
            npix (int): the number of pixels in the map
            addnoise (bool): if True, the noise in the residual map will be added to the final image
        """
        if save_path:
            gamvas.utils.mkdir(save_path)
        if freq_ref is None:
            freq_ref = self.freq_ref
        if freq is None:
            freq = self.freq
        if bnom is None:
            bnom = self.bnom

        bmin, bmaj, bpa = bnom

        mrng = uvf.mrng.value
        npix = npix

        prms = self.prms
        nmod = int(np.round(prms["nmod"]))
        if pol:
            pprms = self.pprms
        else:
            pprms = False

        if plot_resi and save_name:
            # reconstruct residual maps
            save_name_ = save_name.replace("img", "resimap")
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
                self.draw_dirtymap(uvf=uvf, mrng=mrng, npix=npix, uvw=uvf.uvw, plot_resi=True, plotimg=False, save_path=save_path, save_name=save_name_, save_form=save_form)

        # reconstruct model+residual map
        self.draw_dirtymap(uvf=uvf, mrng=mrng, npix=npix, uvw=uvf.uvw, plot_resi=addnoise, plotimg=False)
        self.generate_image(uvf=uvf, pol=pol, freq_ref=freq_ref, freq=freq, prms=prms, pprms=pprms, ifsingle=ifsingle, set_spectrum=set_spectrum, spectrum=self.spectrum)
        image = self.image.copy()
        image = self.convolve_image(uvf=uvf, npix=npix, image=image, bnom=bnom)
        if addnoise:
            resim = uvf.resid
            image += resim

            rms = gamvas.utils.cal_rms(resim)
            if rms == 0:
                rms = 0.01 * np.max(image)

            levels = [mindr * rms]
            if mindr*rms < np.max(image)*np.sqrt(2):
                while levels[-1] < np.max(image):
                    levels.append(levels[-1]*np.sqrt(2))
            if pol:
                levels_n = [-mindr * rms]
                if mindr * rms < np.abs(np.min(image)) * np.sqrt(2):
                    while np.abs(levels_n[-1]) < np.abs(np.min(image)):
                        levels_n.append(levels_n[-1] * np.sqrt(2))


        if genlevels:
            levels = [maxlev]
            while levels[-1] > minlev:
                levels.append(levels[-1] / step)
            levels = np.sort(np.max(image) * np.array(levels))

        xgrid = uvf.xgrid
        ygrid = uvf.ygrid
        self.restore_img = image

        xu, yu = 1/37, 1/41
        fig_map = plt.figure(figsize=(fsize*(yu/xu),fsize))
        ax_map = fig_map.add_axes([5*xu, 9*yu, 30*xu, 30*yu])
        ax_cba = fig_map.add_axes([5*xu, 3*yu, 30*xu, 1*yu])
        ax_map.set_rasterized(True)
        ax_cba.set_rasterized(True)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("Relative R.A (mas)", fontsize=15, fontweight="bold")
        ax_map.set_ylabel("Relative Dec (mas)", fontsize=15, fontweight="bold")
        ax_map.set_xlim(-mrng, +mrng)
        ax_map.set_ylim(-mrng, +mrng)

        if show_title and not title is None:
            fig_map.suptitle(title, fontweight="bold")

        ax_map.contour (xgrid, ygrid, image, levels=levels, colors="lightgrey", linewidths=contourw)
        if pol:
            ax_map.contour (xgrid, ygrid, -image, levels=np.abs(levels_n), colors="red", linewidths=contourw, linestyles="--")
        cb_map = ax_map.contourf(xgrid, ygrid, image, levels=101, vmin=0, vmax=np.max(image), cmap='gist_heat')
        cb = fig_map.colorbar(cb_map, cax=ax_cba, orientation="horizontal")
        cb.set_label("Intensity (Jy/beam)", fontsize=15, fontweight="bold")
        ax_map.tick_params("both", labelsize=12)
        ax_cba.tick_params("x", labelsize=12)
        if mrng >= 6:
            ax_map.xaxis.set_major_locator(MultipleLocator(5.0))
            ax_map.yaxis.set_major_locator(MultipleLocator(5.0))
        else:
            ax_map.xaxis.set_major_locator(MultipleLocator(2.0))
            ax_map.yaxis.set_major_locator(MultipleLocator(2.0))
        ax_map.xaxis.set_minor_locator(MultipleLocator(1.0))
        ax_map.yaxis.set_minor_locator(MultipleLocator(1.0))

        if np.max(np.abs(image)) > 10:
            ax_cba.xaxis.set_major_locator(MultipleLocator(5.0))
            ax_cba.xaxis.set_minor_locator(MultipleLocator(1.0))
        if np.max(np.abs(image)) <= 10:
            ax_cba.xaxis.set_major_locator(MultipleLocator(1.0))
            ax_cba.xaxis.set_minor_locator(MultipleLocator(0.1))
        if np.max(np.abs(image)) <= 2:
            ax_cba.xaxis.set_major_locator(MultipleLocator(0.5))
            ax_cba.xaxis.set_minor_locator(MultipleLocator(0.1))
        if np.max(np.abs(image)) <= 0.5:
            ax_cba.xaxis.set_major_locator(MultipleLocator(0.10))
            ax_cba.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax_map.invert_xaxis()
        beam = patches.Ellipse((+0.9*mrng-bmaj/2, -0.9*mrng+bmaj/2),
                                bmin, bmaj, angle=-bpa, fc='grey', ec='yellow', lw=1.0)
        ax_map.add_patch(beam)

        for i in range(nmod):
            if i == 0:
                ra, dec = 0, 0
            else:
                ra, dec = prms[f"{i + 1}_l"], prms[f"{i + 1}_m"]
            a = prms[f"{i + 1}_a"]
            Gmodel = patches.Ellipse((ra, dec), a, a, angle=0, fc="none", ec="cyan", lw=1.0)
            stick1 = patches.ConnectionPatch(xyA=(ra -a / 2, dec), xyB=(ra +a / 2, dec), coordsA="data", color="cyan", lw=1.0)
            stick2 = patches.ConnectionPatch(xyA=(ra, dec -a / 2), xyB=(ra, dec +a / 2), coordsA="data", color="cyan", lw=1.0)
            ax_map.add_patch(Gmodel)
            ax_map.add_patch(stick1)
            ax_map.add_patch(stick2)

        if save_path and save_name:
            fig_map.savefig(f"{save_path}" + f"{save_name}.{save_form}", format=save_form, dpi=300)
        if plotimg:
            plt.show()

        if returned:
            close_figure(fig_map)
            return (image, resim)
        if outfig:
            close_figure(fig_map)
            return (fig_map, ax_map)
        close_figure(fig_map)


    def draw_fits_image(self,
        uvf, select="i", rms=None, xlim=False, ylim=False, cmap_snr_i=3, cmap_snr_p=3,
        fsize=6, contourw=0.3, pagap=30, plotimg=True, show_title=False,
        save_path=False, save_name=False, save_form="png"
    ):
        if select.lower() == "i":
            image = uvf.fits_image_vi
            if rms is None:
                rms = gamvas.utils.cal_rms(image)
            cntr = gamvas.utils.make_cntr(image, rms=rms)
            cmap = "gist_heat"
            vmin = cmap_snr_i*rms
            vmax = np.max(image)
        elif select.lower() == "q":
            image = uvf.fits_image_q
            if rms is None:
                rms = gamvas.utils.cal_rms(image)
            cntr = gamvas.utils.make_cntr(image, rms=rms)
            cmap = "gist_heat"
            peak_min = np.nanmin(image)
            peak_max = np.nanmax(image)
            if np.abs(peak_min) > np.abs(peak_max):
                vmin = peak_min
                vmax = cmap_snr_p*rms
            else:
                vmin = cmap_snr_p*rms
                vmax = peak_max
        elif select.lower() == "u":
            image = uvf.fits_image_u
            if rms is None:
                rms = gamvas.utils.cal_rms(image)
            cntr = gamvas.utils.make_cntr(image, rms=rms)
            cmap = "gist_heat"
            peak_min = np.nanmin(image)
            peak_max = np.nanmax(image)
            if np.abs(peak_min) > np.abs(peak_max):
                vmin = peak_min
                vmax = cmap_snr_p*rms
            else:
                vmin = cmap_snr_p*rms
                vmax = peak_max
        elif select.lower() == "p":
            image = uvf.fits_image_vi
            if rms is None:
                rms = gamvas.utils.cal_rms(image)
            cntr = gamvas.utils.make_cntr(image, rms=rms)
            cmap = "gist_heat"
            vmin = cmap_snr_p*rms
            vmax = np.nanmax(image)
            image_p = uvf.fits_image_vp * 1e3
            image_f = uvf.fits_image_vfp * 1e2
            rms_p = uvf.fits_image_rms_p * 1e3
            rms_f = rms_p/rms/1e3
            cmap_p = polcba
            vmin_p = cmap_snr_p * rms_p
            vmax_p = np.nanmax(image_p)

            if vmin_p > 3 * vmax_p:
                vmin_p = rms_p

            if np.isnan(vmax_p):
                vmax_p = 100 * rms_p

            vmin_f = cmap_snr_p*rms_f
            vmax_f = 100
            pa_x = uvf.fits_image_evpa_x
            pa_y = uvf.fits_image_evpa_y
            pa_set = uvf.fits_image_evpa_set
        elif select.lower() == "rr":
            image = uvf.fits_image_rr
            rms = uvf.fits_imgrms_rr
            cntr = uvf.fits_imgcntr_rr
            cmap = "gist_heat"
            vmin = cmap_snr_i*rms
            vmax = np.nanmax(image)
        elif select.lower() == "ll":
            image = uvf.fits_image_ll
            rms = uvf.fits_imgrms_ll
            cntr = uvf.fits_imgcntr_ll
            cmap = "gist_heat"
            vmin = cmap_snr_i*rms
            vmax = np.nanmax(image)

        ra = uvf.fits_grid_ra*u.deg.to(u.mas)
        dec = uvf.fits_grid_dec*u.deg.to(u.mas)
        norm_i = mpl.colors.Normalize(vmin=np.abs(vmin), vmax=np.abs(vmax))
        colormapping_i = cm.ScalarMappable(norm=norm_i, cmap=cmap)
        if select.lower() != "p":
            fig_fits, ax_imap = plt.subplots(1, 1, figsize=(fsize*4/3, fsize))
            ax_imap.set_rasterized(True)
        else:
            fig_fits, axes = plt.subplots(1, 2, figsize=(fsize*2.5, fsize))
            ax_pmap = axes[0] ; ax_pmap.set_rasterized(True)
            ax_fmap = axes[1] ; ax_fmap.set_rasterized(True)
            cmap_p = polcba
            norm_p = mpl.colors.LogNorm(vmin=np.abs(vmin_p), vmax=np.abs(vmax_p))
            colormapping_p = cm.ScalarMappable(norm=norm_p, cmap=cmap_p)
            cmap_f = "terrain_r"
            cmap_f = "GnBu"
            norm_f = mpl.colors.LogNorm(vmin=np.abs(vmin_f), vmax=np.abs(vmax_f))
            colormapping_f = cm.ScalarMappable(norm=norm_f, cmap=cmap_f)
        if select.lower() == "p":
            ax_pmap.set_aspect("equal")
            ax_pmap.contour(ra, dec, image, levels=cntr[0], colors="black", linewidths=contourw)
            ax_pmap.pcolor (ra, dec, np.abs(image_p), norm=norm_p, cmap=cmap_p)
            ax_pmap.tick_params(labelsize=15, right=True, top=True)
            cbar_p = fig_fits.colorbar(colormapping_p, ax=ax_pmap, orientation="vertical")
            cbar_p.set_label(r"$I_{\rm p}~{\rm (mJy/beam)}$", fontsize=15, fontweight="bold")
            ax_pmap.set_xlabel("Relative R.A (mas)", fontsize=20, fontweight="bold")
            ax_pmap.quiver(
                ra[::pagap, ::pagap], dec [::pagap, ::pagap],
                pa_x[::pagap, ::pagap], pa_y[::pagap, ::pagap], **pa_set, zorder=2)

            ax_fmap.set_aspect("equal")
            ax_fmap.contour(ra, dec, image, levels=cntr[0], colors="black", linewidths=contourw)
            ax_fmap.pcolor (ra, dec, np.abs(image_f), norm=norm_f, cmap=cmap_f)
            ax_fmap.tick_params(labelsize=15, right=True, top=True)
            cbar_f = fig_fits.colorbar(colormapping_f, ax=ax_fmap, orientation="vertical")
            cbar_f.set_label(r"$m_{\rm p}~{(\%)}$", fontsize=15, fontweight="bold")
            ax_fmap.set_xlabel("Relative R.A (mas)", fontsize=20, fontweight="bold")
            ax_fmap.quiver(
                ra[::pagap, ::pagap], dec [::pagap, ::pagap],
                pa_x[::pagap, ::pagap], pa_y[::pagap, ::pagap], **pa_set, zorder=2)
        elif select.lower() in ["i", "rr", "ll"]:
            ax_imap.set_aspect("equal")
            ax_imap.contour(ra, dec, image, levels=cntr[0], colors="lightgrey", linewidths=contourw)
            ax_imap.pcolor (ra, dec, np.abs(image), norm=norm_i, cmap=cmap)
            ax_imap.tick_params(labelsize=15, right=True, top=True)
            cbar_i = fig_fits.colorbar(colormapping_i, ax=ax_imap, orientation="vertical")
            cbar_i.set_label(r"$I_{I}~{\rm (Jy/beam)}$", fontsize=15, fontweight="bold")
            ax_imap.set_xlabel("Relative R.A (mas)", fontsize=20, fontweight="bold")
            # rect_patch = patches.Rectangle((-4, -4), 8, 8, linewidth=2, edgecolor='cyan', facecolor="none", ls="--")
            # ax_imap.add_patch(rect_patch)

        if select.lower() != "p":
            ec = "yellow"
        else:
            ec = "red"

        if show_title:
            fig_fits.suptitle(
                f"{uvf.fits_source}  |  Intsru. : {uvf.fits_intrum}\nDate : {uvf.fits_date}  |  {uvf.fits_freq:.3f} GHz",
                fontsize=15, fontweight="bold"
            )

        if not type(xlim) in [type([]), type(())] or\
            not type(ylim) in [type([]), type(())]:
                maxlim = np.abs(np.max(ra))
                xlim = [-maxlim, +maxlim]
                ylim = [-maxlim, +maxlim]

        beam =\
            patches.Ellipse(
                (+0.9 * xlim[1] - uvf.fits_bmaj / 2, 0.9 * ylim[0] + uvf.fits_bmaj / 2),
                uvf.fits_bmin,
                uvf.fits_bmaj,
                angle=-uvf.fits_bpa,
                fc='grey',
                ec=ec,
                lw=1.0
            )
        if select.lower() != "p":
            ax_imap.set_xlim(xlim[0], xlim[1])
            ax_imap.set_ylim(ylim[0], ylim[1])
            ax_imap.invert_xaxis()
            ax_imap.add_patch(beam)
        else:
            ax_pmap.set_xlim(xlim[0], xlim[1])
            ax_pmap.set_ylim(ylim[0], ylim[1])
            ax_fmap.set_xlim(xlim[0], xlim[1])
            ax_fmap.set_ylim(ylim[0], ylim[1])
            ax_pmap.invert_xaxis()
            ax_fmap.invert_xaxis()
            ax_pmap.add_patch(beam)
        fig_fits.supylabel("Relative Dec (mas)", fontsize=20, fontweight="bold")

        fig_fits.tight_layout()
        if all([save_path, save_name]):
            fig_fits.savefig(f"{save_path}" + f"{save_name}.{save_form}", format=save_form, dpi=500)
        if plotimg:
            plt.show()
        close_figure(fig_fits)


def close_figure(fig):
    """
    close figure
    """
    plt.close(fig)
    plt.close('all')
    gc.collect()

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
