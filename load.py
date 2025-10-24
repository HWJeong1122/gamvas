
import gc
import os
import sys
import copy
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import itertools as it
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.special import comb
from sklearn.cluster import DBSCAN as dbs
from astropy import constants as C
from astropy import units as u
from astropy.time import Time as Ati
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

import gamvas

cr = (1.00, 0.25, 0.25)
cg = (0.10, 0.90, 0.10)
cb = (0.25, 0.25, 1.00)

nan = np.nan
r2m = u.rad.to(u.mas)
d2m = u.deg.to(u.mas)
d2r = u.deg.to(u.rad)
m2d = u.mas.to(u.deg)
m2r = u.mas.to(u.rad)


class open_fits:
    """
    Load the fits file and extract the information
    Attributes:
        path (str): path to the fits file
        file (str): fits file name
        npixel1 (int): number of pixels in the x-axis
        npixel2 (int): number of pixels in the y-axis
        bmin (float): minor axis of the beam
        bmaj (float): major axis of the beam
        bpa (float): position angle of the beam
        fluxmax (float): maximum intensity of the image
        fluxmin (float): minimum intensity of the image
        noise (float): theoretical thermal noise
        freq (float): observing frequency
        stokes (int): Stokes parameter information
        date (str): observation date
        data (2D-array): uv-fits data
        units (str): units of the data
        select (str): polarization type
        select_if (str): number(s) of IFs
        uvinfo (bool): mask if uv-fits information is loaded
        mrng (float): maximum range of the image
        save_path (str): path to save the image
        save_file (str): file name to save the image
        xlim (list): x-axis limits
        ylim (list): y-axis limits
        source (str): source name
        fits_image (2D-array): reconstructed fits image
        pangle (bool): mask if parallactic angle is updated
        load_uvf (bool): toggle option if load uv-fits
    """
    def __init__(self,
                 path="", file="", npixel1=False, npixel2=False,
                 bmin=False, bmaj=False, bpa=False, fluxmax=False,
                 fluxmin=False, noise=False, freq=False, stokes=None,
                 date=None, data=None, units=None, model=None,
                 fits_model=None, select="i", select_if="all",
                 mrng=10*u.mas, load_uvf=False, load_fits=False):
        self.path = path
        self.file = file
        self.npixel1 = npixel1
        self.npixel2 = npixel2
        self.npixel = None

        self.bmin = bmin
        self.bmaj = bmaj
        self.bpa = bpa
        self.fluxmax = fluxmax
        self.fluxmin = fluxmin
        self.noise = noise
        self.freq = freq

        self.stokes = stokes
        self.date = date
        self.data = data
        self.units = units
        self.model = model
        self.fits_model = fits_model

        self.select = select
        self.select_if = select_if
        self.uvinfo = False
        self.mrng = mrng

        self.save_path = None
        self.save_file = None
        self.xlim = None
        self.ylim = None
        self.source = False
        self.fits_image = False
        self.pangle = False

        self.ploter = gamvas.plotting.plotter()
        self.modeling = gamvas.modeling.modeling()

        if load_uvf:
            self.load_uvf()
        if load_fits:
            self.load_fits()

    def load_fits(self,
                  file=False, align_index=0, align_model=False, units="deg",
                  prt=True, contour_snr=3):
        """
        Load fits image file and extract the information
            Arguments:
                file (str): fits file name
                align_index (int): index of the model for position align
                align_model (bool): toggle option if align fits models
                units (str): units of the model
                prt (bool): toggle option if print the information
                contour_snr (float): signal-to-noise ratio for the contour
        """
        if file:
            self.file = file
        uname = self.file.upper()
        if uname.endswith(".FITS"):
            fits_file = fits.open(self.path + self.file)
        elif uname.endswith(".UVF"):
            fits_file = fits.open(self.path + self.file[:-4] + ".fits")
        elif uname.endswith(".UVFITS"):
            fits_file = fits.open(self.path + self.file[:-7] + ".fits")
        h1 = fits_file["PRIMARY"]
        h2 = fits_file["AIPS CC"]

        self.fits_file = fits_file

        self.fits_ra = h1.header["CRVAL1"]
        self.fits_dec = h1.header["CRVAL2"]

        # units in GHz
        self.fits_freq = h1.header["CRVAL3"] / 1e9

        self.fits_date = h1.header["DATE-OBS"]

        # CRPIX=+1 to +4: I, Q, U, V
        self.fits_stokes = int(h1.header["CRVAL4"])

        self.fits_name = h1.header["OBJECT"]

        # recontructred image from the models
        self.fits_image = h1.data[0, 0, :, :]

        # information on CLEAN or Gaussian models
        self.fits_model = h2.data
        # self.fits_model = h1.header["NITER"]

        # min and max intensities of the image [Jy/beam]
        self.fits_fmin = h1.header["DATAMIN"]
        self.fits_fmax = h1.header["DATAMAX"]

        # synthesis beam parameters
        self.fits_bmin = h1.header["BMIN"]
        self.fits_bmaj = h1.header["BMAJ"]
        self.fits_bpa = h1.header["BPA"]

        # theoretically estimated noise (!= image rms)
        self.fits_noise = h1.header["NOISE"]

        # image rms
        self.fits_image_rms = gamvas.utils.cal_rms(self.fits_image)

        # image contours
        self.fits_imgcntr = gamvas.utils.make_cntr(
            self.fits_image,
            contour_snr=contour_snr)

        # the number of pixel (1-D)
        self.fits_npix = self.fits_image.shape[0]

        # pixel size
        self.fits_psize = np.abs(h1.header["CDELT2"])

        self.fits_range_ra = np.arange(
            +self.fits_psize * self.fits_npix / 2,
            -self.fits_psize * self.fits_npix / 2,
            -self.fits_psize,
            dtype="f4")

        self.fits_range_dec = np.arange(
            -self.fits_psize * self.fits_npix / 2,
            +self.fits_psize * self.fits_npix / 2,
            +self.fits_psize,
            dtype="f4")

        self.fits_grid_ra, self.fits_grid_dec =\
            np.meshgrid(self.fits_range_ra, self.fits_range_dec)

        if self.fits_stokes in [-1, 1]:
            self.fits_model_i = self.fits_model
            self.fits_image_vi = self.fits_image
            self.fits_image_rms_i = self.fits_image_rms
            self.fits_image_vpeak_i = np.max(self.fits_image)
            self.fits_clean_vflux_i = np.sum(self.fits_model["FLUX"])

            sig_peak, sig_tot, sig_size =\
                cal_clean_error(
                    s_peak=np.max(self.fits_image_vi),
                    s_tot=self.fits_clean_vflux_i,
                    sig_rms=self.fits_image_rms_i
                )

            self.fits_image_dpeak_i = 0.1 * self.fits_image_vpeak_i

            self.fits_image_di = np.ones((self.fits_npix, self.fits_npix)) \
                * (0.1 * self.fits_image_vi)

            self.fits_clean_dflux_i = 0.1 * self.fits_clean_vflux_i
            self.fits_imgcntr_i = self.fits_imgcntr
        elif self.fits_stokes == 2:
            peaks = [np.min(self.fits_image), np.max(self.fits_image)]

            self.fits_model_q = self.fits_model
            self.fits_image_vq = self.fits_image
            self.fits_image_rms_q = self.fits_image_rms
            self.fits_image_vpeak_q = np.max(self.fits_image)
            self.fits_clean_vflux_q = np.sum(self.fits_model["FLUX"])

            sig_peak, sig_tot, sig_size = cal_clean_error(
                s_peak=np.max(self.fits_image_vq),
                s_tot=self.fits_clean_vflux_q,
                sig_rms=self.fits_image_rms_q)

            self.fits_image_dpeak_q = 0.1 * np.abs(self.fits_image_vpeak_q)

            self.fits_image_dq =\
                np.ones((self.fits_npix, self.fits_npix)) *\
                (0.1 * np.abs(self.fits_image_vq))

            self.fits_clean_dflux_q = 0.1 * np.abs(self.fits_clean_vflux_q)
            self.fits_imgcntr_q = self.fits_imgcntr

        elif self.fits_stokes == 3:
            peaks = [np.min(self.fits_image), np.max(self.fits_image)]

            self.fits_model_u = self.fits_model
            self.fits_image_vu = self.fits_image
            self.fits_image_rms_u = self.fits_image_rms
            self.fits_image_vpeak_u = np.max(self.fits_image)
            self.fits_clean_vflux_u = np.sum(self.fits_model["FLUX"])

            sig_peak, sig_tot, sig_size = cal_clean_error(
                s_peak=np.max(self.fits_image_vu),
                s_tot=self.fits_clean_vflux_u,
                sig_rms=self.fits_image_rms_u)

            self.fits_image_dpeak_u = 0.1 * np.abs(self.fits_image_vpeak_u)

            self.fits_image_du = np.ones((self.fits_npix, self.fits_npix)) \
                * (0.1 * np.abs(self.fits_image_vu))

            self.fits_clean_dflux_u = 0.1 * np.abs(self.fits_clean_vflux_u)
            self.fits_imgcntr_u = self.fits_imgcntr

        elif self.fits_stokes == 4:
            peaks = [np.min(self.fits_image), np.max(self.fits_image)]

            self.fits_model_v = self.fits_model
            self.fits_image_vv = self.fits_image
            self.fits_image_rms_v = self.fits_image_rms
            self.fits_image_vpeak_v = np.max(self.fits_image)
            self.fits_clean_vflux_v = np.sum(self.fits_model["FLUX"])

            sig_peak, sig_tot, sig_size = cal_clean_error(
                s_peak=np.max(self.fits_image_vv),
                s_tot=self.fits_clean_vflux_v,
                sig_rms=self.fits_image_rms_v)

            self.fits_image_dpeak_v = 0.1 * np.abs(self.fits_image_vpeak_v)
            self.fits_image_dv = np.ones((self.fits_npix, self.fits_npix)) \
                * (0.1 * np.abs(self.fits_image_vv))
            self.fits_clean_dflux_v = 0.1 * np.abs(self.fits_clean_vflux_v)
            self.fits_imgcntr_v = self.fits_imgcntr

        if self.fits_model["MAJOR AX"][0] == 0:
            modty = "CLEAN"
        if self.fits_model["MAJOR AX"][0] != 0:
            modty = "Gaussian"

        if align_model:
            self.fits_model["DELTAX"] -= self.fits_model["DELTAX"][align_index]
            self.fits_model["DELTAY"] -= self.fits_model["DELTAY"][align_index]

        if units == "deg":
            self.fits_model = self.fits_model
        elif units == "mas":
            self.fits_model["DELTAX"] *= u.deg.to(u.mas)
            self.fits_model["DELTAY"] *= u.deg.to(u.mas)
            self.fits_model["MAJOR AX"] *= u.deg.to(u.mas)
            self.fits_model["MINOR AX"] *= u.deg.to(u.mas)

        if self.fits_stokes in [-1, 1] and prt:
            out1 = self.fits_name
            out2 = self.fits_date
            out3 = self.fits_freq
            out4 = np.sum(self.fits_model["FLUX"])
            out_txt =\
                f"\nloading fits file.."\
                f"  {out1} ({out2}, {out3:.3f} GHz, {out4:.2f} Jy)"
            print(out_txt)

        if self.uvinfo:
            uvu = self.data["u"]
            uvv = self.data["v"]
            uvr = np.sqrt(uvu**2 + uvv**2)

            major = self.fits_model["MAJOR AX"]
            minor = self.fits_model["MINOR AX"]

            mod_S = self.fits_model["FLUX"]
            mod_a = u.deg.to(u.mas) * (2 * major + 1 * minor) / 3
            mod_l = u.deg.to(u.mas) * self.fits_model["DELTAX"]
            mod_m = u.deg.to(u.mas) * self.fits_model["DELTAY"]

            vism = np.zeros(len(uvr), dtype="c8")
            for i in range(self.fits_Nmodel):
                vism += gamvas.functions.gvis(
                    (uvu, uvv), mod_S[i], mod_a[i], mod_l[i], mod_m[i])

            vism_s = []
            if self.fits_stokes == 1:
                self.vism_i = vism
                vism_s.append(vism)
            if self.fits_stokes == 2:
                self.vism_q = vism
                vism_s.append(vism)
            if self.fits_stokes == 3:
                vism_s.append(vism)
            if self.fits_stokes == 4:
                vism_s.append(vism)

            self.vism = vism_s

    def load_uvf(self,
                 select="i", select_if="all", uvw="natural", mrng=None,
                 uvave="none", scanlen=600, gaptime=60, d=0.0, m=0.0,
                 dosyscal=True, doscatter=False, set_clq=True, set_pang=False,
                 minclq=True, prt=True):
        """
        Load uv-fits file and extract the information
            Arguments:
                select (str): polarization type
                select_if (str): number(s) of IFs
                uvw (str): uv-weighting // u:uniform, n:natural
                uvave (int): averaging time [sec]
                scanlen (int): scan length [sec]
                doscatter (bool): toggle option if to compute sigma
                                  from standard deviation
                set_clq (bool): toggle option if to set closure quantities
                set_pang (bool): toggle option if to set parallactic angle
                prt (bool): toggle option if to print the information
        """
        self.avgtime = uvave
        self.gaptime = gaptime
        self.scanlen = scanlen
        self.uvw = uvw
        self.uvinfo = True
        self.select_if = select_if
        self.minclq = minclq
        if mrng is not None:
            self.mrng = mrng

        self.d = d
        self.m = m

        if not os.path.isfile(self.path + self.file):
            out_txt =\
                f"File not found: {self.path + self.file}"
            raise Exception(out_txt)

        uvf_file = fits.open(self.path + self.file)
        self.uvf_file = uvf_file

        h1 = uvf_file["PRIMARY"]
        h2 = uvf_file["AIPS FQ"]
        h3 = uvf_file["AIPS AN"]
        freq = h1.header["CRVAL4"] / 1e9
        freq0 = h1.header["CRVAL4"] / 1e9
        stokes = int(h1.header["CRVAL3"])

        data1 = h1.data
        data2 = h2.data
        data3 = h3.data

        if not self.source:
            self.source = h1.header["OBJECT"]

        if not self.freq:
            self.freq = freq

        self.ra = h1.header["CRVAL6"]
        self.dec = h1.header["CRVAL7"]
        self.nvis = h1.header["GCOUNT"]
        self.no_if = h2.header["NO_IF"]
        self.refGST = h3.header["GSTIA0"]
        self.f2w = (C.c / (self.freq * u.GHz)).to(u.m).value

        if isinstance(self.select_if, (list, tuple)):
            ifs = np.array(self.select_if)
        elif self.select_if == "all":
            ifs = np.arange(self.no_if)
        else:
            ifs = np.array([self.select_if]).astype(int) - 1
        self.no_if = ifs.size

        list_uu = ["UU---SIN", "UU--", "UU"]
        list_vv = ["VV---SIN", "VV--", "VV"]
        list_ww = ["WW---SIN", "WW--", "WW"]
        for i in range(len(list_uu)):
            if list_uu[i] in data1.dtype.names:
                idx_uu = list_uu[i]
        for i in range(len(list_vv)):
            if list_vv[i] in data1.dtype.names:
                idx_vv = list_vv[i]
        for i in range(len(list_ww)):
            if list_ww[i] in data1.dtype.names:
                idx_ww = list_ww[i]
        uu = data1[idx_uu] * self.freq * 1e9
        vv = data1[idx_vv] * self.freq * 1e9
        ww = data1[idx_ww] * self.freq * 1e9

        DATE = data1["DATE"].astype(np.float64)
        _DATE = data1["_DATE"].astype(np.float64)
        date = Ati(DATE + _DATE, format="jd").iso
        mjd = Ati(DATE + _DATE, format="jd").mjd
        min_jd = int(np.min(DATE + _DATE) - 2400000.5)
        time = (DATE + _DATE - 2400000.5 - min_jd) * 24

        ant_ = data3.field(0)
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

        self.ant_dict_name2num = dict(zip(ant_, anum))
        self.ant_dict_num2name =\
            {val: key for key, val in self.ant_dict_name2num.items()}

        mjd = Ati(date, format="iso").mjd
        year = Ati(date, format="iso").byear

        tnames = uvf_file["AIPS AN"].data["ANNAME"]
        tnums = uvf_file["AIPS AN"].data["NOSTA"] - 1
        xyz = np.real(uvf_file["AIPS AN"].data["STABXYZ"])
        try:
            sefdr = np.real(uvf_file["AIPS AN"].data["SEFD"])
            sefdl = np.real(uvf_file["AIPS AN"].data["SEFD"])
        except KeyError:
            sefdr = np.zeros(len(tnames))
            sefdl = np.zeros(len(tnames))

        fr_par = np.zeros(len(tnames))
        fr_el = np.zeros(len(tnames))
        fr_off = np.zeros(len(tnames))
        dr = np.zeros(len(tnames)) + 1j * np.zeros(len(tnames))
        dl = np.zeros(len(tnames)) + 1j * np.zeros(len(tnames))

        tsets =\
            [
                tnames, xyz[:, 0], xyz[:, 1], xyz[:, 2],
                sefdr, sefdl, dr, dl,
                fr_par, fr_el, fr_off
            ]

        theads =\
            [
                "name", "x", "y", "z",
                "sefd_r", "sefd_l", "d_r", "d_l",
                "fr_par", "fr_el", "fr_off"
            ]

        ttypes =\
            [
                "U32", "f8", "f8", "f8",
                "f8", "f8", "c16", "c16",
                "f8", "f8", "f8"
            ]

        tncol = len(tnames)
        tarr =\
            np.empty(
                tncol,
                dtype=[(theads[i], ttypes[i]) for i in range(len(tsets))]
            )

        for i in range(len(tsets)):
            tarr[theads[i]] = tsets[i]

        tarr["name"] =\
            np.array(
                list(map(
                    lambda x: x.replace(" ", ""),
                    tarr["name"]
                ))
            )

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
            ** weight<=0 indicates that the visibility measurement is flagged
               and that the values may not be in any way meaningful
        """
        if int(h1.header["CRVAL3"]) == -8:
            self.select = "YX"
        if int(h1.header["CRVAL3"]) == -7:
            self.select = "XY"
        if int(h1.header["CRVAL3"]) == -6:
            self.select = "YY"
        if int(h1.header["CRVAL3"]) == -5:
            self.select = "XX"
        if int(h1.header["CRVAL3"]) == -4:
            self.select = "LR"
        if int(h1.header["CRVAL3"]) == -3:
            self.select = "RL"
        if int(h1.header["CRVAL3"]) == -2:
            self.select = "LL"
        if int(h1.header["CRVAL3"]) == -1:
            self.select = "RR"
        if int(h1.header["CRVAL3"]) == +1:
            self.select = "I"
        if int(h1.header["CRVAL3"]) == +2:
            self.select = "Q"
        if int(h1.header["CRVAL3"]) == +3:
            self.select = "U"
        if int(h1.header["CRVAL3"]) == +4:
            self.select = "V"

        nstokes = data1["DATA"].shape[5]
        r_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 0]
        i_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 1]
        w_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 2]
        r_1 = r_1.reshape(self.nvis, self.no_if, 1)
        i_1 = i_1.reshape(self.nvis, self.no_if, 1)
        w_1 = w_1.reshape(self.nvis, self.no_if, 1)

        if nstokes == 2:
            r_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 0]
            i_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 1]
            w_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 2]
            r_2 = r_2.reshape(self.nvis, self.no_if, 1)
            i_2 = i_2.reshape(self.nvis, self.no_if, 1)
            w_2 = w_2.reshape(self.nvis, self.no_if, 1)
            r_3 = r_1 * 0.0
            i_3 = i_1 * 0.0
            w_3 = w_1 * 0.0
            r_4 = r_1 * 0.0
            i_4 = i_1 * 0.0
            w_4 = w_1 * 0.0
        elif nstokes == 4:
            r_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 0]
            i_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 1]
            w_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 2]
            r_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 0]
            i_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 1]
            w_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 2]
            r_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 0]
            i_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 1]
            w_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 2]
            r_2 = r_2.reshape(self.nvis, self.no_if, 1)
            i_2 = i_2.reshape(self.nvis, self.no_if, 1)
            w_2 = w_2.reshape(self.nvis, self.no_if, 1)
            r_3 = r_3.reshape(self.nvis, self.no_if, 1)
            i_3 = i_3.reshape(self.nvis, self.no_if, 1)
            w_3 = w_3.reshape(self.nvis, self.no_if, 1)
            r_4 = r_4.reshape(self.nvis, self.no_if, 1)
            i_4 = i_4.reshape(self.nvis, self.no_if, 1)
            w_4 = w_4.reshape(self.nvis, self.no_if, 1)
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

        maskw_1 = (np.isnan(w_1)) | (np.isinf(w_1)) | (w_1 == 0)
        maskw_2 = (np.isnan(w_2)) | (np.isinf(w_2)) | (w_2 == 0)
        maskw_3 = (np.isnan(w_3)) | (np.isinf(w_3)) | (w_3 == 0)
        maskw_4 = (np.isnan(w_4)) | (np.isinf(w_4)) | (w_4 == 0)

        if nstokes == 4:
            maskw = (maskw_1) | (maskw_2)
        elif nstokes == 2:
            maskw = (maskw_1) | (maskw_2)
        elif nstokes == 1:
            maskw = maskw_1

        w_1[maskw] = np.nan
        w_2[maskw] = np.nan
        w_3[maskw] = np.nan
        w_4[maskw] = np.nan

        w_1[maskw_1] = np.nan
        w_2[maskw_2] = np.nan
        w_3[maskw_3] = np.nan
        w_4[maskw_4] = np.nan

        maskwn = np.isnan(w_1)
        nvis = np.sum(np.sum(~maskwn, axis=2), axis=1)

        iws_1 = np.nansum(np.nansum(1 / w_1, axis=2), axis=1)
        iws_2 = np.nansum(np.nansum(1 / w_2, axis=2), axis=1)
        iws_3 = np.nansum(np.nansum(1 / w_3, axis=2), axis=1)
        iws_4 = np.nansum(np.nansum(1 / w_4, axis=2), axis=1)
        iws_1 = np.where(iws_1 <= 0, np.nan, iws_1)
        iws_2 = np.where(iws_2 <= 0, np.nan, iws_2)
        iws_3 = np.where(iws_3 <= 0, np.nan, iws_3)
        iws_4 = np.where(iws_4 <= 0, np.nan, iws_4)

        e_1 = np.sqrt(iws_1) / nvis
        e_2 = np.sqrt(iws_2) / nvis
        e_3 = np.sqrt(iws_3) / nvis
        e_4 = np.sqrt(iws_4) / nvis
        maske_1 = np.isnan(e_1) | (e_1 <= 0)
        maske_2 = np.isnan(e_2) | (e_2 <= 0)
        maske_3 = np.isnan(e_3) | (e_3 <= 0)
        maske_4 = np.isnan(e_4) | (e_4 <= 0)
        if nstokes == 4:
            maske = (maske_1) | (maske_2)
        elif nstokes == 2:
            maske = (maske_1) | (maske_2)
        elif nstokes == 1:
            maske = maske_1

        e_1 = e_1[~maske]
        e_2 = e_2[~maske]
        e_3 = e_3[~maske]
        e_4 = e_4[~maske]

        v_1 = r_1 + 1j * i_1
        v_2 = r_2 + 1j * i_2
        v_3 = r_3 + 1j * i_3
        v_4 = r_4 + 1j * i_4
        v_1 = v_1.reshape(self.nvis, self.no_if, 1)
        v_2 = v_2.reshape(self.nvis, self.no_if, 1)
        v_3 = v_3.reshape(self.nvis, self.no_if, 1)
        v_4 = v_4.reshape(self.nvis, self.no_if, 1)

        v_1[maskw] = np.nan
        v_2[maskw] = np.nan
        v_3[maskw] = np.nan
        v_4[maskw] = np.nan

        uu = uu[~maske]
        vv = vv[~maske]
        ww = ww[~maske]
        time = time[~maske]
        mjd = mjd[~maske]
        try:
            tints = uvf_file[0].data["INTTIM"][~maske]
        except KeyError:
            tints = np.zeros(len(~maske))

        ant1 = uvf_file[0].data["BASELINE"][~maske].astype(int) // 256
        ant2 = uvf_file[0].data["BASELINE"][~maske].astype(int) - ant1 * 256
        ant1 = ant1 - 1
        ant2 = ant2 - 1

        ant_name1 =\
            np.array(
                [tarr[np.where(tnums == i)[0][0]]["name"] for i in ant1]
            )

        ant_name2 =\
            np.array(
                [tarr[np.where(tnums == i)[0][0]]["name"] for i in ant2]
            )

        uant = np.unique(np.append(ant1, ant2))

        v_1 = np.nanmean(np.nanmean(v_1, axis=2), axis=1)[~maske]
        v_2 = np.nanmean(np.nanmean(v_2, axis=2), axis=1)[~maske]
        v_3 = np.nanmean(np.nanmean(v_3, axis=2), axis=1)[~maske]
        v_4 = np.nanmean(np.nanmean(v_4, axis=2), axis=1)[~maske]

        uvdist = np.sqrt(uu**2 + vv**2)

        self.time = time
        self.mjd = mjd
        self.uvu = uu
        self.uvv = vv
        self.ww = ww
        self.uvdist = uvdist

        self.vis_1 = v_1
        self.vis_2 = v_2
        self.vis_3 = v_3
        self.vis_4 = v_4

        self.sig_1 = e_1.astype("f8")
        self.sig_2 = e_2.astype("f8")
        self.sig_3 = e_3.astype("f8")
        self.sig_4 = e_4.astype("f8")

        self.tint = tints
        self.ant_name1 = ant_name1
        self.ant_name2 = ant_name2
        self.stokes = stokes
        self.nstokes = nstokes

        self.tarr = tarr
        self.tkey = {self.tarr[i]["name"]: i for i in range(len(self.tarr))}

        self.select = select

        self.sort_uvvis(
            type="tb&ant", reverse=False, set_uvvis=False, set_closure=False,
            minclq=minclq
        )

        self.set_uvvis(set_pang=set_pang)
        self.set_clq = set_clq
        if set_clq:
            self.set_closure(minclq=self.minclq)

        if self.avgtime in ["none", "scan"]:
            binning = self.scanlen
        else:
            binning = self.avgtime

        if dosyscal:
            self.systype = []
            self.cal_systematics(binning=binning, type="vis")
            if len(uant) >= 4 and set_clq:
                self.cal_systematics(binning=binning, type="clamp")
            if len(uant) >= 3 and set_clq:
                self.cal_systematics(binning=binning, type="clphs")

        self.uvave(
            uvave=uvave,
            scanlen=scanlen,
            doscatter=doscatter,
            set_clq=set_clq,
            set_pang=set_pang
        )

        self.date = Ati(self.data["mjd"], format="mjd").iso[0][:10]
        self.fit_beam(uvw=uvw)
        bmin, bmaj, bpa = self.bprms
        source = self.source.replace(" ", "")

        if prt:
            out1 = self.source
            out2 = self.date
            out3 = self.freq
            out4 = bmin
            out5 = bmaj
            out6 = bpa

            if self.nstokes == 4:
                i = self.data["vis_i"]
                p = self.data["vis_p"]
                self.m = np.abs(np.nanmedian(p/i))
                out7 = self.m * 100
                out_txt =\
                    f"\nloading uvf file.." \
                    f"  {out1:10s} ({out2}, {out3:7.3f} GHz" \
                    f", [{out4:5.2f} X {out5:5.2f} mas^2, {out6:5.1f} deg]" \
                    f", m_p={out7:.2f}% / rough estimation)"
            else:
                self.m = m
                out_txt =\
                    f"\nloading uvf file.." \
                    f"  {out1:10s} ({out2}, {out3:7.3f} GHz" \
                    f", [{out4:5.2f} X {out5:5.2f} mas^2, {out6:5.1f} deg])"

            print(out_txt)

        self.bmin = bmin
        self.bmaj = bmaj
        self.bpa = bpa

    def uvave(self,
              uvave="none", scanlen=300, doscatter=False,
              set_clq=True, set_pang=True):
        """
        Average the uv-visibility data
            Arguments:
                uvave (str, float): averaging interval time in second
                                    (availables: 'none', 'scan', or float)
                scanlen (float): scan length in second
                doscatter (bool): if True,
                                  replace sigma with standard deviation
                                  of visibility amplitude
                set_clq (bool): if True, make the closure quantities
                set_pang (bool): if True, calculate parallactic angle
                prt (bool): if True, allow printing message
        """
        if not uvave:
            pass
        elif uvave == "none":
            pass
        else:
            if uvave == "scan":
                avgtime = scanlen
            else:
                avgtime = uvave

            out_txt =\
                f"Averaging data in every {avgtime:.1f} sec."
            print(out_txt)

            data = self.data

            tarr = self.tarr

            time_ = data["time"]
            tint_ = data["tint"]
            mjd_ = data["mjd"]
            ant_num1_ = data["ant_num1"]
            ant_num2_ = data["ant_num2"]
            ant_name1_ = data["ant_name1"]
            ant_name2_ = data["ant_name2"]
            uvu_ = data["u"]
            uvv_ = data["v"]
            ww_ = self.ww
            time_sec = data["time"] * 3600

            utime_sec = np.unique(time_sec)
            select = self.select.lower()

            db =\
                dbs(eps=self.gaptime, min_samples=1) \
                .fit(time_sec.reshape(-1, 1))
            scannum = db.labels_

            scannum =\
                np.where(
                    scannum < 0,
                    scannum + np.max(scannum)+np.abs(scannum),
                    scannum
                )

            uscan = np.unique(scannum)

            uant_name1 = np.unique(ant_name1_)
            uant_name2 = np.unique(ant_name2_)

            def filter_vis(x):
                return (("vis" in x) & ("vism" not in x)) \
                       & (x != "vis")

            def filter_pols(x):
                return x.split("_")[1]

            pols_ = list(filter(filter_vis, data.dtype.names))
            pols_ = list(map(filter_pols, pols_))

            circs = ["rr", "ll", "rl", "lr"]
            pols = []
            for circ in circs:
                if circ in pols_:
                    pols.append(circ)

            timer = np.arange(0, scanlen + avgtime, avgtime)
            time = []
            mjd = []
            tint = []
            ant_num1 = []
            ant_num2 = []
            ant_name1 = []
            ant_name2 = []
            uvu = []
            uvv = []
            ww = []
            vis_1 = []
            vis_2 = []
            vis_3 = []
            vis_4 = []
            sig_1 = []
            sig_2 = []
            sig_3 = []
            sig_4 = []

            for nstoke, pol in enumerate(pols):
                outvis, outsig = [], []

                for nscan, scan in enumerate(uscan):
                    mask_scan = scannum == scan
                    mintime_sec = np.min(time_sec[mask_scan])
                    time_scan = time_sec - mintime_sec
                    for ntime in range(len(timer) - 1):
                        mask_time =\
                            (timer[ntime + 0] <= time_scan) \
                            & (time_scan < timer[ntime + 1] + 1)

                        data_mask1 = data[mask_scan & mask_time]
                        uant_num1 = np.unique(data_mask1["ant_num1"])
                        uant_num2 = np.unique(data_mask1["ant_num2"])
                        uant_nums = np.unique(np.append(uant_num1, uant_num2))
                        upair = list(it.combinations(uant_nums.tolist(), 2))
                        for upair_ in upair:
                            mask_ant_num1 = ant_num1_ == upair_[0]
                            mask_ant_num2 = ant_num2_ == upair_[1]

                            mask_tot =\
                                mask_scan \
                                & mask_time \
                                & mask_ant_num1 \
                                & mask_ant_num2

                            mask_vis =\
                                len(data[f"vis_{pol}"][mask_tot]) == 0

                            if mask_vis:
                                continue

                            data_mask2 = data[mask_tot]
                            getvis = data_mask2[f"vis_{pol}"]
                            getsig = data_mask2[f"sigma_{pol}"]
                            nvis = len(getvis)

                            if nvis <= 1:
                                continue

                            weight = 1 / getsig**2

                            mask_w =\
                                np.any(np.isnan(weight)) \
                                | np.any(np.isinf(weight))
                            if mask_w:
                                avg_vis = np.mean(getvis)
                            else:
                                avg_vis =\
                                    np.average(getvis, weights=weight)

                            if doscatter:
                                outsig_ =\
                                    np.std(np.abs(getvis)) / np.sqrt(nvis)
                                outsig.append(outsig_)
                                outvis.append(avg_vis)
                            else:
                                if np.sum(getsig) == 0:
                                    outsig.append(0.0)
                                    outvis.append(avg_vis)
                                else:
                                    weight = 1 / getsig**2
                                    outsig.append(
                                        np.sqrt(
                                            (np.sum(getsig**2) / nvis**2)
                                        )
                                    )
                                    outvis.append(avg_vis)

                            if nstoke == 0:
                                time.append(np.mean(data_mask1["time"]))
                                tint.append(np.sum(data_mask2["tint"]))
                                mjd.append(np.mean(mjd_[mask_tot]))
                                ant_num1.append(upair_[0])
                                ant_num2.append(upair_[1])
                                ant_name1.append(tarr["name"][upair_[0] - 1])
                                ant_name2.append(tarr["name"][upair_[1] - 1])
                                uvu.append(np.mean(uvu_[mask_tot]))
                                uvv.append(np.mean(uvv_[mask_tot]))
                                ww.append(np.mean(ww_[mask_tot]))

                if nstoke == 0:
                    vis_1, sig_1 = outvis, outsig
                elif nstoke == 1:
                    vis_2, sig_2 = outvis, outsig
                elif nstoke == 2:
                    vis_3, sig_3 = outvis, outsig
                elif nstoke == 3:
                    vis_4, sig_4 = outvis, outsig

            if self.nstokes not in [1, 2, 4]:
                out_txt =\
                    f"Unexpected polarization type number"\
                    f"(input:{self.nstokes})"
                raise Exception(out_txt)

            self.vis_1 = np.array(vis_1)
            self.vis_2 = np.array(vis_2)
            self.vis_3 = np.array(vis_3)
            self.vis_4 = np.array(vis_4)
            self.sig_1 = np.array(sig_1)
            self.sig_2 = np.array(sig_2)
            self.sig_3 = np.array(sig_3)
            self.sig_4 = np.array(sig_4)

            if self.nstokes == 1:
                self.vis_2 = np.full(len(vis_1), np.nan)
                self.vis_3 = np.full(len(vis_1), np.nan)
                self.vis_4 = np.full(len(vis_1), np.nan)
                self.sig_2 = np.full(len(vis_1), 0.0)
                self.sig_3 = np.full(len(vis_1), 0.0)
                self.sig_4 = np.full(len(vis_1), 0.0)
            if self.nstokes == 2:
                self.vis_3 = np.full(len(vis_1), np.nan)
                self.vis_4 = np.full(len(vis_1), np.nan)
                self.sig_3 = np.full(len(vis_1), 0.0)
                self.sig_4 = np.full(len(vis_1), 0.0)

            self.time = np.array(time)
            self.tint = np.array(tint)
            self.mjd = np.array(mjd)
            self.ant_name1 = np.array(ant_name1)
            self.ant_name2 = np.array(ant_name2)
            self.uvu = np.array(uvu)
            self.uvv = np.array(uvv)
            self.ww = np.array(ww)
            self.set_uvvis(set_pang=set_pang)
            if set_pang:
                self.cal_pangle()
            if set_clq:
                self.set_closure(minclq=self.minclq)

    def set_uvvis(self, set_pang=True):
        """
        Set the uv-visibility data
        """
        time = self.time
        tint = self.tint
        mjd = self.mjd
        ant_name1 = self.ant_name1
        ant_name2 = self.ant_name2
        uvu = self.uvu
        uvv = self.uvv
        uvdist = np.sqrt(self.uvu**2 + self.uvv**2)
        freq = np.full(len(self.vis_1), self.freq)
        vis_1 = self.vis_1
        vis_2 = self.vis_2
        vis_3 = self.vis_3
        vis_4 = self.vis_4
        sig_1 = self.sig_1
        sig_2 = self.sig_2
        sig_3 = self.sig_3
        sig_4 = self.sig_4
        pair =\
            list(map(
                lambda x, y: f"{x}-{y}",
                ant_name1, ant_name2
            ))

        # set circular-hand visibilities
        vis_i = (vis_1 + vis_2) * 0.5
        vis_q = (vis_3 + vis_4) * 0.5
        vis_u = (vis_3 - vis_4) * 0.5 / 1j
        vis_v = (vis_1 - vis_2) * 0.5
        vis_p = (vis_q + 1j * vis_u) * 1.0
        sig_i = np.sqrt(sig_1**2 + sig_2**2) * 0.5
        sig_q = np.sqrt(sig_3**2 + sig_4**2) * 0.5
        sig_u = np.sqrt(sig_3**2 + sig_4**2) * 0.5
        sig_v = np.sqrt(sig_1**2 + sig_2**2) * 0.5
        sig_p = np.sqrt(sig_q**2 + sig_u**2) * 1.0

        ant_dict_name2num = dict(zip(
            self.tarr["name"],
            np.arange(len(self.tarr), dtype=int) + 1)
        )
        ant_num1 =\
            np.array(
                list(map(
                    ant_dict_name2num.get,
                    ant_name1
                ))
            )
        ant_num2 =\
            np.array(
                list(map(
                    ant_dict_name2num.get,
                    ant_name2
                ))
            )


        time_sec = (time * u.hour.to(u.second))

        if self.nstokes == 4:
            dheads =\
                [
                    "time", "freq", "tint", "mjd", "u", "v",
                    "ant_num1", "ant_num2", "ant_name1", "ant_name2", "pair",
                    "vis_i", "vis_q", "vis_u", "vis_v", "vis_p",
                    "sigma_i", "sigma_q", "sigma_u", "sigma_v", "sigma_p",
                    "vis_rr", "vis_ll", "vis_rl", "vis_lr",
                    "sigma_rr", "sigma_ll", "sigma_rl", "sigma_lr"
                ]

            dtypes =\
                [
                    "f8", "f8", "f8", "f8", "f8", "f8",
                    "i", "i", "U32", "U32", "U32",
                    "c16", "c16", "c16", "c16", "c16",
                    "f8", "f8", "f8", "f8", "f8",
                    "c16", "c16", "c16", "c16",
                    "f8", "f8", "f8", "f8"
                ]

            dataset =\
                [
                    time, freq, tint, mjd, uvu, uvv,
                    ant_num1, ant_num2, ant_name1, ant_name2, pair,
                    vis_i, vis_q, vis_u, vis_v, vis_p,
                    sig_i, sig_q, sig_u, sig_v, sig_p,
                    vis_1, vis_2, vis_3, vis_4,
                    sig_1, sig_2, sig_3, sig_4
                ]

            data = gamvas.utils.sarray(dataset, dheads, dtypes)
        elif self.nstokes == 2:
            dheads =\
                [
                    "time", "freq", "tint", "mjd", "u", "v",
                    "ant_num1", "ant_num2", "ant_name1", "ant_name2", "pair",
                    "vis_i", "vis_rr", "vis_ll",
                    "sigma_i", "sigma_rr", "sigma_ll"
                ]

            dtypes =\
                [
                    "f8", "f8", "f8", "f8", "f8", "f8",
                    "i", "i", "U32", "U32", "U32",
                    "c16", "c16", "c16",
                    "f8", "f8", "f8"
                ]

            dataset =\
                [
                    time, freq, tint, mjd, uvu, uvv,
                    ant_num1, ant_num2, ant_name1, ant_name2, pair,
                    vis_i, vis_1, vis_2,
                    sig_i, sig_1, sig_2
                ]

            data = gamvas.utils.sarray(dataset, dheads, dtypes)
        elif self.nstokes == 1:
            dheads =\
                [
                    "time", "freq", "tint", "mjd", "u", "v",
                    "ant_num1", "ant_num2", "ant_name1", "ant_name2", "pair",
                    f"vis_{self.select.lower()}", f"sigma_{self.select.lower()}"
                ]

            dtypes =\
                [
                    "f8", "f8", "f8", "f8", "f8", "f8",
                    "i", "i", "U32", "U32", "U32",
                    "c16", "f8"
                ]

            dataset =\
                [
                    time, freq, tint, mjd, uvu, uvv,
                    ant_num1, ant_num2, ant_name1, ant_name2, pair,
                    vis_1, sig_1
                ]

            data = gamvas.utils.sarray(dataset, dheads, dtypes)

        data =\
            rfn.append_fields(
                data,
                "vis",
                data[f"vis_{self.select.lower()}"],
                usemask=False
            )

        data =\
            rfn.append_fields(
                data,
                "sigma",
                data[f"sigma_{self.select.lower()}"],
                usemask=False
            )

        self.data = data
        if set_pang:
            self.cal_pangle()

    def set_closure_(self, minclq=False):
        """
        Set the closure quantities
        """
        if "vis" not in self.data.dtype.names:
            self.set_uvvis()
        data = self.data
        tarr = self.tarr
        times = np.unique(data["time"])

        torder = {tel: idx for idx, tel in enumerate(tarr["name"])}

        ant_names =\
            np.unique(
                np.append(data["ant_name1"], data["ant_name2"])
            )

        field_vis = f"vis"
        field_sig = f"sigma"
        field_cav = f"clamp"
        field_cas = f"sigma_clamp"
        field_cpv = f"clphs"
        field_cps = f"sigma_clphs"

        uvvis =\
            dict(zip(
                tuple(zip(data["u"].tolist(), data["v"].tolist())),
                data[field_vis]
            ))

        uvsig =\
            dict(zip(
                tuple(zip(data["u"].tolist(), data["v"].tolist())),
                data[field_sig]
            ))

        utimes = np.unique(data["time"])

        field_amp =\
            [
                "time", "freq", "quadra",
                "u12", "v12", "u34", "v34", "u13", "v13", "u24", "v24",
                "vis12", "vis34", "vis13", "vis24",
                "sig12", "sig34", "sig13", "sig24",
                field_cav, field_cas
            ]

        dtype_amp =\
            [
                "f8", "f8", "U32",
                "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8",
                "c8", "c8", "c8", "c8",
                "f8", "f8", "f8", "f8",
                "f8", "f8"
            ]

        field_phs =\
            [
                "time", "freq", "triangle",
                "u12", "v12", "u23", "v23", "u31", "v31",
                "vis12", "vis23", "vis31",
                "sig12", "sig23", "sig31",
                field_cpv, field_cps
            ]

        dtype_phs =\
            [
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

        for ut, time in enumerate(
            tqdm(
                utimes,
                desc="Setting closure relations",
                leave=False,
                bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}"
            )):
            mask_time = data["time"] == time
            data_ = data[data["time"] == time]

            ant_names_ =\
                np.array(sorted(
                    np.unique(
                        np.append(data_["ant_name1"], data_["ant_name2"])
                    ),
                    key=lambda x: torder[x]
                ))

            Nant = len(ant_names_)
            if Nant >= 4:
                crit = int(round(0.5 * Nant * (Nant - 3) / 2))
                crit_count = 0
                quadrangles = list(it.combinations(ant_names_.tolist(), 4))

                for quadrangle in quadrangles:
                    if minclq and crit_count >= crit:
                        continue

                    for nvert in range(2):
                        if nvert == 0:
                            quadra_ =\
                                [
                                    quadrangle[0], quadrangle[1],
                                    quadrangle[2], quadrangle[3]
                                ]

                            mask_uv12 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[0]) \
                                & (data["ant_name2"] == quadrangle[1])
                            mask_uv34 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[2]) \
                                & (data["ant_name2"] == quadrangle[3])
                            mask_uv13 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[0]) \
                                & (data["ant_name2"] == quadrangle[2])
                            mask_uv24 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[1]) \
                                & (data["ant_name2"] == quadrangle[3])
                        else:
                            quadra_ =\
                                [
                                    quadrangle[0], quadrangle[2],
                                    quadrangle[1], quadrangle[3]
                                ]

                            mask_uv12 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[0]) \
                                & (data["ant_name2"] == quadrangle[2])
                            mask_uv34 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[1]) \
                                & (data["ant_name2"] == quadrangle[3])
                            mask_uv13 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[0]) \
                                & (data["ant_name2"] == quadrangle[3])
                            mask_uv24 =\
                                (mask_time) \
                                & (data["ant_name1"] == quadrangle[1]) \
                                & (data["ant_name2"] == quadrangle[2])

                        mask_uv1234 =\
                            np.sum(mask_uv12) \
                            + np.sum(mask_uv34) \
                            + np.sum(mask_uv13) \
                            + np.sum(mask_uv24)

                        if mask_uv1234 == 4:
                            out_quadrangle = "-".join(quadra_)

                            out_uv12 =\
                                (
                                    data["u"][mask_uv12][0],
                                    data["v"][mask_uv12][0]
                                )
                            out_uv34 =\
                                (
                                    data["u"][mask_uv34][0],
                                    data["v"][mask_uv34][0]
                                )
                            out_uv13 =\
                                (
                                    data["u"][mask_uv13][0],
                                    data["v"][mask_uv13][0]
                                )
                            out_uv24 =\
                                (
                                    data["u"][mask_uv24][0],
                                    data["v"][mask_uv24][0]
                                )

                            crit_count += 1
                            outca_time.append(time)
                            outca_freq.append(self.freq)
                            outca_quad.append(out_quadrangle)
                            outca_u12.append(out_uv12[0])
                            outca_v12.append(out_uv12[1])
                            outca_u34.append(out_uv34[0])
                            outca_v34.append(out_uv34[1])
                            outca_u13.append(out_uv13[0])
                            outca_v13.append(out_uv13[1])
                            outca_u24.append(out_uv24[0])
                            outca_v24.append(out_uv24[1])
            if Nant >= 3:
                crit = int(round(0.5 * (Nant - 1) * (Nant - 2)))
                crit_count = 0
                triangles = list(it.combinations(ant_names_.tolist(), 3))

                for triangle in triangles:
                    if minclq and crit_count >= crit:
                        continue

                    mask_uv12 =\
                        (mask_time) \
                        & (data["ant_name1"] == triangle[0]) \
                        & (data["ant_name2"] == triangle[1])
                    mask_uv23 =\
                        (mask_time) \
                        & (data["ant_name1"] == triangle[1]) \
                        & (data["ant_name2"] == triangle[2])
                    mask_uv31 =\
                        (mask_time) \
                        & (data["ant_name1"] == triangle[0]) \
                        & (data["ant_name2"] == triangle[2])

                    mask_uv123 =\
                        np.sum(mask_uv12) \
                        + np.sum(mask_uv23) \
                        + np.sum(mask_uv31)

                    if mask_uv123 == 3:
                        out_times = time
                        out_frequency = self.freq
                        out_triangle = "-".join(triangle)

                        out_uv12 =\
                            (
                                data["u"][mask_uv12][0],
                                data["v"][mask_uv12][0]
                            )
                        out_uv23 =\
                            (
                                data["u"][mask_uv23][0],
                                data["v"][mask_uv23][0]
                            )
                        out_uv31 =\
                            (
                                data["u"][mask_uv31][0],
                                data["v"][mask_uv31][0]
                            )

                        crit_count += 1
                        outcp_time.append(time)
                        outcp_freq.append(self.freq)
                        outcp_tri.append(out_triangle)
                        outcp_u12.append(out_uv12[0])
                        outcp_v12.append(out_uv12[1])
                        outcp_u23.append(out_uv23[0])
                        outcp_v23.append(out_uv23[1])
                        outcp_u31.append(out_uv31[0])
                        outcp_v31.append(out_uv31[1])

        if len(outca_time) == 0:
            tmpl_clamp =\
                gamvas.utils.sarray(
                    [[np.nan] for i in range(len(field_amp))],
                    field=field_amp,
                    dtype=dtype_amp
                )
            flag_clamp = True
        else:
            tmpl_clamp =\
                gamvas.utils.sarray(
                    [
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
            tmpl_clphs =\
                gamvas.utils.sarray(
                    [[np.nan] for i in range(len(field_phs))],
                    field=field_phs,
                    dtype=dtype_phs
                )
            flag_clphs = True
        else:
            tmpl_clphs =\
                gamvas.utils.sarray(
                    [
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
            tmpl_clamp =\
                gamvas.utils.sarray(
                    tmpl_clamp,
                    field=field_amp,
                    dtype=dtype_amp
                )
            flag_clamp = True
        if tmpl_clphs.size == 0:
            tmpl_clphs = [np.nan] * len(field_phs)
            tmpl_clphs =\
                gamvas.utils.sarray(
                    tmpl_clphs,
                    field=field_phs,
                    dtype=dtype_phs
                )
            flag_clphs = True
        self.tmpl_clamp = copy.deepcopy(tmpl_clamp)
        self.tmpl_clphs = copy.deepcopy(tmpl_clphs)
        if not flag_clamp:
            clamp_ = tmpl_clamp
            clamp_["vis12"] =\
                list(map(
                    uvvis.get,
                    tuple(zip(clamp_["u12"], clamp_["v12"]))
                ))
            clamp_["vis34"] =\
                list(map(
                    uvvis.get,
                    tuple(zip(clamp_["u34"], clamp_["v34"]))
                ))
            clamp_["vis13"] =\
                list(map(
                    uvvis.get,
                    tuple(zip(clamp_["u13"], clamp_["v13"]))
                ))
            clamp_["vis24"] =\
                list(map(
                    uvvis.get,
                    tuple(zip(clamp_["u24"], clamp_["v24"]))
                ))
            clamp_["sig12"] =\
                list(map(
                    uvsig.get,
                    tuple(zip(clamp_["u12"], clamp_["v12"]))
                ))
            clamp_["sig34"] =\
                list(map(
                    uvsig.get,
                    tuple(zip(clamp_["u34"], clamp_["v34"]))
                ))
            clamp_["sig13"] =\
                list(map(
                    uvsig.get,
                    tuple(zip(clamp_["u13"], clamp_["v13"]))
                ))
            clamp_["sig24"] =\
                list(map(
                    uvsig.get,
                    tuple(zip(clamp_["u24"], clamp_["v24"]))
                ))
            amp12 = np.abs(clamp_["vis12"])
            amp34 = np.abs(clamp_["vis34"])
            amp13 = np.abs(clamp_["vis13"])
            amp24 = np.abs(clamp_["vis24"])
            amp12 =\
                np.where(
                    amp12 < clamp_["sig12"],
                    0,
                    np.sqrt(np.abs(clamp_["vis12"])**2 - clamp_["sig12"]**2)
                )
            amp34 =\
                np.where(
                    amp34 < clamp_["sig34"],
                    0,
                    np.sqrt(np.abs(clamp_["vis34"])**2 - clamp_["sig34"]**2)
                )
            amp13 =\
                np.where(
                    amp13 < clamp_["sig13"],
                    0,
                    np.sqrt(np.abs(clamp_["vis13"])**2 - clamp_["sig13"]**2)
                )
            amp24 =\
                np.where(
                    amp24 < clamp_["sig24"],
                    0,
                    np.sqrt(np.abs(clamp_["vis24"])**2 - clamp_["sig24"]**2)
                )
            snr12 = amp12 / np.abs(clamp_["sig12"])
            snr34 = amp34 / np.abs(clamp_["sig34"])
            snr13 = amp13 / np.abs(clamp_["sig13"])
            snr24 = amp24 / np.abs(clamp_["sig24"])
            clamp_[field_cav] = (amp12 * amp34) / (amp13 * amp24)
            clamp_[field_cas] =\
                np.sqrt(snr12**-2 + snr34**-2 + snr13**-2 + snr24**-2)

        if not flag_clphs:
            clphs_ = tmpl_clphs
            clphs_["vis12"] =\
                list(map(
                    uvvis.get,
                    tuple(zip(clphs_["u12"], clphs_["v12"]))
                ))
            clphs_["vis23"] =\
                list(map(
                    uvvis.get,
                    tuple(zip(clphs_["u23"], clphs_["v23"]))
                ))
            clphs_["vis31"] =\
                list(map(
                    uvvis.get,
                    tuple(zip(clphs_["u31"], clphs_["v31"]))
                ))
            clphs_["sig12"] =\
                list(map(
                    uvsig.get,
                    tuple(zip(clphs_["u12"], clphs_["v12"]))
                ))
            clphs_["sig23"] =\
                list(map(
                    uvsig.get,
                    tuple(zip(clphs_["u23"], clphs_["v23"]))
                ))
            clphs_["sig31"] =\
                list(map(
                    uvsig.get,
                    tuple(zip(clphs_["u31"], clphs_["v31"]))
                ))
            phs12 = np.angle(clphs_["vis12"])
            phs23 = np.angle(clphs_["vis23"])
            phs31 = np.angle(clphs_["vis31"].conj())
            snr12 = np.abs(clphs_["vis12"]) / np.abs(clphs_["sig12"])
            snr23 = np.abs(clphs_["vis23"]) / np.abs(clphs_["sig23"])
            snr31 = np.abs(clphs_["vis31"]) / np.abs(clphs_["sig31"])

            clphs_v = phs12 + phs23 + phs31
            clphs_[field_cpv] = clphs_v
            clphs_[field_cps] = np.sqrt(snr12**-2 + snr23**-2 + snr31**-2)

        if not flag_clamp:
            fields = ["time", "quadra", "freq", field_cav, field_cas]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            datas = [clamp_[fields[nf]] for nf in range(len(fields))]
            clamp_ = gamvas.utils.sarray(datas, fields, dtypes)
        else:
            fields = ["time", "quadra", "freq", field_cav, field_cas]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            clamp_ =\
                gamvas.utils.sarray(
                    [np.nan for i in range(len(fields))],
                    fields,
                    dtypes
                )

        if not flag_clphs:
            fields = ["time", "triangle", "freq", field_cpv, field_cps]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            datas = [clphs_[fields[nf]] for nf in range(len(fields))]
            clphs_ = gamvas.utils.sarray(datas, fields, dtypes)
        else:
            fields = ["time", "triangle", "freq", field_cpv, field_cps]
            dtypes = ["f8", "U32", "f8", "f8", "f8"]
            clphs_ =\
                gamvas.utils.sarray(
                    [np.nan for i in range(len(fields))],
                    fields,
                    dtypes
                )

        clamp = clamp_
        clphs = clphs_

        if not flag_clamp:
            self.clamp = clamp
            self.clamp_check = True
        else:
            fields =\
                [
                    "time", "quadra", "freq",
                    "clamp", "sigma_clamp",
                    "clamp_i", "sigma_clamp_i",
                    "clamp_q", "sigma_clamp_q",
                    "clamp_u", "sigma_clamp_u",
                    "clamp_v", "sigma_clamp_v",
                    "clamp_p", "sigma_clamp_p"
                ]
            dtypes =\
                [
                    "f8", "U32", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8"
                ]
            self.clamp = gamvas.utils.sarray(
                data=[np.nan for i in range(len(fields))],
                field=fields,
                dtype=dtypes)
            self.clamp_check = False

        if not flag_clphs:
            self.clphs = clphs
            self.clphs_check = True
        else:
            fields =\
                [
                    "time", "triangle", "freq",
                    "clphs", "sigma_clphs",
                    "clphs_i", "sigma_clphs_i",
                    "clphs_q", "sigma_clphs_q",
                    "clphs_u", "sigma_clphs_u",
                    "clphs_v", "sigma_clphs_v",
                    "clphs_p", "sigma_clphs_p"
                ]

            dtypes =\
                [
                    "f8", "U32", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8",
                    "f8", "f8"
                ]

            self.clphs =\
                gamvas.utils.sarray(
                    data=[np.nan for i in range(len(fields))],
                    field=fields,
                    dtype=dtypes
                )

            self.clphs_check = False
        self.ploter.clq_obs =\
            (
                copy.deepcopy(self.clamp),
                copy.deepcopy(self.clphs)
            )

    def set_closure(self, minclq=True):
        """
        Set the closure quantities
        """
        if not minclq:
            self.set_closure_(minclq=minclq)
        else:
            if "vis" not in self.data.dtype.names:
                self.set_uvvis()
            data = self.data
            tarr = self.tarr
            times = np.unique(data["time"])

            torder = {tel: idx for idx, tel in enumerate(tarr["name"])}

            ant_names =\
                np.array(sorted(
                    np.unique(
                        np.append(data["ant_name1"], data["ant_name2"])
                    ),
                    key=lambda x: torder[x]
                ))

            field_vis = f"vis"
            field_sig = f"sigma"
            field_cav = f"clamp"
            field_cas = f"sigma_clamp"
            field_cpv = f"clphs"
            field_cps = f"sigma_clphs"

            uvvis =\
                dict(zip(
                    tuple(zip(data["u"].tolist(), data["v"].tolist())),
                    data[field_vis]
                ))

            uvsig =\
                dict(zip(
                    tuple(zip(data["u"].tolist(), data["v"].tolist())),
                    data[field_sig]
                ))

            utimes = np.unique(data["time"])

            field_amp =\
                [
                    "time", "freq", "quadra",
                    "u12", "v12", "u34", "v34", "u13", "v13", "u24", "v24",
                    "vis12", "vis34", "vis13", "vis24",
                    "sig12", "sig34", "sig13", "sig24",
                    field_cav, field_cas
                ]

            dtype_amp =\
                [
                    "f8", "f8", "U32",
                    "f8", "f8", "f8", "f8", "f8", "f8", "f8", "f8",
                    "c16", "c16", "c16", "c16",
                    "f8", "f8", "f8", "f8",
                    "f8", "f8"
                ]

            field_phs =\
                [
                    "time", "freq", "triangle",
                    "u12", "v12", "u23", "v23", "u31", "v31",
                    "vis12", "vis23", "vis31",
                    "sig12", "sig23", "sig31",
                    field_cpv, field_cps
                ]

            dtype_phs =\
                [
                    "f8", "f8", "U32",
                    "f8", "f8", "f8", "f8", "f8", "f8",
                    "c16", "c16", "c16",
                    "f8", "f8", "f8",
                    "f8", "f8"
                ]

            tmpl_clamp = np.array([])
            tmpl_clphs = np.array([])
            for ut, time in enumerate(
                tqdm(
                    utimes,
                    desc="Setting closure relations",
                    leave=False,
                    bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}"
                )):
                data_ = data[data["time"] == time]

                ant_nums_ =\
                    np.unique(
                        np.append(data_["ant_num1"], data_["ant_num2"])
                    )

                ant_names_ =\
                    np.unique(
                        np.append(data_["ant_name1"], data_["ant_name2"])
                    )

                Nant = len(ant_names_)
                if Nant >= 4:
                    pairs_obs =\
                        np.array(list(map(
                            ",".join,
                            tuple(zip(
                                data_["ant_num1"].astype(str),
                                data_["ant_num2"].astype(str)
                            ))
                        )))

                    pairs_obs = np.sort(pairs_obs)
                    matrix_clamp, pairs_full =\
                        set_min_matrix_clamp(Nant, ant_nums_.astype(str))

                    row = matrix_clamp.shape[0]
                    for i in range(row):
                        mask_add = matrix_clamp[i] == +1
                        mask_sub = matrix_clamp[i] == -1
                        pair_add = pairs_full[mask_add]
                        pair_sub = pairs_full[mask_sub]
                        mask_tot =\
                            (pair_add[0] in pairs_obs) \
                            & (pair_add[1] in pairs_obs) \
                            & (pair_sub[0] in pairs_obs) \
                            & (pair_sub[1] in pairs_obs)

                        pair_add = [val.split(",") for val in pair_add]
                        pair_sub = [val.split(",") for val in pair_sub]
                        if mask_tot:
                            pair_ants =\
                                np.array([
                                    int(pair_add[0][0]),
                                    int(pair_add[0][1]),
                                    int(pair_add[1][0]),
                                    int(pair_add[1][1])
                                ])
                            out_times = time
                            out_frequency = self.freq
                            out_quadrangle =\
                                "-".join(list(map(
                                    self.ant_dict_num2name.get, pair_ants
                                )))

                            loc_uv1 =\
                                (data_["ant_num1"] == int(pair_add[0][0])) \
                                & (data_["ant_num2"] == int(pair_add[0][1]))

                            loc_uv2 =\
                                (data_["ant_num1"] == int(pair_add[1][0])) \
                                & (data_["ant_num2"] == int(pair_add[1][1]))

                            loc_uv3 =\
                                (data_["ant_num1"] == int(pair_sub[0][0])) \
                                & (data_["ant_num2"] == int(pair_sub[0][1]))

                            loc_uv4 =\
                                (data_["ant_num1"] == int(pair_sub[1][0])) \
                                & (data_["ant_num2"] == int(pair_sub[1][1]))

                            out_uv1 =\
                                (
                                    data_["u"][loc_uv1][0],
                                    data_["v"][loc_uv1][0]
                                )

                            out_uv2 =\
                                (
                                    data_["u"][loc_uv2][0],
                                    data_["v"][loc_uv2][0]
                                )

                            out_uv3 =\
                                (
                                    data_["u"][loc_uv3][0],
                                    data_["v"][loc_uv3][0]
                                )

                            out_uv4 =\
                                (
                                    data_["u"][loc_uv4][0],
                                    data_["v"][loc_uv4][0]
                                )

                            tmpl_clamp_ = gamvas.utils.sarray(
                                [
                                    out_times,
                                    out_frequency,
                                    out_quadrangle,
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

                            if tmpl_clamp.size == 0:
                                tmpl_clamp = tmpl_clamp_
                            else:
                                tmpl_clamp =\
                                    rfn.stack_arrays((
                                        tmpl_clamp, tmpl_clamp_
                                    ))
                if Nant >= 3:
                    pairs_obs =\
                        np.array(list(map(
                            ",".join,
                            tuple(zip(
                                data_["ant_num1"].astype(str),
                                data_["ant_num2"].astype(str)
                            ))
                        )))

                    matrix_clphs, pairs_full =\
                        set_min_matrix_clphs(Nant, ant_nums_.astype(str))

                    row = matrix_clphs.shape[0]
                    for i in range(row):
                        mask_add = matrix_clphs[i] == +1
                        mask_sub = matrix_clphs[i] == -1
                        pair_add = pairs_full[mask_add]
                        pair_sub = pairs_full[mask_sub]
                        mask_tot =\
                            (pair_add[0] in pairs_obs) \
                            & (pair_add[1] in pairs_obs) \
                            & (pair_sub[0] in pairs_obs)

                        pair_add = [val.split(",") for val in pair_add]
                        pair_sub = [val.split(",") for val in pair_sub]
                        if mask_tot:
                            pair_ants =\
                                np.array([
                                    int(pair_add[0][0]),
                                    int(pair_add[0][1]),
                                    int(pair_sub[0][1])
                                ])

                            out_times = time
                            out_frequency = self.freq
                            out_triangle =\
                                "-".join(list(map(
                                    self.ant_dict_num2name.get, pair_ants
                                )))

                            loc_uv1 =\
                                (data_["ant_num1"] == int(pair_add[0][0])) \
                                & (data_["ant_num2"] == int(pair_add[0][1]))

                            loc_uv2 =\
                                (data_["ant_num1"] == int(pair_add[1][0])) \
                                & (data_["ant_num2"] == int(pair_add[1][1]))

                            loc_uv3 =\
                                (data_["ant_num1"] == int(pair_sub[0][0])) \
                                & (data_["ant_num2"] == int(pair_sub[0][1]))

                            out_uv1 =\
                                (
                                    data_["u"][loc_uv1][0],
                                    data_["v"][loc_uv1][0]
                                )

                            out_uv2 =\
                                (
                                    data_["u"][loc_uv2][0],
                                    data_["v"][loc_uv2][0]
                                )

                            out_uv3 =\
                                (
                                    data_["u"][loc_uv3][0],
                                    data_["v"][loc_uv3][0]
                                )

                            tmpl_clphs_ = gamvas.utils.sarray(
                                [
                                    out_times,
                                    out_frequency,
                                    out_triangle,
                                    out_uv1[0], out_uv1[1],
                                    out_uv2[0], out_uv2[1],
                                    out_uv3[0], out_uv3[1],
                                    0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0
                                ],
                                field=field_phs,
                                dtype=dtype_phs
                            )
                            if tmpl_clphs.size == 0:
                                tmpl_clphs = tmpl_clphs_
                            else:
                                tmpl_clphs =\
                                    rfn.stack_arrays(
                                        (tmpl_clphs, tmpl_clphs_)
                                    )
                gc.collect()

            flag_clamp = False
            flag_clphs = False

            tmpl_clamp = tmpl_clamp.reshape(-1)
            tmpl_clphs = tmpl_clphs.reshape(-1)

            if tmpl_clamp.size == 0:
                tmpl_clamp = [np.nan] * len(field_amp)
                tmpl_clamp =\
                    gamvas.utils.sarray(
                        tmpl_clamp,
                        field=field_amp,
                        dtype=dtype_amp
                    )

                flag_clamp = True
            if tmpl_clphs.size == 0:
                tmpl_clphs = [np.nan] * len(field_phs)
                tmpl_clphs =\
                    gamvas.utils.sarray(
                        tmpl_clphs,
                        field=field_phs,
                        dtype=dtype_phs
                    )

                flag_clphs = True
            self.tmpl_clamp = copy.deepcopy(tmpl_clamp)
            self.tmpl_clphs = copy.deepcopy(tmpl_clphs)
            if not flag_clamp:
                clamp_ = tmpl_clamp
                clamp_["vis12"] =\
                    list(map(
                        uvvis.get,
                        tuple(zip(clamp_["u12"], clamp_["v12"]))
                    ))

                clamp_["vis34"] =\
                    list(map(
                        uvvis.get,
                        tuple(zip(clamp_["u34"], clamp_["v34"]))
                    ))

                clamp_["vis13"] =\
                    list(map(
                        uvvis.get,
                        tuple(zip(clamp_["u13"], clamp_["v13"]))
                    ))

                clamp_["vis24"] =\
                    list(map(
                        uvvis.get,
                        tuple(zip(clamp_["u24"], clamp_["v24"]))
                    ))

                clamp_["sig12"] =\
                    list(map(
                        uvsig.get,
                        tuple(zip(clamp_["u12"], clamp_["v12"]))
                    ))

                clamp_["sig34"] =\
                    list(map(
                        uvsig.get,
                        tuple(zip(clamp_["u34"], clamp_["v34"]))
                    ))

                clamp_["sig13"] =\
                    list(map(
                        uvsig.get,
                        tuple(zip(clamp_["u13"], clamp_["v13"]))
                    ))

                clamp_["sig24"] =\
                    list(map(
                        uvsig.get,
                        tuple(zip(clamp_["u24"], clamp_["v24"]))
                    ))

                amp12 = np.abs(clamp_["vis12"])
                amp34 = np.abs(clamp_["vis34"])
                amp13 = np.abs(clamp_["vis13"])
                amp24 = np.abs(clamp_["vis24"])

                # debiasing visibility amplitude
                #  (Eq. 19 in EHTC+2019, III)
                amp12 =\
                    np.where(
                        amp12 < clamp_["sig12"],
                        0,
                        np.sqrt(
                            np.abs(clamp_["vis12"])**2 - clamp_["sig12"]**2
                        )
                    )

                amp34 =\
                    np.where(
                        amp34 < clamp_["sig34"],
                        0,
                        np.sqrt(
                            np.abs(clamp_["vis34"])**2 - clamp_["sig34"]**2
                        )
                    )

                amp13 =\
                    np.where(
                        amp13 < clamp_["sig13"],
                        0,
                        np.sqrt(
                            np.abs(clamp_["vis13"])**2 - clamp_["sig13"]**2
                        )
                    )

                amp24 =\
                    np.where(
                        amp24 < clamp_["sig24"],
                        0,
                        np.sqrt(
                            np.abs(clamp_["vis24"])**2 - clamp_["sig24"]**2
                        )
                    )

                snr12 = amp12 / np.abs(clamp_["sig12"])
                snr34 = amp34 / np.abs(clamp_["sig34"])
                snr13 = amp13 / np.abs(clamp_["sig13"])
                snr24 = amp24 / np.abs(clamp_["sig24"])
                clamp_[field_cav] = (amp12 * amp34) / (amp13 * amp24)
                clamp_[field_cas] =\
                    np.sqrt(snr12**-2 + snr34**-2 + snr13**-2 + snr24**-2)

            if not flag_clphs:
                clphs_ = tmpl_clphs
                clphs_["vis12"] =\
                    list(
                        map(uvvis.get,
                            tuple(zip(clphs_["u12"], clphs_["v12"]))
                        )
                    )

                clphs_["vis23"] =\
                    list(
                        map(uvvis.get,
                            tuple(zip(clphs_["u23"], clphs_["v23"]))
                        )
                    )

                clphs_["vis31"] =\
                    list(
                        map(uvvis.get,
                            tuple(zip(clphs_["u31"], clphs_["v31"]))
                        )
                    )

                clphs_["sig12"] =\
                    list(
                        map(uvsig.get,
                            tuple(zip(clphs_["u12"], clphs_["v12"]))
                        )
                    )

                clphs_["sig23"] =\
                    list(
                        map(uvsig.get,
                            tuple(zip(clphs_["u23"], clphs_["v23"]))
                        )
                    )

                clphs_["sig31"] =\
                    list(
                        map(uvsig.get,
                            tuple(zip(clphs_["u31"], clphs_["v31"]))
                        )
                    )

                phs12 = np.angle(clphs_["vis12"])
                phs23 = np.angle(clphs_["vis23"])
                phs31 = np.angle(clphs_["vis31"].conj())
                snr12 = np.abs(clphs_["vis12"]) / np.abs(clphs_["sig12"])
                snr23 = np.abs(clphs_["vis23"]) / np.abs(clphs_["sig23"])
                snr31 = np.abs(clphs_["vis31"]) / np.abs(clphs_["sig31"])

                clphs_v = phs12 + phs23 + phs31
                clphs_[field_cpv] = clphs_v
                clphs_[field_cps] =\
                    np.sqrt(snr12**-2 + snr23**-2 + snr31**-2)

            if not flag_clamp:
                fields = ["time", "quadra", "freq", field_cav, field_cas]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                datas = [clamp_[fields[nf]] for nf in range(len(fields))]
                clamp_ = gamvas.utils.sarray(datas, fields, dtypes)
            else:
                fields = ["time", "quadra", "freq", field_cav, field_cas]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                clamp_ =\
                    gamvas.utils.sarray(
                        [np.nan for i in range(len(fields))],
                        fields,
                        dtypes
                    )

            if not flag_clphs:
                fields = ["time", "triangle", "freq", field_cpv, field_cps]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                datas = [clphs_[fields[nf]] for nf in range(len(fields))]
                clphs_ = gamvas.utils.sarray(datas, fields, dtypes)
            else:
                fields = ["time", "triangle", "freq", field_cpv, field_cps]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                clphs_ =\
                    gamvas.utils.sarray(
                        [np.nan for i in range(len(fields))],
                        fields,
                        dtypes
                    )

            clamp = clamp_
            clphs = clphs_

            if not flag_clamp:
                self.clamp = clamp
                self.clamp_check = True
            else:
                fields =\
                    [
                        "time", "quadra", "freq",
                        "clamp", "sigma_clamp",
                        "clamp_i", "sigma_clamp_i",
                        "clamp_q", "sigma_clamp_q",
                        "clamp_u", "sigma_clamp_u",
                        "clamp_v", "sigma_clamp_v",
                        "clamp_p", "sigma_clamp_p"
                    ]
                dtypes =\
                    [
                        "f8", "U32", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8"
                    ]
                self.clamp =\
                    gamvas.utils.sarray(
                        data=[np.nan for i in range(len(fields))],
                        field=fields,
                        dtype=dtypes
                    )

                self.clamp_check = False

            if not flag_clphs:
                self.clphs = clphs
                self.clphs_check = True
            else:
                fields =\
                    [
                        "time", "triangle", "freq",
                        "clphs", "sigma_clphs",
                        "clphs_i", "sigma_clphs_i",
                        "clphs_q", "sigma_clphs_q",
                        "clphs_u", "sigma_clphs_u",
                        "clphs_v", "sigma_clphs_v",
                        "clphs_p", "sigma_clphs_p"
                    ]

                dtypes =\
                    [
                        "f8", "U32", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8",
                        "f8", "f8"
                    ]

                self.clphs =\
                    gamvas.utils.sarray(
                        data=[np.nan for i in range(len(fields))],
                        field=fields,
                        dtype=dtypes
                    )

                self.clphs_check = False
            self.ploter.clq_obs =\
                (
                    copy.deepcopy(self.clamp),
                    copy.deepcopy(self.clphs)
                )

    def sort_uvvis(self,
        type="tb&ant", reverse=False, set_uvvis=False, set_closure=False,
        minclq=True):
        """
        Sort the uv-visibilities in order of time and antenna number
            Arguments:
                type (str): sorting type
                    available options:
                        'time' or 'tb' : sort uv-visibilities in order of time
                        'ant' or 'antenna' : sort uv-visibilities
                                             in order of antenna number
                        'tb&ant' or 'time&antenna' : sort uv-visibilities
                                                     in order of time and
                                                     antenna number
                        'snr' : sort uv-visibilities in order of SNR
                reverse (bool): toggle option for sorting in the reverse order
        """
        tarr = self.tarr
        time = self.time
        tint = self.tint
        mjd = self.mjd
        ant_name1 = self.ant_name1
        ant_name2 = self.ant_name2
        uvu = self.uvu
        uvv = self.uvv
        ww = self.ww
        vis_1 = self.vis_1
        vis_2 = self.vis_2
        vis_3 = self.vis_3
        vis_4 = self.vis_4
        sig_1 = self.sig_1
        sig_2 = self.sig_2
        sig_3 = self.sig_3
        sig_4 = self.sig_4

        ant_dict_name2num =\
            dict(zip(
                tarr["name"],
                np.arange(len(tarr), dtype=int) + 1)
            )
        ant_num1 = np.array(list(map(ant_dict_name2num.get, ant_name1)))
        ant_num2 = np.array(list(map(ant_dict_name2num.get, ant_name2)))

        if type == "time" or type == "tb":
            order = np.argsort(time)
        elif type == "ant" or type == "antenna":
            order = np.lexsort((ant_num2, ant_num1))
        elif type == "tb&ant" or type == "time&antenna":
            order = np.lexsort((ant_num2, ant_num1, time))
        elif type == "snr":
            if self.nstokes == 1:
                vis = vis_1
                sig = sig_1
            elif self.nstokes == 2:
                if self.select.lower() == "rr":
                    vis = vis_1
                    sig = sig_1
                elif self.select.lower() == "ll":
                    vis = vis_2
                    sig = sig_2
                elif self.select.lower() == "i":
                    vis = (vis_1 + vis_2) * 0.5
                    sig = np.sqrt(sig_1**2 + sig_2**2) * 0.5
                else:
                    out_txt =\
                        f"Unexpected polarization type is given: "\
                        f"{self.select.upper()} "\
                        f"(available: 'RR', 'LL', 'I')"
                    raise Exception(out_txt)
            elif self.nstokes == 4:
                if self.select.lower() == "rr":
                    vis = vis_1
                    sig = sig_1
                elif self.select.lower() == "ll":
                    vis = vis_2
                    sig = sig_2
                elif self.select.lower() == "rl":
                    vis = vis_3
                    sig = sig_3
                elif self.select.lower() == "lr":
                    vis = vis_4
                    sig = sig_4
                elif self.select.lower() == "i":
                    vis = (vis_1 + vis_2) * 0.5
                    sig = np.sqrt(sig_1**2 + sig_2**2) * 0.5
                elif self.select.lower() == "q":
                    vis = (vis_3 + vis_4) * 0.5
                    sig = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                elif self.select.lower() == "u":
                    vis = (vis_3 - vis_4) * 0.5 / 1j
                    sig = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                elif self.select.lower() == "v":
                    vis = (vis_1 - vis_2) * 0.5
                    sig = np.sqrt(sig_1**2 + sig_2**2) * 0.5
                elif self.select.lower() == "p":
                    vis_q = (vis_3 + vis_4) * 0.5
                    vis_u = (vis_3 - vis_4) * 0.5 / 1j
                    sig_q = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                    sig_u = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                    vis = (vis_q + 1j * vis_u) * 1.0
                    sig = np.sqrt(sig_q**2 + sig_u**2) * 1.0
            inverse_snr = 1 / (np.abs(vis) / np.abs(sig))
            order = np.argsort(inverse_snr)

        if reverse:
            order = order[::-1]

        self.time = time[order]
        self.tint = tint[order]
        self.mjd = mjd[order]
        self.ant_name1 = ant_name1[order]
        self.ant_name2 = ant_name2[order]
        self.uvu = uvu[order]
        self.uvv = uvv[order]
        self.ww = ww[order]
        self.vis_1 = vis_1[order]
        self.vis_2 = vis_2[order]
        self.vis_3 = vis_3[order]
        self.vis_4 = vis_4[order]
        self.sig_1 = sig_1[order]
        self.sig_2 = sig_2[order]
        self.sig_3 = sig_3[order]
        self.sig_4 = sig_4[order]

        if set_uvvis:
            self.set_uvvis()

        if set_closure:
            self.set_closure()

    def flag_uvvis(self,
                   type=None, value=None, unit="m", prt=True):
        """
        Flag the uv-visibilities
            Arguments:
                type (str): flagging type
                    available options:
                        'time' (list, float): flag uv-visibilities
                                              in the given time range in UTC
                        'sigma' (float): flag uv-visibilities having sigma
                                         higher than the given value
                        'snr' (float): flag uv-visibilities having SNR
                                       lower than the given value
                        'ant' or 'antenna' (list, str): flag uv-visibilities
                                                        of the given antenna(s)
                        'nant' (int): flag uv-visibilities in scans
                                      having lower number of antennas
                                      than the given value
                        'uvr' (list, float): flag uv-visibilities
                                             in the given uv-distance range
                unit (str, type=='uvr' only): unit of the value.
                                              Default is 'mega-lambda'
                    available options:
                        'k' or 'kilo': kilo-lambda
                        'm' or 'mega': mega-lambda
                        'g' or 'giga': giga-lambda
                prt (bool): toggle option for printing the information
        """
        time = self.time
        tint = self.tint
        mjd = self.mjd
        ant_name1 = self.ant_name1
        ant_name2 = self.ant_name2
        uvu = self.uvu
        uvv = self.uvv
        ww = self.ww
        vis_1 = self.vis_1
        vis_2 = self.vis_2
        vis_3 = self.vis_3
        vis_4 = self.vis_4
        sig_1 = self.sig_1
        sig_2 = self.sig_2
        sig_3 = self.sig_3
        sig_4 = self.sig_4

        ndat = len(time)
        utime = np.unique(time)
        uant = np.unique(np.append(ant_name1, ant_name2))
        uvdist = np.sqrt(self.uvu**2 + self.uvv**2)

        if type is None or type not in \
                ["time", "sigma", "snr", "ant", "nant", "uvr"]:
            out_txt =\
                f"Given type '{type}' cannot be assigned." \
                " Please give appropriate type!" \
                " (available: 'time', 'sigma', 'snr', 'ant', 'nant', 'uvr')"
            raise Exception(out_txt)

        if value is None:
            out_txt =\
                "Please give appropriate value for flagging!"
            raise Exception(out_txt)
        else:
            if type == "time" and not isinstance(value, (list, tuple)):
                out_txt =\
                    "Please give flagging value as a list of floats!" \
                    " ([float1, float2])"
                raise Exception(out_txt)

            elif type in ["sigma", "snr", "nant"] \
                    and not isinstance(value, (int, float)):
                out_txt =\
                    "Please give flagging value" \
                    " as a float or an integer!"
                raise Exception(out_txt)

            elif type in ["ant", "antenna"] \
                    and not isinstance(value, (str, list, tuple)):
                out_txt =\
                    "Please give flagging value" \
                    " as a string or a list of strings!"
                raise Exception(out_txt)

            elif type == "uvr" and not isinstance(value, (list, tuple)):
                out_txt =\
                    "Please give flagging value as a list of floats!" \
                    " ([float1, float2])"
                raise Exception(out_txt)

        if type == "time":
            mask = ((value[0] <= time) & (time <= value[1]))
        elif type == "sigma":
            if self.select.lower() == "rr":
                sig = sig_1
            elif self.select.lower() == "ll":
                sig = sig_2
            elif self.select.lower() == "rl":
                sig = sig_3
            elif self.select.lower() == "lr":
                sig = sig_4
            elif self.select.lower() == "i":
                sig = np.sqrt(sig_1**2 + sig_2**2) * 0.5
            elif self.select.lower() == "q":
                sig = np.sqrt(sig_3**2 + sig_4**2) * 0.5
            elif self.select.lower() == "u":
                sig = np.sqrt(sig_3**2 + sig_4**2) * 0.5
            elif self.select.lower() == "v":
                sig = np.sqrt(sig_1**2 + sig_2**2) * 0.5
            elif self.select.lower() == "p":
                sig_q = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                sig_u = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                sig = np.sqrt(sig_q**2 + sig_u**2) * 1.0
            mask = sig >= value

        elif type == "snr":
            if self.select.lower() == "rr":
                vis = vis_1
                sig = sig_1
            elif self.select.lower() == "ll":
                vis = vis_2
                sig = sig_2
            elif self.select.lower() == "rl":
                vis = vis_3
                sig = sig_3
            elif self.select.lower() == "lr":
                vis = vis_4
                sig = sig_4
            elif self.select.lower() == "i":
                vis = (vis_1 + vis_2) * 0.5
                sig = np.sqrt(sig_1**2 + sig_2**2) * 0.5
            elif self.select.lower() == "q":
                vis = (vis_3 + vis_4) * 0.5
                sig = np.sqrt(sig_3**2 + sig_4**2) * 0.5
            elif self.select.lower() == "u":
                vis = (vis_3 - vis_4) * 0.5 / 1j
                sig = np.sqrt(sig_3**2 + sig_4**2) * 0.5
            elif self.select.lower() == "v":
                vis = (vis_1 - vis_2) * 0.5
                sig = np.sqrt(sig_1**2 + sig_2**2) * 0.5
            elif self.select.lower() == "p":
                vis_q = (vis_3 + vis_4) * 0.5
                vis_u = (vis_3 - vis_4) * 0.5 / 1j
                sig_q = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                sig_u = np.sqrt(sig_3**2 + sig_4**2) * 0.5
                vis = (vis_q + 1j * vis_u) * 1.0
                sig = np.sqrt(sig_q**2 + sig_u**2) * 1.0
            snr = np.abs(vis) / sig
            mask = snr <= value

        elif type in ["ant", "antenna"]:
            if isinstance(value, str):
                ant_name1 = self.ant_name1
                ant_name2 = self.ant_name2
                mask =\
                    (ant_name1 == value.upper()) \
                    | (ant_name2 == value.upper())

                if np.sum(mask) == 0 and prt:
                    out_txt =\
                        f"None of '{value.upper()}' visibility." \
                        " Skip flagging on that antenna."
                    print(out_txt)

            elif isinstance(value, list):
                mask = np.zeros(ndat, dtype=bool)
                for ant in value:
                    mask_ =\
                        (ant_name1 == ant.upper()) \
                        | (ant_name2 == ant.upper())
                    if np.sum(mask_) == 0 and prt:
                        out_txt =\
                            f"None of '{ant.upper()}' visibility." \
                            " Skip flagging on that antenna."
                        print(out_txt)
                    else:
                        mask = mask | mask_

        elif type == "nant":
            mask = np.zeros(ndat, dtype=bool)
            for time_ in utime:
                mask_time = time == time_
                ant_name1_ = ant_name1[mask_time]
                ant_name2_ = ant_name2[mask_time]
                nant = len(np.unique(np.append(ant_name1_, ant_name2_)))
                if nant < value or len(ant_name1_) < value * (value - 1) / 2:
                    mask[mask_time] = True

        elif type == "uvr":
            if unit.lower() in ["k", "kilo"]:
                factor = 1e3
            elif unit.lower() in ["m", "mega"]:
                factor = 1e6
            elif unit.lower() in ["g", "giga"]:
                factor = 1e9
            uvdist_ = uvdist / factor
            mask = ((value[0] <= uvdist_) & (uvdist_ <= value[1]))

        if prt:
            if type != "uvr":
                out_txt =\
                    f"# Flag {np.sum(mask)}/{ndat} visibilities" \
                    f" (type='{type}', value={value})"
                print(out_txt)
            else:
                out_txt =\
                    f"# Flag {np.sum(mask)}/{ndat} visibilities" \
                    f"(type='{type}', value={value}, unit='{unit}')"
                print(out_txt)

        self.time = time[~mask]
        self.tint = tint[~mask]
        self.mjd = mjd[~mask]
        self.ant_name1 = ant_name1[~mask]
        self.ant_name2 = ant_name2[~mask]
        self.uvu = uvu[~mask]
        self.uvv = uvv[~mask]
        self.ww = ww[~mask]
        self.vis_1 = vis_1[~mask]
        self.vis_2 = vis_2[~mask]
        self.vis_3 = vis_3[~mask]
        self.vis_4 = vis_4[~mask]
        self.sig_1 = sig_1[~mask]
        self.sig_2 = sig_2[~mask]
        self.sig_3 = sig_3[~mask]
        self.sig_4 = sig_4[~mask]

        if np.sum(mask) != 0:
            self.set_uvvis()
            if self.set_clq:
                self.set_closure(minclq=self.minclq)
            self.fit_beam(uvw=self.uvw)

    def selfcal(self,
                type="phs", gnorm=False, tint=False,
                startmod=False, lm=(0, 0), selfflag=True, prt=True):
        """
        Do self-calibration based on model visibility
            Arguments:
                type (str): self-calibration type // amp, phs, a&p, gscale
                gnorm (bool): if True, normalize gain amplitude solutions
                tint (list, float): interval times [s]
                startmod (bool): if True, do phase self-calibration
                                 using 1 Jy point-like source
                lm (tuple): position of the point source
        """
        if prt:
            out_txt =\
                f"# Self-calibration" \
                f" (startmod={startmod}, type={type}, {self.freq:.1f} GHz)"
            print(out_txt)
            if startmod:
                out_txt =\
                    f"# Startmod : Self-calibrating" \
                    f" to 1 Jy point source at ({lm[0]}, {lm[1]})"
                print(out_txt)

        def cal_nll(theta, indata, inobs, inmod, inants):
            nant = int(len(theta) / 2)
            vsigma = np.abs(indata["sigma"])
            gamp = dict(map(lambda i, j: (i, j), inants, [*theta[:nant]]))
            gphs = dict(map(lambda i, j: (i, j), inants, [*theta[nant:]]))
            ant1 = indata["ant_name1"]
            ant2 = indata["ant_name2"]
            gain1 =\
                (
                    np.array(list(map(gamp.get, ant1)))
                    * np.exp(1j * np.array(list(map(gphs.get, ant1))))
                )

            gain2 =\
                (
                    np.array(list(map(gamp.get, ant2)))
                    * np.exp(1j * np.array(list(map(gphs.get, ant2))))
                ).conj()

            if type == "gscale":
                gain1 = np.abs(gain1)
                gain2 = np.abs(gain2)
                crmod = (gain1 * gain2) * inmod
                amp_obs = np.abs(inobs)
                amp_mod = np.abs(crmod)
                sig_amp = vsigma
                out_nll =\
                    0.5 * np.sum(
                        (amp_mod - amp_obs)**2 / sig_amp**2
                        + np.log(2 * np.pi * sig_amp**2)
                    )

            elif type == "amp":
                crmod = (gain1 * gain2) * inmod
                amp_obs = np.abs(inobs)
                amp_mod = np.abs(crmod)
                sig_amp = vsigma
                out_nll =\
                    0.5 * np.sum(
                        (amp_mod - amp_obs)**2 / sig_amp**2
                        + np.log(2 * np.pi * sig_amp**2)
                    )

            elif type == "phs":
                crmod = (gain1 * gain2) * inmod
                phs_obs = np.angle(inobs)
                phs_mod = np.angle(crmod)
                sig_phs = vsigma / np.abs(inobs)
                out_nll =\
                    0.5 * np.sum(
                        np.abs(
                            np.exp(1j * phs_mod) - np.exp(1j * phs_obs)
                        )**2 / sig_phs**2
                        + np.log(2 * np.pi * sig_phs**2)
                    )

            elif type == "a&p":
                crmod = (gain1 * gain2) * inmod
                out_nll =\
                    0.5 * np.sum(
                        np.abs(crmod - inobs)**2 / vsigma**2
                        + np.log(2 * np.pi * vsigma**2)
                    )

            return out_nll

        if not tint:
            tint =\
                [
                    2880, 1920, 1280, 853, 569, 379, 253, 169, 113, 75, 50,
                    33, 22, 15, 10, 7, 5, 3, 2, 1, 30/60, 10/60, 5/60
                ]

            if not isinstance(self.avgtime, str):
                tint = list(filter(lambda x: x >= self.avgtime / 60, tint))
        if type == "gscale":
            tint = [2880, 1920, 1280, 850, 570, 380, 250, 180]
        tint = np.repeat(np.array(tint), 1).tolist()

        if not isinstance(tint, list):
            tint = np.array([tint])

        if isinstance(tint, list):
            tint = np.array(tint)

        time = self.data["time"]
        time_sec = time * 3600
        out_cgain1 = np.ones(self.data.shape[0]) * np.exp(1j * 0)
        out_cgain2 = np.ones(self.data.shape[0]) * np.exp(1j * 0)

        for nt, t in enumerate(tint):
            data = self.data
            db = dbs(eps=self.gaptime, min_samples=1) \
                .fit((time * 3600).reshape(-1, 1))
            scannums = db.labels_
            uscan = np.unique(scannums)

            cgain1 = None
            cgain2 = None
            for nscan, scannum in enumerate(uscan):
                mask_scan = scannums == scannum
                time_norm = time_sec - np.min(time_sec[mask_scan])
                timer = np.arange(0, np.max(time_norm) + 2 * t * 60, t * 60)

                for ntime in range(len(timer) - 1):
                    mask_time =\
                        (timer[ntime + 0] <= time_norm) \
                        & (time_norm < timer[ntime + 1])
                    mask_tot = (mask_scan & mask_time)

                    data_ = data[mask_tot]
                    time_ = time[mask_tot]

                    ants = np.unique(
                        np.append(data_["ant_name1"], data_["ant_name2"])
                    )

                    if len(data_) == 0:
                        continue

                    if len(ants) < 3:
                        gain1 = np.ones(len(data_)) * np.exp(1j * 0)
                        gain2 = np.ones(len(data_)) * np.exp(1j * 0)
                    elif len(ants) == 3 and selfflag and type in ["amp", "a&p"]:
                        gain1 = np.ones(len(data_)) * np.exp(1j * 0)
                        gain2 = np.ones(len(data_)) * np.exp(1j * 0)
                    else:
                        if startmod:
                            obs = data_["vis"]
                            sig = data_["sigma"]
                            r = np.abs(lm[1] + 1j * lm[0])
                            p = np.angle(lm[1] + 1j * lm[0], deg=True)

                            mod = np.ones(len(obs), dtype="c8")
                        else:
                            obs = data_["vis"]
                            sig = data_["sigma"]
                            mod = data_["vism"]

                        if len(ants) == 3 and t < self.scanlen / 60:
                            gain1 = np.ones(len(data_)) * np.exp(1j * 0)
                            gain2 = np.ones(len(data_)) * np.exp(1j * 0)
                        else:
                            init =\
                                np.append(
                                    np.ones(len(ants)),
                                    np.zeros(len(ants))
                                )

                            bound1 = [[+0.5, +1.5] for i in range(len(ants))]
                            bound2 =\
                                [[-np.pi, +np.pi] for i in range(len(ants))]
                            bounds = bound1 + bound2

                            def nll(*args):
                                return cal_nll(*args)

                            soln =\
                                optimize.minimize(
                                    nll, init, args=(data_, obs, mod, ants),
                                    bounds=bounds, method="Powell"
                                )

                            gamp =\
                                dict(map(
                                    lambda i, j: \
                                        (i, j), ants, [*soln.x[:len(ants)]]
                                ))

                            gphs =\
                                dict(map(
                                    lambda i, j: \
                                        (i, j), ants, [*soln.x[len(ants):]]
                                ))

                            ant1 = data_["ant_name1"]
                            ant2 = data_["ant_name2"]
                            gain1 =\
                                (
                                    np.array(list(map(gamp.get, ant1)))
                                    * np.exp(1j *
                                        np.array(list(map(gphs.get, ant1)))
                                    )
                                )

                            gain2 =\
                                (
                                    np.array(list(map(gamp.get, ant2)))
                                    * np.exp(1j *
                                        np.array(list(map(gphs.get, ant2)))
                                    )
                                ).conj()

                            if type in ["phs"] or startmod:
                                gain1 = gain1 / np.abs(gain1)
                                gain2 = gain2 / np.abs(gain2)
                            elif type in ["amp", "gscale"]:
                                if type == "gscale" and gnorm:
                                    gains =\
                                        np.unique(
                                            np.append(
                                                np.abs(gain1), np.abs(gain2))
                                        )

                                    gain1 =\
                                        np.abs(gain1) \
                                        / np.prod(gains)**(1/len(gains))

                                    gain2 =\
                                        np.abs(gain2) \
                                        / np.prod(gains)**(1/len(gains))

                                gain1 = np.abs(gain1) * np.exp(1j * 0)
                                gain2 = np.abs(gain2) * np.exp(1j * 0)
                            elif type in ["a&p"]:
                                gain1 = gain1
                                gain2 = gain2

                    if cgain1 is None and cgain2 is None:
                        cgain1 = gain1
                        cgain2 = gain2
                    else:
                        cgain1 = np.hstack((cgain1, gain1))
                        cgain2 = np.hstack((cgain2, gain2))

            self.data["vis"] /= (cgain1*cgain2)
            self.data["sigma"] /= (np.abs(cgain1) * np.abs(cgain2))
            self.vis_1 /= (cgain1*cgain2)
            self.sig_1 /= (np.abs(cgain1) * np.abs(cgain2))

            if self.nstokes >= 2:
                self.vis_2 /= (cgain1*cgain2)
                self.sig_2 /= (np.abs(cgain1) * np.abs(cgain2))

            if self.nstokes == 4:
                self.vis_3 /= (cgain1*cgain2)
                self.vis_4 /= (cgain1*cgain2)
                self.sig_3 /= (np.abs(cgain1) * np.abs(cgain2))
                self.sig_4 /= (np.abs(cgain1) * np.abs(cgain2))

            out_cgain1 *= cgain1
            out_cgain2 *= cgain2

        if not startmod:
            vism = self.data["vism"]

        self.time = self.data["time"]
        self.tint = self.data["tint"]
        self.mjd = self.data["mjd"]
        self.ant_name1 = self.data["ant_name1"]
        self.ant_name2 = self.data["ant_name2"]
        self.uvu = self.data["u"]
        self.uvv = self.data["v"]
        self.set_uvvis()

        if not startmod:
            self.data = rfn.append_fields(
                self.data, "vism", vism, usemask=False
            )

        self.cgain1 = out_cgain1
        self.cgain2 = out_cgain2

    def fit_beam(self,
                 npix=256, uvw="natural"):
        """
        Fit beam parameters under Gaussian approximation.
        Given observed (u,v)-coordinates, the beam response near its peak
        is approximated as a Gaussian
        (TMS, Interferometry and Synthesis, 2017, Thompson, Moran, Swenson):
            B_0(l,m) \approx
                1 - 2*pi^2/N * (l^2*sum(u^2) - 2*l*m*sum(u*v) + m^2*sum(v^2))

        Beam parameters can be determined by fitting the latter one to
        a 2D Gaussian function:
            F(x,y) = exp(-(a*x^2 + b*y^2 + c*x*y)),
        where (a,b,c) are functions of beam parameters, (b_min, b_maj, b_pa).

        Arguments:
            npix (int): number of pixels for the beam fitting
            uvw (str): UV weighting option
                - n: natural weighting
                - u: uniform weighting
        """

        uvu = self.data["u"]
        uvv = self.data["v"]
        vis = self.data["vis"]
        sig = self.data["sigma"]

        def fit_abc(beamparams, in_abc):
            """
            FWHM = 2 * sqrt(2 * ln(2)) * sigma
            b_min**2 = 4 * 2 * ln(2) * sig_x**2
            """
            (b_maj, b_min, b_pa) = beamparams
            sig_x2 = b_min**2 / (4 * 2 * np.log(2))
            sig_y2 = b_maj**2 / (4 * 2 * np.log(2))
            a =\
                (
                    np.cos(b_pa)**2 / sig_x2
                    + np.sin(b_pa)**2 / sig_y2
                ) * 0.5

            b =\
                (
                    (1 / sig_y2 - 1 / sig_x2)
                    * np.cos(b_pa) * np.sin(b_pa) * 0.5
                )

            c =\
                (
                    np.sin(b_pa)**2 / sig_x2
                    + np.cos(b_pa)**2 / sig_y2
                ) * 0.5

            out_abc = np.array([a, b, c])
            out = np.sum((in_abc - out_abc)**2)
            return out

        if uvw in ["n", "natural"]:
            wfn = 1 / sig**2
        else:
            wfn = np.ones(vis.shape)

        rsc = 1e-9  # Rescaling factor for fitting

        abc =\
            np.array([
                np.sum(wfn * uvu**2),
                np.sum(wfn * uvu * uvv),
                np.sum(wfn * uvv**2)
                ]) \
            * 2 * np.pi**2 / np.sum(wfn)

        abc *= rsc ** 2

        # Fit the beam
        guess = np.array([1, 1, 0])

        Gprms =\
            optimize.minimize(
                fit_abc, guess, args=abc,
                method="Powell"
            )

        if Gprms.x[0] > Gprms.x[1]:
            fwhm_maj = rsc * Gprms.x[0]
            fwhm_min = rsc * Gprms.x[1]
            theta = np.mod(Gprms.x[2], np.pi)
        else:
            fwhm_maj = rsc * Gprms.x[1]
            fwhm_min = rsc * Gprms.x[0]
            theta = np.mod(Gprms.x[2] + np.pi / 2, np.pi)

        bprms = np.array((fwhm_maj, fwhm_min, theta))
        bprms[0] *= u.rad.to(u.mas)
        bprms[1] *= u.rad.to(u.mas)
        bprms[2] *= u.rad.to(u.deg)

        self.bmin = bprms[1]
        self.bmaj = bprms[0]
        self.bpa = bprms[2]
        self.bprms = (self.bmin, self.bmaj, self.bpa)
        self.ploter.bprms = self.bprms

    def apply_systematics(self, d=0.0, m=0.0):
        types = self.systype

        for systype in types:
            if systype in ["vis", "amp"]:
                data = self.data
                systematics = self.systematics_vis
                label_sigma = "sigma"
                label_pair = "pair"
            elif systype == "clamp":
                data = self.clamp
                systematics = self.systematics_clamp
                label_sigma = "sigma_clamp"
                label_pair = "quadra"
            elif systype == "clphs":
                systematics = self.systematics_clphs
                data = self.clphs
                label_sigma = "sigma_clphs"
                label_pair = "triangle"

            db = dbs(eps=self.gaptime, min_samples=1) \
                .fit((data["time"] * 3600).reshape(-1, 1))
            scannums = db.labels_
            uscan = np.unique(scannums)

            out = np.zeros(len(data))
            count = np.zeros(len(data))

            for nscan in range(len(systematics)):
                time1_ = systematics["time_beg"][nscan]
                time2_ = systematics["time_end"][nscan]
                pair_ = systematics["pair"][nscan]
                systematics_ = systematics["systematics"][nscan]

                mask_time1 = time1_ <= data["time"]
                mask_time2 = data["time"] <= time2_
                mask_pair = data[label_pair] == pair_
                mask = mask_time1 & mask_time2 & mask_pair
                if np.sum(mask) == 0:
                    continue
                else:
                    out[mask] = systematics_
                    count[mask] += 1

            if np.max(count) >= 2:
                out_txt =\
                    f"WARNING:" \
                    f" Some duplicates are found in applying systematics." \
                    f" (type: '{systype}')"
                print(out_txt)

            if systype == "vis":
                self.data["sigma"] =\
                    np.sqrt(self.data["sigma"]**2 + out**2) \
                    + np.abs(self.data["vis"]) * np.sqrt(2 * 2) * d * m

                self.sig_1 =\
                    np.sqrt(self.sig_1**2 + out**2) \
                    + np.abs(self.data["vis"]) * np.sqrt(2 * 2) * d * m

                self.sig_2 =\
                    np.sqrt(self.sig_2**2 + out**2) \
                    + np.abs(self.data["vis"]) * np.sqrt(2 * 2) * d * m

                if self.nstokes == 4:
                    self.sig_3 =\
                        np.sqrt(self.sig_3**2 + out**2) \
                        + np.abs(self.data["vis"]) * np.sqrt(2 * 2) * d * m

                    self.sig_4 =\
                        np.sqrt(self.sig_4**2 + out**2) \
                        + np.abs(self.data["vis"]) * np.sqrt(2 * 2) * d * m

            elif systype == "clamp":
                self.clamp["sigma_clamp"] =\
                    np.sqrt(self.clamp["sigma_clamp"]**2 + out**2) \
                    + (
                        np.sqrt(2 * 4) * d * m
                        * np.abs(np.log(self.clamp["clamp"]))
                    )
                self.ploter.clq_obs =\
                    (
                        copy.deepcopy(self.clamp),
                        copy.deepcopy(self.clphs)
                    )

            elif systype == "clphs":
                self.clphs["sigma_clphs"] =\
                    np.sqrt(self.clphs["sigma_clphs"]**2 + out**2) \
                    + (np.sqrt(2 * 3) * d * m)
                self.ploter.clq_obs =\
                    (
                        copy.deepcopy(self.clamp),
                        copy.deepcopy(self.clphs)
                    )

    def append_visibility_model(self,
                                freq_ref, freq, theta, pol=False,
                                model="gaussian", fitset="sf",
                                spectrum="spl", set_spectrum=True, args=None):
        """
        Append the model visibility to the data
            Arguments:
                freq_ref (float): reference frequency [GHz]
                freq (float): frequency [GHz]
                theta (dict): model parameters
                pol (bool): if True, assumes polarization model
                fitset (str): frequency setting
                              - sf: single-frequency
                              - mf: multi-frequency
                spectrum (str): spectrum type
                                - spl: simple power-law
                                - cpl: curved power-law
                                - ssa: synchrotron self-absorption
                args (tuple): arguments for the model
        """
        if "1_l" in theta.dtype.names:
            relmod = False
        else:
            relmod = True

        uvdat = self.data
        nvis = uvdat.size
        vism = np.zeros(nvis, dtype="c8")
        if pol:
            theta = theta.tolist()
            nmod = len(theta)
            if model == "gaussian":
                for i in range(nmod):
                    vism +=\
                        gamvas.polarization.functions.gvis(
                            (
                                args[0], args[1],
                                args[2][i], args[3][i], args[4][i]
                            ),
                            theta[i]
                        )

            elif model == "delta":
                for i in range(nmod):
                    vism +=\
                        gamvas.polarization.functions.dvis(
                            (
                                args[0], args[1],
                                args[2][i], args[3][i]
                            ),
                            theta[i]
                        )

        else:
            nmod = int(np.round(theta["nmod"]))
            if model == "gaussian":
                for i in range(nmod):
                    if fitset == "sf":
                        if i == 0:
                            args = (uvdat["u"], uvdat["v"])
                            if relmod:
                                vism =\
                                    vism + gamvas.functions.gvis0(
                                        args,
                                        theta[f"{i+1}_S"],
                                        theta[f"{i+1}_a"]
                                    )
                            else:
                                vism =\
                                    vism + gamvas.functions.gvis(
                                        args,
                                        theta[f"{i+1}_S"],
                                        theta[f"{i+1}_a"],
                                        theta[f"{i+1}_l"],
                                        theta[f"{i+1}_m"]
                                    )
                        else:
                            args = (uvdat["u"], uvdat["v"])
                            vism =\
                                vism + gamvas.functions.gvis(
                                    args,
                                    theta[f"{i+1}_S"],
                                    theta[f"{i+1}_a"],
                                    theta[f"{i+1}_l"],
                                    theta[f"{i+1}_m"]
                                )

                    elif fitset == "mf":
                        if set_spectrum:
                            if i == 0:
                                if spectrum == "spl":
                                    args =\
                                        (
                                            freq_ref,
                                            freq,
                                            uvdat["u"],
                                            uvdat["v"]
                                        )

                                    if relmod:
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_spl0(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_alpha"]
                                            )
                                    else:
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_spl(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_l"],
                                                theta[f"{i+1}_m"],
                                                theta[f"{i+1}_alpha"]
                                            )
                                elif spectrum == "cpl":
                                    args = (freq, uvdat["u"], uvdat["v"])
                                    if relmod:
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_cpl0(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_alpha"],
                                                theta[f"{i+1}_freq"]
                                            )
                                    else:
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_cpl(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_l"],
                                                theta[f"{i+1}_m"],
                                                theta[f"{i+1}_alpha"],
                                                theta[f"{i+1}_freq"]
                                            )
                                elif spectrum == "ssa":
                                    args = (freq, uvdat["u"], uvdat["v"])
                                    if relmod:
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_ssa0(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_alpha"],
                                                theta[f"{i+1}_freq"]
                                            )
                                    else:
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_ssa(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_l"],
                                                theta[f"{i+1}_m"],
                                                theta[f"{i+1}_alpha"],
                                                theta[f"{i+1}_freq"]
                                            )
                            else:
                                if spectrum == "spl":
                                    args =\
                                        (
                                            freq_ref,
                                            freq,
                                            uvdat["u"],
                                            uvdat["v"]
                                        )

                                    vism =\
                                        vism \
                                        + gamvas.functions.gvis_spl(
                                            args,
                                            theta[f"{i+1}_S"],
                                            theta[f"{i+1}_a"],
                                            theta[f"{i+1}_l"],
                                            theta[f"{i+1}_m"],
                                            theta[f"{i+1}_alpha"]
                                        )
                                elif int(np.round(theta[f"{i+1}_thick"])) == 0:
                                    args =\
                                        (
                                            freq_ref,
                                            freq,
                                            uvdat["u"],
                                            uvdat["v"]
                                        )

                                    vism =\
                                        vism \
                                        + gamvas.functions.gvis_spl(
                                            args,
                                            theta[f"{i+1}_S"],
                                            theta[f"{i+1}_a"],
                                            theta[f"{i+1}_l"],
                                            theta[f"{i+1}_m"],
                                            theta[f"{i+1}_alpha"]
                                        )
                                else:
                                    if spectrum == "cpl":
                                        args = (freq, uvdat["u"], uvdat["v"])
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_cpl(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_l"],
                                                theta[f"{i+1}_m"],
                                                theta[f"{i+1}_alpha"],
                                                theta[f"{i+1}_freq"]
                                            )
                                    elif spectrum == "ssa":
                                        args = (freq, uvdat["u"], uvdat["v"])
                                        vism =\
                                            vism \
                                            + gamvas.functions.gvis_ssa(
                                                args,
                                                theta[f"{i+1}_S"],
                                                theta[f"{i+1}_a"],
                                                theta[f"{i+1}_l"],
                                                theta[f"{i+1}_m"],
                                                theta[f"{i+1}_alpha"],
                                                theta[f"{i+1}_freq"]
                                            )
                        else:
                            if i == 0:
                                args = (uvdat["u"], uvdat["v"])
                                if relmod:
                                    vism =\
                                        vism \
                                        + gamvas.functions.gvis0(
                                            args,
                                            theta[f"{i+1}_S"],
                                            theta[f"{i+1}_a"]
                                        )
                                else:
                                    vism =\
                                        vism \
                                        + gamvas.functions.gvis(
                                            args,
                                            theta[f"{i+1}_S"],
                                            theta[f"{i+1}_a"],
                                            theta[f"{i+1}_l"],
                                            theta[f"{i+1}_m"]
                                        )
                            else:
                                args = (uvdat["u"], uvdat["v"])
                                vism =\
                                    vism \
                                    + gamvas.functions.gvis(
                                        args,
                                        theta[f"{i+1}_S"],
                                        theta[f"{i+1}_a"],
                                        theta[f"{i+1}_l"],
                                        theta[f"{i+1}_m"]
                                    )
            elif model == "delta":
                for i in range(nmod):
                    if fitset == "sf":
                        if i == 0:
                            args = (uvdat["u"], uvdat["v"])
                            if relmod:
                                vism =\
                                    vism \
                                    + gamvas.functions.dvis0(
                                        args,
                                        theta[f"{i+1}_S"]
                                    )
                            else:
                                vism =\
                                    vism \
                                    + gamvas.functions.dvis(
                                        args,
                                        theta[f"{i+1}_S"],
                                        theta[f"{i+1}_l"],
                                        theta[f"{i+1}_m"]
                                    )
                        else:
                            args = (uvdat["u"], uvdat["v"])
                            vism =\
                                vism \
                                + gamvas.functions.dvis(
                                    args,
                                    theta[f"{i+1}_S"],
                                    theta[f"{i+1}_l"],
                                    theta[f"{i+1}_m"]
                                )

        uvdat = rfn.append_fields(uvdat, "vism", vism, usemask=False)

        self.data = uvdat

    def drop_visibility_model(self):
        """
        Drop the model visibility from the data
        """
        uvdat = self.data

        if "vism" in uvdat.dtype.names:
            uvdat = rfn.drop_fields(uvdat, "vism")

        self.data = uvdat

    def add_error_fraction(self,
                           fraction=0.01, time="all", antenna="all",
                           selfant=False, selffrac=0.1,
                           set_vis=True, set_clq=False):
        """
        Add the error fractionally to the visibility amplitude
            Arguments:
                fraction (float): fraction of the error to be added
                time (list, float): time range to apply
                antenna (list, str): antenna name or list of antenna names
                                     to apply
                set_vis (bool): if True, set complex visibility
                set_clq (bool): if True, set closure quantities
        """
        data = self.data

        if self.nstokes == 1:
            if "vis_rr" in data.dtype.names:
                vis_1 = data["vis_rr"]
                sig_1 = data["sigma_rr"]
            else:
                vis_1 = data["vis_ll"]
                sig_1 = data["sigma_ll"]

            vis_2 = self.vis_2
            vis_3 = self.vis_3
            vis_4 = self.vis_4
            sig_2 = self.sig_2
            sig_3 = self.sig_3
            sig_4 = self.sig_4
        if self.nstokes == 2:
            vis_1 = data["vis_rr"]
            vis_2 = data["vis_ll"]
            vis_3 = self.vis_3
            vis_4 = self.vis_4
            sig_1 = data["sigma_rr"]
            sig_2 = data["sigma_ll"]
            sig_3 = self.sig_3
            sig_4 = self.sig_4
        if self.nstokes == 4:
            vis_1 = data["vis_rr"]
            vis_2 = data["vis_ll"]
            vis_3 = data["vis_rl"]
            vis_4 = data["vis_lr"]
            sig_1 = data["sigma_rr"]
            sig_2 = data["sigma_ll"]
            sig_3 = data["sigma_rl"]
            sig_4 = data["sigma_lr"]

        if antenna == "all":
            mask_ant = np.ones(len(sig_1), dtype=bool)
        else:
            if isinstance(antenna, list):
                mask_ant = np.zeros(len(sig_1), dtype=bool)
                for nant, antenna_ in enumerate(antenna):
                    mask_ =\
                        (data["ant_name1"] == antenna_.upper()) \
                        | (data["ant_name2"] == antenna_.upper())

                    mask_ant[mask_] = True
            else:
                mask_ant =\
                    (data["ant_name1"] == antenna.upper()) \
                    | (data["ant_name2"] == antenna.upper())

        if time == "all":
            mask_time = np.ones(len(sig_1), dtype=bool)
        else:
            mask_time = (time[0] < data["time"]) & (data["time"] < time[1])

        mask = mask_ant & mask_time
        self.sig_1[mask] += fraction * np.abs(vis_1[mask])
        self.sig_2[mask] += fraction * np.abs(vis_2[mask])
        self.sig_3[mask] += fraction * np.abs(vis_3[mask])
        self.sig_4[mask] += fraction * np.abs(vis_4[mask])

        if selfant:
            utime = np.unique(data["time"])
            for i in range(len(utime)):
                mask_time = data["time"] == utime[i]
                data_ = data[mask_time]
                ant1_ = data_["ant_name1"]
                ant2_ = data_["ant_name2"]
                uant_ = np.unique(np.concatenate([ant1_, ant2_]))
                mask_ant = len(uant_) < 4
                if mask_ant:
                    self.sig_1[mask_time] +=\
                        selffrac * np.abs(vis_1[mask_time])
                    self.sig_2[mask_time] +=\
                        selffrac * np.abs(vis_2[mask_time])
                    self.sig_3[mask_time] +=\
                        selffrac * np.abs(vis_3[mask_time])
                    self.sig_4[mask_time] +=\
                        selffrac * np.abs(vis_4[mask_time])

        if set_vis:
            self.set_uvvis()
        if set_clq:
            self.set_closure(minclq=self.minclq)

    def add_error_factor(self,
                         factor=1, time="all", antenna="all",
                         selfant=False, selffactor=0.1,
                         set_vis=True, set_clq=False):
        """
        Add the error by a factor
            Arguments:
                factor (float): factor of the error to be added
                time (list, float): time range to apply
                antenna (list, str): antenna name or list of antenna names
                                     to apply
                set_vis (bool): if True, set complex visibility
                set_clq (bool): if True, set closure quantities
        """
        data = self.data

        if self.nstokes == 1:
            if "sigma_rr" in data.dtype.names:
                sig_1 = data["sigma_rr"]
            else:
                sig_1 = data["sigma_ll"]
            sig_2 = self.sig_2
            sig_3 = self.sig_3
            sig_4 = self.sig_4

        if self.nstokes == 2:
            sig_1 = data["sigma_rr"]
            sig_2 = data["sigma_ll"]
            sig_3 = self.sig_3
            sig_4 = self.sig_4

        if self.nstokes == 4:
            sig_1 = data["sigma_rr"]
            sig_2 = data["sigma_ll"]
            sig_3 = data["sigma_rl"]
            sig_4 = data["sigma_lr"]

        if antenna == "all":
            mask_ant = np.ones(len(sig_1), dtype=bool)
        else:
            if isinstance(antenna, list):
                mask_ant = np.zeros(len(sig_1), dtype=bool)
                for nant, antenna_ in enumerate(antenna):
                    mask_ =\
                        (data["ant_name1"] == antenna_.upper()) \
                        | (data["ant_name2"] == antenna_.upper())

                    mask_ant[mask_] = True
            else:
                mask_ant =\
                    (data["ant_name1"] == antenna.upper()) \
                    | (data["ant_name2"] == antenna.upper())

        if time == "all":
            mask_time = np.ones(len(sig_1), dtype=bool)
        else:
            mask_time = (time[0] < data["time"]) & (data["time"] < time[1])

        mask = mask_ant & mask_time
        self.sig_1[mask] += factor * np.abs(sig_1[mask])
        self.sig_2[mask] += factor * np.abs(sig_2[mask])
        self.sig_3[mask] += factor * np.abs(sig_3[mask])
        self.sig_4[mask] += factor * np.abs(sig_4[mask])

        if selfant:
            utime = np.unique(data["time"])
            for i in range(len(utime)):
                mask_time = data["time"] == utime[i]
                data_ = data[mask_time]
                ant1_ = data_["ant_name1"]
                ant2_ = data_["ant_name2"]
                uant_ = np.unique(np.concatenate([ant1_, ant2_]))
                mask_ant = len(uant_) < 4
                if mask_ant:
                    self.sig_1[mask_time] +=\
                        selffactor * np.abs(sig_1[mask_time])
                    self.sig_2[mask_time] +=\
                        selffactor * np.abs(sig_2[mask_time])
                    self.sig_3[mask_time] +=\
                        selffactor * np.abs(sig_3[mask_time])
                    self.sig_4[mask_time] +=\
                        selffactor * np.abs(sig_4[mask_time])

        if set_vis:
            self.set_uvvis()
        if set_clq:
            self.set_closure(minclq=self.minclq)

    def cal_pangle(self):
        """
        Compute the parallactic angle
        """
        data = self.data
        self.pangle = True

        tarr = self.tarr.copy()
        x = tarr["x"]
        y = tarr["y"]
        z = tarr["z"]
        r = np.sqrt(x**2 + y**2 + z**2)

        ant_geocent = EarthLocation(x=x*u.m, y=y*u.m, z=z*u.m)
        lon, lat, h = ant_geocent.to_geodetic()

        lat = lat.degree
        lon = lon.degree
        height = h.value

        mask_llh =\
            [
                field not in tarr.dtype.names
                for field in ["lat", "lon", "height"]
            ]

        if np.all(mask_llh):
            dtype_ =\
                np.dtype({
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

        obstime = Ati(data["mjd"], format="mjd").iso
        src_coord = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg)
        ants = self.tarr["name"]

        ant1 = data["ant_name1"]
        ant2 = data["ant_name2"]
        lat1 = np.zeros(data.shape[0])
        lon1 = np.zeros(data.shape[0])
        lat2 = np.zeros(data.shape[0])
        lon2 = np.zeros(data.shape[0])
        az1 = np.zeros(data.shape[0])
        el1 = np.zeros(data.shape[0])
        az2 = np.zeros(data.shape[0])
        el2 = np.zeros(data.shape[0])
        for tidx in range(len(ants)):
            ant_ = tarr["name"][tidx]
            lat_ = tarr["lat"][tidx]
            lon_ = tarr["lon"][tidx]
            h_ = tarr["height"][tidx]

            antpos =\
                EarthLocation(
                    lat=lat_ * u.deg,
                    lon=lon_ * u.deg,
                    height=h_ * u.m
                )

            src_azel =\
                src_coord.transform_to(
                    AltAz(obstime=obstime, location=antpos)
                )

            az1 = np.where(ant1 == ant_, src_azel.az.degree, az1)
            el1 = np.where(ant1 == ant_, src_azel.alt.degree, el1)
            az2 = np.where(ant2 == ant_, src_azel.az.degree, az2)
            el2 = np.where(ant2 == ant_, src_azel.alt.degree, el2)
            lat1 = np.where(ant1 == ant_, lat_, lat1)
            lon1 = np.where(ant1 == ant_, lon_, lon1)
            lat2 = np.where(ant2 == ant_, lat_, lat2)
            lon2 = np.where(ant2 == ant_, lon_, lon2)

        ra = np.pi / 180 * self.ra
        dec = np.pi / 180 * self.dec
        lat1_rad = np.pi / 180 * lat1
        lat2_rad = np.pi / 180 * lat2
        az1_rad = np.pi / 180 * az1
        az2_rad = np.pi / 180 * az2
        el1_rad = np.pi / 180 * el1
        el2_rad = np.pi / 180 * el2
        sin_pa1 = np.cos(lat1_rad) / np.cos(dec) * np.sin(az1_rad)

        cos_pa1 =\
            (np.sin(lat1_rad) - np.sin(dec) * np.sin(el1_rad)) \
            / (np.cos(dec) * np.cos(el1_rad))

        sin_pa2 = np.cos(lat2_rad) / np.cos(dec) * np.sin(az2_rad)

        cos_pa2 =\
            (np.sin(lat2_rad) - np.sin(dec) * np.sin(el2_rad)) \
            / (np.cos(dec) * np.cos(el2_rad))

        p_angle1 = -np.angle(cos_pa1 + 1j * sin_pa1)
        p_angle2 = -np.angle(cos_pa2 + 1j * sin_pa2)

        azel =\
            gamvas.utils.sarray(
                data=[self.data["time"], az1, az2, el1, el2],
                field=["time", "az1", "az2", "el1", "el2"],
                dtype=["f8", "f8", "f8", "f8", "f8"]
            )
        self.azel = azel


        if "phi1" in data.dtype.names:
            data["phi1"] = p_angle1
        else:
            data = rfn.append_fields(data, "phi1", p_angle1, usemask=False)

        if "phi2" in data.dtype.names:
            data["phi2"] = p_angle2
        else:
            data = rfn.append_fields(data, "phi2", p_angle2, usemask=False)

        self.data = data

    def cal_systematics(self,
                        binning=None, type=None):

        self.systype.append(type)

        def cal_s(s, X, sigma_th):
            Y = X / np.sqrt(sigma_th**2 + s**2)
            mad = np.nanmedian(np.abs(Y - np.nanmedian(Y)))
            out = np.abs(mad - 1/1.483)
            if np.isnan(out):
                return np.inf
            else:
                return out

        if type == "vis":
            data = self.data
            X = np.abs(data["vis"])
            pair = data["pair"]
            time = data["time"]
            sigma_th = data["sigma"]
        elif type == "clamp":
            data = self.clamp
            X = np.log(data["clamp"])
            pair = data["quadra"]
            time = data["time"]
            sigma_th = data["sigma_clamp"]
        elif type == "clphs":
            data = self.clphs
            X = data["clphs"]
            pair = data["triangle"]
            time = data["time"]
            sigma_th = data["sigma_clphs"]
        else:
            out_txt =\
                "Please provide appropriate type!" \
                " (availables: 'vis', 'clamp', 'clphs')"
            raise ValueError(out_txt)

        if not np.all(np.isnan(data["time"])):
            db =\
                dbs(eps=self.gaptime, min_samples=1) \
                .fit((time * 3600).reshape(-1, 1))
            scannums = db.labels_
            uscan = np.unique(scannums)
            upair = np.unique(pair)

            out_systematics = []
            out_pair = []
            out_time1 = []
            out_time2 = []
            for nscan, scan_ in enumerate(uscan):
                for npair, pair_ in enumerate(upair):
                    mask = (pair == pair_) & (scannums == scan_)
                    if np.sum(mask) == 0:
                        continue

                    def fn(*args):
                        return cal_s(*args)

                    soln =\
                        optimize.minimize(
                            fn,
                            [0.01],
                            args=(X[mask], sigma_th[mask]),
                            method="Powell"
                        )

                    out_time1.append(np.min(time[mask]))
                    out_time2.append(np.max(time[mask]))
                    out_pair.append(pair_)
                    out_systematics.append(np.abs(soln.x[0]))
            out =\
                gamvas.utils.sarray(
                    [out_time1, out_time2, out_pair, out_systematics],
                    dtype=["f8", "f8", "U32", "f8"],
                    field=["time_beg", "time_end", "pair", "systematics"]
                )

            if type == "vis":
                self.systematics_vis = out
            if type == "clamp":
                self.systematics_clamp = out
            if type == "clphs":
                self.systematics_clphs = out

    def cal_polarization(self,
                         snr_i=5, snr_p=3, evpalength=1, evpawidth=1):
        """
        Compute the polarization data
            Arguments:
                snr_i (float): signal-to-noise ratio of the total intensity
                snr_p (float): signal-to-noise ratio of the polarization
                evpalength (float): length of the EVPA
                evpawidth (float): width of the EVPA
        """
        self.fits_image_vp =\
            np.sqrt(self.fits_image_vq**2 + self.fits_image_vu**2)

        self.fits_image_rms_p =\
            (self.fits_image_rms_q + self.fits_image_rms_q) / 2

        self.fits_image_vpeak_p = np.nanmax(self.fits_image_vp)
        self.fits_clean_vflux_p =\
            np.sqrt(self.fits_clean_vflux_q**2 + self.fits_clean_vflux_u**2)

        sig_peak, sig_tot, sig_size =\
            cal_clean_error(
                s_peak=np.nanmax(self.fits_image_vp),
                s_tot=self.fits_clean_vflux_p,
                sig_rms=self.fits_image_rms_p
            )

        self.fits_image_dpeak_p = 0.1 * self.fits_image_vpeak_p
        self.fits_clean_dflux_p = 0.1 * self.fits_clean_vflux_p
        self.fits_clean_vfp =\
            self.fits_clean_vflux_p / self.fits_clean_vflux_i
        self.fits_clean_vevpa =\
            0.5 * np.arctan2(self.fits_clean_vflux_u, self.fits_clean_vflux_q)

        if self.fits_clean_vevpa > np.pi:
            self.fits_clean_vevpa -= np.pi
        if self.fits_clean_vevpa < 0:
            self.fits_clean_vevpa += np.pi

        ui = ufloat(self.fits_clean_vflux_i, self.fits_clean_dflux_i)
        uq = ufloat(self.fits_clean_vflux_q, self.fits_clean_dflux_q)
        uu = ufloat(self.fits_clean_vflux_u, self.fits_clean_dflux_u)
        up = ufloat(self.fits_clean_vflux_p, self.fits_clean_dflux_p)
        self.fits_clean_dfp = unp.std_devs(up / ui)
        self.fits_clean_devpa = unp.std_devs(0.5*unp.arctan2(uu, uq))

        self.fits_image_vp =\
            np.sqrt(self.fits_image_vq**2 + self.fits_image_vu**2)

        self.fits_image_vfp = self.fits_image_vp / self.fits_image_vi
        self.fits_image_vevpa =\
            0.5 * np.arctan2(self.fits_image_vu, self.fits_image_vq)

        self.fits_image_vevpa =\
            np.where(
                self.fits_image_vevpa > np.pi,
                self.fits_image_vevpa - np.pi,
                self.fits_image_vevpa
            )

        self.fits_image_vevpa =\
            np.where(
                self.fits_image_vevpa < 0,
                self.fits_image_vevpa + np.pi,
                self.fits_image_vevpa
            )

        self.fits_image_dp =\
            np.ones((self.fits_npix, self.fits_npix)) \
            * self.fits_image_rms_p

        self.fits_image_dfp =\
            np.ones((self.fits_npix, self.fits_npix)) \
            * np.abs((self.fits_image_rms_p / self.fits_image_vi))

        self.fits_image_devpa =\
            np.ones((self.fits_npix, self.fits_npix)) \
            * (self.fits_image_rms_p / (2*self.fits_image_vp))

        self.cal_evpa(
            snr_i=snr_i,
            snr_p=snr_p,
            evpalength=evpalength,
            evpawidth=evpawidth
        )

    def cal_evpa(self,
                 snr_i=5, snr_p=3, evpalength=1, evpawidth=1):
        """
        Compute electric vector position angle (EVPA) data
            Arguments:
                snr_i (float): signal-to-noise ratio of the total intensity
                snr_p (float): signal-to-noise ratio of the polarization
                evpalength (float): length of the EVPA
                evpawidth (float): width of the EVPA
        """
        evpalength = evpalength
        evpawidth = evpawidth * self.fits_psize * u.deg.to(u.mas)
        scale = 1 / evpalength
        self.fits_image_evpa_set =\
            dict(
                color="black", pivot="middle", units="xy",
                scale=scale, width=evpawidth,
                headlength=0, headwidth=0, headaxislength=0
            )

        fits_evpa_x, fits_evpa_y =\
            np.sin(self.fits_image_vevpa), -np.cos(self.fits_image_vevpa)

        mask =\
            (self.fits_image_vi >= snr_i * self.fits_image_rms_i) \
            & (self.fits_image_vp >= snr_p * self.fits_image_rms_p)

        self.fits_image_evpa_x = np.where(mask, fits_evpa_x, np.nan)
        self.fits_image_evpa_y = np.where(mask, fits_evpa_y, np.nan)
        self.fits_image_vevpa = np.where(mask, self.fits_image_vevpa, np.nan)
        self.fits_image_vp = np.where(mask, self.fits_image_vp, np.nan)
        self.fits_image_vfp = np.where(mask, self.fits_image_vfp, np.nan)

    def cal_Tb(self):
        """
        Compute the brightness temperature
        """
        if self.fits_clean_vflux_i is not None:
            freq = self.fits_freq
            psize = np.round(self.fits_psize * u.deg.to(u.mas), 5)

            beam =\
                self.fits_bmaj * self.fits_bmin \
                * np.pi * (u.deg.to(u.mas))**2

            factor = (1 * u.Jy / 2 / C.k_B * C.c**2).to(u.K * u.Hz**2).value
            vtb = factor * self.fits_image_vi / freq**2
            dtb = factor * self.fits_image_vi / freq**2
            self.fits_image_vtb = vtb
            self.fits_image_dtb = dtb
        else:
            out_txt =\
                "FITS-image is not provided." \
                " Please check if you load UV-fits file."
            raise Exception(out_txt)

    def cal_fits_model_error(self,
                             select=None, uvw="u", mrng=10, r_limit=10,
                             prt=True, fitset="sf", spectrum="spl"):
        """
        Compute the error of the fits model
         NOTE: This method is based on signal-to-noise ratio (SNR) of the model
               (Fomalont 1999; Image Analysis)
            Arguments:
                select (str): polarization type
                              (i, q, u, v, rr, ll, rl, lr)
                uvw (str): uv-weight
                           - u: uniform weighting
                           - n: natural weighting
                mrng (float): map range
                r_limit (float): limit of the radius
                prt (bool): if True, print corresponding information
                fitset (str): frequency setting
                              - sf: single-frequency
                              - mf: multi-frequency
                spectrum (str): spectrum type
                                - spl: simple power-law
                                - cpl: curved power-law
                                - ssa: synchrotron self-absorption
        """
        if select is None:
            out_txt =\
                "Please assign polarization type into 'select'."\
                " (available: i, q, u, v, rr, ll, rl, lr)"
            raise Exception(out_txt)

        self.fits_model["DELTAX"] *= d2m
        self.fits_model["DELTAY"] *= d2m
        self.fits_model["MINOR AX"] *= d2m
        self.fits_model["MAJOR AX"] *= d2m
        mjd = Ati(self.fits_date, format="iso").mjd

        l_ = self.fits_model["DELTAX"]
        m_ = self.fits_model["DELTAY"]
        r_ = np.sqrt(l_**2 + m_**2)
        mask_ = r_ < r_limit

        model = self.fits_model.copy()[mask_]
        nmod = model.shape[0]

        flux = model["FLUX"]
        fwhm = (2 * model["MAJOR AX"] + 1 * model["MINOR AX"]) / 3
        lml = model["DELTAX"]
        lmm = model["DELTAY"]
        r = np.sqrt(lml**2 + lmm**2)
        pa = np.arctan2(lml, lmm) * 180 / np.pi

        self.load_uvf(select=select, set_clq=False)
        self.uvave(
            uvave="scan",
            set_clq=False, set_pang=False
        )
        self.fit_beam(uvw=uvw)

        thetas = np.array([])
        fields = np.array([])
        dtypes = np.array([])
        for i in range(nmod):
            thetas =\
                np.append(
                    thetas,
                    [flux[i], fwhm[i], lml[i], lmm[i]]
                )

            fields =\
                np.append(
                    fields,
                    [f"{i + 1}_S", f"{i + 1}_a", f"{i + 1}_l", f"{i + 1}_m"]
                )

            dtypes =\
                np.append(
                    dtypes,
                    ["f8", "f8", "f8", "f8"]
                )

        prms = gamvas.utils.sarray(thetas, fields, dtypes)
        self.append_visibility_model(
            self.freq,
            self.freq,
            prms,
            fitset=fitset,
            spectrum=spectrum,
            set_spectrum=False
        )

        self.ploter.draw_dirtymap(
            uvf=self,
            mrng=mrng,
            npix=512,
            uvw="n",
            plot_resi=True,
            plotimg=False
        )

        self.drop_visibility_model()

        image = self.fits_image
        resid = self.resid.T
        psize1 = 2 * np.round(np.max(self.fits_grid_ra)*d2m, 2)/image.shape[0]
        psize2 = 2 * np.round(np.max(self.mrng.value), 2)/resid.shape[0]
        imagecp = image.copy()
        residcp = resid.copy()

        peak, sigma_rms = np.array([]), np.array([])
        for i in range(nmod):
            centerx1 = int(image.shape[0] / 2 - m[i] / psize1)
            centery1 = int(image.shape[0] / 2 - l[i] / psize1)
            centerx2 = int(resid.shape[0] / 2 - m[i] / psize2)
            centery2 = int(resid.shape[0] / 2 - l[i] / psize2)
            size_ = fwhm[i] / 2
            range1 = size_ / psize1
            range2 = size_ / psize2
            if (range1 < 10) | (range2 < 10):
                range1 = 0.1 / psize1
                range2 = 0.1 / psize2

            residcp[
                int(centerx2 - range2):int(centerx2 + range2),
                int(centery2 - range2):int(centery2 + range2)
            ] = np.nan

            imagecp[
                int(centerx1 - range1):int(centerx1 + range1),
                int(centery1 - range1):int(centery1 + range1)
            ] = np.nan

            image_ =\
                image[
                    int(centerx1 - range1):int(centerx1 + range1),
                    int(centery1 - range1):int(centery1 + range1)
                ]

            resid_ =\
                resid[
                    int(centerx2 - range2):int(centerx2 + range2),
                    int(centery2 - range2):int(centery2 + range2)
                ]

            peak_ = np.abs(np.nanmax(image_))
            sigma_rms_ = np.nanstd(resid_)
            peak = np.append(peak, peak_)
            sigma_rms = np.append(sigma_rms, sigma_rms_)

        snr = peak / sigma_rms
        dmin = 2 / np.pi * np.sqrt(
            np.pi
            * self.bprms[0]
            * self.bprms[1]
            * np.log(2)
            * np.log(snr / (snr - 1))
        )

        dmin_bool = fwhm <= dmin
        fwhm = np.where(dmin_bool, dmin, fwhm)
        dmin_bool = np.where(dmin_bool, 1, 0)

        model["DELTAX"] -= model["DELTAX"][0]
        model["DELTAY"] -= model["DELTAY"][0]

        model["DELTAX"] *= d2m
        model["DELTAY"] *= d2m
        model["MAJOR AX"] *= d2m
        model["MINOR AX"] *= d2m

        flux = model["FLUX"]
        dist = np.sqrt(model["DELTAX"]**2 + model["DELTAY"]**2)
        phi = np.angle(model["DELTAY"] + 1j*model["DELTAX"], deg=True)

        sigma_peak = sigma_rms * (1 + peak / sigma_rms)**0.5
        sigma_flux = sigma_peak * (1 + (flux / peak)**2)**0.5
        sigma_fwhm = sigma_peak * fwhm / peak
        sigma_dist = sigma_fwhm * 0.5
        sigma_phi = np.arctan(sigma_dist / dist) * 180 / np.pi

        uflux = unp.uarray([flux, sigma_flux])
        ufwhm = unp.uarray([fwhm, sigma_fwhm])

        utb = 1.22e+12 * uflux / (ufwhm * self.freq)**2
        tb, dtb = unp.nominal_values(utb), unp.std_devs(utb)
        mjds = np.full(nmod, mjd)

        fields =\
            [
                "mjd",
                "flux", "dflux",
                "size", "dsize",
                "radius", "dradius",
                "phi", "dphi",
                "tb", "dtb",
                "dmin"
            ]

        dtypes = ["f8" for i in range(len(fields))]
        model_cal =\
            gamvas.utils.sarray(
                [
                    mjds,
                    flux, sigma_flux,
                    fwhm, sigma_fwhm,
                    dist, sigma_dist,
                    phi, sigma_phi,
                    tb, dtb,
                    dmin_bool
                ],
                dtype=dtypes,
                field=fields
            )
        self.fits_model_cal = model_cal

    def save_uvfits(self,
                     save_path=False, save_name=False):
        """
        Save new fits file
            Arguments:
                save_name (str): name of the new fits file
                save_path (str): path of the new fits file
        """
        hdul_cp = copy.deepcopy(self.uvf_file)
        hdul = copy.deepcopy(self.uvf_file)

        staxof = hdul["AIPS AN"].data["STAXOF"]

        data = self.data.copy()
        ufq = np.unique(data["freq"])

        if len(ufq) == 1:
            tarr = self.tarr
        else:
            for i in range(len(ufq)):
                mask_fq = data["freq"] == ufq[i]
                self.ant_name1[mask_fq] =\
                    np.char.add(self.ant_name1[mask_fq], f"{i+1}")
                self.ant_name2[mask_fq] =\
                    np.char.add(self.ant_name2[mask_fq], f"{i+1}")

                if i != 0:
                    self.ant_num1[mask_fq] = self.ant_num1[mask_fq] + nant * i
                    self.ant_num2[mask_fq] = self.ant_num2[mask_fq] + nant * i
                nant = len(self.tarr)

                if i == 0:
                    tarr = self.tarr.copy()
                    tarr["name"] = np.char.add(tarr["name"], "1")
                else:
                    tarr_ = self.tarr.copy()
                    tarr_["name"] = np.char.add(tarr_["name"], f"{i+1}")
                    tarr  = np.append(tarr, tarr_)

        nant = len(tarr)
        self.tarr = tarr
        self.sort_uvvis(
            type="tb&ant", reverse=False, set_uvvis=True,
            set_closure=False, minclq=True
        )

        data = self.data


        # PRIMARY table
        nstokes = self.nstokes
        data_pr = hdul["PRIMARY"].data
        cols_pr = hdul["PRIMARY"].columns
        pars_pr = hdul["PRIMARY"].data.parnames
        ncol_pr = len(cols_pr)

        ndata = len(data)
        nfield = len(cols_pr.names)
        dims_pr = (ndata, 1, 1, 1, 1, nstokes, 3)

        data_ = np.empty(dims_pr)
        lst_real =\
            [
                self.vis_1.real, self.vis_2.real,
                self.vis_3.real, self.vis_4.real
            ]
        lst_imag =\
            [
                self.vis_1.imag, self.vis_2.imag,
                self.vis_3.imag, self.vis_4.imag
            ]
        lst_wght =\
            [
                1 / self.sig_1**2, 1 / self.sig_2**2,
                1 / self.sig_3**2, 1 / self.sig_4**2
            ]

        for i in range(nstokes):
            data_[:, 0, 0, 0, 0, i, :] =\
                np.stack([lst_real[i], lst_imag[i], lst_wght[i]], axis=-1)

        ant1 = data["ant_num1"]
        ant2 = data["ant_num2"]
        time = data["time"]
        jd = int(np.min(Ati(self.date, format="iso").jd + time / 24 - 0.5))

        UU = data["u"] / (self.freq * 1e9)
        VV = data["v"] / (self.freq * 1e9)
        if len(self.ww) == ndata:
            WW = self.ww / (self.freq * 1e9)
        else:
            WW = np.zeros(ndata)
        BASELINE = ant1 * 256 + ant2
        DATE = np.full(ndata, jd) - hdul["PRIMARY"].header["PZERO5"]
        _DATE = time / 24

        INTTIM = data["tint"]
        DATA = data_

        if len(pars_pr) >= 8:
            pars_pr = pars_pr[:7]

        data_pr = [UU, VV, WW, BASELINE, DATE, _DATE, INTTIM]
        cols_pr_ =\
            fits.GroupData(
                data_, bitpix=-32,
                parnames=pars_pr,
                pardata=data_pr
            )
        hdu_pr = fits.GroupsHDU(cols_pr_)
        hdu_pr.name = "PRIMARY"

        # AN table
        cols_an = hdul["AIPS AN"].columns
        ncol_an = len(cols_an)

        ANNAME = tarr["name"]
        STABXYZ =\
            np.array(
                [
                    (tarr["x"][i], tarr["y"][i], tarr["z"][i]) \
                    for i in range(nant)
                ]
            )
        ORBPARM = [[] for i in range(nant)]
        NOSTA = np.array([i+1 for i in range(nant)], dtype=np.int32)
        MNTSTA = np.array([0 for i in range(nant)])
        STAXOF = staxof
        POLTYA = np.array(["R" for i in range(nant)])
        POLAA = np.array([0. for i in range(nant)])
        POLCALA = np.array([[0., 0.] for i in range(nant)])
        POLTYB = np.array(["L" for i in range(nant)])
        POLAB = np.array([0. for i in range(nant)])
        POLCALB = np.array([[0., 0.] for i in range(nant)])

        dict_an = dict(zip(cols_an.names, cols_an.formats))
        dict_an_units = dict(zip(cols_an.names, cols_an.units))

        dict_an_ =\
            {
                "ANNAME":[ANNAME, "8A"], "STABXYZ":[STABXYZ, "3D"],
                "ORBPARM":[ORBPARM, "0D"], "NOSTA":[NOSTA, "1J"],
                "MNTSTA":[MNTSTA, "1J"], "STAXOF":[STAXOF, "1E"],
                "POLTYA":[POLTYA, "1A"],  "POLAA":[POLAA, "1E"],
                "POLCALA":[POLCALA, "2E"], "POLTYB":[POLTYB, "1A"],
                "POLAB":[POLAB, "1E"], "POLCALB":[POLCALB, "2E"]
            }

        cols_an_ = [0 for i in range(len(dict_an_))]
        for ncol, cname in enumerate(list(dict_an_.keys())):
            if cname in ["POLCALA", "POLCALB"]\
                and dict_an[cname] == "0E":
                    cols_an_[ncol] =\
                        fits.Column(
                            name=cname,
                            format="0E",
                            array=[[] for i in range(nant)],
                            unit=dict_an_units[cname]
                        )
            else:
                cols_an_[ncol] =\
                    fits.Column(
                        name=cname,
                        format=dict_an_[cname][1],
                        array=dict_an_[cname][0],
                        unit=dict_an_units[cname]
                    )

        cols_an_ = fits.ColDefs(cols_an_)

        hdu_an = fits.BinTableHDU.from_columns(cols_an_)
        hdu_an.name = "AIPS AN"

        for i, h in enumerate(hdul):
            if h.name == "PRIMARY":
                hdul[i] = hdu_pr
                break
        else:
            hdul.append(hdu_pr)

        for i, h in enumerate(hdul):
            if h.name == "AIPS AN":
                hdul[i] = hdu_an
                break
        else:
            hdul.append(hdu_an)

        hdul["PRIMARY"].header["CRVAL4"] = self.freq * 1e9

        hdr_pr = hdul["PRIMARY"].header
        hdr_cp = hdul_cp["PRIMARY"].header
        keys = list(hdr_cp.keys())
        comments = list(hdr_cp.comments)
        for i in range(len(keys)):
            if "HISTORY" in keys[i]:
                break
            hdr_pr[keys[i]] = (hdr_cp[keys[i]], comments[i].replace("\n", " "))

        hdr_an = hdul["AIPS AN"].header
        hdr_cp = hdul_cp["AIPS AN"].header
        keys = list(hdr_cp.keys())
        comments = list(hdr_cp.comments)
        for i in range(len(keys)):
            hdr_an[keys[i]] = (hdr_cp[keys[i]], comments[i].replace("\n", " "))

        hdul.writeto(
            save_path + save_name,
            overwrite=True, output_verify="warn")

    def uvshift(self,
                deltal=0, deltam=0):
        """
        Shift the uv-visibility data
            Arguments:
                deltal (float, mas): shift in the RA-direction
                deltam (float, mas): shift in the DEC-direction
        """
        deltal = deltal * m2r
        deltam = deltam * m2r

        uvu = self.data["u"]
        uvv = self.data["v"]

        self.vis_1 =\
            self.vis_1 \
            * np.exp(+2j * np.pi * uvu * deltal) \
            * np.exp(+2j * np.pi * uvv * deltam)

        self.vis_2 =\
            self.vis_2 \
            * np.exp(+2j * np.pi * uvu * deltal) \
            * np.exp(+2j * np.pi * uvv * deltam)

        self.vis_3 =\
            self.vis_3 \
            * np.exp(+2j * np.pi * uvu * deltal) \
            * np.exp(+2j * np.pi * uvv * deltam)

        self.vis_4 =\
            self.vis_4 \
            * np.exp(+2j * np.pi * uvu * deltal) \
            * np.exp(+2j * np.pi * uvv * deltam)

        self.set_uvvis()


    def get_data(self, type=None, query=None):
        out_txt =\
            f"Please assign appropriate data type (given: {type}). \n" \
            "availables: " \
            "'ant_name1', 'ant_name2', 'phi', 'azimuth', 'elevation', pair', " \
            "'time', 'time_angle', 'mjd', 'u', 'v', " \
            "'vism', 'ampm', 'phsm', " \
            "'vis', 'amp', 'phs', 'sigma', 'sigma_amp', 'sigma_phs', " \
            "'vis_rr', 'vis_ll', 'vis_rl', 'vis_lr', " \
            "'vis_i', 'vis_q', 'vis_u', 'vis_v', 'vis_p', " \
            "'amp_rr', 'amp_ll', 'amp_rl', 'amp_lr', " \
            "'amp_i', 'amp_q', 'amp_u', 'amp_v', 'amp_p', " \
            "'phs_rr', 'phs_ll', 'phs_rl', 'phs_lr', " \
            "'phs_i', 'phs_q', 'phs_u', 'phs_v', 'phs_p', " \
            "'sigma_rr', 'sigma_ll', 'sigma_rl', 'sigma_lr', " \
            "'sigma_i', 'sigma_q', 'sigma_u', 'sigma_v', 'sigma_p', " \
            "'sigma_amp_rr', 'sigma_amp_ll', 'sigma_amp_rl', 'sigma_amp_lr', " \
            "'sigma_amp_i', 'sigma_amp_q', 'sigma_amp_u', 'sigma_amp_v', " \
            "'sigma_amp_p', " \
            "'sigma_phs_rr', 'sigma_phs_ll', 'sigma_phs_rl', 'sigma_phs_lr', " \
            "'sigma_phs_i', 'sigma_phs_q', 'sigma_phs_u', 'sigma_phs_v', " \
            "'sigma_phs_p', " \
            "'time_cla', 'quadra', 'clamp', 'sigma_clamp', " \
            "'time_clp', 'triangle', 'clphs', 'sigma_clphs' " \

        if type is None:
            raise Exception(out_txt)

        if type in \
            [
                "ant_name1", "ant_name2", "pair", "time", "mjd", "u", "v",
                "vism", "ampm", "phsm",
                "vis", "amp", "phs", "sigma", "sigma_amp", "sigma_phs",
                "vis_rr", "vis_ll", "vis_rl", "vis_lr",
                "vis_i", "vis_q", "vis_u", "vis_v", "vis_p",
                "amp_rr", "amp_ll", "amp_rl", "amp_lr",
                "amp_i", "amp_q", "amp_u", "amp_v", "amp_p",
                "phs_rr", "phs_ll", "phs_rl", "phs_lr",
                "phs_i", "phs_q", "phs_u", "phs_v", "phs_p",
                "sigma_rr", "sigma_ll", "sigma_rl", "sigma_lr",
                "sigma_i", "sigma_q", "sigma_u", "sigma_v", "sigma_p",
                "sigma_amp_rr", "sigma_amp_ll", "sigma_amp_rl", "sigma_amp_lr",
                "sigma_amp_i", "sigma_amp_q", "sigma_amp_u", "sigma_amp_v",
                "sigma_phs_rr", "sigma_phs_ll", "sigma_phs_rl", "sigma_phs_lr",
                "sigma_phs_i", "sigma_phs_q", "sigma_phs_u", "sigma_phs_v",
                "sigma_phs_p"
            ]:
                data = copy.deepcopy(self.data)

                if type in ["vism", "ampm", "phsm"] \
                    and type not in data.dtype.names:
                    out_txt = "Model visibility is not established."
                    raise Exception(out_txt)

                if query is not None:
                    if isinstance(query, (list, tuple)) and len(query) == 2:
                        query = list(map(str.upper, query))
                        baseline = "-".join(query)
                    elif isinstance(query, str) and "-" in query:
                        baseline = query.upper()
                    else:
                        out_txt =\
                            f"Please assign appropriate 'query'."
                        raise Exception(out_txt)
                    mask =\
                        data["pair"] == baseline
                    data = data[mask]

                if "phs" in type:
                    if "sigma" in type:
                        amp = np.abs(data[type.replace("sigma_phs", "vis")])
                        sig = data[type.replace("sigma_phs", "sigma")]
                        out = sig / amp
                    else:
                        out = np.angle(data[type.replace("phs", "vis")])
                elif "amp" in type:
                    if "sigma" in type:
                        out = data[type.replace("sigma_amp", "sigma")]
                    else:
                        out = np.abs(data[type.replace("amp", "vis")])
                else:
                    out = data[type]

        elif type in \
            [
                "time_angle",
                "pangle", "phi",
                "azimuth", "az",
                "elevation", "el"
            ]:
                data1 = copy.deepcopy(self.data)
                data2 = copy.deepcopy(self.azel)

                ant1 = copy.deepcopy(data1["ant_name1"])
                ant2 = copy.deepcopy(data1["ant_name2"])
                uant = np.unique(np.append(ant1, ant2))


                if query is not None:
                    if isinstance(query, (list, tuple)) and len(query) >= 2:
                        out_txt =\
                            f"Please assign a single element on 'query'."
                        raise Exception(out_txt)

                    if isinstance(query, (list, tuple)):
                        query = query[0].upper()
                    else:
                        query = query.upper()

                    if (query not in ant1) & (query not in ant2):
                        out_txt =\
                            f"Please assign appropriate element on 'query'. \n" \
                            f"availables: {uant} // given: '{query}'"
                        raise Exception(out_txt)

                    mask = (ant1 == query) | (ant2 == query)
                    data1 = data1[mask]
                    data2 = data2[mask]

                    ant1 = data1["ant_name1"]
                    ant2 = data1["ant_name2"]
                    mask1 = ant1 == query
                    mask2 = ant2 == query

                if type == "time_angle":
                    out = data1["time"]

                if type in ["pangle", "phi"]:
                    out =\
                        np.select(
                            [mask1, mask2],
                            [data1["phi1"], data1["phi2"]]
                        )

                if type in ["azimuth", "az"]:
                    out =\
                        np.select(
                            [mask1, mask2],
                            [data2["az1"], data2["az2"]]
                        ) / 180 * np.pi

                if type in ["elevation", "el"]:
                    out =\
                        np.select(
                            [mask1, mask2],
                            [data2["el1"], data2["el2"]]
                        ) / 180 * np.pi

        elif type in \
            [
                "time_cla", "quadra", "clamp", "sigma_clamp",
            ]:
                data = copy.deepcopy(self.clamp)
                if query is not None:
                    if isinstance(query, (list, tuple)) and len(query) == 4:
                        query = list(map(str.upper, query))
                        quadrangle = "-".join(query)
                    elif isinstance(query, str) and "-" in query:
                        quadrangle = query.upper()
                    else:
                        out_txt =\
                            f"Please assign appropriate 'query'."
                        raise Exception(out_txt)

                    mask =\
                        data["quadra"] == quadrangle
                    data = data[mask]

                if type == "time_cla":
                    out = data["time"]
                else:
                    out = data[type]


        elif type in \
            [
                "time_clp", "triangle", "clphs", "sigma_clphs",
            ]:
                data = copy.deepcopy(self.clphs)
                if query is not None:
                    if isinstance(query, (list, tuple)) and len(query) == 3:
                        query = list(map(str.upper, query))
                        triangle = "-".join(query)
                    elif isinstance(query, str) and "-" in query:
                        triangle = query.upper()
                    else:
                        out_txt =\
                            f"Please assign appropriate 'query'."
                        raise Exception(out_txt)

                    mask =\
                        data["triangle"] == triangle
                    data = data[mask]

                if type == "time_clp":
                    out = data["time"]
                else:
                    out = data[type]

        else:
            raise Exception(out_txt)

        return out


    def get_sblf(self):
        """
        Get the shortest-baseline flux level
        """
        data = self.data
        uvd = np.sqrt(data["u"]**2 + data["v"]**2)

        mask = uvd == np.min(uvd)
        sbl = data[mask]

        sbl_ant1 = sbl["ant_name1"][0]
        sbl_ant2 = sbl["ant_name2"][0]

        mask_sbl =\
            (data["freq"] == np.min(data["freq"])) \
            & (data["ant_name1"] == sbl_ant1) \
            & (data["ant_name2"] == sbl_ant2)

        sbl = data[mask_sbl]
        sblf = np.median(np.abs(sbl["vis"]))
        self.sblf = sblf
        self.sbl = (sbl_ant1, sbl_ant2)
        return self.sblf, self.sbl

    def dterm_transfer(self,
                       uncal=None, niter=1, drarr=None, dlarr=None,
                       plotimg=False, saveimg=False, savename=None,
                       saveuvf=False):
        if drarr is None or dlarr is None:
            out_txt = "Either 'drarr' or 'dlarr' is not given."
            raise Exception(out_txt)

        if isinstance(drarr, list):
            drarr = np.array(drarr)

        if isinstance(dlarr, list):
            dlarr = np.array(dlarr)

        def fit_vis(s, vis, sigma, lml, lmm, a, uvu, uvv):
            nmod = len(s)
            ndat = len(uvu)
            a /= (2 * (2 * unp.log(2))**0.5)
            vism = np.zeros(ndat, dtype="c8")
            for i in range(nmod):
                vism +=\
                    gamvas.functions.gvis(
                        (uvu, uvv),
                        s[i], a[i], lml[i], lmm[i]
                    )

            out =\
                0.5 * np.sum(
                    np.abs(
                        ((vis - vism)/sigma)**2
                    )
                )

            return out

        def fit_dphs(
            dphs, vis, vism, sigma,
            dterm_r, dterm_l, phi1, phi2,
            ant1, ant2
        ):

            vir_rr, virm_rr, sigma_rr = vis[0], vism[0], sigma[0]
            vis_rl, vism_rl, sigma_rl = vis[1], vism[1], sigma[1]
            vis_lr, vism_lr, sigma_lr = vis[2], vism[2], sigma[2]
            vis_ll, vism_ll, sigma_ll = vis[3], vism[3], sigma[3]

            dterm_r_ = dterm_r * np.exp(+2j * dphs)
            dterm_l_ = dterm_l * np.exp(-2j * dphs)

            d_r1 = dterm_r_[ant1 - 1]
            d_l1 = dterm_l_[ant1 - 1]
            d_r2 = dterm_r_[ant2 - 1]
            d_l2 = dterm_l_[ant2 - 1]

            cal_rl =\
                vism_rl \
                + d_r1 * vis_ll * np.exp(+2j * phi1) \
                + d_l2.conj() * vis_rr * np.exp(+2j * phi2) \
                + d_r1 * d_l2.conj() * vism_lr * np.exp(+2j * (phi1 + phi2))

            cal_lr =\
                vism_lr \
                + d_l1 * vis_rr * np.exp(-2j * phi1) \
                + d_r2.conj() * vis_ll * np.exp(-2j * phi2) \
                + d_l1 * d_r2.conj() * vism_rl * np.exp(-2j * (phi1 + phi2))

            out_rl = 0.5 * np.sum(np.abs((vis_rl - cal_rl) / sigma_rl)**2)
            out_lr = 0.5 * np.sum(np.abs((vis_lr - cal_lr) / sigma_lr)**2)
            return out_rl + out_lr

        if self.fits_model is None:
            out_txt =\
                "FITS model is not provided." \
                " Please check if you load UV-fits file."
            raise Exception(out_txt)
        fits_model = self.fits_model.copy()
        mask = self.fits_model["FLUX"] > 0
        fits_model = fits_model[mask]
        s = fits_model["FLUX"]
        lml = fits_model["DELTAX"] * u.deg.to(u.mas)
        lmm = fits_model["DELTAY"] * u.deg.to(u.mas)
        a =\
            (2 * fits_model["MAJOR AX"] + 1 * fits_model["MINOR AX"]) \
            / 3 * u.deg.to(u.mas)

        uvu = self.data["u"]
        uvv = self.data["v"]
        uvr = np.sqrt(uvu**2 + uvv**2)

        ant1 = self.data["ant_num1"]
        ant2 = self.data["ant_num2"]
        phi1 = self.data["phi1"]
        phi2 = self.data["phi2"]

        drarr1 = drarr.copy()[ant1 - 1]
        dlarr1 = dlarr.copy()[ant1 - 1]
        drarr2 = drarr.copy()[ant2 - 1]
        dlarr2 = dlarr.copy()[ant2 - 1]

        sigma_rr = self.data["sigma_rr"]
        sigma_rl = self.data["sigma_rl"]
        sigma_lr = self.data["sigma_lr"]
        sigma_ll = self.data["sigma_ll"]

        dphs = 0
        for i in range(niter):
            if i == 0:
                vis_rr = self.data["vis_rr"].copy()
                vis_rl = self.data["vis_rl"].copy()
                vis_lr = self.data["vis_lr"].copy()
                vis_ll = self.data["vis_ll"].copy()
            else:
                drarr1_ = drarr1.copy() * np.exp(+2j * dphs_)
                dlarr1_ = dlarr1.copy() * np.exp(-2j * dphs_)
                drarr2_ = drarr2.copy() * np.exp(+2j * dphs_)
                dlarr2_ = dlarr2.copy() * np.exp(-2j * dphs_)

                vis_rr = vis_rr

                vis_rl =\
                    vis_rl \
                    - drarr1 * vism_ll * np.exp(+2j * phi1) \
                    - drarr2.conj() * vism_rr * np.exp(+2j * phi2) \
                    - (
                        drarr1
                        * drarr2.conj()
                        * vism_lr
                        * np.exp(+2j * (phi1 + phi2))
                    )

                vis_lr =\
                    vis_lr \
                    - drarr1 * vism_rr * np.exp(-2j * phi1) \
                    - drarr2.conj() * vism_ll * np.exp(-2j * phi2) \
                    - (
                        drarr1
                        * drarr2.conj()
                        * vism_rl
                        * np.exp(-2j * (phi1 + phi2))
                    )

                vis_ll = vis_ll

            args_rr = (vis_rr, sigma_rr, lml, lmm, a, uvu, uvv)
            args_rl = (vis_rl, sigma_rl, lml, lmm, a, uvu, uvv)
            args_lr = (vis_lr, sigma_lr, lml, lmm, a, uvu, uvv)
            args_ll = (vis_ll, sigma_ll, lml, lmm, a, uvu, uvv)

            bounds = np.stack([-s, s], axis=1)

            def nll(*args):
                return fit_vis(*args)

            init = s * self.m

            soln_rr =\
                optimize.minimize(
                    fun=nll,
                    x0=init,
                    args=args_rr,
                    bounds=bounds,
                    method="Powell"
                )

            soln_rl =\
                optimize.minimize(
                    fun=nll,
                    x0=init,
                    args=args_rl,
                    bounds=bounds,
                    method="Powell"
                )

            soln_lr =\
                optimize.minimize(
                    fun=nll,
                    x0=init,
                    args=args_lr,
                    bounds=bounds,
                    method="Powell"
                )

            soln_ll =\
                optimize.minimize(
                    fun=nll,
                    x0=init,
                    args=args_ll,
                    bounds=bounds,
                    method="Powell"
                )

            nmod = len(s)
            ndat = len(uvu)
            vism_rr = np.zeros(ndat, dtype="c8")
            vism_rl = np.zeros(ndat, dtype="c8")
            vism_lr = np.zeros(ndat, dtype="c8")
            vism_ll = np.zeros(ndat, dtype="c8")
            for i in range(nmod):
                vism_rr +=\
                    gamvas.functions.gvis(
                        (uvu, uvv),
                        soln_rr.x[i], a[i], l[i], m[i]
                    )

                vism_rl +=\
                    gamvas.functions.gvis(
                        (uvu, uvv),
                        soln_rl.x[i], a[i], l[i], m[i]
                    )

                vism_lr +=\
                    gamvas.functions.gvis(
                        (uvu, uvv),
                        soln_lr.x[i], a[i], l[i], m[i]
                    )

                vism_ll +=\
                    gamvas.functions.gvis(
                        (uvu, uvv),
                        soln_ll.x[i], a[i], l[i], m[i]
                    )

            def nll_dphs(*args):
                return fit_dphs(*args)

            vis = [vis_rr, vis_rl, vis_lr, vis_ll]
            vism = [vism_rr, vism_rl, vism_lr, vism_ll]
            sigma = [sigma_rr, sigma_rl, sigma_lr, sigma_ll]

            args_dphs =\
                (
                    vis, vism, sigma,
                    drarr, dlarr, phi1, phi2,
                    ant1, ant2
                )

            init_dphs = [0]

            soln_dphs =\
                optimize.minimize(
                    fun=nll_dphs,
                    x0=init_dphs,
                    args=args_dphs,
                    bounds=[[-np.pi, +np.pi]],
                    method="Powell"
                )

            dphs_ = soln_dphs.x[0]
            dphs += dphs_

        out_txt =\
            f"{self.source:8}" \
            f" ({int(np.round(self.freq))} GHz, {self.date}):" \
            f" {dphs * 180 / np.pi:.3f} [deg]"
        print(out_txt)
        self.dterm_dphs = dphs

        d_r1 = drarr.copy()[ant1-1] * np.exp(+2j*dphs)
        d_l1 = dlarr.copy()[ant1-1] * np.exp(-2j*dphs)
        d_r2 = drarr.copy()[ant2-1] * np.exp(+2j*dphs)
        d_l2 = dlarr.copy()[ant2-1] * np.exp(-2j*dphs)

        cal_rr = self.data["vis_rr"]

        cal_rl =\
            self.data["vis_rl"] \
            - d_r1 * vis_ll * np.exp(+2j * phi1) \
            - d_l2.conj() * vis_rr * np.exp(+2j * phi2) \
            - d_r1 * d_l2.conj() * vism_lr * np.exp(+2j * (phi1 + phi2))

        cal_lr =\
            self.data["vis_lr"] \
            - d_l1 * vis_rr * np.exp(-2j * phi1) \
            - d_r2.conj() * vis_ll * np.exp(-2j * phi2) \
            - d_l1 * d_r2.conj() * vism_rl * np.exp(-2j * (phi1 + phi2))

        cal_ll = self.data["vis_ll"]

        obs_rlrr = self.data["vis_rl"] / self.data["vis_rr"]
        obs_rlll = self.data["vis_rl"] / self.data["vis_ll"]
        obs_lrrr = self.data["vis_lr"] / self.data["vis_rr"]
        obs_lrll = self.data["vis_lr"] / self.data["vis_ll"]

        cal_rlrr = cal_rl / cal_rr
        cal_rlll = cal_rl / cal_ll
        cal_lrrr = cal_lr / cal_rr
        cal_lrll = cal_lr / cal_ll

        axlim1 = max(
            np.max(np.abs(obs_rlrr)),
            np.max(np.abs(obs_rlll)),
            np.max(np.abs(obs_lrrr)),
            np.max(np.abs(obs_lrll))
        )

        axlim2 = max(
            np.max(np.abs(cal_rlrr)),
            np.max(np.abs(cal_rlll)),
            np.max(np.abs(cal_lrrr)),
            np.max(np.abs(cal_lrll))
        )

        axlim = 1.1 * np.max([axlim1, axlim2])

        if plotimg:
            fig, axes =\
                plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
            axes[0, 0].set_aspect("equal")
            axes[0, 1].set_aspect("equal")
            axes[1, 0].set_aspect("equal")
            axes[1, 1].set_aspect("equal")
            axes[0, 0].set_title("RL/RR", fontsize=12)
            axes[0, 1].set_title("RL/LL", fontsize=12)
            axes[1, 0].set_title("LR/RR", fontsize=12)
            axes[1, 1].set_title("LR/LL", fontsize=12)
            axes[0, 0].scatter(
                obs_rlrr.real, obs_rlrr.imag,
                c="red", marker="x", s=10, zorder=1
            )

            axes[0, 1].scatter(
                obs_rlll.real, obs_rlll.imag,
                c="red", marker="x", s=10, zorder=1
            )

            axes[1, 0].scatter(
                obs_lrrr.real, obs_lrrr.imag,
                c="red", marker="x", s=10, zorder=1
            )

            axes[1, 1].scatter(
                obs_lrll.real, obs_lrll.imag,
                c="red", marker="x", s=10, zorder=1
            )

            axes[0, 0].scatter(
                cal_rlrr.real, cal_rlrr.imag,
                c="black", marker="x", s=10, zorder=2
            )

            axes[0, 1].scatter(
                cal_rlll.real, cal_rlll.imag,
                c="black", marker="x", s=10, zorder=2
            )

            axes[1, 0].scatter(
                cal_lrrr.real, cal_lrrr.imag,
                c="black", marker="x", s=10, zorder=2
            )

            axes[1, 1].scatter(
                cal_lrll.real, cal_lrll.imag,
                c="black", marker="x", s=10, zorder=2
            )

            axes[0, 0].set_xlim(-axlim, +axlim)
            axes[0, 1].set_xlim(-axlim, +axlim)
            axes[1, 0].set_xlim(-axlim, +axlim)
            axes[1, 1].set_xlim(-axlim, +axlim)
            axes[0, 0].set_ylim(-axlim, +axlim)
            axes[0, 1].set_ylim(-axlim, +axlim)
            axes[1, 0].set_ylim(-axlim, +axlim)
            axes[1, 1].set_ylim(-axlim, +axlim)
            axes[0, 0].grid(True)
            axes[0, 1].grid(True)
            axes[1, 0].grid(True)
            axes[1, 1].grid(True)
            fig.supxlabel("Real", fontsize=15)
            fig.supylabel("Imag", fontsize=15)
            fig.suptitle(f"{self.source}", fontsize=15)
            fig.tight_layout()
            if saveimg:
                fig.savefig(
                    self.path
                    + f"dterm.{int(np.round(self.freq))}.{self.source}.png",
                    dpi=300)
            plt.show()

        if saveuvf:
            if savename is None:
                savename = f"no-name.{int(np.round(self.freq))}" \
                    f".{self.source}.{self.date}.uvf"
            path = self.path
            file = uncal

            open_uvf = fits.open(path + file)

            nifs = open_uvf["PRIMARY"].data["DATA"].shape[3]
            nstokes = open_uvf["PRIMARY"].data["DATA"].shape[5]

            ant1 =\
                open_uvf["PRIMARY"].data["BASELINE"][:].astype(int) \
                // 256

            ant2 =\
                open_uvf["PRIMARY"].data["BASELINE"][:].astype(int) \
                - ant1 * 256

            ant1 -= 1
            ant2 -= 1

            ant_name1 = self.tarr["name"][ant1]
            ant_name2 = self.tarr["name"][ant2]

            ndat = open_uvf["PRIMARY"].data["DATA"].shape[0]
            vism = np.zeros(ndat, dtype="c8")

            list_uvu = ["UU---SIN", "UU--", "UU"]
            list_uvv = ["VV---SIN", "VV--", "VV"]
            for i in range(len(list_uvu)):
                if list_uvu[i] in open_uvf["PRIMARY"].data.dtype.names:
                    idx_uvu = list_uvu[i]
            for i in range(len(list_uvv)):
                if list_uvv[i] in open_uvf["PRIMARY"].data.dtype.names:
                    idx_uvv = list_uvv[i]
            uvu = open_uvf["PRIMARY"].data[idx_uvu] * self.freq * 1e9
            uvv = open_uvf["PRIMARY"].data[idx_uvv] * self.freq * 1e9

            vism_rl = np.zeros(ndat, dtype="c8")
            vism_lr = np.zeros(ndat, dtype="c8")
            for i in range(nmod):
                vism_rl +=\
                    gamvas.functions.gvis(
                        (uvu, uvv),
                        soln_rl.x[i], a[i], l[i], m[i]
                    )

                vism_lr +=\
                    gamvas.functions.gvis(
                        (uvu, uvv),
                        soln_lr.x[i], a[i], l[i], m[i]
                    )

            mjd =\
                Ati(
                    open_uvf["PRIMARY"].data["DATE"]
                    + open_uvf["PRIMARY"].data["_DATE"],
                    format="jd"
                ).mjd

            tarr = self.tarr
            x = tarr["x"]
            y = tarr["y"]
            z = tarr["z"]
            r = np.sqrt(x**2 + y**2 + z**2)

            ant_geocent = EarthLocation(x=x*u.m, y=y*u.m, z=z*u.m)
            lon, lat, h = ant_geocent.to_geodetic()

            lat = lat.degree
            lon = lon.degree
            height = h.value

            obstime = Ati(mjd, format="mjd").iso
            src_coord = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg)
            ants = self.tarr["name"]

            lat1 = np.zeros(ndat)
            lon1 = np.zeros(ndat)
            lat2 = np.zeros(ndat)
            lon2 = np.zeros(ndat)
            az1 = np.zeros(ndat)
            el1 = np.zeros(ndat)
            az2 = np.zeros(ndat)
            el2 = np.zeros(ndat)
            for tidx in range(len(ants)):
                ant_ = tarr["name"][tidx]
                lat_ = tarr["lat"][tidx]
                lon_ = tarr["lon"][tidx]
                h_ = tarr["height"][tidx]

                antpos =\
                    EarthLocation(
                        lat=lat_ * u.deg,
                        lon=lon_ * u.deg,
                        height=h_ * u.m
                    )

                src_azel =\
                    src_coord.transform_to(
                        AltAz(obstime=obstime, location=antpos)
                    )

                az1 = np.where(ant_name1 == ant_, src_azel.az.degree, az1)
                el1 = np.where(ant_name1 == ant_, src_azel.alt.degree, el1)
                az2 = np.where(ant_name2 == ant_, src_azel.az.degree, az2)
                el2 = np.where(ant_name2 == ant_, src_azel.alt.degree, el2)
                lat1 = np.where(ant_name1 == ant_, lat_, lat1)
                lon1 = np.where(ant_name1 == ant_, lon_, lon1)
                lat2 = np.where(ant_name2 == ant_, lat_, lat2)
                lon2 = np.where(ant_name2 == ant_, lon_, lon2)

            ra = np.pi / 180 * self.ra
            dec = np.pi / 180 * self.dec
            lat1_rad = np.pi / 180 * lat1
            lat2_rad = np.pi / 180 * lat2
            az1_rad = np.pi / 180 * az1
            az2_rad = np.pi / 180 * az2
            el1_rad = np.pi / 180 * el1
            el2_rad = np.pi / 180 * el2
            sin_pa1 = np.cos(lat1_rad) / np.cos(dec) * np.sin(az1_rad)

            cos_pa1 =\
                (np.sin(lat1_rad) - np.sin(dec) * np.sin(el1_rad)) \
                / (np.cos(dec) * np.cos(el1_rad))

            sin_pa2 = np.cos(lat2_rad) / np.cos(dec) * np.sin(az2_rad)

            cos_pa2 =\
                (np.sin(lat2_rad) - np.sin(dec) * np.sin(el2_rad)) \
                / (np.cos(dec) * np.cos(el2_rad))

            phi1 = -np.angle(cos_pa1 + 1j * sin_pa1)
            phi2 = -np.angle(cos_pa2 + 1j * sin_pa2)

            d_r1 = drarr.copy()[ant1] * np.exp(+2j*dphs)
            d_l1 = dlarr.copy()[ant1] * np.exp(-2j*dphs)
            d_r2 = drarr.copy()[ant2] * np.exp(+2j*dphs)
            d_l2 = dlarr.copy()[ant2] * np.exp(-2j*dphs)

            for ifnum in range(nifs):
                for stokesnum in range(nstokes):
                    vis_real =\
                        open_uvf["PRIMARY"].data["DATA"][
                            :, 0, 0, ifnum, 0, stokesnum, 0
                        ].copy()

                    vis_imag =\
                        open_uvf["PRIMARY"].data["DATA"][
                            :, 0, 0, ifnum, 0, stokesnum, 1
                        ].copy()

                    if stokesnum == 0:
                        vis_rr = vis_real + 1j * vis_imag
                        continue

                    if stokesnum == 1:
                        vis_ll = vis_real + 1j * vis_imag
                        continue

                    if stokesnum == 2:
                        vis_rl = vis_real + 1j * vis_imag
                        corr =\
                            d_r1 * vis_ll * np.exp(+2j * phi1) \
                            + d_l2.conj() * vis_rr * np.exp(+2j * phi2) \
                            + (
                                d_r1
                                * d_l2.conj()
                                * vism_lr
                                * np.exp(+2j * (phi1 + phi2))
                            )
                        out = vis_rl - corr

                    if stokesnum == 3:
                        vis_lr = vis_real + 1j * vis_imag
                        corr =\
                            d_l1 * vis_rr * np.exp(-2j * phi1) \
                            + d_r2.conj() * vis_ll * np.exp(-2j * phi2) \
                            + (
                                d_l1
                                * d_r2.conj()
                                * vism_rl
                                * np.exp(-2j * (phi1 + phi2))
                            )
                        out = vis_lr - corr

                    open_uvf["PRIMARY"].data["DATA"][
                        :, 0, 0, ifnum, 0, stokesnum, 0
                    ] = out.real

                    open_uvf["PRIMARY"].data["DATA"][
                        :, 0, 0, ifnum, 0, stokesnum, 1
                    ] = out.imag

            open_uvf.writeto(
                path + savename,
                overwrite=True,
                output_verify="warn"
            )


def cal_clean_error(s_peak=0, s_tot=0, sig_rms=0, size=0):
    """
    Compute the error of the clean image
        Arguments:
            s_peak (float): peak intensity
            s_tot (float): total flux density
            sig_rms (float): rms noise level
            size (float): model size
        Returns:
            sig_peak (float): error of the peak intensity
            sig_tot (float): error of the total flux density
            sig_size (float): error of the model size
    """
    sig_peak = sig_rms * np.sqrt(1 + s_peak / sig_rms)
    sig_tot = sig_peak * np.sqrt(1 + (s_tot / s_peak)**2)
    sig_size = size * (sig_peak / s_peak)
    return sig_peak, sig_tot, sig_size

def set_matrix_visphs(N):
    """
    Set the matrix for the visibility phase
        Arguments:
            N (int): number of antennas
        Returns:
            out (np.array): matrix for the visibility phase
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


def set_min_matrix_clphs(N, ant_nums):
    """
    Set the minimum matrix for the closure phase
        Arguments:
            N (int): number of antennas
            ant_nums (array): array of antenna numbers
        Returns:
            out (array): matrix for the closure phase
            pairs (array): array of antenna pairs
    """
    pairs = np.array(list(map(",".join, list(it.combinations(ant_nums, 2)))))
    out = np.array([[1, -1, 1]])
    if N == 3:
        return out, pairs
    else:
        phi = set_matrix_visphs(N - 1)
        Is = np.eye(int(comb(N - 1, 2)))
        out = np.concatenate((phi, Is), axis=1)
        return out, pairs


def set_min_matrix_clamp(N, ant_nums):
    """
    Set the minimum matrix for the closure amplitude
        Arguments:
            N (int): number of antennas
            ant_nums (array): array of antenna numbers
        Returns:
            out (array): matrix for the closure amplitude
            pairs (array): array of antenna pairs
    """
    pairs =\
        np.array(
            list(map(
                ",".join,
                list(it.combinations(ant_nums, 2))
            ))
        )
    out = np.array([
        [0, 1, -1, -1, 1, 0],
        [1, 0, -1, -1, 0, 1]]
    )

    if N == 4:
        return out, pairs
    else:
        for i in range(5, N + 1):
            ant_nums_ = ant_nums[:i]
            m1 = np.ones((i - 2, 1))
            m0 = np.zeros((int((i-1) * (i - 4) / 2), i-1))
            xn = np.concatenate((np.eye(i - 2), -m1), axis=1)
            yn = np.zeros((i - 2, int(comb(i-1, 2))))

            yn_comb =\
                np.array(
                list(map(
                    ",".join,
                    list(it.combinations(ant_nums_, 2))
                ))
            )[i - 1:]

            for j in range(i-2):
                if j != i-3:
                    idx1 = f"{ant_nums_[j + 1]},{ant_nums_[j + 2]}"
                    idx2 = f"{ant_nums_[j + 2]},{ant_nums_[-1]}"
                    loc1 = np.where(yn_comb == idx1)[0][0]
                    loc2 = np.where(yn_comb == idx2)[0][0]
                else:
                    idx1 = f"{ant_nums_[1]},{ant_nums_[-2]}"
                    idx2 = f"{ant_nums_[1]},{ant_nums_[-1]}"
                    loc1 = np.where(yn_comb == idx1)[0][0]
                    loc2 = np.where(yn_comb == idx2)[0][0]
                yn[j, loc1] = -1
                yn[j, loc2] = +1
            upper = np.concatenate((xn, yn), axis=1)

            if i == 4:
                out = upper
            else:
                m0 = np.zeros((int((i - 1) * (i - 4) / 2), i - 1))
                cm0 = out
                lower = np.concatenate((m0, cm0), axis=1)
                out = np.concatenate((upper, lower), axis=0)
        return out, pairs
