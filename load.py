
import gc
import sys
import copy
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
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.coordinates import solar_system_ephemeris, Angle
from astropy.coordinates import get_body_barycentric, get_body
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
        path (str): The path to the fits file
        file (str): The fits file name
        npixel1 (int): The number of pixels in the x-axis
        npixel2 (int): The number of pixels in the y-axis
        bmin (float): The minor axis of the beam
        bmaj (float): The major axis of the beam
        bpa (float): The position angle of the beam
        fluxmax (float): The maximum intensity of the image
        fluxmin (float): The minimum intensity of the image
        noise (float): The theoretical thermal noise
        freq (float): The observing frequency
        stokes (int): The Stokes parameter information
        date (str): The observation date
        data (2D-array): The uv-fits data
        units (str): The units of the data
        select (str): The polarization type
        select_if (str): The number(s) of IFs
        uvinfo (bool): The mask if uv-fits information is loaded
        mrng (float): The maximum range of the image
        save_path (str): The path to save the image
        save_file (str): The file name to save the image
        xlim (list): The x-axis limits
        ylim (list): The y-axis limits
        source (str): The source name
        fits_image (2D-array): The reconstructed fits image
        pangle (bool): The mask if parallactic angle is updated
        load_uvf (bool): The toggle option if load uv-fits
    """
    def __init__(self,
        path=None, file=None, npixel1=False, npixel2=False,
        bmin=False, bmaj=False, bpa=False, fluxmax=False, fluxmin=False, noise=False, freq=False,
        stokes=None, date=None, data=None, units=None, model=None,
        select="i", select_if="all", mrng=10*u.mas, load_uvf=False, load_fits=False
    ):
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
        file=False, align_index=0, align_model=False, units="deg", pinfo=True, contour_snr=3
    ):
        """
        Load fits image file and extract the information
            Arguments:
                file (str): The fits file name
                align_index (int): The index of the model for position align
                align_model (bool): The toggle option if align fits models
                units (str): The units of the model
                pinfo (bool): The toggle option if print the information
                contour_snr (float): The signal-to-noise ratio for the contour
        """
        if file : self.file = file
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
        self.fits_freq = h1.header["CRVAL3"] / 1e9  # units in GHz
        self.fits_date = h1.header["DATE-OBS"]
        self.fits_stokes = int(h1.header["CRVAL4"]) # CRPIX=+1 to +4: I, Q, U, V
        self.fits_source = h1.header["OBJECT"]
        self.fits_project = h1.header["OBSERVER"]
        try:
            self.fits_instrument = h1.header["INSTRUME"]
        except KeyError:
            self.fits_project = h1.header["TELESCOP"]

        self.fits_image = h1.data[0, 0, :, :]                           # recontructred image from the models
        self.fits_model = h2.data                                       # information on CLEAN or Gaussian models
        self.fits_modcol = list(map(str.lower, h2.data.dtype.names))    # columns of model data
        self.fits_Nmodel = h1.header["NITER"]
        self.fits_fmin = h1.header["DATAMIN"]                           # min intensity of the image    [Jy/beam]
        self.fits_fmax = h1.header["DATAMAX"]                           # max intensity of the image    [Jy/beam]
        self.fits_bmin = h1.header["BMIN"]                              # minor axis of the beam        [degrees]
        self.fits_bmaj = h1.header["BMAJ"]                              # major axis of the beam        [degrees]
        self.fits_bpa = h1.header["BPA"]                                # position angle of the beam    [degrees]
        self.fits_noise = h1.header["NOISE"]                            # theoretically estimated noise (not image r.m.s.)
        self.fits_image_rms = gamvas.utils.cal_rms(self.fits_image)
        self.fits_imgcntr =\
            gamvas.utils.make_cntr(
                self.fits_image,
                contour_snr=contour_snr
            )

        self.fits_npix = self.fits_image.shape[0]                       # number of pixel   [-]
        self.fits_psize = np.abs(h1.header["CDELT2"])                   # pixel size        [degrees]
        self.fits_range_ra =\
            np.arange(
                +self.fits_psize * self.fits_npix / 2,
                -self.fits_psize * self.fits_npix / 2,
                -self.fits_psize,
                dtype="f4"
            )
        self.fits_range_dec =\
            np.arange(
                -self.fits_psize * self.fits_npix / 2,
                +self.fits_psize * self.fits_npix / 2,
                +self.fits_psize,
                dtype="f4"
            )
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
            self.fits_image_di = np.ones((self.fits_npix, self.fits_npix)) * (0.1 * self.fits_image_vi)
            self.fits_clean_dflux_i = 0.1 * self.fits_clean_vflux_i
            self.fits_imgcntr_i = self.fits_imgcntr
        elif self.fits_stokes == 2:
            peaks = [np.min(self.fits_image), np.max(self.fits_image)]
            self.fits_model_q = self.fits_model
            self.fits_image_vq = self.fits_image
            self.fits_image_rms_q = self.fits_image_rms
            self.fits_image_vpeak_q = np.max(self.fits_image)
            self.fits_clean_vflux_q = np.sum(self.fits_model["FLUX"])
            sig_peak, sig_tot, sig_size = cal_clean_error(s_peak=np.max(self.fits_image_vq), s_tot=self.fits_clean_vflux_q, sig_rms=self.fits_image_rms_q)
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
            sig_peak, sig_tot, sig_size =\
                cal_clean_error(
                    s_peak=np.max(self.fits_image_vu),
                    s_tot=self.fits_clean_vflux_u,
                    sig_rms=self.fits_image_rms_u
                )
            self.fits_image_dpeak_u = 0.1 * np.abs(self.fits_image_vpeak_u)
            self.fits_image_du = np.ones((self.fits_npix, self.fits_npix)) * (0.1 * np.abs(self.fits_image_vu))
            self.fits_clean_dflux_u = 0.1 * np.abs(self.fits_clean_vflux_u)
            self.fits_imgcntr_u = self.fits_imgcntr
        elif self.fits_stokes == 4:
            peaks = [np.min(self.fits_image), np.max(self.fits_image)]
            self.fits_model_v = self.fits_model
            self.fits_image_vv = self.fits_image
            self.fits_image_rms_v = self.fits_image_rms
            self.fits_image_vpeak_v = np.max(self.fits_image)
            self.fits_clean_vflux_v = np.sum(self.fits_model["FLUX"])
            sig_peak, sig_tot, sig_size =\
                cal_clean_error(
                    s_peak=np.max(self.fits_image_vv),
                    s_tot=self.fits_clean_vflux_v,
                    sig_rms=self.fits_image_rms_v
                )
            self.fits_image_dpeak_v = 0.1 * np.abs(self.fits_image_vpeak_v)
            self.fits_image_dv = np.ones((self.fits_npix, self.fits_npix)) * (0.1 * np.abs(self.fits_image_vv))
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

        if self.fits_stokes in [-1, 1] and pinfo:
            print(" " + "-"*50)
            print(" | Fits-image file information   ({0:10s})".format(self.fits_date) + " "*5 + "|")
            print(" " + "-"*50)
            print(" | Observing Source         : {0:20s}|".format(self.fits_source))
            print(" | Observing Frequency      : {0:-8.3f} (GHz)      |".format(self.fits_freq))
            print(" | Total Number of Model    : {0:-8d} ({1})".format(self.fits_Nmodel, modty) + " "*(9-len(modty))+"|")
            print(" | Total CLEAN Flux         : {0:-8.2f} (Jy)       |".format(np.sum(self.fits_model["FLUX"])))
            print(" | Beam Minor Axis          : {0:-8.2f} (mas)      |".format(self.fits_bmin *u.deg.to(u.mas) ))
            print(" | Beam Major Axis          : {0:-8.2f} (mas)      |".format(self.fits_bmaj *u.deg.to(u.mas) ))
            print(" | Beam Position Angle Axis : {0:-8.2f} (deg)      |".format(self.fits_bpa))
            print(" | Minimum Intensity        : {0:-8.2f} (Jy/Beam)  |".format(self.fits_fmin))
            print(" | Maximum Intensity        : {0:-8.2f} (Jy/Beam)  |".format(self.fits_fmax))
            print(" | Theoretical Noise        : {0:-8.2f} (mJy)      |".format(self.fits_noise *1e3))
            print(" | Image-estimated Noise    : {0:-8.2f} (mJy)      |".format(self.fits_image_rms*1e3))
            print(" | Dynamic Range            : {0:-8.2f}            |".format(self.fits_fmax/self.fits_image_rms))
            print(" " + "-"*50)
            print("\n")

        if self.uvinfo:
            uu = self.uu
            vv = self.vv
            uvd = np.sqrt(uu**2 + vv**2)

            major = self.fits_model["MAJOR AX"]
            minor = self.fits_model["MINOR AX"]
            S = self.fits_model["FLUX"]
            a = u.deg.to(u.rad) * (2 * major + 1 * minor)/3
            l = u.deg.to(u.rad) * self.fits_model["DELTAX"]
            m = u.deg.to(u.rad) * self.fits_model["DELTAY"]
            vism = 0+0j
            for i in range(self.fits_Nmodel):
                vis =\
                    S[i] *\
                    np.exp(-2 * (np.pi * a[i])**2 * (uu**2 + vv**2)) *\
                    np.exp(-2j * np.pi * (uu * (-l[i]) - vv * m[i]))
                vism += vis

            if self.fits_stokes == 1:
                self.vism_i = vism
            if self.fits_stokes == 2:
                self.vism_q = vism
            if self.fits_stokes == 3:
                self.vism_u = vism
            if self.fits_stokes == 3:
                self.vism_v = vism
            self.vism_p = self.vism_q + 1j * self.vism_u

            vism_s = [self.vism_i, self.vism_q, self.vism_u, self.vism_v, self.vism_p]
            vnames = list(self.vism.dtype.names)
            for i in range(len(vism_s)):
                self.vism[vnames[i]] = vism_s[i]


    def load_uvf(self,
        select="i", select_if="all", uvw="natural", uvave="none", scanlen=600, d=0.0, m=0.0,
        doscatter=False, timeflag=None, snrflag=0, set_clq=True, set_pang=True, reorder=True, pinfo=True,
    ):
        """
        Load uv-fits file and extract the information
            Arguments:
                select (str): The polarization type
                select_if (str): The number(s) of IFs
                uvw (str): The uv-weighting // u:uniform, n:natural
                uvave (int): The averaging time [sec]
                scanlen (int): The scan length [sec]
                doscatter (bool): The toggle option if to compute sigma from standard
                snrflag (float): The signal-to-noise ratio for the flagging
                set_clq (bool): The toggle option if to set closure quantities
                set_pang (bool): The toggle option if to set parallactic angle
                reorder (bool): The toggle option if to reorder the data by baseline
                pinfo (bool): The toggle option if to print the information
        """
        self.avgtime = uvave
        self.scanlen = scanlen
        self.snrflag = snrflag
        self.uvw = uvw
        self.uvinfo = True
        self.select_if = select_if

        self.d = d
        self.m = m

        try:
            uvf_file = fits.open(self.path + self.file)
        except:
            uname = self.file.upper()
            if uname.endswith(".FITS"):
                uvf_file = fits.open(self.path + self.file[:-4] + "uvf")
            elif uname.endswith(".UVF") | uname.endswith(".UVFITS"):
                uvf_file = fits.open(self.path + self.file)
            elif uname.endswith(".UVP"):
                uvf_file = fits.open(self.path + self.file)
            else:
                try:
                    uvf_file = fits.open(self.path + self.file)
                except:
                    uvf_file = fits.open(self.path + self.file + ".uvf")

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
            self.source=h1.header["OBJECT"]
        if not self.freq:
            self.freq=freq

        self.ra = h1.header["CRVAL6"]
        self.dec = h1.header["CRVAL7"]
        self.nvis = h1.header["GCOUNT"]
        self.no_if = h2.header["NO_IF"]
        self.refGST = h3.header["GSTIA0"]
        self.f2w = (C.c / (self.freq * u.GHz)).to(u.m).value

        if self.select_if == "all":
            ifs = np.arange(self.no_if)
        elif self.select_if in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            ifs = np.array([self.select_if]).astype(int)-1
        else:
            ifs = np.array(self.select_if)
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

        bline = data1["BASELINE"]
        date = Ati(data1["DATE"] + data1["_DATE"], format="jd").iso
        mjd = Ati(data1["DATE"] + data1["_DATE"], format="jd").mjd
        min_jd = int(np.min(data1["DATE"] + data1["_DATE"]) - 2400000.5)
        time = (data1["DATE"] + data1["_DATE"] - 2400000.5 - min_jd) * 24

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
        self.ant_dict_num2name = {val: key for key, val in self.ant_dict_name2num.items()}

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

        tsets = [
            tnames, xyz[:,0], xyz[:,1], xyz[:,2],
            sefdr, sefdl, dr, dl,
            fr_par, fr_el, fr_off]
        theads = [
            "name", "x", "y", "z",
            "sefd_r", "sefd_l", "d_r", "d_l",
            "fr_par", "fr_el", "fr_off"]
        ttypes = [
            "U32", "f8", "f8", "f8",
            "f8", "f8", "c16", "c16",
            "f8", "f8", "f8"]
        tncol = len(tnames)
        tarr = np.empty(tncol, dtype=[(theads[i], ttypes[i]) for i in range(len(tsets))])
        for i in range(len(tsets)):
            tarr[theads[i]] = tsets[i]
        tarr["name"] = np.array(list(map(lambda x : x.replace(" ",""), tarr["name"])))

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
        r_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 0].reshape(self.nvis, self.no_if, 1)
        i_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 1].reshape(self.nvis, self.no_if, 1)
        w_1 = data1["DATA"][:, 0, 0, ifs, 0, 0, 2].reshape(self.nvis, self.no_if, 1)

        if nstokes == 2:
            r_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 0].reshape(self.nvis, self.no_if, 1)
            i_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 1].reshape(self.nvis, self.no_if, 1)
            w_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 2].reshape(self.nvis, self.no_if, 1)
            r_3 = r_1 * 0.0
            i_3 = i_1 * 0.0
            w_3 = w_1 * 0.0
            r_4 = r_1 * 0.0
            i_4 = i_1 * 0.0
            w_4 = w_1 * 0.0
        elif nstokes == 4:
            r_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 0].reshape(self.nvis, self.no_if, 1)
            i_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 1].reshape(self.nvis, self.no_if, 1)
            w_2 = data1["DATA"][:, 0, 0, ifs, 0, 1, 2].reshape(self.nvis, self.no_if, 1)
            r_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 0].reshape(self.nvis, self.no_if, 1)
            i_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 1].reshape(self.nvis, self.no_if, 1)
            w_3 = data1["DATA"][:, 0, 0, ifs, 0, 2, 2].reshape(self.nvis, self.no_if, 1)
            r_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 0].reshape(self.nvis, self.no_if, 1)
            i_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 1].reshape(self.nvis, self.no_if, 1)
            w_4 = data1["DATA"][:, 0, 0, ifs, 0, 3, 2].reshape(self.nvis, self.no_if, 1)
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

        maskw_1 = (np.isnan(w_1) | np.isinf(w_1)) | (w_1 == 0)
        maskw_2 = (np.isnan(w_2) | np.isinf(w_2)) | (w_2 == 0)
        maskw_3 = (np.isnan(w_3) | np.isinf(w_3)) | (w_3 == 0)
        maskw_4 = (np.isnan(w_4) | np.isinf(w_4)) | (w_4 == 0)

        if nstokes == 4:
            maskw = ~(((maskw_1 | maskw_2) | maskw_3) | maskw_4)
        elif nstokes == 2:
            maskw = ~(maskw_1 | maskw_2)
        elif nstokes == 1:
            maskw = ~maskw_1

        w_1 = np.where(maskw, w_1, np.nan)
        w_2 = np.where(maskw, w_2, np.nan)
        w_3 = np.where(maskw, w_3, np.nan)
        w_4 = np.where(maskw, w_4, np.nan)

        maskwn = ~np.isnan(w_1)
        nvis = np.sum(np.sum(maskwn, axis=2), axis=1)

        e_1 = np.sqrt(np.nansum(np.nansum(1/w_1, axis=2), axis=1)) / nvis
        e_2 = np.sqrt(np.nansum(np.nansum(1/w_2, axis=2), axis=1)) / nvis
        e_3 = np.sqrt(np.nansum(np.nansum(1/w_3, axis=2), axis=1)) / nvis
        e_4 = np.sqrt(np.nansum(np.nansum(1/w_4, axis=2), axis=1)) / nvis
        maske_1 = np.isnan(e_1) | (e_1== 0)
        maske_2 = np.isnan(e_2) | (e_2== 0)
        maske_3 = np.isnan(e_3) | (e_3== 0)
        maske_4 = np.isnan(e_4) | (e_4== 0)
        if nstokes == 4:
            maske = ~(((maske_1 | maske_2) | maske_3) | maske_4)
        elif nstokes == 2:
            maske = ~(maske_1 | maske_2)
        elif nstokes == 1:
            maske = ~maske_1
        e_1 = e_1[maske]
        e_2 = e_2[maske]
        e_3 = e_3[maske]
        e_4 = e_4[maske]

        v_1 = r_1 + 1j * i_1
        v_2 = r_2 + 1j * i_2
        v_3 = r_3 + 1j * i_3
        v_4 = r_4 + 1j * i_4
        v_1 = v_1.reshape(self.nvis, self.no_if, 1)
        v_2 = v_2.reshape(self.nvis, self.no_if, 1)
        v_3 = v_3.reshape(self.nvis, self.no_if, 1)
        v_4 = v_4.reshape(self.nvis, self.no_if, 1)

        v_1[~maskw] = np.nan
        v_2[~maskw] = np.nan
        v_3[~maskw] = np.nan
        v_4[~maskw] = np.nan

        uu = uu[maske]
        vv = vv[maske]
        ww = ww[maske]
        time = time[maske]
        mjd = mjd[maske]
        try:
            tints = uvf_file[0].data["INTTIM"][maske]
        except KeyError:
            tints = np.zeros(len(maske))

        ant1 = uvf_file[0].data["BASELINE"][maske].astype(int) // 256
        ant2 = uvf_file[0].data["BASELINE"][maske].astype(int) - ant1 * 256
        ant1 = ant1 - 1
        ant2 = ant2 - 1

        ant_name1 = np.array([tarr[np.where(tnums == i)[0][0]]["name"] for i in ant1])
        ant_name2 = np.array([tarr[np.where(tnums == i)[0][0]]["name"] for i in ant2])

        v_1 = np.nanmean(np.nanmean(v_1, axis=2), axis=1)[maske]
        v_2 = np.nanmean(np.nanmean(v_2, axis=2), axis=1)[maske]
        v_3 = np.nanmean(np.nanmean(v_3, axis=2), axis=1)[maske]
        v_4 = np.nanmean(np.nanmean(v_4, axis=2), axis=1)[maske]

        uvdist = np.sqrt(uu**2 + vv**2)

        if not timeflag is None:
            mask_time = (timeflag[0] <= time) & (time <= timeflag[1])
            time = time[~mask_time]
            tints = tints[~mask_time]
            mjd = mjd[~mask_time]
            uu = uu[~mask_time]
            vv = vv[~mask_time]
            ww = ww[~mask_time]
            uvdist = uvdist[~mask_time]

            v_1 = v_1[~mask_time]
            v_2 = v_2[~mask_time]
            v_3 = v_3[~mask_time]
            v_4 = v_4[~mask_time]

            e_1 = e_1[~mask_time]
            e_2 = e_2[~mask_time]
            e_3 = e_3[~mask_time]
            e_4 = e_4[~mask_time]

            ant_name1 = ant_name1[~mask_time]
            ant_name2 = ant_name2[~mask_time]

        self.time = time
        self.mjd = mjd
        self.uu = uu
        self.vv = vv
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

        self.flag_snr(
            snrflag=self.snrflag,
            pinfo=pinfo,
            set_uvvis=False
        )

        self.set_uvvis()
        self.set_closure()

        if self.avgtime in ["none", "scan"]:
            binning = self.scanlen
        else:
            binning = self.avgtime

        self.cal_systematics(binning=binning, type="vis")
        if len(self.tarr) >= 4:
            self.cal_systematics(binning=binning, type="clamp")
        if len(self.tarr) >= 3:
            self.cal_systematics(binning=binning, type="clphs")

        self.uvave(
            uvave=uvave,
            scanlen=scanlen,
            doscatter=doscatter,
            set_clq=set_clq,
            set_pang=set_pang,
            pinfo=pinfo
        )

        if reorder:
            self.reorder_baseline()

        self.date = Ati(self.data["mjd"], format="mjd").iso[0][:10]
        self.fit_beam(uvw=uvw)
        bmin, bmaj, bpa = self.beam_prms
        source = self.source.replace(" ", "")
        if pinfo:
            print(" " + "-" * 29)
            print(" | UV-fits file information  |")
            print(" " + "-" * 29)
            print(" | Date      : {0:10s}   ".format(self.date) + " |")
            print(" | Source    : {0:10s}   ".format(source) + " |")
            print(" | R.A       : {0:-7.3f} ".format(self.ra) + "(deg) |")
            print(" | Dec       : {0:-7.3f} ".format(self.dec) + "(deg) |")
            print(" | Frequency : {0:-7.3f} ".format(self.freq) + "(GHz) |")
            print(" | B.A-minor : {0:-7.3f} ".format(bmin) + "(mas) |")
            print(" | B.A-major : {0:-7.3f} ".format(bmaj) + "(mas) |")
            print(" | B.P-Angle : {0:-7.3f} ".format(bpa) + "(deg) |")
            if uvw in ["n", "natural"]:
                print(" | UV-weight : 'natural' " + "    |")
            else:
                print(" | UV-weight : 'uniform' " + "    |")
            print(" " + "-"*29)
            if self.nstokes==4:
                i = self.data["vis_i"]
                p = self.data["vis_p"]
                fp = p/i
                print(f" # Degree of Linear Polarization (rough estimation) : {np.abs(np.nanmean(fp))*100:.2f} %\n")
                self.m = np.abs(np.nanmean(fp))
            else:
                print("\n")

        self.bmin = bmin
        self.bmaj = bmaj
        self.bpa  = bpa


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

        if np.all([field not in tarr.dtype.names for field in ["lat", "lon", "height"]]):
            dtype_ = np.dtype({
                "names":["lat", "lon", "height"],
                "formats":["f8", "f8", "f8"]
            })
            dtype_ = np.dtype((tarr.dtype.descr + dtype_.descr))
            tarr_ = []
            for i in range(tarr.size):
                tarr_.append(
                    np.array(
                        tuple([tarr[i][j] for j in range(len(tarr[i]))] + [lat[i], lon[i], height[i]]),
                        dtype=dtype_))
            tarr = np.array(tarr_)
            self.tarr = tarr

        obstime = Ati(data["mjd"], format="mjd").iso
        src_coord = SkyCoord(ra=self.ra*u.deg, dec=self.dec*u.deg)
        ants = np.unique(np.append(data["ant_name1"],data["ant_name2"]))

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

            antpos = EarthLocation(
                lat=lat_ * u.deg,
                lon=lon_ * u.deg,
                height=h_ * u.m
            )
            src_azel = src_coord.transform_to(AltAz(obstime=obstime, location=antpos))
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
        cos_pa1 = np.sin(lat1_rad) - np.sin(dec) * np.sin(el1_rad) / np.cos(dec) / np.cos(el1_rad)
        sin_pa2 = np.cos(lat2_rad) / np.cos(dec) * np.sin(az2_rad)
        cos_pa2 = np.sin(lat2_rad) - np.sin(dec) * np.sin(el2_rad) / np.cos(dec) / np.cos(el2_rad)
        p_angle1 = np.angle(cos_pa1 + 1j * sin_pa1)
        p_angle2 = np.angle(cos_pa2 + 1j * sin_pa2)

        data = rfn.append_fields(data, "phi1", p_angle1, usemask=False)
        data = rfn.append_fields(data, "phi2", p_angle2, usemask=False)
        self.data = data


    def append_visibility_model(self,
        freq_ref, freq, theta, pol=False, fitset="sf", spectrum="spl", set_spectrum=True, args=None
    ):
        """
        Append the model visibility to the data
            Arguments:
                freq_ref (float): The reference frequency [GHz]
                freq (float): The frequency [GHz]
                theta (dict): The model parameters
                pol (bool): The toggle option if the model is polarized
                fitset (str): The fit setting option // sf:single-frequency, mf:multi-frequency
                spectrum (str): The spectrum type // spl:simple power-law, cpl:curved power-law, ssa:synchrotron self-absorption
                args (tuple): The arguments for the model
        """
        uvdat = self.data
        nvis = uvdat.size
        vism = np.zeros(nvis, dtype="c8")
        if pol:
            theta = theta.tolist()
            nmod = len(theta)
            for i in range(nmod):
                vism +=\
                    gamvas.polarization.functions.gvis(
                        (args[0], args[1], args[2][i], args[3][i], args[4][i]),
                        theta[i]
                    )
        else:
            nmod = int(np.round(theta["nmod"]))
            for i in range(nmod):
                if fitset == "sf":
                    if i == 0:
                        args = (uvdat["u"], uvdat["v"])
                        vism =\
                            vism + gamvas.functions.gvis0(
                                args,
                                theta[f"{i+1}_S"],
                                theta[f"{i+1}_a"]
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
                                args = (freq_ref, freq, uvdat["u"], uvdat["v"])
                                vism =\
                                    vism + gamvas.functions.gvis_spl0(
                                        args,
                                        theta[f"{i+1}_S"],
                                        theta[f"{i+1}_a"],
                                        theta[f"{i+1}_alpha"]
                                    )
                            elif spectrum == "cpl":
                                args = (freq, uvdat["u"], uvdat["v"])
                                vism =\
                                    vism + gamvas.functions.gvis_cpl0(
                                        args,
                                        theta[f"{i+1}_S"],
                                        theta[f"{i+1}_a"],
                                        theta[f"{i+1}_alpha"],
                                        theta[f"{i+1}_freq"]
                                    )
                            elif spectrum == "ssa":
                                args = (freq, uvdat["u"], uvdat["v"])
                                vism =\
                                    vism + gamvas.functions.gvis_ssa0(
                                        args,
                                        theta[f"{i+1}_S"],
                                        theta[f"{i+1}_a"],
                                        theta[f"{i+1}_alpha"],
                                        theta[f"{i+1}_freq"]
                                    )
                        else:
                            if int(np.round(theta[f"{i+1}_thick"])) == 0 or spectrum == "spl":
                                args = (freq_ref, freq, uvdat["u"], uvdat["v"])
                                vism =\
                                    vism + gamvas.functions.gvis_spl(
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
                                        vism + gamvas.functions.gvis_cpl(
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
                                        vism + gamvas.functions.gvis_ssa(
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
                            vism =\
                                vism + gamvas.functions.gvis0(
                                    args,
                                    theta[f"{i+1}_S"],
                                    theta[f"{i+1}_a"]
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


    def selfcal(self,
        type="phs", gnorm=True, tint=False, scanlen=300, startmod=False, lm=(0,0), refant=None
    ):
        """
        Do self-calibration based on model visibility
            Arguments:
                type (str): The self-calibration type // amp, phs, a&p, gscale
                gnorm (bool): The toggle option for gain normalization
                tint (list, float): The interval times [s]
                scanlen (float): The scan length [s]
                startmod (bool): The toggle option if do phase self-calibration using 1 Jy point at (0,0)
                lm (tuple): The position of the point source
        """
        print(f"# Self-calibration (startmod={startmod}, type={type}, {self.freq:.1f} GHz)")
        if startmod:
            print(f"# Startmod : Self-calibrating to 1 Jy point source at ({lm[0]},{lm[1]})")


        def cal_nll(theta, indata, inobs, inmod, inants):
            nant = int(len(theta) / 2)
            vsigma = np.abs(indata["sigma"])
            gamp = dict(map(lambda i, j: (i, j), inants, [*theta[:nant]]))
            gphs = dict(map(lambda i, j: (i, j), inants, [*theta[nant:]]))
            ant1 = indata["ant_name1"]
            ant2 = indata["ant_name2"]
            gain1 = (np.array(list(map(gamp.get, ant1))) * np.exp(1j * np.array(list(map(gphs.get, ant1)))))
            gain2 = (np.array(list(map(gamp.get, ant2))) * np.exp(1j * np.array(list(map(gphs.get, ant2))))).conj()
            if type == "gscale":
                gain1 = np.abs(gain1)
                gain2 = np.abs(gain2)
                crmod = (gain1 * gain2) * inmod
                amp_obs = np.abs(inobs)
                amp_mod = np.abs(crmod)
                sig_amp = vsigma
                out_nll = 0.5 * np.sum((amp_mod - amp_obs)**2 / sig_amp**2 + np.log(2 * np.pi * sig_amp**2))
            elif type == "amp":
                crmod = (gain1 * gain2) * inmod
                amp_obs = np.abs(inobs)
                amp_mod = np.abs(crmod)
                sig_amp = vsigma
                out_nll = 0.5 * np.sum((amp_mod - amp_obs)**2 / sig_amp**2 + np.log(2 * np.pi * sig_amp**2))
            elif type == "phs":
                crmod = (gain1 * gain2) * inmod
                phs_obs = np.angle(inobs)
                phs_mod = np.angle(crmod)
                sig_phs = vsigma / np.abs(inobs)
                out_nll = 0.5 * np.sum(np.abs(np.exp(1j * phs_mod) - np.exp(1j * phs_obs))**2 / sig_phs**2 + np.log(2 * np.pi * sig_phs**2))
            elif type == "a&p":
                crmod = (gain1 * gain2) * inmod
                out_nll = 0.5 * np.sum(np.abs(crmod - inobs)**2 / vsigma**2 + np.log(2 * np.pi * vsigma**2))
            return out_nll

        if not tint:
            tint = [2880, 1920, 1280, 853, 569, 379, 253, 169, 113, 75, 50, 33, 22, 15, 10, 7, 5, 3, 2, 1]
        if type == "gscale":
            tint = [2880]
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
            db = dbs(eps=t * 60, min_samples=2).fit((time * 3600).reshape(-1, 1))
            scannums = db.labels_
            uscan = np.unique(scannums)

            cgain1 = None
            cgain2 = None
            for nscan, scannum in enumerate(uscan):
                mask_scan = scannums == scannum
                time_norm = time_sec - np.min(time_sec[mask_scan])
                timer = np.arange(0, np.max(time_norm) + 2 * t * 60, t * 60)

                for ntime in range(len(timer) - 1):
                    mask_time = (timer[ntime + 0] <= time_norm) & (time_norm < timer[ntime + 1])
                    mask_tot = (mask_scan & mask_time)

                    data_ = data[mask_tot]
                    time_ = time[mask_tot]
                    ants = np.unique(np.append(data_["ant_name1"], data_["ant_name2"]))
                    if len(data_) == 0:
                        continue

                    if len(ants) < 3:
                        gain1 = np.ones(len(data_)) * np.exp(1j * 0)
                        gain2 = np.ones(len(data_)) * np.exp(1j * 0)
                    elif len(ants) == 3 and type in ["amp", "a&p", "gscale"]:
                        gain1 = np.ones(len(data_)) * np.exp(1j * 0)
                        gain2 = np.ones(len(data_)) * np.exp(1j * 0)
                    else:
                        if startmod:
                            obs = data_["vis"]
                            sig = data_["sigma"]
                            r = np.abs(lm[1] + 1j * lm[0])
                            p = np.angle(lm[1] + 1j * lm[0], deg=True)
                            mod = gamvas.functions.gvis((data_["u"], data_["v"]), 1, 0.1, r, p)
                            mod = mod/np.abs(mod)
                        else:
                            obs = data_["vis"]
                            sig = data_["sigma"]
                            mod = data_["vism"]
                        init = np.append(np.ones(len(ants)), np.zeros(len(ants)))
                        bound1 = [[+0.5, +1.5] for i in range(len(ants))]
                        bound2 = [[-np.pi, +np.pi] for i in range(len(ants))]
                        bounds = bound1 + bound2

                        nll = lambda *args : cal_nll(*args)
                        soln = optimize.minimize(nll, init, args=(data_, obs, mod, ants), bounds=bounds, method="Powell")
                        gamp = dict(map(lambda i, j: (i, j), ants, [*soln.x[:len(ants)]]))
                        gphs = dict(map(lambda i, j: (i, j), ants, [*soln.x[len(ants):]]))
                        ant1 = data_["ant_name1"]
                        ant2 = data_["ant_name2"]
                        gain1 = (np.array(list(map(gamp.get, ant1))) * np.exp(1j * np.array(list(map(gphs.get, ant1)))))
                        gain2 = (np.array(list(map(gamp.get, ant2))) * np.exp(1j * np.array(list(map(gphs.get, ant2))))).conj()

                        if type in ["phs"] or startmod:
                            gain1 = gain1 / np.abs(gain1)
                            gain2 = gain2 / np.abs(gain2)
                        elif type in ["amp", "gscale"]:
                            if type == "gscale" and gnorm:
                                gains = np.unique(np.append(np.abs(gain1), np.abs(gain2)))
                                gain1 = np.abs(gain1) / np.prod(gains)**(1/len(gains))
                                gain2 = np.abs(gain2) / np.prod(gains)**(1/len(gains))
                            gain1 = np.abs(gain1 ) * np.exp(1j * 0)
                            gain2 = np.abs(gain2 ) * np.exp(1j * 0)
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
        self.uu = self.data["u"]
        self.vv = self.data["v"]
        self.set_uvvis()
        # self.set_closure()

        if not startmod:
            self.data = rfn.append_fields(self.data, "vism", vism, usemask=False)
        self.cgain1 = out_cgain1
        self.cgain2 = out_cgain2


    def fit_beam(self,
        npix=128, uvw="natural"
    ):
        """
        Fit the beam parameters for the input UVF data
         (imported from eht-imaging)
         (https://achael.github.io/eht-imaging/; Chael+2018, ApJ, 857, 23C)
            Arguments:
                npix (int): The number of pixels for the beam fitting
                uvw (str): The UV weighting option // natural (n), uniform (u)
        """
        data = self.data

        uu = data["u"]
        vv = data["v"]
        vis = data["vis"]
        sig = data["sigma"]

        def fit_chisq(beamparams, db_coeff):
            (fwhm_maj2, fwhm_min2, theta) = beamparams
            a = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_min2 + np.sin(theta)**2 / fwhm_maj2)
            b = 4 * np.log(2) * (np.cos(theta)**2 / fwhm_maj2 + np.sin(theta)**2 / fwhm_min2)
            c = 8 * np.log(2) * np.cos(theta) * np.sin(theta) * (1.0 / fwhm_maj2 - 1.0 / fwhm_min2)
            gauss_coeff = np.array((a, b, c))
            chisq = np.sum((np.array(db_coeff) - gauss_coeff)**2)
            return chisq

        rng_map = self.mrng
        rng_axis = np.linspace(-rng_map, +rng_map, npix)
        xgrid, ygrid = np.meshgrid(-rng_axis, rng_axis)
        if uvw in ["n", "natural"]:
            wfn = 1/sig**2
        else:
            wfn = np.ones(vis.shape)

        abc = np.array([np.sum(wfn * uu**2),
                        np.sum(wfn * vv**2),
                        2 * np.sum(wfn * uu * vv)])
        abc *= (2. * np.pi**2 / np.sum(wfn))
        abc *= 1e-20    # Decrease size of coefficients

        # Fit the beam
        guess = [(50)**2, (50)**2, 0.0]
        params = optimize.minimize(fit_chisq, guess, args=(abc,), method="Powell")

        if params.x[0] > params.x[1]:
            fwhm_maj = 1e-10 * np.sqrt(params.x[0])
            fwhm_min = 1e-10 * np.sqrt(params.x[1])
            theta = np.mod(params.x[2], np.pi)
        else:
            fwhm_maj = 1e-10 * np.sqrt(params.x[1])
            fwhm_min = 1e-10 * np.sqrt(params.x[0])
            theta = np.mod(params.x[2] + np.pi / 2.0, np.pi)

        gparams = np.array((fwhm_maj, fwhm_min, theta))
        gparams[0] *= u.rad.to(u.mas)
        gparams[1] *= u.rad.to(u.mas)
        gparams[2] *= u.rad.to(u.deg)

        self.beam_bmin = gparams[1]
        self.beam_bmaj = gparams[0]
        self.beam_bpa = gparams[2]
        self.beam_prms = (self.beam_bmin, self.beam_bmaj, self.beam_bpa)
        self.ploter.bnom = self.beam_prms


    def flag_sigma(self,
        sigma=None, pinfo=True, set_uvvis=True
    ):
        """
        Flag the visibilities based on the sigma level
            Arguments:
                sigma (float): The sigma level for the flagging
                pinfo (bool): The toggle option for printing the information
                set_uvvis (bool): The toggle option for setting the UV and visibility data
        """
        data = self.data

        time_ = self.time
        tint_ = self.tint
        mjd_ = self.mjd
        ant_name1_ = self.ant_name1
        ant_name2_ = self.ant_name2
        uu_ = self.uu
        vv_ = self.vv
        uvdist_ = np.sqrt(self.uu**2 + self.vv**2)
        vis_1_ = self.vis_1
        vis_2_ = self.vis_2
        vis_3_ = self.vis_3
        vis_4_ = self.vis_4
        sig_1_ = self.sig_1
        sig_2_ = self.sig_2
        sig_3_ = self.sig_3
        sig_4_ = self.sig_4

        if sigma is None:
            sigma1 = np.percentile(sig_1_, 90)
            sigma2 = np.percentile(sig_2_, 90)
        else:
            sigma1 = sigma
            sigma2 = sigma

        mask_sigma1 = sig_1_ < sigma1
        mask_sigma2 = sig_2_ < sigma2

        if self.nstokes == 1:
            mask_sigma = mask_sigma1
        if self.nstokes == 2:
            mask_sigma = mask_sigma1
        if self.nstokes == 4:
            mask_sigma = (mask_sigma1 & mask_sigma2)

        if pinfo:
            nflag = mask_sigma.shape[0] - np.sum(mask_sigma.astype(int))
            ntot = mask_sigma.shape[0]
            print(f"# Flag {nflag}/{ntot} visibilities (applied sigma={sigma1:.2f}, {sigma2:.2f})")

        vis_1 = vis_1_[mask_sigma]
        vis_2 = vis_2_[mask_sigma]
        vis_3 = vis_3_[mask_sigma]
        vis_4 = vis_4_[mask_sigma]
        sig_1 = sig_1_[mask_sigma]
        sig_2 = sig_2_[mask_sigma]
        sig_3 = sig_3_[mask_sigma]
        sig_4 = sig_4_[mask_sigma]
        time = time_[mask_sigma]
        tint = tint_[mask_sigma]
        mjd = mjd_[mask_sigma]
        ant_name1 = ant_name1_[mask_sigma]
        ant_name2 = ant_name2_[mask_sigma]
        uu = uu_[mask_sigma]
        vv = vv_[mask_sigma]

        if self.nstokes == 1:
            mask_nan = (~np.isnan(vis_1) | ~np.isnan(sig_1))
        if self.nstokes == 2:
            mask_nan1 = (~np.isnan(vis_1) | ~np.isnan(sig_1))
            mask_nan2 = (~np.isnan(vis_2) | ~np.isnan(sig_2))
            mask_nan = mask_nan1 & mask_nan2
        if self.nstokes == 4:
            mask_nan1 = (~np.isnan(vis_1) | ~np.isnan(sig_1))
            mask_nan2 = (~np.isnan(vis_2) | ~np.isnan(sig_2))
            mask_nan3 = (~np.isnan(vis_3) | ~np.isnan(sig_3))
            mask_nan4 = (~np.isnan(vis_4) | ~np.isnan(sig_4))
            mask_nan = mask_nan1 & mask_nan2 & mask_nan3 & mask_nan4
        if pinfo and np.sum(mask_nan) > 0:
            nflag = mask_nan.shape[0] - np.sum(mask_nan)
            ntot = mask_sigma.shape[0]
            print(f"# Flag NaN-values: {nflag}/{ntot} visibilities")

        self.vis_1 = vis_1[mask_nan]
        self.vis_2 = vis_2[mask_nan]
        self.vis_3 = vis_3[mask_nan]
        self.vis_4 = vis_4[mask_nan]
        self.sig_1 = sig_1[mask_nan]
        self.sig_2 = sig_2[mask_nan]
        self.sig_3 = sig_3[mask_nan]
        self.sig_4 = sig_4[mask_nan]
        self.time = time[mask_nan]
        self.tint = tint[mask_nan]
        self.mjd = mjd[mask_nan]
        self.ant_name1 = ant_name1[mask_nan]
        self.ant_name2 = ant_name2[mask_nan]
        self.uu = uu[mask_nan]
        self.vv = vv[mask_nan]

        if set_uvvis:
            self.set_uvvis()

        if set_clq:
            self.set_closure()


    def flag_snr(self,
        snrflag=0, pinfo=True, set_uvvis=True, set_clq=True
    ):
        """
        Flag the visibilities based on the signal-to-noise ratio
            Arguments:
                snrflag (float): The signal-to-noise ratio for the flagging
                pinfo (bool): The toggle option for printing the information
                set_uvvis (bool): The toggle option for setting the UV and visibility data
                set_clq (bool): The toggle option for setting the closure phase and amplitude data
        """
        if self.data is None:
            time_ = self.time
            tint_ = self.tint
            mjd_ = self.mjd
            ant_name1_ = self.ant_name1
            ant_name2_ = self.ant_name2
            uu_ = self.uu
            vv_ = self.vv
            uvdist_ = np.sqrt(self.uu**2 + self.vv**2)
            vis_1_ = self.vis_1
            vis_2_ = self.vis_2
            vis_3_ = self.vis_3
            vis_4_ = self.vis_4
            sig_1_ = self.sig_1
            sig_2_ = self.sig_2
            sig_3_ = self.sig_3
            sig_4_ = self.sig_4

            mask_snr1 = np.abs(vis_1_) / sig_1_ > snrflag
            mask_snr2 = np.abs(vis_2_) / sig_2_ > snrflag

            if self.nstokes == 1:
                mask_snr = mask_snr1
            if self.nstokes == 2:
                mask_snr = mask_snr1
            if self.nstokes == 4:
                mask_snr = (mask_snr1 & mask_snr2)

            if pinfo:
                nflag = mask_snr.shape[0] - np.sum(mask_snr.astype(int))
                ntot = mask_snr.shape[0]
                print(f"# Flag {nflag}/{ntot} visibilities (applied SNR={snrflag})")

            vis_1 = vis_1_[mask_snr]
            vis_2 = vis_2_[mask_snr]
            vis_3 = vis_3_[mask_snr]
            vis_4 = vis_4_[mask_snr]
            sig_1 = sig_1_[mask_snr]
            sig_2 = sig_2_[mask_snr]
            sig_3 = sig_3_[mask_snr]
            sig_4 = sig_4_[mask_snr]
            time = time_[mask_snr]
            tint = tint_[mask_snr]
            mjd = mjd_[mask_snr]
            ant_name1 = ant_name1_[mask_snr]
            ant_name2 = ant_name2_[mask_snr]
            uu = uu_[mask_snr]
            vv = vv_[mask_snr]

            if self.nstokes == 1:
                mask_nan = (~np.isnan(vis_1) | ~np.isnan(sig_1))
            if self.nstokes == 2:
                mask_nan1 = (~np.isnan(vis_1) | ~np.isnan(sig_1))
                mask_nan2 = (~np.isnan(vis_2) | ~np.isnan(sig_2))
                mask_nan = mask_nan1 & mask_nan2
            if self.nstokes == 4:
                mask_nan1 = (~np.isnan(vis_1) | ~np.isnan(sig_1))
                mask_nan2 = (~np.isnan(vis_2) | ~np.isnan(sig_2))
                mask_nan3 = (~np.isnan(vis_3) | ~np.isnan(sig_3))
                mask_nan4 = (~np.isnan(vis_4) | ~np.isnan(sig_4))
                mask_nan = mask_nan1 & mask_nan2 & mask_nan3 & mask_nan4
            if pinfo and np.sum(mask_nan) > 0:
                nflag = mask_nan.shape[0] - np.sum(mask_nan)
                ntot = mask_snr.shape[0]
                print(f"# Flag NaN-values: {nflag}/{ntot} visibilities")

            self.vis_1 = vis_1[mask_nan]
            self.vis_2 = vis_2[mask_nan]
            self.vis_3 = vis_3[mask_nan]
            self.vis_4 = vis_4[mask_nan]
            self.sig_1 = sig_1[mask_nan]
            self.sig_2 = sig_2[mask_nan]
            self.sig_3 = sig_3[mask_nan]
            self.sig_4 = sig_4[mask_nan]
            self.time = time[mask_nan]
            self.tint = tint[mask_nan]
            self.mjd = mjd[mask_nan]
            self.ant_name1 = ant_name1[mask_nan]
            self.ant_name2 = ant_name2[mask_nan]
            self.uu = uu[mask_nan]
            self.vv = vv[mask_nan]

            if set_uvvis:
                self.set_uvvis()
        else:
            data = self.data

            time_ = data["time"]
            tint_ = data["tint"]
            mjd_ = data["mjd"]
            ant_name1_ = data["ant_name1"]
            ant_name2_ = data["ant_name2"]
            uu_ = data["u"]
            vv_ = data["v"]
            uvdist_ = np.sqrt(uu_**2 + vv_**2)

            if self.nstokes == 1:
                try:
                    vis_1_ = data["vis_rr"]
                    vis_2_ = np.full(len(vis_1_), np.nan)
                    vis_3_ = np.full(len(vis_1_), np.nan)
                    vis_4_ = np.full(len(vis_1_), np.nan)
                    sig_1_ = data["sigma_rr"]
                    sig_2_ = np.full(len(vis_1_), np.nan)
                    sig_3_ = np.full(len(vis_1_), np.nan)
                    sig_4_ = np.full(len(vis_1_), np.nan)
                except:
                    vis_1_ = data["vis_ll"]
                    vis_2_ = np.full(len(vis_1_), np.nan)
                    vis_3_ = np.full(len(vis_1_), np.nan)
                    vis_4_ = np.full(len(vis_1_), np.nan)
                    sig_1_ = data["sigma_ll"]
                    sig_2_ = np.full(len(vis_1_), np.nan)
                    sig_3_ = np.full(len(vis_1_), np.nan)
                    sig_4_ = np.full(len(vis_1_), np.nan)
                mask_snr1 = np.abs(vis_1_) / sig_1_ > snrflag
                mask_snr = mask_snr1
            elif self.nstokes == 2:
                vis_1_ = data["vis_rr"]
                vis_2_ = data["vis_ll"]
                vis_3_ = np.full(len(vis_1_), np.nan)
                vis_4_ = np.full(len(vis_1_), np.nan)
                sig_1_ = data["sigma_rr"]
                sig_2_ = data["sigma_ll"]
                sig_3_ = np.full(len(vis_1_), np.nan)
                sig_4_ = np.full(len(vis_1_), np.nan)
                mask_snr1 = np.abs(vis_1_) / sig_1_ > snrflag
                mask_snr2 = np.abs(vis_2_) / sig_2_ > snrflag
                mask_snr = mask_snr1
            elif self.nstokes == 4:
                vis_1_ = data["vis_rr"]
                vis_2_ = data["vis_ll"]
                vis_3_ = data["vis_rl"]
                vis_4_ = data["vis_lr"]
                sig_1_ = data["sigma_rr"]
                sig_2_ = data["sigma_ll"]
                sig_3_ = data["sigma_rl"]
                sig_4_ = data["sigma_lr"]
                mask_snr1 = np.abs(vis_1_) / sig_1_ > snrflag
                mask_snr2 = np.abs(vis_2_) / sig_2_ > snrflag
                mask_snr = (mask_snr1 & mask_snr2)

            if pinfo:
                nflag = mask_snr.shape[0] - np.sum(mask_snr.astype(int))
                ntot = mask_snr.shape[0]
                print(f"# Flag {nflag}/{ntot} visibilities (applied SNR={snrflag})")

            if self.nstokes == 1:
                vis_1 = vis_1_[mask_snr]
                vis_2 = np.full(len(vis_1), np.nan)
                vis_3 = np.full(len(vis_1), np.nan)
                vis_4 = np.full(len(vis_1), np.nan)
                sig_1 = sig_1_[mask_snr]
                sig_2 = np.full(len(vis_1), np.nan)
                sig_3 = np.full(len(vis_1), np.nan)
                sig_4 = np.full(len(vis_1), np.nan)
            elif self.nstokes == 2:
                vis_1 = vis_1_[mask_snr]
                vis_2 = vis_2_[mask_snr]
                vis_3 = np.full(len(vis_1), np.nan)
                vis_4 = np.full(len(vis_1), np.nan)
                sig_1 = sig_1_[mask_snr]
                sig_2 = sig_2_[mask_snr]
                sig_3 = np.full(len(vis_1), np.nan)
                sig_4 = np.full(len(vis_1), np.nan)
            elif self.nstokes == 4:
                vis_1 = vis_1_[mask_snr]
                vis_2 = vis_2_[mask_snr]
                vis_3 = vis_3_[mask_snr]
                vis_4 = vis_4_[mask_snr]
                sig_1 = sig_1_[mask_snr]
                sig_2 = sig_2_[mask_snr]
                sig_3 = sig_3_[mask_snr]
                sig_4 = sig_4_[mask_snr]

            time = time_[mask_snr]
            tint = tint_[mask_snr]
            mjd = mjd_[mask_snr]
            ant_name1 = ant_name1_[mask_snr]
            ant_name2 = ant_name2_[mask_snr]
            uu = uu_[mask_snr]
            vv = vv_[mask_snr]

            if self.nstokes == 1:
                mask_nan = (~np.isnan(vis_1) | ~np.isnan(sig_1))
            if self.nstokes == 2:
                mask_nan1 = (~np.isnan(vis_1) | ~np.isnan(sig_1))
                mask_nan2 = (~np.isnan(vis_2) | ~np.isnan(sig_2))
                mask_nan = mask_nan1 & mask_nan2
            if self.nstokes == 4:
                mask_nan1 = (~np.isnan(vis_1) | ~np.isnan(sig_1))
                mask_nan2 = (~np.isnan(vis_2) | ~np.isnan(sig_2))
                mask_nan3 = (~np.isnan(vis_3) | ~np.isnan(sig_3))
                mask_nan4 = (~np.isnan(vis_4) | ~np.isnan(sig_4))
                mask_nan = mask_nan1 & mask_nan2 & mask_nan3 & mask_nan4
            if pinfo and np.sum(mask_nan) > 0:
                nflag = mask_nan.shape[0] - np.sum(mask_nan)
                ntot = mask_snr.shape[0]
                print(f"# Flag NaN-values: {nflag}/{ntot} visibilities")

            self.vis_1 = vis_1[mask_nan]
            self.vis_2 = vis_2[mask_nan]
            self.vis_3 = vis_3[mask_nan]
            self.vis_4 = vis_4[mask_nan]
            self.sig_1 = sig_1[mask_nan]
            self.sig_2 = sig_2[mask_nan]
            self.sig_3 = sig_3[mask_nan]
            self.sig_4 = sig_4[mask_nan]
            self.time = time[mask_nan]
            self.tint = tint[mask_nan]
            self.mjd = mjd[mask_nan]
            self.ant_name1 = ant_name1[mask_nan]
            self.ant_name2 = ant_name2[mask_nan]
            self.uu = uu[mask_nan]
            self.vv = vv[mask_nan]

            if set_uvvis:
                self.set_uvvis()

            if set_clq:
                self.set_closure()


    def flag_nant(self, nant=4):
        """
        Flag the visibilities based on the number of antennas
            Arguments:
                nant (int): The number of antennas for the flagging
        """
        data = self.data
        atime = data["time"]
        utime = np.unique(atime)
        mask = np.array([], dtype=bool)

        for ntime, time in enumerate(utime):
            mask_time = atime == time
            data_ = data[mask_time]
            ndata = len(data_)
            ants = np.unique(np.append(data_["ant_name1"], data_["ant_name2"]))
            if len(data_) < nant*(nant-1)/2:
                mask = np.append(mask, np.zeros(ndata, dtype=bool))
            else:
                mask = np.append(mask, np.ones(ndata, dtype=bool))

        self.time = self.time[mask]
        self.tint = self.tint[mask]
        self.mjd = self.mjd[mask]
        self.ant_name1 = self.ant_name1[mask]
        self.ant_name2 = self.ant_name2[mask]
        self.uu = self.uu[mask]
        self.vv = self.vv[mask]
        self.vis_1 = self.vis_1[mask]
        self.vis_2 = self.vis_2[mask]
        self.vis_3 = self.vis_3[mask]
        self.vis_4 = self.vis_4[mask]
        self.sig_1 = self.sig_1[mask]
        self.sig_2 = self.sig_2[mask]
        self.sig_3 = self.sig_3[mask]
        self.sig_4 = self.sig_4[mask]
        self.set_uvvis()
        self.set_closure()


    def cal_systematics(self,
        binning=None, type=None,
    ):
        def cal_s(s, X, sigma_th):
            Y = X / np.sqrt(sigma_th**2 + s**2)
            mad = np.nanmedian(np.abs(Y - np.nanmedian(Y)))
            out = np.abs(mad - 1/1.4826)
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
            raise ValueError("Please provide correct type (availables: 'vis', 'clamp', 'clphs')!")

        if not np.all(np.isnan(data["time"])):
            db = dbs(eps=binning, min_samples=1).fit((time * 3600).reshape(-1, 1))
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

                    fn = lambda *args : cal_s(*args)
                    soln = optimize.minimize(fn, [0.01], args=(X[mask], sigma_th[mask]), method="Powell")
                    out_time1.append(np.min(time[mask]))
                    out_time2.append(np.max(time[mask]))
                    out_pair.append(pair_)
                    out_systematics.append(np.abs(soln.x[0]))
            out = gamvas.utils.sarray(
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


    def apply_systematics(self, type=None, binning=None, d=0.0, m=0.0):
        if type is None:
            raise Exception("Please provide correct type (available: 'clamp', 'clphs')!")

        if binning is None:
            raise Exception("Please provide appropriate binning time!")

        if type == "vis":
            data = self.data
            systematics = self.systematics_vis
            label_sigma = "sigma"
            label_pair = "pair"
        elif type == "clamp":
            data = self.clamp
            systematics = self.systematics_clamp
            label_sigma = "sigma_clamp"
            label_pair = "quadra"
        elif type == "clphs":
            systematics = self.systematics_clphs
            data = self.clphs
            label_sigma = "sigma_clphs"
            label_pair = "triangle"

        db = dbs(eps=binning, min_samples=1).fit((data["time"] * 3600).reshape(-1, 1))
        scannums = db.labels_
        uscan = np.unique(scannums)

        out = np.zeros(len(data))
        for nscan in range(len(systematics)):
            for npair, pair_ in enumerate(np.unique(systematics["pair"])):
                mask = (systematics["time_beg"][nscan] <= data["time"]) & (data["time"] < systematics["time_end"][nscan]) & (data[label_pair] == pair_)
                if np.sum(mask) == 0:
                    continue
                out[mask] = np.sqrt(out[mask]**2 + systematics[nscan]["systematics"]**2)

        if type == "vis":
            self.data["sigma"] = np.sqrt(self.data["sigma"]**2 + out**2) + np.abs(self.data["vis"]) * np.sqrt(2) * d * m
            self.sig_1 = np.sqrt(self.sig_1**2 + out**2) + np.abs(self.data["vis"]) * np.sqrt(2) * d * m
            self.sig_2 = np.sqrt(self.sig_2**2 + out**2) + np.abs(self.data["vis"]) * np.sqrt(2) * d * m
        elif type == "clamp":
            self.clamp["sigma_clamp"] = np.sqrt(self.clamp["sigma_clamp"]**2 + out**2) + np.abs(np.log(self.clamp["clamp"])) * np.sqrt(2 * 4) * d * m
        elif type == "clphs":
            self.clphs["sigma_clphs"] = np.sqrt(self.clphs["sigma_clphs"]**2 + out**2) + np.sqrt(2 * 3) * d * m


    def add_error_fraction(self,
        fraction=0.01, time="all", antenna="all", type="all", setvis=True, setclq=True
    ):
        """
        Add the error fractionally to the visibility amplitude
            Arguments:
                fraction (float): The fraction of the error to be added
                type (str): The type of the error to be added
                            ("all":visibility+closure, "vis":visibility-only)
                setvis (bool): The toggle option for setting the visibility data
                setclq (bool): The toggle option for setting the closure data
        """
        data = self.data

        if self.nstokes == 1:
            try:
                vis_1 = data["vis_rr"]
                sig_1 = data["sigma_rr"]
            except:
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
            mask_ant = (data["ant_name1"] == antenna.upper()) | (data["ant_name2"] == antenna.upper())

        if time == "all":
            mask_time = np.ones(len(sig_1), dtype=bool)
        else:
            mask_time = (time[0] < data["time"]) & (data["time"] < time[1])

        mask = mask_ant & mask_time
        self.sig_1[mask] = sig_1[mask] + fraction * np.abs(vis_1[mask])
        self.sig_2[mask] = sig_2[mask] + fraction * np.abs(vis_2[mask])
        self.sig_3[mask] = sig_3[mask] + fraction * np.abs(vis_3[mask])
        self.sig_4[mask] = sig_4[mask] + fraction * np.abs(vis_4[mask])

        if setvis or type == "all":
            self.set_uvvis()
        if setclq or type == "all":
            self.set_closure()


    def add_error_factor(self,
        factor=1, time="all", antenna="all", type="all", setvis=True, setclq=True
    ):
        """
        Add the error by a factor
            Arguments:
                factor (float): The factor of the error to be added
                                ("all":visibility+closure, "vis":visibility-only)
                setvis (bool): The toggle option for setting the visibility data
                setclq (bool): The toggle option for setting the closure data
        """
        data = self.data

        if self.nstokes == 1:
            try:
                sig_1 = data["sigma_rr"]
            except:
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
            mask_ant = (data["ant_name1"] == antenna.upper()) | (data["ant_name2"] == antenna.upper())

        if time == "all":
            mask_time = np.ones(len(sig_1), dtype=bool)
        else:
            mask_time = (time[0] < data["time"]) & (data["time"] < time[1])

        mask = mask_ant & mask_time
        self.sig_1[mask] = sig_1[mask] + fraction * np.abs(vis_1[mask])
        self.sig_2[mask] = sig_2[mask] + fraction * np.abs(vis_2[mask])
        self.sig_3[mask] = sig_3[mask] + fraction * np.abs(vis_3[mask])
        self.sig_4[mask] = sig_4[mask] + fraction * np.abs(vis_4[mask])

        if setvis or type == "all":
            self.set_uvvis()
        if setclq or type == "all":
            self.set_closure()


    def set_uvvis(self):
        """
        Set the uv-visibility data
        """
        time = self.time
        tint = self.tint
        mjd = self.mjd
        ant_name1 = self.ant_name1
        ant_name2 = self.ant_name2
        uu = self.uu
        vv = self.vv
        uvdist = np.sqrt(self.uu**2 + self.vv**2)
        freq  = np.full(len(self.vis_1), self.freq)
        vis_1 = self.vis_1
        vis_2 = self.vis_2
        vis_3 = self.vis_3
        vis_4 = self.vis_4
        sig_1 = self.sig_1
        sig_2 = self.sig_2
        sig_3 = self.sig_3
        sig_4 = self.sig_4
        pair = list(map(lambda x, y: f"{x}-{y}", ant_name1, ant_name2))
        # set visibility on circular hands
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

        ant_num1 = np.array(list(map(self.ant_dict_name2num.get, ant_name1)))
        ant_num2 = np.array(list(map(self.ant_dict_name2num.get, ant_name2)))

        time_sec = (time * u.hour.to(u.second))

        if self.nstokes == 4:
            dheads = [
                "time", "freq", "tint", "mjd", "u", "v",
                "ant_num1", "ant_num2", "ant_name1", "ant_name2", "pair",
                "vis_i", "vis_q", "vis_u", "vis_v", "vis_p",
                "sigma_i", "sigma_q", "sigma_u", "sigma_v", "sigma_p",
                "vis_rr", "vis_ll", "vis_rl", "vis_lr",
                "sigma_rr", "sigma_ll", "sigma_rl", "sigma_lr"
            ]

            dtypes = [
                "f8", "f8", "f8", "f8", "f8", "f8",
                "i", "i", "U32", "U32", "U32",
                "c16", "c16", "c16", "c16", "c16",
                "f8", "f8", "f8", "f8", "f8",
                "c16", "c16", "c16", "c16",
                "f8", "f8", "f8", "f8"

            ]
            dataset = [
                time, freq, tint, mjd, uu, vv,
                ant_num1, ant_num2, ant_name1, ant_name2, pair,
                vis_i, vis_q, vis_u, vis_v, vis_p,
                sig_i, sig_q, sig_u, sig_v, sig_p,
                vis_1, vis_2, vis_3, vis_4,
                sig_1, sig_2, sig_3, sig_4
            ]

            data = gamvas.utils.sarray(dataset, dheads, dtypes)
        elif self.nstokes == 2:
            dheads = [
                "time", "freq", "tint", "mjd", "u", "v",
                "ant_num1", "ant_num2", "ant_name1", "ant_name2", "pair",
                "vis_i", "vis_rr", "vis_ll",
                "sigma_i", "sigma_rr", "sigma_ll"
            ]

            dtypes = [
                "f8", "f8", "f8", "f8", "f8", "f8",
                "i", "i", "U32", "U32", "U32",
                "c16", "c16", "c16",
                "f8", "f8", "f8"
            ]

            dataset = [
                time, freq, tint, mjd, uu, vv,
                ant_num1, ant_num2, ant_name1, ant_name2, pair,
                vis_i, vis_1, vis_2,
                sig_i, sig_1, sig_2
            ]

            data = gamvas.utils.sarray(dataset, dheads, dtypes)
        elif self.nstokes == 1:
            dheads = [
                "time", "freq", "tint", "mjd", "u", "v",
                "ant_num1", "ant_num2", "ant_name1", "ant_name2", "pair",
                f"vis_{self.select.lower()}", f"sigma_{self.select.lower()}"
            ]

            dtypes = [
                "f8", "f8", "f8", "f8", "f8", "f8",
                "i", "i", "U32", "U32", "U32",
                "c16", "f8"
            ]

            dataset = [
                time, freq, tint, mjd, uu, vv,
                ant_num1, ant_num2, ant_name1, ant_name2, pair,
                vis_1, sig_1
            ]

            data = gamvas.utils.sarray(dataset, dheads, dtypes)

        data = rfn.append_fields(data, "vis", data[f"vis_{self.select.lower()}"], usemask=False)
        data = rfn.append_fields(data, "sigma", data[f"sigma_{self.select.lower()}"], usemask=False)
        self.data = data


    def uvave(self,
        uvave=False, scanlen=300, doscatter=False, docombine=False,
        set_clq=True, set_pang=True, pinfo=True
    ):
        """
        Average the uv-visibility data
            Arguments:
                uvave (bool, float): The toggle option for averaging the UV data
                                     A float will be used as the averaging time
                scanlen (float): The scan length
                doscatter (bool): The toggle option if to compute sigma from standard deviation
                docombine (bool): The toggle option if to combine the sigma (scatter + weight)
                set_clq (bool): The toggle option for setting the closure phase and amplitude data
                set_pang (bool): The toggle option for setting the parallactic angle
                pinfo (bool): The toggle option for printing the information
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
            data = self.data

            time_ = data["time"]
            tint_ = data["tint"]
            mjd_ = data["mjd"]
            ant_name1_ = data["ant_name1"]
            ant_name2_ = data["ant_name2"]
            uu_ = data["u"]
            vv_ = data["v"]
            time_sec = data["time"] * 3600


            utime_sec = np.unique(time_sec)
            select = self.select.lower()

            db = dbs(eps=avgtime, min_samples=2).fit(time_sec.reshape(-1, 1))
            scannum = db.labels_
            scannum = np.where(scannum < 0, scannum + np.max(scannum)+np.abs(scannum), scannum)
            uscan = np.unique(scannum)
            uant_name1 = np.unique(ant_name1_)
            uant_name2 = np.unique(ant_name2_)

            filter_vis = lambda x : (("vis" in x) & (not "vism" in x)) & (x != "vis")
            filter_pols = lambda x : x.split("_")[1]
            pols_ = list(map(filter_pols, (list(filter(filter_vis, data.dtype.names)))))

            circs = ["rr", "ll", "rl", "lr"]
            pols = []
            for circ in circs:
                if circ in pols_:
                    pols.append(circ)

            timer = np.arange(0,scanlen+avgtime,avgtime)
            time = []
            mjd = []
            tint = []
            ant_name1 = []
            ant_name2 = []
            uu = []
            vv = []
            vis_1 = []
            vis_2 = []
            vis_3 = []
            vis_4 = []
            sig_1 = []
            sig_2 = []
            sig_3 = []
            sig_4 = []

            for nstoke, pol in enumerate(pols):
                if nstoke == 0:
                    outvis, outsig = vis_1, sig_1
                elif nstoke == 1:
                    outvis, outsig = vis_2, sig_2
                elif nstoke == 2:
                    outvis, outsig = vis_3, sig_3
                elif nstoke == 3:
                    outvis, outsig = vis_4, sig_4

                for nscan, scan in enumerate(uscan):
                    mask_scan = scannum == scan
                    time_scan = time_sec-np.min(time_sec[mask_scan])
                    for ntime in range(len(timer)-1):
                        mask_time = (timer[ntime+0] <= time_scan) & (time_scan < timer[ntime+1] + 1)
                        data_mask1 = data[mask_scan & mask_time]
                        for uant_name1_ in uant_name1:
                            for uant_name2_ in uant_name2:
                                mask_ant_name1 = ant_name1_ == uant_name1_
                                mask_ant_name2 = ant_name2_ == uant_name2_
                                mask_tot = mask_scan & mask_time & mask_ant_name1 & mask_ant_name2
                                mask_vis = len(data[f"vis_{pol}"][mask_tot]) == 0
                                if mask_vis:
                                    continue
                                if uant_name1_ == uant_name2_:
                                    continue
                                data_mask2 = data[mask_tot]
                                getvis = data_mask2[f"vis_{pol}"]
                                getsig = data_mask2[f"sigma_{pol}"]
                                nvis = len(getvis)

                                weight = 1 / getsig**2
                                avg_vis = np.average(getvis, weights=weight)
                                # avg_vis = np.mean(getvis)
                                if docombine:
                                    outsig_1 = np.std(np.abs(getvis)) / np.sqrt(nvis)
                                    outsig_2 = np.sqrt((np.sum(getsig**2) / nvis**2))
                                    outsig_  = outsig_1 + outsig_2
                                    outsig.append(outsig_)
                                    outvis.append(avg_vis)
                                else:
                                    if doscatter and nvis <= 2:
                                        continue
                                    elif doscatter and nvis > 1:
                                        outsig_ = np.std(np.abs(getvis)) / np.sqrt(nvis)
                                        outsig.append(outsig_)
                                        outvis.append(avg_vis)
                                    else:
                                        if np.sum(getsig) == 0:
                                            outsig.append(0.0)
                                            outvis.append(avg_vis)
                                        else:
                                            weight = 1 / getsig**2
                                            outsig.append(np.sqrt((np.sum(getsig**2) / nvis**2)))
                                            outvis.append(avg_vis)

                                if nstoke == 0:
                                    time.append(np.mean(data_mask1["time"]))
                                    tint.append(np.sum(data_mask2["tint"]))
                                    mjd.append(np.mean(mjd_[mask_tot]))
                                    ant_name1.append(uant_name1_)
                                    ant_name2.append(uant_name2_)
                                    uu.append(np.mean(uu_[mask_tot]))
                                    vv.append(np.mean(vv_[mask_tot]))

            if not self.nstokes in [1,2,4]:
                raise Exception(f"Unexpected number of polarization types (input:{self.nstokes})")

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
            self.uu = np.array(uu)
            self.vv = np.array(vv)
            self.set_uvvis()
        if set_pang:
            self.cal_pangle()
        if set_clq:
            self.set_closure()


    def set_closure(self):
        """
        Set the closure quantities
        """
        if not "vis" in self.data.dtype.names:
            self.set_uvvis()
        data = self.data
        ant_names = np.unique(np.append(data["ant_name1"], data["ant_name2"]))
        times = np.unique(data["time"])

        selects = []
        for ns, select in enumerate(data.dtype.names):
            if  select in ["vis_i", "vis_q", "vis_u", "vis_v", "vis_p"]:
                selects.append(select.split("_")[1])
        selects.append("")

        for ns, select in enumerate(selects):
            if select in ["i", "q", "u", "v", "p"]:
                field_vis = f"vis_{select}"
                field_sig = f"sigma_{select}"
                field_cav = f"clamp_{select}"
                field_cas = f"sigma_clamp_{select}"
                field_cpv = f"clphs_{select}"
                field_cps = f"sigma_clphs_{select}"
            if select == "":
                field_vis = f"vis"
                field_sig = f"sigma"
                field_cav = f"clamp"
                field_cas = f"sigma_clamp"
                field_cpv = f"clphs"
                field_cps = f"sigma_clphs"

            uvvis = dict(zip(tuple(zip(data["u"].tolist(), data["v"].tolist())), data[field_vis]))
            uvsig = dict(zip(tuple(zip(data["u"].tolist(), data["v"].tolist())), data[field_sig]))

            utimes = np.unique(data["time"])

            field_amp = [
                "time", "freq", "quadra",
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

            tmpl_clamp = np.array([])
            tmpl_clphs = np.array([])
            for ut, time in enumerate(utimes):
                data_ = data[data["time"] == time]
                ant_nums_ = np.unique(np.append(data_["ant_num1"], data_["ant_num2"]))
                ant_names_ = np.unique(np.append(data_["ant_name1"], data_["ant_name2"]))
                Nant = len(ant_names_)
                if Nant >= 4:
                    pairs_obs = np.array(list(map(",".join, tuple(zip(data_["ant_num1"].astype(str), data_["ant_num2"].astype(str))))))
                    pairs_obs = np.sort(pairs_obs)
                    matrix_clamp, pairs_full = set_min_matrix_clamp(Nant, ant_nums_.astype(str))

                    row = matrix_clamp.shape[0]
                    for i in range(row):
                        mask_add = matrix_clamp[i] == +1
                        mask_sub = matrix_clamp[i] == -1
                        pair_add = pairs_full[mask_add]
                        pair_sub = pairs_full[mask_sub]
                        mask_tot =\
                            (pair_add[0] in pairs_obs) & (pair_add[1] in pairs_obs) &\
                            (pair_sub[0] in pairs_obs) & (pair_sub[1] in pairs_obs)
                        pair_add = [val.split(',') for val in pair_add]
                        pair_sub = [val.split(',') for val in pair_sub]
                        if mask_tot:
                            pair_ants = np.array([
                                int(pair_add[0][0]), int(pair_add[0][1]),
                                int(pair_add[1][0]), int(pair_add[1][1])
                            ])
                            out_times = time
                            out_frequency = self.freq
                            out_quadrangle = "-".join(list(map(self.ant_dict_num2name.get, pair_ants)))

                            loc_uv1 = (data_["ant_num1"] == int(pair_add[0][0])) & (data_["ant_num2"] == int(pair_add[0][1]))
                            loc_uv2 = (data_["ant_num1"] == int(pair_add[1][0])) & (data_["ant_num2"] == int(pair_add[1][1]))
                            loc_uv3 = (data_["ant_num1"] == int(pair_sub[0][0])) & (data_["ant_num2"] == int(pair_sub[0][1]))
                            loc_uv4 = (data_["ant_num1"] == int(pair_sub[1][0])) & (data_["ant_num2"] == int(pair_sub[1][1]))

                            out_uv1 = (data_["u"][loc_uv1][0], data_["v"][loc_uv1][0])
                            out_uv2 = (data_["u"][loc_uv2][0], data_["v"][loc_uv2][0])
                            out_uv3 = (data_["u"][loc_uv3][0], data_["v"][loc_uv3][0])
                            out_uv4 = (data_["u"][loc_uv4][0], data_["v"][loc_uv4][0])
                            tmpl_clamp_ = gamvas.utils.sarray(
                                [
                                    out_times, out_frequency, out_quadrangle,
                                    out_uv1[0], out_uv1[1], out_uv2[0], out_uv2[1], out_uv3[0], out_uv3[1], out_uv4[0], out_uv4[1],
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                ],
                                field=field_amp,
                                dtype=dtype_amp
                            )

                            if tmpl_clamp.size == 0:
                                tmpl_clamp = tmpl_clamp_
                            else:
                                tmpl_clamp = rfn.stack_arrays((tmpl_clamp, tmpl_clamp_))
                if Nant >= 3:
                    pairs_obs = np.array(list(map(",".join, tuple(zip(data_["ant_num1"].astype(str), data_["ant_num2"].astype(str))))))
                    matrix_clphs, pairs_full = set_min_matrix_clphs(Nant, ant_nums_.astype(str))

                    row = matrix_clphs.shape[0]
                    for i in range(row):
                        mask_add = matrix_clphs[i] == +1
                        mask_sub = matrix_clphs[i] == -1
                        pair_add = pairs_full[mask_add]
                        pair_sub = pairs_full[mask_sub]
                        mask_tot =\
                            (pair_add[0] in pairs_obs) & (pair_add[1] in pairs_obs) & (pair_sub[0] in pairs_obs)
                        pair_add = [val.split(',') for val in pair_add]
                        pair_sub = [val.split(',') for val in pair_sub]
                        if mask_tot:
                            pair_ants = np.array([int(pair_add[0][0]), int(pair_add[0][1]), int(pair_sub[0][1])])
                            out_times = time
                            out_frequency = self.freq
                            out_triangle = "-".join(list(map(self.ant_dict_num2name.get, pair_ants)))

                            loc_uv1 = (data_["ant_num1"] == int(pair_add[0][0])) & (data_["ant_num2"] == int(pair_add[0][1]))
                            loc_uv2 = (data_["ant_num1"] == int(pair_add[1][0])) & (data_["ant_num2"] == int(pair_add[1][1]))
                            loc_uv3 = (data_["ant_num1"] == int(pair_sub[0][0])) & (data_["ant_num2"] == int(pair_sub[0][1]))

                            out_uv1 = (data_["u"][loc_uv1][0], data_["v"][loc_uv1][0])
                            out_uv2 = (data_["u"][loc_uv2][0], data_["v"][loc_uv2][0])
                            out_uv3 = (data_["u"][loc_uv3][0], data_["v"][loc_uv3][0])
                            tmpl_clphs_ = gamvas.utils.sarray(
                                [
                                    out_times, out_frequency, out_triangle,
                                    out_uv1[0], out_uv1[1], out_uv2[0], out_uv2[1], out_uv3[0], out_uv3[1],
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                                ],
                                field=field_phs,
                                dtype=dtype_phs
                            )
                            if tmpl_clphs.size == 0:
                                tmpl_clphs = tmpl_clphs_
                            else:
                                tmpl_clphs = rfn.stack_arrays((tmpl_clphs, tmpl_clphs_))

            flag_clamp = False
            flag_clphs = False

            tmpl_clamp = tmpl_clamp.reshape(-1)
            tmpl_clphs = tmpl_clphs.reshape(-1)

            if tmpl_clamp.size == 0:
                tmpl_clamp = [np.nan] * len(field_amp)
                tmpl_clamp = gamvas.utils.sarray(tmpl_clamp, field=field_amp, dtype=dtype_amp)
                flag_clamp = True
            if tmpl_clphs.size == 0:
                tmpl_clphs = [np.nan] * len(field_phs)
                tmpl_clphs = gamvas.utils.sarray(tmpl_clphs, field=field_phs, dtype=dtype_phs)
                flag_clphs = True
            self.tmpl_clamp = copy.deepcopy(tmpl_clamp)
            self.tmpl_clphs = copy.deepcopy(tmpl_clphs)
            if not flag_clamp:
                clamp_ = tmpl_clamp
                clamp_["vis12"] = list( map(uvvis.get, tuple(zip(clamp_["u12"], clamp_["v12"]))) )
                clamp_["vis34"] = list( map(uvvis.get, tuple(zip(clamp_["u34"], clamp_["v34"]))) )
                clamp_["vis13"] = list( map(uvvis.get, tuple(zip(clamp_["u13"], clamp_["v13"]))) )
                clamp_["vis24"] = list( map(uvvis.get, tuple(zip(clamp_["u24"], clamp_["v24"]))) )
                clamp_["sig12"] = list( map(uvsig.get, tuple(zip(clamp_["u12"], clamp_["v12"]))) )
                clamp_["sig34"] = list( map(uvsig.get, tuple(zip(clamp_["u34"], clamp_["v34"]))) )
                clamp_["sig13"] = list( map(uvsig.get, tuple(zip(clamp_["u13"], clamp_["v13"]))) )
                clamp_["sig24"] = list( map(uvsig.get, tuple(zip(clamp_["u24"], clamp_["v24"]))) )
                amp12 = np.abs(clamp_["vis12"])
                amp34 = np.abs(clamp_["vis34"])
                amp13 = np.abs(clamp_["vis13"])
                amp24 = np.abs(clamp_["vis24"])
                amp12 = np.where(amp12 < clamp_["sig12"], 0, np.sqrt(np.abs(clamp_["vis12"])**2 - clamp_["sig12"]**2))
                amp34 = np.where(amp34 < clamp_["sig34"], 0, np.sqrt(np.abs(clamp_["vis34"])**2 - clamp_["sig34"]**2))
                amp13 = np.where(amp13 < clamp_["sig13"], 0, np.sqrt(np.abs(clamp_["vis13"])**2 - clamp_["sig13"]**2))
                amp24 = np.where(amp24 < clamp_["sig24"], 0, np.sqrt(np.abs(clamp_["vis24"])**2 - clamp_["sig24"]**2))
                snr12 = amp12 / np.abs(clamp_["sig12"])
                snr34 = amp34 / np.abs(clamp_["sig34"])
                snr13 = amp13 / np.abs(clamp_["sig13"])
                snr24 = amp24 / np.abs(clamp_["sig24"])
                clamp_[field_cav] = (amp12 * amp34) / (amp13 * amp24)
                clamp_[field_cas] = np.sqrt(snr12**-2 + snr34**-2 + snr13**-2 + snr24**-2)

            if not flag_clphs:
                clphs_ = tmpl_clphs
                clphs_["vis12"] = list( map(uvvis.get, tuple(zip(clphs_["u12"], clphs_["v12"]))) )
                clphs_["vis23"] = list( map(uvvis.get, tuple(zip(clphs_["u23"], clphs_["v23"]))) )
                clphs_["vis31"] = list( map(uvvis.get, tuple(zip(clphs_["u31"], clphs_["v31"]))) )
                clphs_["sig12"] = list( map(uvsig.get, tuple(zip(clphs_["u12"], clphs_["v12"]))) )
                clphs_["sig23"] = list( map(uvsig.get, tuple(zip(clphs_["u23"], clphs_["v23"]))) )
                clphs_["sig31"] = list( map(uvsig.get, tuple(zip(clphs_["u31"], clphs_["v31"]))) )
                phs12 = np.angle(clphs_["vis12"])
                phs23 = np.angle(clphs_["vis23"])
                phs31 = np.angle(clphs_["vis31"].conj())
                snr12 = np.abs(clphs_["vis12"]) / np.abs(clphs_["sig12"])
                snr23 = np.abs(clphs_["vis23"]) / np.abs(clphs_["sig23"])
                snr31 = np.abs(clphs_["vis31"]) / np.abs(clphs_["sig31"])

                clphs_v = phs12 + phs23 + phs31
                # clphs_v = np.where(clphs_v > +np.pi, clphs_v - 2 * np.pi, clphs_v)
                # clphs_v = np.where(clphs_v < -np.pi, clphs_v + 2 * np.pi, clphs_v)
                clphs_[field_cpv] = clphs_v
                clphs_[field_cps] = np.sqrt(snr12**-2 + snr23**-2 + snr31**-2)

            if not flag_clamp:
                fields = ["time", "quadra", "freq", field_cav, field_cas]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                datas  = [clamp_[fields[nf]] for nf in range(len(fields))]
                clamp_ = gamvas.utils.sarray(datas, fields, dtypes)
            else:
                fields = ["time", "quadra", "freq", field_cav, field_cas]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                clamp_ = gamvas.utils.sarray([np.nan for i in range(len(fields))], fields, dtypes)

            if not flag_clphs:
                fields = ["time", "triangle", "freq", field_cpv, field_cps]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                datas  = [clphs_[fields[nf]] for nf in range(len(fields))]
                clphs_ = gamvas.utils.sarray(datas, fields, dtypes)
            else:
                fields = ["time", "triangle", "freq", field_cpv, field_cps]
                dtypes = ["f8", "U32", "f8", "f8", "f8"]
                clphs_ = gamvas.utils.sarray([np.nan for i in range(len(fields))], fields, dtypes)

            if ns == 0:
                clamp = clamp_
                clphs = clphs_
            else:
                if not flag_clamp:
                    clamp = rfn.append_fields(clamp, field_cav, clamp_[field_cav], usemask=False)
                    clamp = rfn.append_fields(clamp, field_cas, clamp_[field_cas], usemask=False)
                if not flag_clphs:
                    clphs = rfn.append_fields(clphs, field_cpv, clphs_[field_cpv], usemask=False)
                    clphs = rfn.append_fields(clphs, field_cps, clphs_[field_cps], usemask=False)


        if not flag_clamp:
            self.clamp = clamp
            self.clamp_check = True
        else:
            fields = [
                "time", "quadra", "freq",
                "clamp", "sigma_clamp",
                "clamp_i", "sigma_clamp_i",
                "clamp_q", "sigma_clamp_q",
                "clamp_u", "sigma_clamp_u",
                "clamp_v", "sigma_clamp_v",
                "clamp_p", "sigma_clamp_p"
            ]
            dtypes = [
                "f8", "U32", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8"
            ]
            self.clamp = gamvas.utils.sarray(data=[np.nan for i in range(len(fields))], field=fields, dtype=dtypes)
            self.clamp_check = False

        if not flag_clphs:
            self.clphs = clphs
            self.clphs_check = True
        else:
            fields = [
                "time", "triangle", "freq",
                "clphs", "sigma_clphs",
                "clphs_i", "sigma_clphs_i",
                "clphs_q", "sigma_clphs_q",
                "clphs_u", "sigma_clphs_u",
                "clphs_v", "sigma_clphs_v",
                "clphs_p", "sigma_clphs_p"
            ]

            dtypes = [
                "f8", "U32", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8",
                "f8", "f8"
            ]

            self.clphs = gamvas.utils.sarray(data=[np.nan for i in range(len(fields))], field=fields, dtype=dtypes)
            self.clphs_check = False
        self.ploter.clq_obs = (self.clamp, self.clphs)


    def reorder_baseline(self):
        """
        Reorder uv-visibility data by baseline number
        (imported from eht-imaging)
        (https://achael.github.io/eht-imaging/; Chael+2018, ApJ, 857, 23C)
        """
        data = self.data.copy()
        dlist = []
        for key, group in it.groupby(data, lambda x: x["time"]):
            dlist.append(np.array([obs for obs in group]))
        obs = []
        for tlist in dlist:
            blpairs = []
            for dat in tlist:
                if not (set((dat["ant_name1"], dat["ant_name2"]))) in blpairs:

                    # Reverse the baseline in the right order for uvfits:
                    if(self.tkey[dat["ant_name2"]] < self.tkey[dat["ant_name1"]]):

                        (dat["ant_name1"], dat["ant2"]) = (dat["ant_name2"], dat["ant_name1"])
                        dat["u"] = -dat["u"]
                        dat["v"] = -dat["v"]

                        dat["vis"] = np.conj(dat["vis"])
                        dat["vis_q"] = np.conj(dat["vis_q"])
                        dat["vis_u"] = np.conj(dat["vis_u"])
                        dat["vis_v"] = np.conj(dat["vis_v"])
                        dat["vis_p"] = np.conj(dat["vis_p"])


                    # Append the data point
                    blpairs.append(set((dat["ant_name1"], dat["ant_name2"])))
                    obs.append(dat)

        obs = np.array(obs, dtype=self.data.dtype)
        obs = obs[np.argsort(obs, order=["time", "ant_name1"])]
        self.data = obs


    def cal_polarization(self,
        snr_i=5, snr_p=3, evpalength=1, evpawidth=1
    ):
        """
        Compute the polarization data
            Arguments:
                snr_i (float): The signal-to-noise ratio for the total intensity
                snr_p (float): The signal-to-noise ratio for the polarization
                evpalength (float): The length of the bar (electric vector position angle)
                evpawidth (float): The width of the bar (electric vector position angle)
        """
        self.fits_image_vp = np.sqrt(self.fits_image_vq**2 + self.fits_image_vu**2)
        self.fits_image_rms_p = (self.fits_image_rms_q + self.fits_image_rms_q) / 2
        self.fits_image_vpeak_p = np.max(self.fits_image_vp)
        self.fits_clean_vflux_p = np.sqrt(self.fits_clean_vflux_q**2 + self.fits_clean_vflux_u**2)
        sig_peak, sig_tot, sig_size =\
            cal_clean_error(
                s_peak=np.max(self.fits_image_vp),
                s_tot=self.fits_clean_vflux_p,
                sig_rms=self.fits_image_rms_p
            )
        self.fits_image_dpeak_p = 0.1 * self.fits_image_vpeak_p
        self.fits_clean_dflux_p = 0.1 * self.fits_clean_vflux_p
        self.fits_clean_vfp = self.fits_clean_vflux_p / self.fits_clean_vflux_i
        self.fits_clean_vevpa = 0.5 * np.arctan2(self.fits_clean_vflux_u, self.fits_clean_vflux_q)

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

        self.fits_image_vp = np.sqrt(self.fits_image_vq**2 + self.fits_image_vu**2)
        self.fits_image_vfp = self.fits_image_vp / self.fits_image_vi
        self.fits_image_vevpa = 0.5*np.arctan2(self.fits_image_vu, self.fits_image_vq)
        self.fits_image_vevpa = np.where(self.fits_image_vevpa > np.pi, self.fits_image_vevpa-np.pi, self.fits_image_vevpa)
        self.fits_image_vevpa = np.where(self.fits_image_vevpa < 0, self.fits_image_vevpa+np.pi, self.fits_image_vevpa)
        self.fits_image_dp = np.ones((self.fits_npix, self.fits_npix)) * self.fits_image_rms_p
        self.fits_image_dfp = np.ones((self.fits_npix, self.fits_npix)) * np.abs((self.fits_image_rms_p / self.fits_image_vi))
        self.fits_image_devpa = np.ones((self.fits_npix, self.fits_npix)) * (self.fits_image_rms_p / (2*self.fits_image_vp))
        self.cal_evpa(snr_i=snr_i, snr_p=snr_p, evpalength=evpalength, evpawidth=evpawidth)


    def cal_evpa(self,
        snr_i=5, snr_p=3, evpalength=1, evpawidth=1
    ):
        """
        Compute electric vector position angle (EVPA) data
            Arguments:
                snr_i (float): The signal-to-noise ratio for the total intensity
                snr_p (float): The signal-to-noise ratio for the polarization
                evpalength (float): The length of the EVPA bar
                evpawidth (float): The width of the EVPA bar
        """
        evpalength = evpalength
        evpawidth = evpawidth * self.fits_psize * u.deg.to(u.mas)
        scale = 1 / evpalength
        self.fits_image_evpa_set = dict(color="black", pivot="middle", units="xy", scale=scale, width=evpawidth,
                             headlength=0, headwidth=0, headaxislength=0)
        fits_evpa_x, fits_evpa_y = np.sin(self.fits_image_vevpa), -np.cos(self.fits_image_vevpa)

        mask = (self.fits_image_vi >= snr_i * self.fits_image_rms_i) & (self.fits_image_vp >= snr_p * self.fits_image_rms_p)
        self.fits_image_evpa_x = np.where(mask, fits_evpa_x, np.nan)
        self.fits_image_evpa_y = np.where(mask, fits_evpa_y, np.nan)
        self.fits_image_vevpa = np.where(mask, self.fits_image_vevpa, np.nan)
        self.fits_image_vp = np.where(mask, self.fits_image_vp, np.nan)
        self.fits_image_vfp = np.where(mask, self.fits_image_vfp, np.nan)


    def cal_Tb(self):
        """
        Compute the brightness temperature
        """
        if not self.fits_clean_vflux_i is None:
            freq = self.fits_freq
            psize = np.round(self.fits_psize * u.deg.to(u.mas), 5)
            beam = self.fits_bmaj * self.fits_bmin * np.pi * (u.deg.to(u.mas))**2
            factor = (1 * u.Jy / 2 / C.k_B * C.c**2).to(u.K * u.Hz**2).value
            vtb = factor * self.fits_image_vi / freq**2
            dtb = factor * self.fits_image_vi / freq**2
            self.fits_image_vtb = vtb
            self.fits_image_dtb = dtb
        else:
            raise Exception("FITS-image is not provided. Please check if you load UV-fits file (see 'load_fits()' method).")


    def cal_fits_model_error(self,
        mrng=10, r_limit=10, pinfo=True, fitset="sf", spectrum="spl"
    ):
        """
        Compute the error of the fits model
         NOTE: This method is based on signal-to-noise ratio (SNR) of the model (Fomalont 1999; Image Analysis)
            Arguments:
                mrng (float): The map range
                r_limit (float): The limit of the radius
                pinfo (bool): Print the information
                fitset (str): The fit setting type // sf:single-frequency, mf:multi-frequency
                spectrum (str): The spectrum type // spl:simple power-law, cpl:curved power-law, ssa:synchrotron self-absorption
        """
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
        l = model["DELTAX"]
        m = model["DELTAY"]
        r = np.sqrt(l**2 + m**2)
        pa = pa = np.arctan2(l, m) * u.rad.to(u.deg)
        try:
            self.load_uvf(select="i", set_clq=False)
            self.uvave(uvave="scan", set_clq=False, set_pang=False, pinfo=pinfo)
            self.fit_beam(uvw="n")
        except:
            self.load_uvf(select="ll", set_clq=False)
            self.uvave(uvave="scan", set_clq=False, set_pang=False, pinfo=pinfo)
            self.fit_beam(uvw="n")

        thetas = np.array([])
        fields = np.array([])
        dtypes = np.array([])
        for i in range(nmod):
            thetas = np.append(thetas, [flux[i], fwhm[i], l[i], m[i]])
            fields = np.append(fields, [f"{i + 1}_S", f"{i + 1}_a", f"{i + 1}_l", f"{i + 1}_m"])
            dtypes = np.append(dtypes, ["f8", "f8", "f8", "f8"])
        prms = gamvas.utils.sarray(thetas, fields, dtypes)
        self.append_visibility_model(self.freq, self.freq, prms, fitset=fitset, spectrum=spectrum, set_spectrum=False)
        self.ploter.draw_dirtymap(uvf=self, mrng=mrng, npix=512, uvw="n", plot_resi=True, plotimg=False)

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

            residcp[int(centerx2 - range2):int(centerx2 + range2), int(centery2 - range2) : int(centery2 + range2)] = np.nan
            imagecp[int(centerx1 - range1):int(centerx1 + range1), int(centery1 - range1) : int(centery1 + range1)] = np.nan
            image_ = image[int(centerx1 - range1):int(centerx1 + range1), int(centery1 - range1):int(centery1 + range1)]
            resid_ = resid[int(centerx2 - range2):int(centerx2 + range2), int(centery2 - range2):int(centery2 + range2)]
            peak_ = np.abs(np.nanmax(image_))
            sigma_rms_ = np.nanstd(resid_)
            peak = np.append(peak, peak_)
            sigma_rms = np.append(sigma_rms, sigma_rms_)

        snr = peak / sigma_rms
        dmin = 2 / np.pi * np.sqrt(np.pi * self.beam_prms[0] * self.beam_prms[1] * np.log(2) * np.log(snr / (snr - 1)))
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
        sigma_phi = np.arctan(sigma_dist / dist) * u.rad.to(u.deg)

        uflux = unp.uarray([flux, sigma_flux])
        ufwhm = unp.uarray([fwhm, sigma_fwhm])

        utb = 1.22e+12 * uflux / (ufwhm * self.freq)**2
        tb, dtb = unp.nominal_values(utb), unp.std_devs(utb)
        mjds = np.full(nmod, mjd)

        fields = ["mjd", "flux", "dflux", "size", "dsize", "radius", "dradius", "phi", "dphi", "tb", "dtb", "dmin"]
        dtypes = ["f8" for i in range(len(fields))]
        model_cal = gamvas.utils.sarray(
            [mjds, flux, sigma_flux, fwhm, sigma_fwhm, dist, sigma_dist, phi, sigma_phi, tb, dtb, dmin_bool],
            dtype=dtypes,
            field=fields
        )
        self.fits_model_cal = model_cal


    def save_newfits(self,
        save_name=False, save_path=False
    ):
        """
        Save new fits file
            Arguments:
                save_name (str): The name of the new fits file
                save_path (str): The path of the new fits file
        """
        self.uvf_file.writeto(save_path + save_name)


    def uvshift(self,
        deltal=0, deltam=0
    ):
        """
        Shift the uv-visibility data
            Arguments:
                deltal (float): The shift in the RA-direction
                deltam (float): The shift in the DEC-direction
        """
        deltal = deltal * m2r
        deltam = deltam * m2r
        select = self.select.lower()
        data = self.data
        vis = data["vis"]
        U, V = data["u"], data["v"]
        self.data[f"vis_{select}"] = vis * np.exp(+2j * np.pi * U * deltal) * np.exp(+2j * np.pi * V * deltam)


    def get_zblf(self):
        """
        Get the zero-baseline flux level
        """
        data = self.data
        uvd = np.sqrt(data["u"]**2+data["v"]**2)
        mask = uvd == np.min(uvd)
        zbl = data[mask]

        zbl_ant1 = zbl["ant_name1"][0]
        zbl_ant2 = zbl["ant_name2"][0]
        mask_zbl = (data["ant_name1"] == zbl_ant1) & (data["ant_name2"] == zbl_ant2)
        zbl = data[mask_zbl]
        zblf = np.median(np.abs(zbl["vis"]))
        self.zblf = zblf
        self.zbl = (zbl_ant1, zbl_ant2)
        return self.zblf, self.zbl


def cal_clean_error(s_peak=0, s_tot=0, sig_rms=0, size=0):
    """
    Compute the error of the clean image
        Arguments:
            s_peak (float): The peak intensity
            s_tot (float): The total flux density
            sig_rms (float): The rms noise level
            size (float): The model size
        Returns:
            sig_peak (float): The error of the peak intensity
            sig_tot (float): The error of the total flux density
            sig_size (float): The error of the model size
    """
    sig_peak = sig_rms * np.sqrt(1 + s_peak / sig_rms)
    sig_tot = sig_peak * np.sqrt(1 + (s_tot / s_peak)**2)
    sig_size = size * (sig_peak / s_peak)
    return sig_peak, sig_tot, sig_size


def set_matrix_visphs(N):
    """
    Set the matrix for the visibility phase
        Arguments:
            N (int): The number of antennas
        Returns:
            out (np.array): The matrix for the visibility phase
    """
    out = np.array([[1, -1]])
    if N == 2:
        return out
    else:
        for i in range(3 ,N + 1):
            m1 = np.ones ((i - 1, 1))
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
            N (int): The number of antennas
            ant_nums (np.array): The array of antenna numbers
        Returns:
            out (np.array): The matrix for the closure phase
            pairs (np.array): The array of antenna pairs
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
            N (int): The number of antennas
            ant_nums (np.array): The array of antenna numbers
        Returns:
            out (np.array): The matrix for the closure amplitude
            pairs (np.array): The array of antenna pairs
    """
    pairs = np.array(list(map(",".join, list(it.combinations(ant_nums, 2)))))
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
            m0 = np.zeros((int((i - 1) * (i - 4) / 2), i - 1))
            xn = np.concatenate((np.eye(i - 2), -m1), axis=1)
            yn = np.zeros((i - 2, int(comb(i - 1, 2))))
            yn_comb = np.array(list(map(",".join, list(it.combinations(ant_nums_, 2)))))[i - 1:]
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
                yn[j,loc1] = -1
                yn[j,loc2] = +1
            upper = np.concatenate((xn, yn), axis=1)
            if i == 4:
                out = upper
            else:
                m0 = np.zeros((int((i - 1) * (i - 4) / 2), i - 1))
                cm0 = out
                lower = np.concatenate((m0, cm0), axis=1)
                out = np.concatenate((upper, lower), axis=0)
        return out, pairs
