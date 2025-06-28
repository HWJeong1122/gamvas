
import os
import sys
import copy
import numpy as np
import gamvas as gv

# TODO: set the number of CPU core (for multi-processing)
ncpu = 1                # use only one CPU core
ncpu = os.cpu_count()   # use maximum number of CPU core

# TODO: set uvf path
path = f"{os.getcwd()}/"
path = f"/Users/hwjeong/PyCoding/gamvas/examples/"
path_uvf = path + "uvfs/"

# NOTE: multi-processing setting for Mac OS
import multiprocessing as mpr
mpr.set_start_method("fork")

bands   = ["K", "Q", "W", "D"]  # bands names
spectrum= "ssa"                 # spectrum type
                                #   spl: simple power-law
                                #   cpl: curved power-law
                                #   ssa: synchrotron self-absorption (Turler+1999)

# image parameters
npix  = 256             # (int) : number of image pixels (one axis)

# model-fit parameters
runfit_set = "mf"       # (str) : model-fitting setting // "sf" for single-frequency, "mf" for multi-frequency
runfit_sf  = True       # (bool) : run single-frequency model-fitting
runfit_mf  = True       # (bool) : run multi-frequency model-fitting
doampcal   = True       # (bool) : amplitude self-calibration
dophscal   = True       # (bool) : phase self-calibration
maxn       = 10         # (float) [mas] : maximum number of model
scanlen    = 300        # (float) [sec] : scan-length
select     = "ll"       # (str) : polarization type
snrflag    = 3          # (float) : snr-flagging value // recommended to be set >= 3
uvave      = 60         # (float) [sec] : uv-averaging // "scan" for scan-average, "none" for no averaging

# set data terms and weights
#    - "vis": complex visibility
#    - "amp": visibility amplitude
#    - "phs": visibility phase
#    - "clamp": closure amplitude
#    - "clphs": closure phase
ftype = ["amp", "clamp", "clphs"]
fwght = [0.1, 1, 1]

# set source-corresponding parameters
source = "1928+738"     # (str) : source name
mrng   = 22.0           # (float) [mas] : map range
bnd_l  = [-5 , +5]      # (list, float) [mas] : boundary condition on Right Ascension
bnd_m  = [-20, +0]      # (list, float) [mas] : boundary condition on Declination
bnd_pa = None           # (list, float / or None) [deg] : boundary condition on jet position angle
width  = 3              # (float) [mas] : maximum width of a Gaussian model



"""
Load uvf files
"""
# set file name
# NOTE: recommend to use calibrated uvf files
file_k = f"edt_f24sl02b_1928+738_{bands[0]}.uvf"
file_q = f"edt_f24sl02b_1928+738_{bands[1]}.uvf"
file_w = f"edt_f24sl02b_1928+738_{bands[2]}.uvf"
file_d = f"edt_f24sl02b_1928+738_{bands[3]}.uvf"

# attach files to gamvas module
uvw = "u"   # uv-weighting // u:uniform, n:natural
uvf1 = gv.load.open_fits(path=path_uvf, file=file_k, mrng=mrng*gv.mas)
uvf2 = gv.load.open_fits(path=path_uvf, file=file_q, mrng=mrng*gv.mas)
uvf3 = gv.load.open_fits(path=path_uvf, file=file_w, mrng=mrng*gv.mas)
uvf4 = gv.load.open_fits(path=path_uvf, file=file_d, mrng=mrng*gv.mas)

# load the attached uvf files
uvf1.load_uvf(select=select, uvw=uvw, scanlen=scanlen, uvave=uvave)
uvf2.load_uvf(select=select, uvw=uvw, scanlen=scanlen, uvave=uvave)
uvf3.load_uvf(select=select, uvw=uvw, scanlen=scanlen, uvave=uvave)
uvf4.load_uvf(select=select, uvw=uvw, scanlen=scanlen, uvave=uvave)

# flag uv-visibility data by SNR
uvf1.flag_uvvis(type="snr", value=snrflag)
uvf2.flag_uvvis(type="snr", value=snrflag)
uvf3.flag_uvvis(type="snr", value=snrflag)
uvf4.flag_uvvis(type="snr", value=snrflag)

# NOTE: flag uv-visibility data by the number of antenna
fant = 4 # flag visibility if the number of antenna is less than this value
uvf1.flag_uvvis(type="nant", value=fant)
uvf2.flag_uvvis(type="nant", value=fant)
uvf3.flag_uvvis(type="nant", value=fant)
uvf4.flag_uvvis(type="nant", value=3)       # assume 129 GHz data contains only three stations

# apply systematics estimated from median absolute deviation (MAD)
# NOTE: four stations are necessary to apply systematics to "clamp"
uvf1.apply_systematics(binning=uvave, types=["vis", "clamp", "clphs"])
uvf2.apply_systematics(binning=uvave, types=["vis", "clamp", "clphs"])
uvf3.apply_systematics(binning=uvave, types=["vis", "clamp", "clphs"])
uvf4.apply_systematics(binning=uvave, types=["vis", "clphs"])

# (optional) add additional fractinoal errors to complex visibility to consider gain self-calibration unceratinty
uvf1.add_error_fraction(0.20)   # add 20% (22 GHz)
uvf2.add_error_fraction(0.20)   # add 20% (43 GHz)
uvf3.add_error_fraction(0.20)   # add 20% (86 GHz)
uvf4.add_error_fraction(0.30)   # add 30% (129 GHz)

# make multi-frequency uv-fits
uvfs = [uvf1, uvf2, uvf3, uvf4]
uvall = gv.utils.set_uvf(uvfs, type=runfit_set)

ufreq = np.array([uvfs[i].freq for i in range(len(uvfs))])  # unique frequency information
date  = uvfs[0].date    # observing date (assuming simultaneous observation)
bnd_f = [uvfs[0].freq, uvfs[-1].freq]   # frequency boundary condition

# generate base directories
path_fig_ = path + f"/{source}/"
gv.utils.mkdir(path_fig_)
path_fig_ = path + f"/{source}/{date}/"
gv.utils.mkdir(path_fig_)
path_fig_ = path + f"/{source}/{date}/Pol_{select.upper()}/"
gv.utils.mkdir(path_fig_)

# check visibility data
uvall.ploter.draw_tplot(uvall, plotimg=True)                    # time-coverage of all stations
uvall.ploter.draw_radplot(uvall, plotimg=True, plotsnr=False)   # radial plot of complex visibility
uvall.ploter.draw_uvcover(uvall, plotimg=True)                  # uv-coverage
uvall.ploter.draw_dirtymap(uvall, plotimg=True)                 # beam pattern (left) & dirty map (right)
uvall.ploter.draw_closure(type="clamp", plotimg=True)           # closure amplitude
uvall.ploter.draw_closure(type="clphs", plotimg=True)           # closure phase

# number of data
nvis  = uvall.data.shape[0]
ncamp = uvall.tmpl_clamp.shape[0]
ncphs = uvall.tmpl_clphs.shape[0]

# print the model-fit parameters
print(f"# Running parameters")
print(f"# Map range : {mrng:.1f} (mas)")
print(f"# B_min : {uvall.beam_prms[0]:.3f} (mas)")
print(f"# B_maj : {uvall.beam_prms[1]:.3f} (mas)")
print(f"# B_pa  : {uvall.beam_prms[2]:.3f} (deg)")
print(f"# Fit-spec : ", spectrum)
print(f"# Fit-type : ", ftype)
print(f"# Fit-wght : ", fwght)
print(f"# Number of complex visibility : ", nvis )
print(f"# Number of closure amplitude  : ", ncamp)
print(f"# Number of closure phase      : ", ncphs)
print(f"# uv-average time : {uvave}")
print(f"# Selected polarization: {select.upper()}")
print(f"# Number of active CPU cores: {ncpu}/{os.cpu_count()}")
print("\n")

# run model-fitting
# available options for sampler:
#   "rwalk" : random walk (fastest but less accurate)
#   "rslice": multivariate random slice (intermediate between "rwalk" and "slice")
#   "slice" : multivariate principle axes slice (slowest but more accurate)
mfu = gv.modeling.modeling(
    uvfs=uvfs, select=select, factor_zblf=1.5, sampler="slice", bound="multi",
    runfit_sf=runfit_sf, runfit_mf=runfit_mf, runfit_set=runfit_set, ftype=ftype, fwght=fwght,
    ufreq=ufreq, bands=bands, spectrum=spectrum, maxn=maxn,
    gacalerr=0.0, dognorm=False, doampcal=doampcal, dophscal=dophscal,
    npix=npix, mrng=mrng, bnd_l=bnd_l, bnd_m=bnd_m, bnd_f=bnd_f, bnd_pa=bnd_pa,
    uvw=uvw, path_fig=path_fig_, source=source, date=date, ncpu=ncpu, width=width
)
mfu.run()
