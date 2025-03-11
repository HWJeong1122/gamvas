
import os
import sys
import numpy as np
import gamvas as gv
import copy

# set the number of CPU core (for multi-processing)
ncpu = 1                # use only one CPU core
ncpu = os.cpu_count()   # use maximum number of CPU core

# set uvf path
path = f"{os.getcwd()}/"
path_uvf = path + "uvfs/"

# multi-processing setting for Mac OS (optional)
import multiprocessing as mpr
mpr.set_start_method("fork")

bands   = ["K", "Q", "W", "D"]  # bands names
spectrum= "ssa"                 # spectrum type
                                #   spl: simple power-law
                                #   cpl: curved power-law
                                #   ssa: synchrotron self-absorption (Turler+1999)

# image parameters
npix  = 256             # number of image pixels (one axis)

# model-fit parameters
runfit_set = "mf"       # model-fitting setting // "sf" for single-frequency, "mf" for multi-frequency
runfit_sf  = True       # run single-frequency model-fitting
runfit_mf  = True       # run multi-frequency model-fitting
doampcal   = True       # amplitude self-calibration
dophscal   = True       # phase self-calibration
maxn       = 10         # maximum number of model
scanlen    = 300        # scan-length [sec]
select     = "ll"       # polarization type
snrflag    = 3          # snr-flagging value
uvave      = 60         # uv-averaging  [sec] // "scan" for scan-average, "none" for no averaging

# set data terms and weights
ftype = ["amp", "clamp", "clphs"]
fwght = [0.01, 1, 1]

# set source-corresponding parameters
source = "1928+738"     # source name
mrng   = 22.0           # map range     [mas]
bnd_l  = [-5 , +5]      # RA boundary   [mas]
bnd_m  = [-20, +0]      # DEC boundary  [mas]



"""
Load uvf files
"""
file_k = f"edt_f24sl02b_1928+738_{bands[0]}.uvf"
file_q = f"edt_f24sl02b_1928+738_{bands[1]}.uvf"
file_w = f"edt_f24sl02b_1928+738_{bands[2]}.uvf"
file_d = f"edt_f24sl02b_1928+738_{bands[3]}.uvf"

# load uv-fits files
uvw = "u"   # uv-weighting // u:uniform, n:natural
uvf1 = gv.load.open_fits(path=path_uvf, file=file_k, mrng=mrng * gv.mas)
uvf2 = gv.load.open_fits(path=path_uvf, file=file_q, mrng=mrng * gv.mas)
uvf3 = gv.load.open_fits(path=path_uvf, file=file_w, mrng=mrng * gv.mas)
uvf4 = gv.load.open_fits(path=path_uvf, file=file_d, mrng=mrng * gv.mas)
uvf1.load_uvf(select=select, uvw=uvw)
uvf2.load_uvf(select=select, uvw=uvw)
uvf3.load_uvf(select=select, uvw=uvw)
uvf4.load_uvf(select=select, uvw=uvw)

# uv-average
# available 'uvave' options:
#    "none": no averaging
#    float : time interval (time-average)
#    "scan": scan-average
uvf1.uvave(uvave=uvave, scanlen=scanlen)
uvf2.uvave(uvave=uvave, scanlen=scanlen)
uvf3.uvave(uvave=uvave, scanlen=scanlen)
uvf4.uvave(uvave=uvave, scanlen=scanlen)

# list of uv-fits files
uvfs = [uvf1, uvf2, uvf3, uvf4]

# unique frequency list
ufreq = np.array([uvfs[i].freq for i in range(len(uvfs))])

# observing date
date  = uvfs[0].date

# turnover frequency boundary condition (from 22 to 129 GHz)
bnd_f = [uvfs[0].freq, uvfs[-1].freq]

# set out path
path_fig_ = path + f"/{source}/"
gv.utils.mkdir(path_fig_)
path_fig_ = path + f"/{source}/{date}/"
gv.utils.mkdir(path_fig_)
path_fig_ = path + f"/{source}/{date}/Pol_{select.upper()}/"
gv.utils.mkdir(path_fig_)

# set multi-frequency uv-fits
uvall = gv.utils.set_uvf(uvfs, type="mf")

# check visibility data
uvall.ploter.draw_radplot(uvall, plotimg=True)
uvall.ploter.draw_uvcover(uvall, plotimg=True)
uvall.ploter.draw_dirtymap(uvall, plotimg=True)
uvall.ploter.draw_closure(type="clamp", plotimg=True)
uvall.ploter.draw_closure(type="clphs", plotimg=True)

# number of data
nvis  = uvall.data.shape[0]
ncamp = uvall.tmpl_clamp.shape[0]
ncphs = uvall.tmpl_clphs.shape[0]

# print the model-fit parameters
print(f"\n# Multi-frequency-synthesized model-fit parameters")
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
print(f"# Selected polarization: {select.upper()}", "\n")

# run model-fitting
# available options for sampler:
#   "rwalk" : random walk (fastest but less accurate)
#   "rslice": multivariate random slice (intermediate between "rwalk" and "slice")
#   "slice" : multivariate principle axes slice (slowest but more accurate)
mfu = gv.modeling.modeling(
    uvfs=uvfs, select=select, factor_zblf=1.5, sampler="slice", bound="multi",
    runfit_sf=runfit_sf, runfit_mf=runfit_mf, runfit_set=runfit_set,
    ftype=ftype, fwght=fwght, ufreq=ufreq, bands=bands, spectrum=spectrum,
    maxn=maxn, doampcal=doampcal, dophscal=dophscal,
    npix=npix, mrng=mrng, bnd_l=bnd_l, bnd_m=bnd_m, bnd_f=bnd_f,
    uvw=uvw, path_fig=path_fig_, source=source, date=date, ncpu=ncpu
)
mfu.run()
