
import os
import sys
import copy
import numpy as np
import gamvas as gv

# TODO: set the number of CPU core (for multi-processing)
ncpu = 1                # use a single CPU core
ncpu = os.cpu_count()   # use maximum number of CPU core

# TODO: set uvf path
path = f"{os.getcwd()}/"
path_uvf = f"{path}/uvfs/"

# NOTE: multi-processing setting for Mac OS (optional)
import multiprocessing as mpr
mpr.set_start_method("fork")

bands = ["22", "43", "86", "129"]   # frequency names (used for saving name)
spectrum= "ssa" # spectrum type
                #     spl: simple power-law
                #     cpl: curved power-law
                #     ssa: synchrotron self-absorption (Turler+1999)

# image parameters
npix  = 512     # (int) : number of image pixels

# model-fit parameters
runfit_set = "mf"       # (str): model-fitting setting
                        #   - 'sf': single-frequency
                        #   - 'mf': multi-frequency
runfit_sf  = False      # (bool): run single-frequency model-fitting
runfit_mf  = True       # (bool): run multi-frequency model-fitting
doampcal   = True       # (bool): amplitude self-calibration
dophscal   = True       # (bool): phase self-calibration
maxn       = 10         # (int): maximum number of model
fixnmod    = False      # (bool): if True, fix the number of models to 'maxn'
scanlen    = 300        # (float) [sec]: scan length
select     = "ll"       # (str): polarization type
snrflag    = 3          # (float): snr-flagging value
uvave      = 10         # (float) [sec]: uv-averaging
                        #   - 'scan': scan-average
                        #   - 'none': skip averaging
gaptime    = 60         # (float) [sec]: gap time between scans

# NOTE: it is recommended to set snrflag >= 3 (at least),
#       otherwise closure quantities significantly devidate
#       from Gaussian distribution.
#           (see Section 9.3 in TMS;
#               Interferometry and Synthesis in Radio Astronomy)

# TODO: set data terms and weights
# NOTE: when neither 'vis' nor 'phs' is used,
#       the first model is fixed to (0, 0) position,
#       as closure phases give information of the relative position
#    - 'vis': complex visibility
#    - 'amp': visibility amplitude
#    - 'phs': visibility phase
#    - 'clamp': closure amplitude
#    - 'clphs': closure phase
ftype = ["amp", "clamp", "clphs"]
fwght = [0.1, 1, 1]

# TODO: set source name information
# global setting
source = "1928+738"     # (str): source name (for saving)
mrng = 22.0             # (float) [mas]: map range (for plotting)

"""
Load uvf files
"""
# set file name
# NOTE: recommend to use calibrated uvf files
file1 = f"c.22.1928+738.2024-12-13.uvf"
file2 = f"c.43.1928+738.2024-12-13.uvf"
file3 = f"c.86.1928+738.2024-12-13.uvf"
file4 = f"c.129.1928+738.2024-12-13.uvf"

# attach files to gamvas module
uvw = "u"   # uv-weighting // u:uniform, n:natural
uvf1 = gv.load.open_fits(path=path_uvf, file=file1, mrng=mrng*gv.mas)
uvf2 = gv.load.open_fits(path=path_uvf, file=file2, mrng=mrng*gv.mas)
uvf3 = gv.load.open_fits(path=path_uvf, file=file3, mrng=mrng*gv.mas)
uvf4 = gv.load.open_fits(path=path_uvf, file=file4, mrng=mrng*gv.mas)

# load the attached uvf files
# NOTE: closure relations are constructed for an optimal minimum set.
#           (Blackburn+2020; DOI: 10.3847/1538-4357/ab8469)
#       if you want to use full closures, then use 'minclq=False'
uvf1.load_uvf(
    select=select, uvw=uvw, scanlen=scanlen, uvave=uvave, gaptime=gaptime,
) # minclq=False)
uvf2.load_uvf(
    select=select, uvw=uvw, scanlen=scanlen, uvave=uvave, gaptime=gaptime,
) # minclq=False)
uvf3.load_uvf(
    select=select, uvw=uvw, scanlen=scanlen, uvave=uvave, gaptime=gaptime,
) # minclq=False)
uvf4.load_uvf(
    select=select, uvw=uvw, scanlen=scanlen, uvave=uvave, gaptime=gaptime,
) # minclq=False)

# flag uv-visibility data by SNR
uvf1.flag_uvvis(type="snr", value=snrflag)
uvf2.flag_uvvis(type="snr", value=snrflag)
uvf3.flag_uvvis(type="snr", value=snrflag)
uvf4.flag_uvvis(type="snr", value=snrflag)

# flag uv-visibility data by the number of antenna
uvf1.flag_uvvis(type="nant", value=4)
uvf2.flag_uvvis(type="nant", value=4)
uvf3.flag_uvvis(type="nant", value=4)
uvf4.flag_uvvis(type="nant", value=3)

# apply systematics estimated from median absolute deviation (MAD)
# a similar application can be found in EHTC+2019 III.
# NOTE: systematics on 'clamp' is omitted at 129 GHz
uvf1.apply_systematics(binning=uvave, types=["vis", "clamp", "clphs"])
uvf2.apply_systematics(binning=uvave, types=["vis", "clamp", "clphs"])
uvf3.apply_systematics(binning=uvave, types=["vis", "clamp", "clphs"])
uvf4.apply_systematics(binning=uvave, types=["vis", "clphs"])

# add additional fractinoal errors to complex visibility
# to consider the unceratinties on gain amplitude scales
uvf1.add_error_fraction(0.20)   # add 20% (22 GHz)
uvf2.add_error_fraction(0.20)   # add 20% (43 GHz)
uvf3.add_error_fraction(0.20)   # add 20% (86 GHz)
uvf4.add_error_fraction(0.30)   # add 30% (129 GHz)

# concat. multi-frequency uv-fits files
uvfs = [uvf1, uvf2, uvf3, uvf4]
uvall = gv.utils.set_uvf(uvfs, type="mf")

# TODO: set boundary conditions
# global setting
bnd_l = [-1 , +5]       # (list, float) [mas]: boundary condition on R.A
bnd_m = [-15, +1]       # (list, float) [mas]: boundary condition on Dec
bnd_f = [uvfs[0].freq, uvfs[-1].freq]   # (list, float) [GHz]: turnover freq.
width = 3               # (float) [mas]: maximum width of a Gaussian model
bnds = None

# or, you may want to adjust boundary conditions individually,
# rather than a global setting, then, refer to the below.
#   (NOTE: these are an example, assuming three components)
# sblf = uvall.get_sblf()[0]      # extract short-baseline flux
# bnd_S = [[0.5 * sblf, 1.5 * sblf], [0, sblf], [0, sblf]]    # flux density
# bnd_a = [[0, width], [0, width], [0, width]]                # width
# bnd_l = [[-1.0, +1.0], [-3.0, +3.0], [-5.0, +5.0]]          # R.A
# bnd_m = [[-1.0, +0.0], [-5.0, +0.0], [-15.0, +5.0]]         # Dec
# bnd_f = [[22, 129], [22, 129], [22, 129]]                   # turnover freq.
# bnd_i = [[-3.0, +0.0], [-3.0, +0.0], [-3.0, +5.0]]          # spectral index
# bnds = (bnd_S, bnd_a, bnd_l, bnd_m, bnd_f, bnd_i)
# width = 3

# unique frequency information
ufreq = np.array([uvfs[i].freq for i in range(len(uvfs))])

# observing date
#   (assuming simultaneous observation, for saving)
date = uvfs[0].date

# generate base directories
path_fig_ = path + f"/{source}/"
gv.utils.mkdir(path_fig_)
path_fig_ = path + f"/{source}/{date}/"
gv.utils.mkdir(path_fig_)
path_fig_ = path + f"/{source}/{date}/Pol_{select.upper()}/"
gv.utils.mkdir(path_fig_)

# check visibility data
#   time-coverage of all stations
uvall.ploter.draw_tplot(uvall, plotimg=True)

#   radial plot of complex visibility
uvall.ploter.draw_radplot(uvall, plotimg=True, plotsnr=False)

#   uv-coverage
uvall.ploter.draw_uvcover(uvall, plotimg=True)

#   beam pattern (left) & dirty map (right)
uvall.ploter.draw_dirtymap(uvall, plotimg=True)

#   closure amplitude
uvall.ploter.draw_closure(type="clamp", plotimg=True)

#   closure phase
uvall.ploter.draw_closure(type="clphs", plotimg=True)

# run model-fitting
# available options for sampler:
#   'rwalk' : random walk (fastest but not accurate)
#   'rslice': multivariate random slice (between 'rwalk' and 'slice')
#   'slice' : multivariate principle axes slice (slowest but more robust)
#             (recommend)
mfu =\
    gv.modeling.modeling(
        uvfs=uvfs, select=select, uvw=uvw, factor_sblf=1.5, sampler="slice",
        runfit_sf=runfit_sf, runfit_mf=runfit_mf, runfit_set=runfit_set,
        ftype=ftype, fwght=fwght, ufreq=ufreq, bands=bands, spectrum=spectrum,
        maxn=maxn, fixnmod=fixnmod, doampcal=doampcal, dophscal=dophscal,
        npix=npix, mrng=mrng, bnd_l=bnd_l, bnd_m=bnd_m, bnd_f=bnd_f, bnds=bnds,
        width=width, source=source, date=date, path_fig=path_fig_, ncpu=ncpu
    )
mfu.run()

# load and check resultant model parameters
prms = gv.utils.rd_mprms(f"{path_fig_}/mf/model_params.xlsx")
mprms = prms[0] # value
eprms = prms[1] # error
print(mprms)
print(mprms.dtype.names)

# after modeling, the imported uv-fits files are self-calibrated.
# if you need a second run with the calibrated uv-fits files
# using modified data terms and weights, then follow below. (an example)
# mfu.ftype = ["vis", "clamp", "clphs"]
# mfu.fwght = [1, 1, 1]
# mfu.run()


# finally, you can use the model parameters saved in 'model_result.txt'
# in Difmap software just by copying and pasting using the 'edmod' task.
# e.g., at 22 GHz,
# 0> edmod
#   0.184v 0.000v 0.000v 0.198v
#   2.364v 0.510v 177.714v 0.331v
#   0.193v 2.011v 166.600v 0.131v
#   0.080v 7.147v 164.177v 2.920v
#   :wq
# 0>
