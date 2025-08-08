
# Gaussian Multi-frequency VLBI Analyses (GaMVAs) for AGN jets

from gamvas import load
from gamvas import functions
from gamvas import modeling
from gamvas import utils
from gamvas import plotting
from gamvas import carr
from gamvas import antlist
from gamvas import sourcelist
from gamvas import polarization

from astropy import units
uas = units.uas
mas = units.mas
arcs = units.arcsecond
deg = units.deg
rad = units.rad

import warnings
warnings.filterwarnings("ignore")

import matplotlib as mpl
mpl.rcParams["xtick.major.size" ] = 5.0*1.0
mpl.rcParams["xtick.major.width"] = 2.0*1.0
mpl.rcParams["xtick.minor.size" ] = 2.5*1.0
mpl.rcParams["xtick.minor.width"] = 1.0*1.0

mpl.rcParams["ytick.major.size" ] = 5.0*1.0
mpl.rcParams["ytick.major.width"] = 2.0*1.0
mpl.rcParams["ytick.minor.size" ] = 2.5*1.0
mpl.rcParams["ytick.minor.width"] = 1.0*1.0
