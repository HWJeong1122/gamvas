
import os
import sys
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy import units as u

nan = np.nan
r2m = u.rad.to(u.mas)
d2m = u.rad.to(u.mas)


def gvis(args, S):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of the Gaussian model
            fwhm (float): full-width at half maximum of the Gaussian model
            r (float): raidus of the Gaussian model from (0,0) position
            p (float): position angle of the Guassian model (jet direction, north-to-east // top-to-left in RA-DEC map)
        Returns:
            complex visibility of the Gaussian model
    """
    fwhm = args[2]
    l = args[3]
    m = args[4]
    uu = args[0] / r2m
    vv = args[1] / r2m
    a = fwhm / (2 * np.sqrt(2 * np.log(2)))
    visibility = S * np.exp(-2 * np.pi**2 * a**2 * (uu**2 + vv**2) + 2j * np.pi * (uu * l + vv * m))
    return visibility.astype("c8")
