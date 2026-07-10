
import numpy as np
from uncertainties import unumpy as unp
from astropy import units as au

r2m = au.rad.to(au.mas)
d2m = au.rad.to(au.mas)


def dvis(args, S, l, m, phi):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        S (float): flux density
        l (float): right ascension position
        m (float): declination position
        phi (float): polarization angle
    Returns:
        complex visibility of delta-function model
    """
    uu = args[0] / r2m
    vv = args[1] / r2m
    out = (
        S
        * np.exp(2j * np.pi * (uu * l + vv * m))
        * np.exp(2j * phi)
    )

    return out.astype("c8")


def gvis(args, S, fwhm, l, m, phi):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        S (float): flux density
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        phi (float): polarization angle
    Returns:
        complex visibility of Gaussian model
    """
    uu = args[0] / r2m
    vv = args[1] / r2m
    sigma = fwhm / (2 * (2 * unp.log(2))**0.5)
    out = (
        S
        * np.exp(
        -2 * np.pi**2 * sigma**2 * (uu**2 + vv**2)
        + 2j * np.pi * (uu * l + vv * m)
        )
        * np.exp(2j * phi)
    )

    return out.astype("c8")
