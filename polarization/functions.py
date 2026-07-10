
import numpy as np
from astropy import units as au

r2m = au.rad.to(au.mas)
d2m = au.rad.to(au.mas)


def dvis(args, sq, su, l, m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        sq (float): flux density for Stokes Q
        su (float): flux density for Stokes U
        l (float): right ascension position
        m (float): declination position
    Returns:
        complex visibility of delta-function model
    """
    u = args[0] / r2m
    v = args[1] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    sp = (sq**2 + su**2)**0.5
    evpa = 0.5 * np.arctan2(su, sq)
    gvis = (
        sp
        * np.exp(2j * np.pi * (u * l + v * m))
    )
    out_q = gvis * np.cos(2 * evpa)
    out_u = gvis * np.sin(2 * evpa)

    return out_q.astype("c8"), out_u.astype("c8")


def gvis(args, sq, su, fwhm, l, m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        sq (float): flux density for Stokes Q
        su (float): flux density for Stokes U
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
    Returns:
        complex visibility of Gaussian model
    """
    u = args[0] / r2m
    v = args[1] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    sp = (sq**2 + su**2)**0.5
    evpa = 0.5 * np.arctan2(su, sq)
    gvis = (
        sp
        * np.exp(-2 * np.pi**2 * sigma**2 * (u**2 + v**2))
        * np.exp(2j * np.pi * (u * l + v * m))
    )
    out_q = gvis * np.cos(2 * evpa)
    out_u = gvis * np.sin(2 * evpa)

    return out_q.astype("c8"), out_u.astype("c8")
