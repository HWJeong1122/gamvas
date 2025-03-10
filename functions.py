
import os
import sys
import numpy as np
import numpy.lib.recfunctions as rfn
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy import units as u

nan = np.nan
r2m = u.rad.to(u.mas)
d2m = u.rad.to(u.mas)

def linear(x, m, a):
    """
        Arguments:
            x (array): input x-axis data points
            m (float): slope of the linear function
            a (float): offset of the linear function (constant)
        Returns:
            A linear function
    """
    return m * x + a


def gaussian_1d(x, peak, a, mx):
    """
        Arguments:
            x (array): input x-axis data points
            peak (float): peak-value of the Gaussian
            a (float): standard deviation of the Gaussian
            mx (float): offset of the peak in x-axis from the zero-position
        Returns:
            A 1-D Guassian function
    """
    return peak * np.e**(-((x - mx) / a)**2 / 2)


def gaussian_2d(xy, peak, ax, ay, mx, my, theta):
    """
        Arguments:
            xy (2D-array, tuple): input x/y-axis data points
            peak (float): peak-value of the Gaussian
            ax/y (float): standard deviation of the Gaussian in x/y-axis
            mx/y (float): offset of the peak in x/y-axis from the zero-position
            theta (float): position angle of the Gaussian (elliptical Guassian)
        Returns:
            A 2-D Guassian function
    """
    x, y = xy
    a = 0.5 * (np.cos(theta)**2 / ax**2 + np.sin(theta)**2 / ax**2)
    b = 0.5 * (np.sin(theta * 2) / ax**2 - np.sin(theta * 2) /ay**2)
    c = 0.5 * (np.sin(theta)**2 / ax**2 + np.cos(theta)**2 / ax**2)
    return peak * np.e**(-(x - mx)**2 / (2 * ax**2) -(y - my)**2 / (2 * ay**2))


def SSA(nu, Smax, tf, alpha):
    """
    NOTE: This function assumes optically thick spectral index as 2.5
    (Turler+1999, A&A, 349, 45T)
        Arguments:
            nu (array or float): input frequency
            Smax (float): flux density at 'tf'
            tf (float): turnover frequency of the SSA spectrum
            alpha (float): optically thin spectral index
        Returns:
            estimated flux density at nu (nu: float) / SSA spectrm (nu: list or array)
    """
    term_tau = 1.5 * ((1 - (8 * alpha) / 7.5)**0.5 - 1)
    term_nu = (nu / tf)**2.5
    term_frac = (1 - np.e**(-term_tau * (nu / tf)**(alpha - 2.5))) / (1 - np.e**(-term_tau))
    result = Smax * term_nu * term_frac
    return result


def S_spl(nu_ref, nu, Smax, alpha):
    """
        Arguments:
            nu_ref (float): reference frequency, recommended to set at the lowest one
            nu (array or float): input frequency
            Smax (float): flux density at 'nu_ref'
            alpha (float): optically thin spectral index
        Returns:
            estimated flux density at nu (nu: float) / simple power-law spectrum (nu: list or array)
    """
    return 10**(alpha * (unp.log10(nu) - unp.log10(nu_ref)) + unp.log10(Smax))


def S_cpl(nu, Smax, tf, alpha):
    """
        Arguments:
            nu (array or float): input frequency
            Smax (float): flux density at 'tf'
            alpha (float): optically thin spectral index
        Returns:
            estimated flux density at nu (nu: float) / curved power-law spectrum (nu: list or array)
    """
    return Smax * (nu / tf)**(alpha * unp.log10(nu / tf))


def gvis0(args, S, fwhm):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of Gaussian model
            fwhm (float): full-width at half maximum of Gaussian model
        Returns:
            complex visibility of a Gaussian model
    """
    uu = args[0] / r2m
    vv = args[1] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    visibility = S * np.e**(-2*np.pi**2 * a**2 * (uu**2 + vv**2))
    return visibility.astype("c8")


def gvis_spl0(args, Smax, fwhm, alpha):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (float): reference frequency; recommended to set at the lowest one
                args[1] (array or float): input frequency
                args[2] (1D-array): u-axis data points
                args[3] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'args[0]'
            fwhm (float): full-width at half maximum of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model (based on a simple power-law spectrum)
    """
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    S = S_spl(nu_ref, nu, Smax, alpha)
    visibility = S * np.e**(-2 * (np.pi * a)**2 * (uu**2 + vv**2))
    return visibility.astype("c8")


def gvis_cpl0(args, Smax, fwhm, alpha, nu_m):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (array or float): input frequency
                args[1] (1D-array): u-axis data points
                args[2] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model (based on curved power-law spectrum)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    S = S_cpl(nu, Smax, nu_m, alpha)
    visibility = S * np.e**(-2 * (np.pi * a)**2 * (uu**2 + vv**2))
    return visibility.astype("c8")


def gvis_ssa0(args, Smax, fwhm, alpha, nu_m):
    """
    NOTE: This function is intended to fix model position to (0,0)
        Arguments:
            args (tuple): input sub-arguments
                args[0] (array or float): input frequency
                args[1] (1D-array): u-axis data points
                args[2] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
            nu_m (float): turnover frequency
        Returns:
            complex visibility of Gaussian model (based on SSA spectrum; Turler+1999, A&A, 349, 45T)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    S = SSA(nu, Smax, nu_m, alpha)
    visibility = S * np.e**(-2 * (np.pi * a)**2 * (uu**2 + vv**2))
    return visibility.astype("c8")


def gvis(args, S, fwhm, l, m):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of Gaussian model
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
        Returns:
            complex visibility of Gaussian model
    """
    uu = args[0] / r2m
    vv = args[1] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    visibility = S * np.e**(-2 * np.pi**2 * a**2 * (uu**2 + vv**2) + 2j * np.pi * (uu * l + vv * m))
    return visibility.astype("c8")


def gvis_spl(args, Smax, fwhm, l, m, alpha):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (float): reference frequency, recommended to set at the lowest one
                args[1] (array or float): input frequency
                args[2] (1D-array): u-axis data points
                args[3] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'args[0]'
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model (based on simple power-law spectrum)
    """
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    S = S_spl(nu_ref, nu, Smax, alpha)
    visibility = S * np.e**(-2 * (np.pi * a)**2 * (uu**2 + vv**2) + 2j * np.pi * (uu * l + vv * m))
    return visibility.astype("c8")


def gvis_cpl(args, Smax, fwhm, l, m, alpha, nu_m):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (float): reference frequency, recommended to set at the lowest one
                args[1] (array or float): input frequency
                args[2] (1D-array): u-axis data points
                args[3] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
        Returns:
            complex visibility of Gaussian model (based on simple power-law spectrum)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    S = S_cpl(nu, Smax, nu_m, alpha)
    visibility = S * np.e**(-2 * (np.pi * a)**2 * (uu**2 + vv**2) + 2j * np.pi * (uu * l + vv * m))
    return visibility.astype("c8")


def gvis_ssa(args, Smax, fwhm, l, m, alpha, nu_m):
    """
        Arguments:
            args (tuple): input sub-arguments
                args[0] (array or float): input frequency
                args[1] (1D-array): u-axis data points
                args[2] (1D-array): v-axis data points
            Smax (float): flux density of Gaussian model at 'nu_m'
            fwhm (float): full-width at half maximum of Gaussian model
            l (float): right ascension position of Gaussian model
            m (float): declination position of Gaussian model
            alpha (float): optically thin spectral index of Gaussian model
            nu_m (float): turnover frequency
        Returns:
            complex visibility of Gaussian model (based on SSA spectrum)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    a = fwhm / (2 * (2 * unp.log(2))**0.5)
    S = SSA(nu, Smax, nu_m, alpha)
    visibility = S * np.e**(-2 * (np.pi * a)**2 * (uu**2 + vv**2) + 2j * np.pi * (uu * l + vv * m))
    return visibility.astype("c8")
