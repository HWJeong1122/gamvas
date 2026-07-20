
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy import units as au

r2m = au.rad.to(au.mas)
d2m = au.deg.to(au.mas)

def cpl(nu, smax, tf, alpha):
    """
    Args:
        nu (array or float): input frequency
        smax (float): flux density at 'tf'
        alpha (float): optically thin spectral index
    Returns:
        estimated flux density at nu
    """
    out = smax * 10**(alpha * np.log10(nu / tf)**2)

    return out

def dvis(args, S, l, m):
    """
        Args:
            args (tuple): input sub-arguments
                args[0] (1D-array): u-axis data points
                args[1] (1D-array): v-axis data points
            S (float): flux density of delta function model
            l (float): right ascension position of delta function model
            m (float): declination position of delta function model
        Returns:
            complex visibility of delta-function model
    """
    uu = args[0] / r2m
    vv = args[1] / r2m
    out = S * np.exp(2j * np.pi * (uu * l + vv * m))

    return out.astype("c8")

def dvis_cpl(args, smax, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (float): reference frequency (recommended to set at the
                lowest one)
            args[1] (array or float): input frequency
            args[2] (1D-array): u-axis data points
            args[3] (1D-array): v-axis data points
        smax (float): flux density at 'nu_m'
        l (float): right ascension position
        m (float): declination position
        alpha (float): optically thin spectral index
        nu_m (float): turnover frequency
    Returns:
        complex visibility of delta-function model (based on simple power-law
            spectrum)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    S = cpl(nu, smax, nu_m, alpha)
    out = S * np.exp(2j * np.pi * (uu * l + vv * m))

    return out.astype("c8")

def dvis_poly(args, s_ref, l, m, alpha, beta):
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    S = poly(nu_ref, nu, s_ref, alpha, beta)
    out = S * np.exp(2j * np.pi * (uu * l + vv * m))

    return out.astype("c8")

def dvis_spl(args, smax, l, m, alpha):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (float): reference frequency (recommended to set at the
                lowest one)
            args[1] (array or float): input frequency
            args[2] (1D-array): u-axis data points
            args[3] (1D-array): v-axis data points
        smax (float): flux density at 'args[0]'
        l (float): right ascension position
        m (float): declination position
        alpha (float): optically thin spectral index
    Returns:
        complex visibility of delta-function model (based on simple power-law
            spectrum)
    """
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    S = spl(nu_ref, nu, smax, alpha)
    out = S * np.exp(2j * np.pi * (uu * l + vv * m))

    return out.astype("c8")

def dvis_ssa(args, smax, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (array or float): input frequency
            args[1] (1D-array): u-axis data points
            args[2] (1D-array): v-axis data points
        smax (float): flux density  at 'nu_m'
        l (float): right ascension position
        m (float): declination position
        alpha (float): optically thin spectral index
        nu_m (float): turnover frequency
    Returns:
        complex visibility of delta-function model (based on SSA spectrum)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    S = ssa(nu, smax, nu_m, alpha)
    out = S * np.exp(2j * np.pi * (uu * l + vv * m))

    return out.astype("c8")

def gaussian_1d(x, peak, a, mx):
    """
    Args:
        x (array): input x-axis data points
        peak (float): peak-value of the Gaussian
        a (float): standard deviation of the Gaussian
        mx (float): offset of the peak in x-axis from the zero-position
    Returns:
        A 1-D Guassian function
    """
    out = peak * np.exp(-((x - mx) / a)**2 / 2)
    return out

def gaussian_2d(xy, peak, ax, ay, mx, my, theta):
    """
    Args:
        xy (2D-array, tuple): input x/y-axis data points
        peak (float): peak-value of the Gaussian
        ax/y (float): standard deviation of the Gaussian in x/y-axis
        mx/y (float): offset of the peak in x/y-axis from the zero-position
        theta (float): position angle of the Gaussian (elliptical Guassian)
    Returns:
        A 2-D Guassian function
    """
    x, y = xy
    out = peak * np.exp(
        -(x - mx)**2 / (2 * ax**2) - (y - my)**2 / (2 * ay**2)
    )

    return out

def gvis(args, S, fwhm, l, m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        S (float): flux density
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
    Returns:
        complex visibility of Gaussian model
    """
    uu = args[0] / r2m
    vv = args[1] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    out = S * np.exp(
        -2 * np.pi**2 * sigma**2 * (uu**2 + vv**2)
        + 2j * np.pi * (uu * l + vv * m)
    )

    return out.astype("c8")

def gvis_cpl(args, smax, fwhm, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (float): reference frequency (recommended to set at the
                lowest one)
            args[1] (array or float): input frequency
            args[2] (1D-array): u-axis data points
            args[3] (1D-array): v-axis data points
        smax (float): flux density at 'nu_m'
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        alpha (float): optically thin spectral index
        nu_m (float): turnover frequency
    Returns:
        complex visibility of Gaussian model (based on simple power-law
            spectrum)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    S = cpl(nu, smax, nu_m, alpha)
    out = S * np.exp(
        -2 * (np.pi * sigma)**2 * (uu**2 + vv**2)
        + 2j * np.pi * (uu * l + vv * m)
    )

    return out.astype("c8")

def gvis_elipse(args, S, major, minor, l, m, pa):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (1D-array): u-axis data points
            args[1] (1D-array): v-axis data points
        S (float): flux density
        major (float): full-width at half maximum in major axis
        minor (float): full-width at half maximum in minor axis
        l (float): right ascension position
        m (float): declination position
        pa (float): position angle in radian
    Returns:
        complex visibility of Gaussian model
    """
    uu = args[0] / r2m
    vv = args[1] / r2m

    fwhm2sigma = 1 / (2 * (2 * np.log(2))**0.5)
    sigma_major = major * fwhm2sigma
    sigma_minor = minor * fwhm2sigma

    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    u_maj = uu * sin_pa + vv * cos_pa
    u_min = uu * cos_pa - vv * sin_pa
    out = S * np.exp(
            -2 * np.pi**2
            * (sigma_major**2 * u_maj**2 + sigma_minor**2 * u_min**2)
            + 2j * np.pi * (uu * l + vv * m)
        )

    return out.astype("c8")

def gvis_poly(args, s_ref, fwhm, l, m, alpha, beta):
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    S = poly(nu_ref, nu, s_ref, alpha, beta)
    out = S * np.exp(
        -2 * (np.pi * sigma)**2 * (uu**2 + vv**2)
        + 2j * np.pi * (uu * l + vv * m)
    )

    return out.astype("c8")

def gvis_spl(args, smax, fwhm, l, m, alpha):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (float): reference frequency (recommended to set at the
                lowest one)
            args[1] (array or float): input frequency
            args[2] (1D-array): u-axis data points
            args[3] (1D-array): v-axis data points
        smax (float): flux density at 'args[0]'
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        alpha (float): optically thin spectral index
    Returns:
        complex visibility of Gaussian model (based on simple power-law
            spectrum)
    """
    nu_ref = args[0]
    nu = args[1]
    uu = args[2] / r2m
    vv = args[3] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    S = spl(nu_ref, nu, smax, alpha)
    out = S * np.exp(
        -2 * (np.pi * sigma)**2 * (uu**2 + vv**2)
        + 2j * np.pi * (uu * l + vv * m)
    )

    return out.astype("c8")

def gvis_ssa(args, smax, fwhm, l, m, alpha, nu_m):
    """
    Args:
        args (tuple): input sub-arguments
            args[0] (array or float): input frequency
            args[1] (1D-array): u-axis data points
            args[2] (1D-array): v-axis data points
        smax (float): flux density at 'nu_m'
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        alpha (float): optically thin spectral index
        nu_m (float): turnover frequency
    Returns:
        complex visibility of Gaussian model (based on SSA spectrum)
    """
    nu = args[0]
    uu = args[1] / r2m
    vv = args[2] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    S = ssa(nu, smax, nu_m, alpha)
    out = S * np.exp(
        -2 * (np.pi * sigma)**2 * (uu**2 + vv**2)
        + 2j * np.pi * (uu * l + vv * m)
    )

    return out.astype("c8")

def linear(x, m, a):
    """
    Args:
        x (array): input x-axis data points
        m (float): slope of the linear function
        a (float): offset of the linear function (constant)
    Returns:
        A linear function
    """
    out = m * x + a

    return out

def poly(nu_ref, nu, s_ref, alpha, beta):
    x = unp.log(nu / nu_ref)
    out = s_ref * unp.exp(alpha * x + beta * x**2)

    return out

def ring(args, S, r, w):
    uu = args[0] / r2m
    vv = args[1] / r2m
    q2 = uu**2 + vv**2
    thickness = 1 - 2 * w / r
    out = S * (
        1 - np.pi**2 * r**2 * q2 / 10
        * (1 - thickness**5) / (1 - thickness**3)
    )

    return out.astype("c8")

def spl(nu_ref, nu, smax, alpha):
    """
    Args:
        nu_ref (float): reference frequency (recommended to set at the
            lowest one)
        nu (array or float): input frequency
        smax (float): flux density at 'nu_ref'
        alpha (float): optically thin spectral index
    Returns:
        estimated flux density at nu
    """
    out = smax * (nu / nu_ref)**alpha

    return out

def ssa(nu, smax, tf, alpha):
    """
    NOTE: This function assumes optically thick spectral index as 2.5
    (Turler+1999, A&A, 349, 45T)
    Args:
        nu (array or float): input frequency
        smax (float): flux density at 'tf'
        tf (float): turnover frequency of the SSA spectrum
        alpha (float): optically thin spectral index
    Returns:
        estimated flux density at nu
    """
    term_tau = 1.5 * ((1 - (8 * alpha) / 7.5)**0.5 - 1)
    term_nu = (nu / tf)**2.5
    term_frac = (
        (1 - np.e**(-term_tau * (nu / tf)**(alpha - 2.5)))
        / (1 - np.e**(-term_tau))
    )
    out = smax * term_nu * term_frac

    return out
