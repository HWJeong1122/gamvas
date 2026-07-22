
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
from astropy import units as au

r2m = au.rad.to(au.mas) # radian to milli-arcsecond
d2m = au.deg.to(au.mas) # degree to milli-arcsecond

def cpl(nu, smax, tf, alpha):
    """
    Curved power-law spectrum

    Args:
        nu (array or float): input frequency
        smax (float): flux density at 'tf'
        tf (float): turnover frequency
        alpha (float): spectral index

    Returns:
        spectrum or flux density at 'nu'
    """

    out = smax * 10**(alpha * np.log10(nu / tf)**2)

    return out

def dvis(args, s, l, m):
    """
    Complex visibility of delta-function model

    Args:
        args (tuple): input sub-arguments
            - args[0] (1D-array): u-axis data points
            - args[1] (1D-array): v-axis data points
        s (float): flux density of delta function model
        l (float): right ascension position of delta function model
        m (float): declination position of delta function model

    Returns:
        complex visibility of delta-function model
    """

    u = args[0] / r2m
    v = args[1] / r2m
    out = s * np.exp(2j * np.pi * (u * l + v * m))

    return out.astype("c8")

def dvis_cpl(args, smax, l, m, alpha, nu_m):
    """
    Complex visibility of delta-function model with curved power-law spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (float): reference frequency
                (recommended to set at the lowest one)
            - args[1] (array or float): input frequency
            - args[2] (1D-array): u-axis data points
            - args[3] (1D-array): v-axis data points
        smax (float): flux density at 'nu_m'
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index
        nu_m (float): turnover frequency

    Returns:
        complex visibility
    """

    nu = args[0]
    u = args[1] / r2m
    v = args[2] / r2m
    S = cpl(nu, smax, nu_m, alpha)
    out = S * np.exp(2j * np.pi * (u * l + v * m))

    return out.astype("c8")

def dvis_poly(args, s_ref, l, m, alpha, beta):
    """
    Complex visibility of delta-function model with polynomial spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (float): reference frequency
                (recommended to set at the lowest one)
            - args[1] (array or float): input frequency
            - args[2] (1D-array): u-axis data points
            - args[3] (1D-array): v-axis data points
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index
        beta (float): spectrum curvature

    Returns:
        complex visibility
    """
    nu_ref = args[0]
    nu = args[1]
    u = args[2] / r2m
    v = args[3] / r2m
    S = poly(nu_ref, nu, s_ref, alpha, beta)
    out = S * np.exp(2j * np.pi * (u * l + v * m))

    return out.astype("c8")

def dvis_spl(args, smax, l, m, alpha):
    """
    Complex visibility of delta-function model with simple power-law spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (float): reference frequency
                (recommended to set at the lowest one)
            - args[1] (array or float): input frequency
            - args[2] (1D-array): u-axis data points
            - args[3] (1D-array): v-axis data points
        smax (float): flux density at 'args[0]'
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index

    Returns:
        complex visibility
    """

    nu_ref = args[0]
    nu = args[1]
    u = args[2] / r2m
    v = args[3] / r2m
    S = spl(nu_ref, nu, smax, alpha)
    out = S * np.exp(2j * np.pi * (u * l + v * m))

    return out.astype("c8")

def dvis_ssa(args, smax, l, m, alpha, nu_m):
    """
    Complex visibility of delta-function model with synchrotron
    self-absorption spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (array or float): input frequency
            - args[2] (1D-array): u-axis data points
            - args[3] (1D-array): v-axis data points
        smax (float): flux density  at 'nu_m'
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index
        nu_m (float): turnover frequency

    Returns:
        complex visibility
    """

    nu = args[0]
    u = args[1] / r2m
    v = args[2] / r2m
    S = ssa(nu, smax, nu_m, alpha)
    out = S * np.exp(2j * np.pi * (u * l + v * m))

    return out.astype("c8")

def gaussian_1d(x, peak, a, mx):
    """
    1D Gaussian function

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
    2D Gaussian function

    Args:
        xy (2D-array, tuple): input x/y-axis data points
        peak (float): peak-value of the Gaussian
        ax/y (float): standard deviation of a Gaussian in x/y-axis
        mx/y (float): offset of the peak in x/y-axis from the zero-position
        theta (float): position angle of a Gaussian (elliptical Guassian)

    Returns:
        A 2-D Guassian function
    """

    x, y = xy
    out = peak * np.exp(
        -(x - mx)**2 / (2 * ax**2) - (y - my)**2 / (2 * ay**2)
    )

    return out

def gvis(args, s, fwhm, l, m):
    """
    Complex visibility of a Gaussian model with a flat spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (1D-array): u-axis data points
            - args[1] (1D-array): v-axis data points
        s (float): flux density
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position

    Returns:
        complex visibility
    """

    u = args[0] / r2m
    v = args[1] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    out = s * np.exp(
        -2 * np.pi**2 * sigma**2 * (u**2 + v**2)
        + 2j * np.pi * (u * l + v * m)
    )

    return out.astype("c8")

def gvis_cpl(args, smax, fwhm, l, m, alpha, nu_m):
    """
    Complex visibility of a Gaussian model with curved power-law spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (float): reference frequency
                (recommended to set at the lowest one)
            - args[1] (array or float): input frequency
            - args[2] (1D-array): u-axis data points
            - args[3] (1D-array): v-axis data points
        smax (float): flux density at 'nu_m'
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index
        nu_m (float): turnover frequency

    Returns:
        complex visibility
    """

    nu = args[0]
    u = args[1] / r2m
    v = args[2] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    s = cpl(nu, smax, nu_m, alpha)
    out = s * np.exp(
        -2 * (np.pi * sigma)**2 * (u**2 + v**2)
        + 2j * np.pi * (u * l + v * m)
    )

    return out.astype("c8")

def gvis_elipse(args, s, major, minor, l, m, pa):
    """
    Complex visibility of a Gaussian model with elliptical shape

    Args:
        args (tuple): input sub-arguments
            - args[0] (1D-array): u-axis data points
            - args[1] (1D-array): v-axis data points
        s (float): flux density
        major (float): full-width at half maximum in major axis
        minor (float): full-width at half maximum in minor axis
        l (float): right ascension position
        m (float): declination position
        pa (float): position angle in radian

    Returns:
        complex visibility
    """

    u = args[0] / r2m
    v = args[1] / r2m

    fwhm2sigma = 1 / (2 * (2 * np.log(2))**0.5)
    sigma_major = major * fwhm2sigma
    sigma_minor = minor * fwhm2sigma

    cos_pa, sin_pa = np.cos(pa), np.sin(pa)
    u_maj = u * sin_pa + v * cos_pa
    u_min = u * cos_pa - v * sin_pa
    out = s * np.exp(
            -2 * np.pi**2
            * (sigma_major**2 * u_maj**2 + sigma_minor**2 * u_min**2)
            + 2j * np.pi * (u * l + v * m)
        )

    return out.astype("c8")

def gvis_poly(args, s_ref, fwhm, l, m, alpha, beta):
    """
    Complex visibility of a Gaussian model with polynomial spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (float): reference frequency
            - args[1] (array or float): input frequency
            - args[2] (1D-array): u-axis data points
            - args[3] (1D-array): v-axis data points
        s_ref (float): reference flux density
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index
        beta (float): spectrum curvature

    Returns:
        complex visibility
    """

    nu_ref = args[0]
    nu = args[1]
    u = args[2] / r2m
    v = args[3] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    s = poly(nu_ref, nu, s_ref, alpha, beta)
    out = s * np.exp(
        -2 * (np.pi * sigma)**2 * (u**2 + v**2)
        + 2j * np.pi * (u * l + v * m)
    )

    return out.astype("c8")

def gvis_spl(args, smax, fwhm, l, m, alpha):
    """
    Complex visibility of a Gaussian model with simple power-law spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (float): reference frequency
                (recommended to set at the lowest one)
            - args[1] (array or float): input frequency
            - args[2] (1D-array): u-axis data points
            - args[3] (1D-array): v-axis data points
        smax (float): flux density at 'args[0]'
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index

    Returns:
        complex visibility
    """

    nu_ref = args[0]
    nu = args[1]
    u = args[2] / r2m
    v = args[3] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    s = spl(nu_ref, nu, smax, alpha)
    out = s * np.exp(
        -2 * (np.pi * sigma)**2 * (u**2 + v**2)
        + 2j * np.pi * (u * l + v * m)
    )

    return out.astype("c8")

def gvis_ssa(args, smax, fwhm, l, m, alpha, nu_m):
    """
    Complex visibility of a Gaussian model with synchrotron
    self-absorption spectrum

    Args:
        args (tuple): input sub-arguments
            - args[0] (array or float): input frequency
            - args[1] (1D-array): u-axis data points
            - args[2] (1D-array): v-axis data points
        smax (float): flux density at 'nu_m'
        fwhm (float): full-width at half maximum
        l (float): right ascension position
        m (float): declination position
        alpha (float): spectral index
        nu_m (float): turnover frequency

    Returns:
        complex visibility
    """

    nu = args[0]
    u = args[1] / r2m
    v = args[2] / r2m
    sigma = fwhm / (2 * (2 * np.log(2))**0.5)
    s = ssa(nu, smax, nu_m, alpha)
    out = s * np.exp(
        -2 * (np.pi * sigma)**2 * (u**2 + v**2)
        + 2j * np.pi * (u * l + v * m)
    )

    return out.astype("c8")

def linear(x, m, a):
    """
    Linear function

    Args:
        x (array): input x-axis data points
        m (float): slope of the linear function
        a (float): offset of the linear function

    Returns:
        A linear function
    """

    out = m * x + a

    return out

def poly(nu_ref, nu, s_ref, alpha, beta):
    """
    Logarithmic second-order polynomial function

    Args:
        nu_ref (float): reference frequency
        nu (array or float): input frequency
        s_ref (float): reference flux density
        alpha (float): spectral index
        beta (float): spectral curvature

    Returns:
        spectrum or flux density at 'nu'
    """

    x = unp.log(nu / nu_ref)
    out = s_ref * unp.exp(alpha * x + beta * x**2)

    return out

def ring(args, s, r, w):
    """
    Ring function

    Args:
        args (tuple): input sub-arguments
            - args[0] (1D-array): u-axis data points
            - args[1] (1D-array): v-axis data points
        s (float): flux density
        r (float): ring radius
        w (float): ring width

    Returns:
        complex visibility
    """

    u = args[0] / r2m
    v = args[1] / r2m
    q2 = u**2 + v**2
    thickness = 1 - 2 * w / r
    out = s * (
        1 - np.pi**2 * r**2 * q2 / 10
        * (1 - thickness**5) / (1 - thickness**3)
    )

    return out.astype("c8")

def spl(nu_ref, nu, smax, alpha):
    """
    Simple power-law spectrum

    Args:
        nu_ref (float): reference frequency (recommended to set at the
            lowest one)
        nu (array or float): input frequency
        smax (float): flux density at 'nu_ref'
        alpha (float): spectral index

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
        alpha (float): spectral index

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
