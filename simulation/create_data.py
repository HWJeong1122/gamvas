
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from astropy.coordinates import SkyCoord

import gamvas as gv

class CreateData:
    """
    (Demo)
    Create a simulation of radio VLBI data.
    """

    def __init__(
        self,
        source=None, source_coord=None, frequency=None, bandwidth=None,
        array=None, tarr=None, date=None, tstart=None, duration=None,
        scanlen=None, gaptime=None, tint=None, model=None, modeltype="image",
        frequency_ref=None, spectrum=None, ellim=0, sigma_factor=1, sefd=None,
        gain_amplitude=None, gain_err=0.01, bit=2, time_ff=30, sys_err=0.01

    ):
        """
        Args:
            source (str): Name of the source.
            source_coord (astropy.coordinates.SkyCoord): Coordinates of
                the source.
            frequency (float): Frequency of the observation in Hz.
            bandwidth (float): Bandwidth of the observation in Hz.
            array (array, str): A list of antenna objects.
            tarr (gamvas.tarr, array): A gamvas.tarr object.
            date (str): Date of the observation in YYYY-MM-DD format.
            tstart (float): Start time of the observation in seconds.
            duration (float): Duration of the observation in seconds.
            scanlen (float): Scan length in minutes.
            gaptime (float): Gap time between scans in hours.
            tint (float): Integration time in seconds.
            model (float, 2d array): Reconstructed image of the source.
            modeltype (str): Type of model, either "image" or "component".
            ellim (float): Elevation limit in degrees.
            sigma_factor (float): Factor to scale the thermal noise sigma.
                (if 1, scaled to the expected thermal noise level)
            sefd (dictionary): System equivalent flux density in Jy.
            gain_amplitude (dictionary): Gain amplitude offset.
            gain_err (float): Gain error to add to the data. (0.01 = 1%)
            bit (int): Number of bits in data sampling.
            time_ff (float): Solution interval of fringe fit in seconds.
            sys_err (float): Systematic random offset to add
                to the data. (0.01 = 1%)
        """

        self.source = source
        self.source_coord = source_coord
        self.frequency = np.array(frequency)
        self.bandwidth = np.array(bandwidth)
        self.array = array
        self.tarr = tarr
        self.date = date
        self.tstart = tstart
        self.duration = duration
        self.scanlen = scanlen
        self.gaptime = gaptime
        self.tint = tint
        self.model = model
        self.modeltype = modeltype
        self.frequency_ref = frequency_ref
        self.spectrum = spectrum
        self.ellim = ellim
        self.sigma_factor = sigma_factor
        self.gain_amplitude = gain_amplitude
        self.gain_err = gain_err
        self.sefd = None

        if bit not in [1, 2, 4, 8]:
            raise ValueError("Unexpected bit value: {}".format(bit))

        self.bit = bit
        if bit == 1:
            self.eta_q = 2 / np.pi
        elif bit == 2:
            self.eta_q = 0.882518
        elif bit == 4:
            self.eta_q = 0.988
        elif bit == 8:
            self.eta_q = 0.991

        self.time_ff = time_ff
        self.sys_err = sys_err

    def cdata(self):
        mask_fits = False
        if isinstance(self.model, str):
            if not os.path.isfile(self.model):
                raise FileNotFoundError(f"Model file not found: {self.model}")
            try:
                with pyfits.open(self.model) as _hdul:
                    _hdul.verify("exception")
                mask_fits = True
            except (OSError, ValueError, pyfits.verify.VerifyError):
                mask_fits = False

        elif isinstance(self.model, np.ndarray):
            mask_fits = False

        else:
            raise TypeError(
                f"'model' must be a FITS path (str) or mprms ndarray, "
                f"got {type(self.model).__name__}."
            )

        carr = gv.simulation.create_array.CreateArray(
            source=self.source, source_coord=self.source_coord,
            frequency=self.frequency + self.bandwidth / 2, array=self.array,
            tarr=self.tarr, date=self.date, tstart=self.tstart,
            duration=self.duration, scanlen=self.scanlen, gaptime=self.gaptime,
            tint=self.tint, ellim=self.ellim
        )

        carr.carr()
        dict_num2name = carr.ant_dict_num2name
        dict_name2num = carr.ant_dict_name2num

        tarr = carr.tarr
        uvcov = carr.uvcov

        u = uvcov["u"] / 1e6
        v = uvcov["v"] / 1e6
        ant1 = uvcov["ant1"]
        ant2 = uvcov["ant2"]
        ant1_name = np.array(list(map(dict_num2name.get, ant1)))
        ant2_name = np.array(list(map(dict_num2name.get, ant2)))

        # generate visibility from model fits file
        if mask_fits:
            vis = gv.utils.dft_fits(
                path="", file=self.model, uvcov=uvcov, dotype=self.modeltype
            )
        else:
            if self.frequency_ref is None:
                raise ValueError("frequency_ref must be specified")

            if self.modeltype not in ["gaussian", "delta"]:
                raise ValueError(f"Unexpected model type: {self.modeltype}")

            if self.spectrum is None:
                raise ValueError("spectrum must be specified")

            nif = len(self.frequency)

            assert len(u) % nif == 0, "uvcov length not divisible by nif"
            nvis = int(len(u) / nif)

            _u = u.reshape((nif, nvis, 1)) * 1e6
            _v = v.reshape((nif, nvis, 1)) * 1e6
            _freq = np.broadcast_to(
                self.frequency[:, None], (nif, nvis)
            )[..., None].copy() / 1e9

            dshape = _freq.shape
            dtypes = self.model.dtype.names

            in_args = (
                _u, _v,
                self.frequency_ref / 1e9, _freq,
                self.modeltype, self.spectrum, dshape, dtypes
            )

            in_mask = (True, False)
            vis = gv.utils.model_visibility_append(
                in_args,
                self.model,
                in_mask,
            ).flatten()

        # generate UVFITS file
        uvf = gv.load.open_fits()
        uvf.ant_dict_num2name = dict_num2name
        uvf.ant_dict_name2num = dict_name2num

        # Assign metadata to UVFITS
        uvf.ufreq = np.unique(uvcov["frequency"])
        uvf.freq0 = uvf.ufreq[0]
        uvf.freq_mean = float(uvf.ufreq.mean())
        uvf.no_if = len(uvf.ufreq)
        uvf.no_if_original = len(uvf.ufreq)
        uvf.select_if = "all"
        uvf.nstokes = 2
        uvf.stokes = "I"
        uvf.select_pol = "i"
        uvf.gaptime = self.gaptime
        uvf.scanlen = self.scanlen
        uvf.source = f"{self.source}"
        uvf.date = self.date
        uvf.tarr = tarr

        if self.source_coord is not None:
            uvf.ra = float(self.source_coord.ra.deg)
            uvf.dec = float(self.source_coord.dec.deg)
        else:
            uvf.ra = 0.0
            uvf.dec = 0.0

        try:
            from astropy.time import Time as _atime
            uvf.refGST = _atime(
                self.date, format="iso"
            ).sidereal_time("apparent", "greenwich").deg
        except Exception:
            uvf.refGST = 0.0

        uvf.avg_timebin = self.tint

        # Per-IF accumulators: each list collects one 1-D array per IF,
        # stacked at the end into shape (nvis, no_if).
        cols = {
            k: [] for k in (
                "time", "mjd", "freq", "ant1", "ant2", "baseline",
                "u", "v", "w",
                "r_1", "r_2", "r_3", "r_4",
                "i_1", "i_2", "i_3", "i_4",
                "w_1", "w_2", "w_3", "w_4",
            )
        }

        serr = self.sys_err
        gerr = self.gain_err

        for _no_if in range(uvf.no_if):
            mask = uvcov["frequency"] == uvf.ufreq[_no_if]
            nvis = int(mask.sum())
            _vis = vis[mask]
            _ant1_name = ant1_name[mask]
            _ant2_name = ant2_name[mask]

            cols["time"].append(uvcov["time"][mask])
            cols["mjd"].append(uvcov["mjd"][mask])
            cols["freq"].append(uvcov["frequency"][mask])
            cols["ant1"].append(uvcov["ant1"][mask])
            cols["ant2"].append(uvcov["ant2"][mask])
            cols["baseline"].append(
                256 * uvcov["ant1"][mask] + uvcov["ant2"][mask]
            )
            cols["u"].append(uvcov["u"][mask])
            cols["v"].append(uvcov["v"][mask])
            cols["w"].append(uvcov["w"][mask])

            # Baseline gain (deterministic part: G_ant1 * G_ant2).
            if self.gain_amplitude is not None:
                g1 = np.array(list(map(self.gain_amplitude.get, _ant1_name)))
                g2 = np.array(list(map(self.gain_amplitude.get, _ant2_name)))
                gain_base = g1 * g2
            else:
                gain_base = np.ones(nvis)

            if self.sefd is None:
                # No SEFD: no thermal random noise & assume SNR=100
                cols["r_1"].append(_vis.real)
                cols["i_1"].append(_vis.imag)
                cols["r_2"].append(_vis.real)
                cols["i_2"].append(_vis.imag)
                cols["w_1"].append(np.full(nvis, 10000 / np.abs(_vis)))
                cols["w_2"].append(np.full(nvis, 10000 / np.abs(_vis)))
            else:
                # Thermal noise sigma per visibility (real and imag share the
                # same sigma assuming circularly symmetric Gaussian noise).
                sigma = cal_thermal_error(
                    self.sefd, _ant1_name, _ant2_name, self.eta_q,
                    self.bandwidth[_no_if], self.time_ff
                ) * self.sigma_factor

                # Two parallel-hand polarizations (RR-like, LL-like).
                # Each polarization gets its own multiplicative corruption
                # (gain * (1 + sys)) shared by real and imag parts of the same
                # complex visibility — this preserves visibility phase under
                # gain corruption. Thermal noise is independent per (pol, r/i).
                r_1, i_1 = contaminate_vis(_vis, sigma, gain_base, gerr, serr)
                r_2, i_2 = contaminate_vis(_vis, sigma, gain_base, gerr, serr)
                cols["r_1"].append(r_1)
                cols["i_1"].append(i_1)
                cols["r_2"].append(r_2)
                cols["i_2"].append(i_2)

                # Visibility weight = 1 / variance.
                # Complex Gaussian variance = sigma_r^2 + sigma_i^2 = 2 * sigma^2.
                wgt = 1.0 / (2.0 * sigma**2)
                cols["w_1"].append(wgt)
                cols["w_2"].append(wgt)

            # Cross-hands left empty (NaN) in this demo simulator.
            for k in ("r_3", "r_4", "i_3", "i_4"):
                cols[k].append(np.full(nvis, np.nan))
            for k in ("w_3", "w_4"):
                cols[k].append(np.full(nvis, 0.01))


        for k, lst in cols.items():
            arr = np.stack(lst, axis=1)            # (nvis, no_if)

            if k in ["time", "mjd", "baseline", "ant1", "ant2"]:
                setattr(uvf, k, arr[:, 0].reshape(-1))
            elif k in ["freq"]:
                _arr = np.unique(arr).reshape(1, -1)
                setattr(uvf, k, _arr)
            else:
                setattr(uvf, k, arr.reshape(-1, uvf.no_if, 1))

        uvf.nvis = uvf.r_1.shape[0]
        uvf.load_uvf(gaptime=self.gaptime, scanlen=self.scanlen * 60, prt=False)
        uvf.check_w0()

        return uvf

def cal_thermal_error(sefd, name1, name2, eta, bandwidth, time_ff):
    sefd1 = np.array(list(map(sefd.get, name1)))
    sefd2 = np.array(list(map(sefd.get, name2)))
    thermal_sig = (
        np.sqrt(sefd1 * sefd2)
        / (eta * np.sqrt(2 * bandwidth * time_ff))
    )
    return thermal_sig

def contaminate_vis(vis, sigma, gain_base, gain_err, sys_err):
    """
    Apply one shared multiplicative corruption per complex visibility,
    independent thermal noise on real/imag parts.

    Args:
        vis (np.ndarray, complex): true visibilities, shape (n,).
        sigma (np.ndarray): thermal noise sigma per visibility.
        gain_base (np.ndarray): baseline gain product G_ant1 * G_ant2.
        gain_err (float): random fluctuation around gain_base.
        sys_err (float): random fractional offset on signal.

    Returns:
        (real_contaminateed, imag_contaminateed): two real-valued
            arrays of shape (n,).
    """
    n = len(vis)
    gain = np.random.normal(loc=gain_base, scale=gain_err, size=n)
    sys = np.random.normal(loc=0.0, scale=sys_err, size=n)
    contaminate = gain * (1.0 + sys)            # shared by real and imag
    n_real = np.random.normal(0.0, sigma, n)    # thermal noise on real part
    n_imag = np.random.normal(0.0, sigma, n)    # thermal noise on imag part
    return contaminate * vis.real + n_real, contaminate * vis.imag + n_imag
