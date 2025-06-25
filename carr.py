
import os
import sys
import gc
import copy
import itertools
import pandas as pd
import numpy as np
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy import units as u
from astropy import constants as C
from astropy.time import Time as ati
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun

import gamvas

class create_array:
    def __init__(
        self, array=None, tarr=None, date=None, tstart=None, duration=None, scanlen=None, tint=None, gaptime=None,
        source_coord=None, obs_freq=None, ellim=0
    ):
        """
        Arguments:
            array (str): array name
            tarr (array): numpy array of VLBI-array
            date (str): observing date                          (iso-type)
            tstart (float): starting time                       (hour)
            duration (float): total duration of observation     (hour)
            scanlen (float): total length of scan               (minute)
            tint (float): integration time of visibility        (second)
            gaptime (float): gap size between scans             (hour)
            source_coord (object): source coordinate            (astropy.SkyCoord)
            obs_freq (float): observing frequency               (GHz)
        """
        self.array          = array
        self.tarr           = tarr
        self.date           = date
        self.tstart         = tstart
        self.duration       = duration
        self.scanlen        = scanlen
        self.tint           = tint
        self.gaptime        = gaptime
        self.source_coord   = source_coord
        self.obs_freq       = obs_freq
        self.ellim          = ellim


    def carr(self):
        obsw = (C.c/(self.obs_freq*u.GHz)).to(u.m)
        mjd0 = ati(self.date, format="iso").mjd
        scantime = np.arange(self.tstart, self.tstart+self.duration, self.gaptime)

        nvis_scan = int(self.scanlen*60/self.tint)
        time = np.array([])
        for nscan, scantime_ in enumerate(scantime):
            time = np.append(time, scantime_+np.arange(0, nvis_scan*self.tint, self.tint)/3600)

        # Time information in UTC
        times = ati(mjd0 + time/24, format="mjd").iso

        # Load tarr information
        if self.tarr is not None and self.array is not None:
            tarr, arranme = self.tarr, self.array
        elif self.tarr is None and self.array is not None:
            tarr, array = gamvas.antlist.get_vlbi(self.array)
        else:
            raise Exception("Array information is not given properly.")

        self.tarr = tarr
        self.ant_dict_name2num = dict(zip(self.tarr["name"], np.arange(self.tarr.shape[0])+1))
        self.ant_dict_num2name = {val: key for key, val in self.ant_dict_name2num.items()}

        # source ra, dec
        source_coord = self.source_coord
        ra, dec = source_coord.ra, source_coord.dec

        # set baselines
        baselines = list(itertools.combinations(tarr["name"], 2))
        obs_times = []
        obs_ant_num1 = []
        obs_ant_num2 = []
        obs_ant_name1 = []
        obs_ant_name2 = []

        for ntime, time in enumerate(times):
            for baseline in baselines:
                obs_times.append(time)
                obs_ant_num1.append(self.ant_dict_name2num[baseline[0]])
                obs_ant_num2.append(self.ant_dict_name2num[baseline[1]])
                obs_ant_name1.append(baseline[0])
                obs_ant_name2.append(baseline[1])

        dict_lat = dict(zip(tarr["name"], tarr["lat"]))
        dict_lon = dict(zip(tarr["name"], tarr["lon"]))
        dict_hgt = dict(zip(tarr["name"], tarr["height"]))
        arr_lat1 = np.array(list(map(dict_lat.get, obs_ant_name1)))
        arr_lon1 = np.array(list(map(dict_lon.get, obs_ant_name1)))
        arr_hgt1 = np.array(list(map(dict_hgt.get, obs_ant_name1)))
        arr_lat2 = np.array(list(map(dict_lat.get, obs_ant_name2)))
        arr_lon2 = np.array(list(map(dict_lon.get, obs_ant_name2)))
        arr_hgt2 = np.array(list(map(dict_hgt.get, obs_ant_name2)))

        loc1 = EarthLocation(lat=arr_lat1*u.deg, lon=arr_lon1*u.deg, height=arr_hgt1*u.m)
        loc2 = EarthLocation(lat=arr_lat2*u.deg, lon=arr_lon2*u.deg, height=arr_hgt2*u.m)
        scd1 = source_coord.transform_to(AltAz(obstime=obs_times, location=loc1))
        scd2 = source_coord.transform_to(AltAz(obstime=obs_times, location=loc2))

        mask_elevation = (scd1.alt.value >= self.ellim) & (scd2.alt.value >= self.ellim)
        loc1 = loc1[mask_elevation]
        loc2 = loc2[mask_elevation]
        scd1 = scd1[mask_elevation]
        scd2 = scd2[mask_elevation]

        obs_times = np.array(obs_times)[mask_elevation]

        lst1 = ati(obs_times).sidereal_time("apparent", "greenwich")
        lst2 = ati(obs_times).sidereal_time("apparent", "greenwich")
        hangle = (lst1 - ra).to(u.deg)

        Xw = (loc2.x - loc1.x) / obsw
        Yw = (loc2.y - loc1.y) / obsw
        Zw = (loc2.z - loc1.z) / obsw

        obs_uv_u = +Xw*np.sin(hangle)             + Yw*np.cos(hangle)             + Zw*0
        obs_uv_v = -Xw*np.sin(dec)*np.cos(hangle) + Yw*np.sin(dec)*np.sin(hangle) + Zw*np.cos(dec)
        obs_uv_w = +Xw*np.cos(dec)*np.cos(hangle) - Yw*np.cos(dec)*np.sin(hangle) + Zw*np.sin(dec)

        mjd = ati(obs_times, format="iso").mjd
        times = (mjd - int(np.min(mjd))) * 24
        tint = np.ones(times.shape[0]) * self.tint
        obs_ant_num1 = np.array(obs_ant_num1)[mask_elevation]
        obs_ant_num2 = np.array(obs_ant_num2)[mask_elevation]
        obs_ant_name1 = np.array(obs_ant_name1)[mask_elevation]
        obs_ant_name2 = np.array(obs_ant_name2)[mask_elevation]
        phi1 = np.zeros(times.shape[0])
        phi2 = np.zeros(times.shape[0])

        uvcov = sarray(
            data=[times, tint, mjd, obs_ant_num1, obs_ant_num2, obs_ant_name1, obs_ant_name2, obs_uv_u, obs_uv_v, phi1, phi2, scd1.alt.value, scd2.alt.value],
            field=["time", "tint", "mjd", "ant_num1", "ant_num2", "ant_name1", "ant_name2", "u" , "v" , "phi1", "phi2", "elevation1", "elevation2"],
            dtype=["f8", "f8", "f8", "i", "i", "U32", "U32", "f8", "f8", "f8", "f8", "f8", "f8"])
        mask_u0 = uvcov["u"] == 0
        mask_v0 = uvcov["v"] == 0
        mask_uv0 = (mask_u0) & (mask_v0)
        uvcov = uvcov[~mask_uv0]
        self.uvcov = uvcov


def add_vis(inputdat, select):
    out = inputdat.copy()
    out = rfn.append_fields(out, "vis_{0}"  .format(select.lower()), np.zeros(out.shape[0], "c8"), usemask=False)
    out = rfn.append_fields(out, "sigma_{0}".format(select.lower()), np.zeros(out.shape[0], "f8"), usemask=False)
    return out


def apply_gain_error(inputdat, dict_gain, select):
    gainamp1 = np.array(list(map(dict_gain.get, inputdat["ant_name1"])))
    gainamp2 = np.array(list(map(dict_gain.get, inputdat["ant_name2"])))
    inputdat["vis_{0}".format(select.lower())] *= gainamp1 * gainamp2
    return inputdat

def sarray(data, field, dtype):
    data = np.array(data)
    sarray_ = np.zeros(data.shape[1:], dtype=list(zip(field, dtype)))
    for nf,field in enumerate(field):
        sarray_[field] = data[nf]
    return sarray_
