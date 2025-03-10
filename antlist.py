
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun

import gamvas

name_TN, lat_TN, lon_TN,  hgt_TN = "KT", (33*u.deg+17*u.arcmin+20.900*u.arcsec).to(u.deg), (+(126*u.deg+27*u.arcmin+34.400*u.arcsec)).to(u.deg), 452.00*u.m
name_US, lat_US, lon_US,  hgt_US = "KU", (35*u.deg+32*u.arcmin+44.200*u.arcsec).to(u.deg), (+(129*u.deg+14*u.arcmin+59.300*u.arcsec)).to(u.deg), 170.00*u.m
name_YS, lat_YS, lon_YS,  hgt_YS = "KY", (37*u.deg+33*u.arcmin+54.900*u.arcsec).to(u.deg), (+(126*u.deg+56*u.arcmin+27.400*u.arcsec)).to(u.deg), 139.00*u.m
name_PC, lat_PC, lon_PC,  hgt_PC = "KC", (37*u.deg+32*u.arcmin+00.100*u.arcsec).to(u.deg), (+(128*u.deg+26*u.arcmin+55.100*u.arcsec)).to(u.deg), 557.00*u.m
name_SR, lat_SR, lon_SR,  hgt_SR = "KS", (37*u.deg+27*u.arcmin+15.700*u.arcsec).to(u.deg), (+(126*u.deg+57*u.arcmin+19.000*u.arcsec)).to(u.deg), 204.00*u.m
name_SJ, lat_SJ, lon_SJ,  hgt_SJ = "SJ", (36*u.deg+31*u.arcmin+21.800*u.arcsec).to(u.deg), (+(127*u.deg+18*u.arcmin+12.000*u.arcsec)).to(u.deg), 194.62*u.m
name_TR, lat_TR, lon_TR,  hgt_TR = "TR", (36*u.deg+23*u.arcmin+51.162*u.arcsec).to(u.deg), (+(127*u.deg+22*u.arcmin+30.591*u.arcsec)).to(u.deg), 144.00*u.m

name_MP, lat_MP, lon_MP,  hgt_MP = "MP", -31.2678*u.deg                                , 149.0997*u.deg                                 , 860*u.m

name_BR, lat_BR, lon_BR,  hgt_BR = "BR", (48*u.deg+7 *u.arcmin+52.42*u.arcsec).to(u.deg), (-(119*u.deg+ 40*u.arcmin+59.80*u.arcsec)).to(u.deg), 250 *u.m
name_FD, lat_FD, lon_FD,  hgt_FD = "FD", (30*u.deg+38*u.arcmin+6.11 *u.arcsec).to(u.deg), (-(103*u.deg+ 56*u.arcmin+41.34*u.arcsec)).to(u.deg), 1606*u.m
name_HN, lat_HN, lon_HN,  hgt_HN = "HN", (42*u.deg+56*u.arcmin+0.99 *u.arcsec).to(u.deg), (-(71 *u.deg+ 59*u.arcmin+11.69*u.arcsec)).to(u.deg), 296 *u.m
name_KP, lat_KP, lon_KP,  hgt_KP = "KP", (31*u.deg+57*u.arcmin+22.70*u.arcsec).to(u.deg), (-(111*u.deg+ 36*u.arcmin+44.72*u.arcsec)).to(u.deg), 1902*u.m
name_LA, lat_LA, lon_LA,  hgt_LA = "LA", (35*u.deg+46*u.arcmin+30.45*u.arcsec).to(u.deg), (-(106*u.deg+ 14*u.arcmin+44.15*u.arcsec)).to(u.deg), 1962*u.m
name_MK, lat_MK, lon_MK,  hgt_MK = "MK", (19*u.deg+48*u.arcmin+4.97 *u.arcsec).to(u.deg), (-(155*u.deg+ 27*u.arcmin+19.81*u.arcsec)).to(u.deg), 3763*u.m
name_NL, lat_NL, lon_NL,  hgt_NL = "NL", (41*u.deg+46*u.arcmin+17.13*u.arcsec).to(u.deg), (-(91 *u.deg+ 34*u.arcmin+26.88*u.arcsec)).to(u.deg), 222 *u.m
name_OV, lat_OV, lon_OV,  hgt_OV = "OV", (37*u.deg+13*u.arcmin+53.95*u.arcsec).to(u.deg), (-(118*u.deg+ 16*u.arcmin+37.37*u.arcsec)).to(u.deg), 1196*u.m
name_PT, lat_PT, lon_PT,  hgt_PT = "PT", (34*u.deg+18*u.arcmin+3.61 *u.arcsec).to(u.deg), (-(108*u.deg+ 7 *u.arcmin+09.06*u.arcsec)).to(u.deg), 2365*u.m
name_SC, lat_SC, lon_SC,  hgt_SC = "SC", (17*u.deg+45*u.arcmin+23.68*u.arcsec).to(u.deg), (-(64 *u.deg+ 35*u.arcmin+1.07 *u.arcsec)).to(u.deg), -15 *u.m

name_VLA, lat_VLA, lon_VLA,  hgt_VLA = 'VLA', (34*u.deg+4 *u.arcmin+43.75*u.arcsec).to(u.deg), (-(107*u.deg+37*u.arcmin+5.91 *u.arcsec)).to(u.deg), 2115*u.m
name_GB , lat_GB , lon_GB ,  hgt_GB  = 'GB' , (38*u.deg+25*u.arcmin+59.24*u.arcsec).to(u.deg), (-(79 *u.deg+50*u.arcmin+23.41*u.arcsec)).to(u.deg), 824 *u.m
name_EB , lat_EB , lon_EB ,  hgt_EB  = 'EB' , (50*u.deg+31*u.arcmin+29.39*u.arcsec).to(u.deg), (+(6  *u.deg+53*u.arcmin+1.00 *u.arcsec)).to(u.deg), 417 *u.m

name_JCMT , lat_JCMT , lon_JCMT ,  hgt_JCMT = 'JCMT', (19*u.deg+49*u.arcmin+22*u.arcsec).to(u.deg), (-(155*u.deg+28*u.arcmin+37*u.arcsec).to(u.deg)), 4092*u.m

name_ALMA , lat_ALMA , lon_ALMA ,  hgt_ALMA = 'ALMA', -23.029*u.deg, -67.755*u.deg, 5058.7*u.m

dict_lat = {
    "KT":lat_TN, "KU":lat_US, "KY":lat_YS, "KC":lat_PC, "KS":lat_SR, "SJ":lat_SJ, "TR":lat_TR,
    "BR":lat_BR, "FD":lat_FD, "HN":lat_HN, "KP":lat_KP, "LA":lat_LA,
    "MK":lat_MK, "NL":lat_NL, "OV":lat_OV, "PT":lat_PT, "SC":lat_SC,
    "MP":lat_MP, "JCMT":lat_JCMT, "ALMA":lat_ALMA
}

dict_lon = {
    "KT":lon_TN, "KU":lon_US, "KY":lon_YS, "KC":lon_PC, "KS":lon_SR, "SJ":lon_SJ, "TR":lon_TR,
    "BR":lon_BR, "FD":lon_FD, "HN":lon_HN, "KP":lon_KP, "LA":lon_LA,
    "MK":lon_MK, "NL":lon_NL, "OV":lon_OV, "PT":lon_PT, "SC":lon_SC,
    "MP":lon_MP, "JCMT":lon_JCMT, "ALMA":lon_ALMA
}

dict_hgt = {
    "KT":hgt_TN, "KU":hgt_US, "KY":hgt_YS, "KC":hgt_PC, "KS":hgt_SR, "SJ":hgt_SJ, "TR":hgt_TR,
    "BR":hgt_BR, "FD":hgt_FD, "HN":hgt_HN, "KP":hgt_KP, "LA":hgt_LA,
    "MK":hgt_MK, "NL":hgt_NL, "OV":hgt_OV, "PT":hgt_PT, "SC":hgt_SC,
    "MP":hgt_MP, "JCMT":hgt_JCMT, "ALMA":hgt_ALMA
}


def get_station(name):
    " KVN "
    if name.upper() == "PC":
        ant = EarthLocation(lat=lat_PC, lon=lon_PC, height=hgt_PC)
    if name.upper() == "YS":
        ant = EarthLocation(lat=lat_YS, lon=lon_YS, height=hgt_YS)
    if name.upper() == "US":
        ant = EarthLocation(lat=lat_US, lon=lon_US, height=hgt_US)
    if name.upper() == "TN":
        ant = EarthLocation(lat=lat_TN, lon=lon_TN, height=hgt_TN)
    if name.upper() == "TR":
        ant = EarthLocation(lat=lat_TR, lon=lon_TR, height=hgt_TR)

    " NRAO VLBA "
    if name.upper() == "BR":
        ant = EarthLocation(lat=lat_BR, lon=lon_BR, height=hgt_BR)
    if name.upper() == "FD":
        ant = EarthLocation(lat=lat_FD, lon=lon_FD, height=hgt_FD)
    if name.upper() == "HN":
        ant = EarthLocation(lat=lat_HN, lon=lon_HN, height=hgt_HN)
    if name.upper() == "KP":
        ant = EarthLocation(lat=lat_KP, lon=lon_KP, height=hgt_KP)
    if name.upper() == "LA":
        ant = EarthLocation(lat=lat_LA, lon=lon_LA, height=hgt_LA)
    if name.upper() == "MK":
        ant = EarthLocation(lat=lat_MK, lon=lon_MK, height=hgt_MK)
    if name.upper() == "NL":
        ant = EarthLocation(lat=lat_NL, lon=lon_NL, height=hgt_NL)
    if name.upper() == "OV":
        ant = EarthLocation(lat=lat_OV, lon=lon_OV, height=hgt_OV)
    if name.upper() == "PT":
        ant = EarthLocation(lat=lat_PT, lon=lon_PT, height=hgt_PT)
    if name.upper() == "SC":
        ant = EarthLocation(lat=lat_SC, lon=lon_SC, height=hgt_SC)

    " NRAO VLA "
    if name.upper() == "VLA":
        ant = EarthLocation(lat=lat_VLA, lon=lon_VLA, height=hgt_VLA)

    if name.upper() == "GBT":
        ant = EarthLocation(lat=lat_GB, lon=lon_GB, height=hgt_GB)

    " LBA MP"
    if name.upper() == "MP":
        ant = EarthLocation(lat=lat_MP, lon=lon_MP, height=hgt_MP)

    " EVN Yebes"
    if name.upper() == "EB":
        ant = EarthLocation(lat=lat_EB, lon=lon_EB, height=hgt_EB)

    " JCMT "
    if name.upper() == "JCMT":
        ant = EarthLocation(lat=lat_JCMT, lon=lon_JCMT, height=hgt_JCMT)

    " ALMA "
    if name.upper() == "ALMA":
        ant = EarthLocation(lat=lat_ALMA, lon=lon_ALMA, height=hgt_ALMA)

    return ant

def get_vlbi(name):
    arr_xpo = np.array([])
    arr_ypo = np.array([])
    arr_zpo = np.array([])
    arr_lat = np.array([])
    arr_lon = np.array([])
    arr_hgt = np.array([])
    if type(name) == str:
        if name.upper() == "KVN":
            VLBIarr = ["KT", "KU", "KY"]
        elif name.upper() == "EKVN":
            VLBIarr = ["KT", "KU", "KY", "KC"]
        elif name.upper() == "EKVN+TR":
            VLBIarr = ["KT", "KU", "KY", "KC", "TR"]
        elif name.upper() == "EKVN+TR+SJ":
            VLBIarr = ["KT", "KU", "KY", "KC", "SJ", "TR"]
        elif name.upper() == "KVNMP":
            VLBIarr = ["KT", "KU", "KY", "MP"]
        elif name.upper() == "EKVNMP":
            VLBIarr = ["KT", "KU", "KY", "KC", "MP"]
        elif name.upper() == "VLBA":
            VLBIarr = ["BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC"]
    else:
        VLBIarr = name

    arr_nam = np.array(VLBIarr)
    for arr in VLBIarr:
        loc = EarthLocation(lat=dict_lat[arr], lon=dict_lon[arr], height=dict_hgt[arr])
        arr_xpo = np.append(arr_xpo, loc.x.value)
        arr_ypo = np.append(arr_ypo, loc.y.value)
        arr_zpo = np.append(arr_zpo, loc.z.value)
        arr_lat = np.append(arr_lat, dict_lat[arr].value)
        arr_lon = np.append(arr_lon, dict_lon[arr].value)
        arr_hgt = np.append(arr_hgt, dict_hgt[arr].value)
    out = gamvas.utils.sarray(
        data =[arr_nam, arr_xpo, arr_ypo, arr_zpo, arr_lat, arr_lon, arr_hgt],
        field=["name", "x", "y", "z", "lat", "lon", "height"],
        dtype=["U32", "f8", "f8", "f8", "f8", "f8", "f8"])
    return out, name
