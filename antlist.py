
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun

import gamvas


"""
EAVN
"""
name_KC, lat_KC, lon_KC, hgt_KC = "KC", (+(37*u.deg + 32*u.arcmin + 00.100*u.arcsec)).to(u.deg), (+(128*u.deg + 26*u.arcmin + 55.100*u.arcsec)).to(u.deg), +( 557.00)*u.m
name_KT, lat_KT, lon_KT, hgt_KT = "KT", (+(33*u.deg + 17*u.arcmin + 20.900*u.arcsec)).to(u.deg), (+(126*u.deg + 27*u.arcmin + 34.400*u.arcsec)).to(u.deg), +( 452.00)*u.m
name_KU, lat_KU, lon_KU, hgt_KU = "KU", (+(35*u.deg + 32*u.arcmin + 44.200*u.arcsec)).to(u.deg), (+(129*u.deg + 14*u.arcmin + 59.300*u.arcsec)).to(u.deg), +( 170.00)*u.m
name_KY, lat_KY, lon_KY, hgt_KY = "KY", (+(37*u.deg + 33*u.arcmin + 54.900*u.arcsec)).to(u.deg), (+(126*u.deg + 56*u.arcmin + 27.400*u.arcsec)).to(u.deg), +( 139.00)*u.m
name_SJ, lat_SJ, lon_SJ, hgt_SJ = "SJ", (+(36*u.deg + 31*u.arcmin + 21.800*u.arcsec)).to(u.deg), (+(127*u.deg + 18*u.arcmin + 12.000*u.arcsec)).to(u.deg), +( 194.62)*u.m
name_KS, lat_KS, lon_KS, hgt_KS = "KS", (+(37*u.deg + 27*u.arcmin + 15.700*u.arcsec)).to(u.deg), (+(126*u.deg + 57*u.arcmin + 19.000*u.arcsec)).to(u.deg), +( 204.00)*u.m
name_TR, lat_TR, lon_TR, hgt_TR = "TR", (+(36*u.deg + 23*u.arcmin + 51.162*u.arcsec)).to(u.deg), (+(127*u.deg + 22*u.arcmin + 30.591*u.arcsec)).to(u.deg), +( 144.00)*u.m

name_NY, lat_NY, lon_NY, hgt_NY = "NY", (+(35*u.deg + 56*u.arcmin + 40.900*u.arcsec)).to(u.deg), (+(138*u.deg + 28*u.arcmin + 21.200*u.arcsec)).to(u.deg), +(1350.00)*u.m
name_TK, lat_TK, lon_TK, hgt_TK = "TK", (+(36*u.deg + 41*u.arcmin + 54.500*u.arcsec)).to(u.deg), (+(140*u.deg + 41*u.arcmin + 41.000*u.arcsec)).to(u.deg), +( 117.10)*u.m
name_HT, lat_HT, lon_HT, hgt_HT = "HT", (+(36*u.deg + 41*u.arcmin + 50.800*u.arcsec)).to(u.deg), (+(140*u.deg + 41*u.arcmin + 31.600*u.arcsec)).to(u.deg), +( 120.20)*u.m
name_YM, lat_YM, lon_YM, hgt_YM = "YM", (+(34*u.deg + 12*u.arcmin + 57.700*u.arcsec)).to(u.deg), (+(131*u.deg + 33*u.arcmin + 25.500*u.arcsec)).to(u.deg), +( 133.00)*u.m
name_T6, lat_T6, lon_T6, hgt_T6 = "T6", (+(31*u.deg +  5*u.arcmin + 31.600*u.arcsec)).to(u.deg), (+(121*u.deg +  8*u.arcmin +  9.400*u.arcsec)).to(u.deg), +(  49.20)*u.m
name_SH, lat_SH, lon_SH, hgt_SH = "SH", (+(31*u.deg +  5*u.arcmin + 57.000*u.arcsec)).to(u.deg), (+(121*u.deg + 11*u.arcmin + 58.800*u.arcsec)).to(u.deg), +(  29.40)*u.m
name_UR, lat_UR, lon_UR, hgt_UR = "UR", (+(43*u.deg + 28*u.arcmin + 15.600*u.arcsec)).to(u.deg), (+( 87*u.deg + 10*u.arcmin + 40.400*u.arcsec)).to(u.deg), +(2029.40)*u.m
name_KM, lat_KM, lon_KM, hgt_KM = "KM", (+(25*u.deg +  1*u.arcmin + 40.800*u.arcsec)).to(u.deg), (+(102*u.deg + 47*u.arcmin + 45.600*u.arcsec)).to(u.deg), +(1974.00)*u.m
name_VM, lat_VM, lon_VM, hgt_VM = "VM", (+(39*u.deg +  8*u.arcmin +  0.700*u.arcsec)).to(u.deg), (+(141*u.deg +  7*u.arcmin + 57.300*u.arcsec)).to(u.deg), +( 116.60)*u.m
name_VR, lat_VR, lon_VR, hgt_VR = "VR", (+(31*u.deg + 44*u.arcmin + 52.400*u.arcsec)).to(u.deg), (+(130*u.deg + 26*u.arcmin + 23.600*u.arcsec)).to(u.deg), +( 573.60)*u.m
name_VO, lat_VO, lon_VO, hgt_VO = "VO", (+(27*u.deg +  5*u.arcmin + 30.500*u.arcsec)).to(u.deg), (+(142*u.deg + 12*u.arcmin + 59.800*u.arcsec)).to(u.deg), +( 273.10)*u.m
name_VS, lat_VS, lon_VS, hgt_VS = "VS", (+(24*u.deg + 24*u.arcmin + 43.800*u.arcsec)).to(u.deg), (+(124*u.deg + 10*u.arcmin + 15.600*u.arcsec)).to(u.deg), +(  65.10)*u.m



"""
EVN
"""
name_JB, lat_JB, lon_JB, hgt_JB = "JB", (+(53*u.deg + 14*u.arcmin +  2.281*u.arcsec)).to(u.deg), (-(  2*u.deg + 18*u.arcmin + 14.031*u.arcsec)).to(u.deg), +( 143.77)*u.m
name_EF, lat_EF, lon_EF, hgt_EF = "EF", (+(50*u.deg + 31*u.arcmin + 29.410*u.arcsec)).to(u.deg), (+(  6*u.deg + 53*u.arcmin +  1.022*u.arcsec)).to(u.deg), +( 416.72)*u.m
name_MC, lat_MC, lon_MC, hgt_MC = "MC", (+(44*u.deg + 31*u.arcmin + 13.788*u.arcsec)).to(u.deg), (+( 11*u.deg + 38*u.arcmin + 48.984*u.arcsec)).to(u.deg), +(  67.14)*u.m
name_NT, lat_NT, lon_NT, hgt_NT = "NT", (+(36*u.deg + 52*u.arcmin + 33.796*u.arcsec)).to(u.deg), (+( 14*u.deg + 59*u.arcmin + 20.594*u.arcsec)).to(u.deg), +( 143.21)*u.m
name_O6, lat_O6, lon_O6, hgt_O6 = "O6", (+(57*u.deg + 23*u.arcmin + 45.023*u.arcsec)).to(u.deg), (+( 11*u.deg + 55*u.arcmin + 34.902*u.arcsec)).to(u.deg), +(  59.35)*u.m
name_TR, lat_TR, lon_TR, hgt_TR = "TR", (+(53*u.deg +  5*u.arcmin + 43.673*u.arcsec)).to(u.deg), (+( 18*u.deg + 33*u.arcmin + 50.635*u.arcsec)).to(u.deg), +( 133.61)*u.m
name_MH, lat_MH, lon_MH, hgt_MH = "MH", (+(60*u.deg + 13*u.arcmin +  4.122*u.arcsec)).to(u.deg), (+( 24*u.deg + 23*u.arcmin + 35.227*u.arcsec)).to(u.deg), +(  80.02)*u.m
name_YS, lat_YS, lon_YS, hgt_YS = "YS", (+(40*u.deg + 31*u.arcmin + 28.810*u.arcsec)).to(u.deg), (-(  3*u.deg +  5*u.arcmin + 12.683*u.arcsec)).to(u.deg), +( 988.96)*u.m
name_HH, lat_HH, lon_HH, hgt_HH = "HH", (-(25*u.deg + 53*u.arcmin + 23.091*u.arcsec)).to(u.deg), (+( 27*u.deg + 41*u.arcmin +  7.430*u.arcsec)).to(u.deg), +(1415.72)*u.m
name_SR, lat_SR, lon_SR, hgt_SR = "SR", (+(39*u.deg + 29*u.arcmin + 35.028*u.arcsec)).to(u.deg), (+(  9*u.deg + 14*u.arcmin + 42.543*u.arcsec)).to(u.deg), +( 671.47)*u.m
name_RO, lat_RO, lon_RO, hgt_RO = "RO", (+(40*u.deg + 25*u.arcmin + 52.362*u.arcsec)).to(u.deg), (-(  4*u.deg + 14*u.arcmin + 52.817*u.arcsec)).to(u.deg), +( 864.90)*u.m


"""
NRAO
"""
name_BR, lat_BR, lon_BR, hgt_BR = "BR", (+(48*u.deg +  7*u.arcmin + 52.401*u.arcsec)).to(u.deg), (-(119*u.deg + 40*u.arcmin + 59.815*u.arcsec)).to(u.deg), +( 250.47)*u.m
name_FD, lat_FD, lon_FD, hgt_FD = "FD", (+(30*u.deg + 38*u.arcmin +  6.102*u.arcsec)).to(u.deg), (-(103*u.deg + 56*u.arcmin + 41.366*u.arcsec)).to(u.deg), +(1606.42)*u.m
name_HN, lat_HN, lon_HN, hgt_HN = "HN", (+(42*u.deg + 56*u.arcmin +  0.999*u.arcsec)).to(u.deg), (-( 71*u.deg + 59*u.arcmin + 11.704*u.arcsec)).to(u.deg), +( 295.57)*u.m
name_KP, lat_KP, lon_KP, hgt_KP = "KP", (+(31*u.deg + 57*u.arcmin + 22.689*u.arcsec)).to(u.deg), (-(111*u.deg + 36*u.arcmin + 44.733*u.arcsec)).to(u.deg), +(1901.99)*u.m
name_LA, lat_LA, lon_LA, hgt_LA = "LA", (+(35*u.deg + 46*u.arcmin + 30.439*u.arcsec)).to(u.deg), (-(106*u.deg + 14*u.arcmin + 44.158*u.arcsec)).to(u.deg), +(1962.43)*u.m
name_MK, lat_MK, lon_MK, hgt_MK = "MK", (+(19*u.deg + 48*u.arcmin +  5.000*u.arcsec)).to(u.deg), (-(155*u.deg + 27*u.arcmin + 19.864*u.arcsec)).to(u.deg), +(3762.99)*u.m
name_NL, lat_NL, lon_NL, hgt_NL = "NL", (+(41*u.deg + 46*u.arcmin + 17.128*u.arcsec)).to(u.deg), (-( 91*u.deg + 34*u.arcmin + 26.911*u.arcsec)).to(u.deg), +( 222.21)*u.m
name_OV, lat_OV, lon_OV, hgt_OV = "OV", (+(37*u.deg + 13*u.arcmin + 53.938*u.arcsec)).to(u.deg), (-(118*u.deg + 16*u.arcmin + 37.414*u.arcsec)).to(u.deg), +(1196.31)*u.m
name_PT, lat_PT, lon_PT, hgt_PT = "PT", (+(34*u.deg + 18*u.arcmin +  3.657*u.arcsec)).to(u.deg), (-(108*u.deg +  7*u.arcmin +  9.095*u.arcsec)).to(u.deg), +(2364.68)*u.m
name_SC, lat_SC, lon_SC, hgt_SC = "SC", (+(17*u.deg + 45*u.arcmin + 23.703*u.arcsec)).to(u.deg), (-( 64*u.deg + 35*u.arcmin +  1.069*u.arcsec)).to(u.deg), -(  15.02)*u.m
name_YY, lat_YY, lon_YY, hgt_YY = "YY", (+(34*u.deg +  4*u.arcmin + 43.725*u.arcsec)).to(u.deg), (-(107*u.deg + 37*u.arcmin +  6.013*u.arcsec)).to(u.deg), +(2114.48)*u.m
name_GB, lat_GB, lon_GB, hgt_GB = "GB", (+(38*u.deg + 25*u.arcmin + 59.267*u.arcsec)).to(u.deg), (-( 79*u.deg + 50*u.arcmin + 23.433*u.arcsec)).to(u.deg), +( 823.66)*u.m
name_AL, lat_AL, lon_AL, hgt_AL = "AL", -23.029*u.deg, -67.755*u.deg, 5058.7*u.m


"""
LBA
"""
name_AT, lat_AT, lon_AT, hgt_AT = "AT", (-(30*u.deg + 18*u.arcmin + 46.333*u.arcsec)).to(u.deg), (+(149*u.deg + 33*u.arcmin + 53.158*u.arcsec)).to(u.deg), +( 252.02)*u.m
name_CD, lat_CD, lon_CD, hgt_CD = "CD", (-(31*u.deg + 52*u.arcmin +  3.642*u.arcsec)).to(u.deg), (+(133*u.deg + 48*u.arcmin + 35.420*u.arcsec)).to(u.deg), +( 164.62)*u.m
name_HO, lat_HO, lon_HO, hgt_HO = "HO", (-(42*u.deg + 48*u.arcmin + 12.864*u.arcsec)).to(u.deg), (+(147*u.deg + 26*u.arcmin + 25.880*u.arcsec)).to(u.deg), +(  65.08)*u.m
name_MP, lat_MP, lon_MP, hgt_MP = "MP", (-(31*u.deg + 16*u.arcmin +  4.071*u.arcsec)).to(u.deg), (+(149*u.deg +  5*u.arcmin + 58.743*u.arcsec)).to(u.deg), +( 867.32)*u.m
name_PA, lat_PA, lon_PA, hgt_PA = "PA", (-(32*u.deg + 59*u.arcmin + 54.263*u.arcsec)).to(u.deg), (+(148*u.deg + 15*u.arcmin + 48.636*u.arcsec)).to(u.deg), +( 410.80)*u.m
name_TD, lat_TD, lon_TD, hgt_TD = "TD", (-(35*u.deg + 24*u.arcmin +  8.730*u.arcsec)).to(u.deg), (+(148*u.deg + 58*u.arcmin + 52.560*u.arcsec)).to(u.deg), +( 688.80)*u.m


"""
JCMT
"""
name_JC, lat_JC, lon_JC, hgt_JC = "JC", (+(19*u.deg + 49*u.arcmin + 22.000*u.arcsec)).to(u.deg), (-(155*u.deg + 28*u.arcmin + 37.000*u.arcsec)).to(u.deg), +(4092.00)*u.m


dict_lat = {
    "KC":lat_KC, "KT":lat_KT, "KU":lat_KU, "KY":lat_KY, "KS":lat_KS, "SJ":lat_SJ, "TR":lat_TR,
    "NY":lat_NY, "TK":lat_TK, "HT":lat_HT, "YM":lat_YM, "T6":lat_T6, "SH":lat_SH, "UR":lat_UR, "KM":lat_KM, "VM":lat_VM, "VR":lat_VR, "VO":lat_VO, "VS":lat_VS,
    "JB":lat_JB, "EF":lat_EF, "MC":lat_MC, "NT":lat_NT, "O6":lat_O6, "TR":lat_TR, "MH":lat_MH, "YS":lat_YS, "HH":lat_HH, "SR":lat_SR, "RO":lat_RO,
    "BR":lat_BR, "FD":lat_FD, "HN":lat_HN, "KP":lat_KP, "LA":lat_LA, "MK":lat_MK, "NL":lat_NL, "OV":lat_OV, "PT":lat_PT, "SC":lat_SC,
    "AT":lat_AT, "CD":lat_CD, "HO":lat_HO, "MP":lat_MP, "PA":lat_PA, "TD":lat_TD,
    "YY":lat_YY, "GB":lat_GB, "AL":lat_AL,
    "JC":lat_JC
}

dict_lon = {
    "KC":lon_KC, "KT":lon_KT, "KU":lon_KU, "KY":lon_KY, "KS":lon_KS, "SJ":lon_SJ, "TR":lon_TR,
    "NY":lon_NY, "TK":lon_TK, "HT":lon_HT, "YM":lon_YM, "T6":lon_T6, "SH":lon_SH, "UR":lon_UR, "KM":lon_KM, "VM":lon_VM, "VR":lon_VR, "VO":lon_VO, "VS":lon_VS,
    "JB":lon_JB, "EF":lon_EF, "MC":lon_MC, "NT":lon_NT, "O6":lon_O6, "TR":lon_TR, "MH":lon_MH, "YS":lon_YS, "HH":lon_HH, "SR":lon_SR, "RO":lon_RO,
    "BR":lon_BR, "FD":lon_FD, "HN":lon_HN, "KP":lon_KP, "LA":lon_LA, "MK":lon_MK, "NL":lon_NL, "OV":lon_OV, "PT":lon_PT, "SC":lon_SC,
    "AT":lon_AT, "CD":lon_CD, "HO":lon_HO, "MP":lon_MP, "PA":lon_PA, "TD":lon_TD,
    "YY":lon_YY, "GB":lon_GB, "AL":lon_AL,
    "JC":lon_JC
}

dict_hgt = {
    "KC":hgt_KC, "KT":hgt_KT, "KU":hgt_KU, "KY":hgt_KY, "KS":hgt_KS, "SJ":hgt_SJ, "TR":hgt_TR,
    "NY":hgt_NY, "TK":hgt_TK, "HT":hgt_HT, "YM":hgt_YM, "T6":hgt_T6, "SH":hgt_SH, "UR":hgt_UR, "KM":hgt_KM, "VM":hgt_VM, "VR":hgt_VR, "VO":hgt_VO, "VS":hgt_VS,
    "JB":hgt_JB, "EF":hgt_EF, "MC":hgt_MC, "NT":hgt_NT, "O6":hgt_O6, "TR":hgt_TR, "MH":hgt_MH, "YS":hgt_YS, "HH":hgt_HH, "SR":hgt_SR, "RO":hgt_RO,
    "BR":hgt_BR, "FD":hgt_FD, "HN":hgt_HN, "KP":hgt_KP, "LA":hgt_LA, "MK":hgt_MK, "NL":hgt_NL, "OV":hgt_OV, "PT":hgt_PT, "SC":hgt_SC,
    "AT":hgt_AT, "CD":hgt_CD, "HO":hgt_HO, "MP":hgt_MP, "PA":hgt_PA, "TD":hgt_TD,
    "YY":hgt_YY, "GB":hgt_GB, "AL":hgt_AL,
    "JC":hgt_JC
}


def get_station(name):
    " EAVN "
    if name.upper() == "KC":
        ant = EarthLocation(lat=lat_KC, lon=lon_KC, height=hgt_KC)
    if name.upper() == "KT":
        ant = EarthLocation(lat=lat_KT, lon=lon_KT, height=hgt_KT)
    if name.upper() == "KU":
        ant = EarthLocation(lat=lat_KU, lon=lon_KU, height=hgt_KU)
    if name.upper() == "KY":
        ant = EarthLocation(lat=lat_KY, lon=lon_KY, height=hgt_KY)
    if name.upper() == "SJ":
        ant = EarthLocation(lat=lat_SJ, lon=lon_SJ, height=hgt_SJ)
    if name.upper() == "KS":
        ant = EarthLocation(lat=lat_KS, lon=lon_KS, height=hgt_KS)
    if name.upper() == "TR":
        ant = EarthLocation(lat=lat_TR, lon=lon_TR, height=hgt_TR)
    if name.upper() == "NY":
        ant = EarthLocation(lat=lat_NY, lon=lon_NY, height=hgt_NY)
    if name.upper() == "TK":
        ant = EarthLocation(lat=lat_TK, lon=lon_TK, height=hgt_TK)
    if name.upper() == "HT":
        ant = EarthLocation(lat=lat_HT, lon=lon_HT, height=hgt_HT)
    if name.upper() == "YM":
        ant = EarthLocation(lat=lat_YM, lon=lon_YM, height=hgt_YM)
    if name.upper() == "T6":
        ant = EarthLocation(lat=lat_T6, lon=lon_T6, height=hgt_T6)
    if name.upper() == "SH":
        ant = EarthLocation(lat=lat_SH, lon=lon_SH, height=hgt_SH)
    if name.upper() == "UR":
        ant = EarthLocation(lat=lat_UR, lon=lon_UR, height=hgt_UR)
    if name.upper() == "KM":
        ant = EarthLocation(lat=lat_KM, lon=lon_KM, height=hgt_KM)
    if name.upper() == "VM":
        ant = EarthLocation(lat=lat_VM, lon=lon_VM, height=hgt_VM)
    if name.upper() == "VR":
        ant = EarthLocation(lat=lat_VR, lon=lon_VR, height=hgt_VR)
    if name.upper() == "VO":
        ant = EarthLocation(lat=lat_VO, lon=lon_VO, height=hgt_VO)
    if name.upper() == "VS":
        ant = EarthLocation(lat=lat_VS, lon=lon_VS, height=hgt_VS)

    " EVN "
    if name.upper() == "JB":
        ant = EarthLocation(lat=lat_JB, lon=lon_JB, height=hgt_JB)
    if name.upper() == "EF":
        ant = EarthLocation(lat=lat_EF, lon=lon_EF, height=hgt_EF)
    if name.upper() == "MC":
        ant = EarthLocation(lat=lat_MC, lon=lon_MC, height=hgt_MC)
    if name.upper() == "NT":
        ant = EarthLocation(lat=lat_NT, lon=lon_NT, height=hgt_NT)
    if name.upper() == "O6":
        ant = EarthLocation(lat=lat_O6, lon=lon_O6, height=hgt_O6)
    if name.upper() == "TR":
        ant = EarthLocation(lat=lat_TR, lon=lon_TR, height=hgt_TR)
    if name.upper() == "MH":
        ant = EarthLocation(lat=lat_MH, lon=lon_MH, height=hgt_MH)
    if name.upper() == "YS":
        ant = EarthLocation(lat=lat_YS, lon=lon_YS, height=hgt_YS)
    if name.upper() == "HH":
        ant = EarthLocation(lat=lat_HH, lon=lon_HH, height=hgt_HH)
    if name.upper() == "SR":
        ant = EarthLocation(lat=lat_SR, lon=lon_SR, height=hgt_SR)
    if name.upper() == "RO":
        ant = EarthLocation(lat=lat_RO, lon=lon_RO, height=hgt_RO)

    " NRAO "
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
    if name.upper() == "YY":
        ant = EarthLocation(lat=lat_YY, lon=lon_YY, height=hgt_YY)
    if name.upper() == "GB":
        ant = EarthLocation(lat=lat_GB, lon=lon_GB, height=hgt_GB)
    if name.upper() == "AL":
        ant = EarthLocation(lat=lat_AL, lon=lon_AL, height=hgt_AL)

    " LBA "
    if name.upper() == "AT":
        ant = EarthLocation(lat=lat_AT, lon=lon_AT, height=hgt_AT)
    if name.upper() == "CD":
        ant = EarthLocation(lat=lat_CD, lon=lon_CD, height=hgt_CD)
    if name.upper() == "HO":
        ant = EarthLocation(lat=lat_HO, lon=lon_HO, height=hgt_HO)
    if name.upper() == "MP":
        ant = EarthLocation(lat=lat_MP, lon=lon_MP, height=hgt_MP)
    if name.upper() == "PA":
        ant = EarthLocation(lat=lat_PA, lon=lon_PA, height=hgt_PA)
    if name.upper() == "TD":
        ant = EarthLocation(lat=lat_TD, lon=lon_TD, height=hgt_TD)

    "JCMT"
    if name.upper() == "JC":
        ant = EarthLocation(lat=lat_JC, lon=lon_JC, height=hgt_JC)
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
        elif name.upper() == "KAVA":
            VLBIarr = ["KC", "KT", "KU", "KY", "VM", "VR", "VO", "VS"]
        elif name.upper() == "EAVN22":
            # VLBIarr = ["KC", "KT", "KU", "KY", "VM", "VR", "VO", "VS", "T6", "UR", "NY", "TK"]
            VLBIarr = ["KC", "KT", "KU", "KY", "VM", "VR", "VO", "VS", "UR"]
        elif name.upper() == "EAVN43":
            VLBIarr = ["KC", "KT", "KU", "KY", "VM", "VR", "VO", "VS", "T6", "NY"]
        elif name.upper() == "EKVN+LBA":
            VLBIarr = [
            "KC", "KT", "KU", "KY",                                                 # KVN
            "AT", "CD", "HO", "MP", "TD",                                           # LBA
            ]
        elif name.upper() == "EKVN+EVN":
            VLBIarr = [
            "JB", "EF", "MC", "NT", "O6", "T6", "UR", "TR", "MH", "YS", "HH", "SR", # EVN
            "KC", "KT", "KU", "KY",                                                 # KVN
            ]
        elif name.upper() == "GVA22":
            VLBIarr = [
            "JB", "EF", "MC", "NT", "O6", "T6", "UR", "TR", "MH", "YS", "HH", "SR", # EVN
            "BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC",             # VLBA
            "KC", "KT", "KU", "KY",                                                 # KVN
            "AT", "CD", "HO", "MP", "TD",                                           # LBA
            ]
        elif name.upper() == "GVA43":
            VLBIarr = [
                "EF", "NT", "O6", "T6", "YS", # EVN
                "BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC",             # VLBA
                "KC", "KT", "KU", "KY",                                                 # KVN
                "MP"
            ]
        elif name.upper() == "EVN":
            VLBIarr = ["JB", "EF", "MC", "NT", "O6", "T6", "UR", "TR", "MH", "YS", "HH", "SR"]
        elif name.upper() == "VLBA":
            VLBIarr = ["BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC"]
        elif name.upper() == "NRAO":
            VLBIarr = ["BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC", "YY", "GB"]
        elif name.upper() == "LBA":
            VLBIarr = ["AT", "CD", "HO", "MP", "PA", "TD"]
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
