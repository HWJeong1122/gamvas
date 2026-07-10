
import numpy as np
from astropy import units as au
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.coordinates import AltAz
from astropy.coordinates import get_sun

import gamvas as gv

"""
EAVN
"""
name_KC, lat_KC, lon_KC, hgt_KC = (
    "KC",
    (+(37 * au.deg + 32 * au.arcmin + 00.100 * au.arcsec)).to(au.deg),
    (+(128 * au.deg + 26 * au.arcmin + 55.100 * au.arcsec)).to(au.deg),
    +( 557.00) * au.m
)   # KVN Pyeongchang
name_KT, lat_KT, lon_KT, hgt_KT = (
    "KT",
    (+(33 * au.deg + 17 * au.arcmin + 20.900 * au.arcsec)).to(au.deg),
    (+(126 * au.deg + 27 * au.arcmin + 34.400 * au.arcsec)).to(au.deg),
    +( 452.00) * au.m
)   # KVN Tamna
name_KU, lat_KU, lon_KU, hgt_KU = (
    "KU",
    (+(35 * au.deg + 32 * au.arcmin + 44.200 * au.arcsec)).to(au.deg),
    (+(129 * au.deg + 14 * au.arcmin + 59.300 * au.arcsec)).to(au.deg),
    +( 170.00) * au.m
)   # KVN Ulsan
name_KY, lat_KY, lon_KY, hgt_KY = (
    "KY",
    (+(37 * au.deg + 33 * au.arcmin + 54.900 * au.arcsec)).to(au.deg),
    (+(126 * au.deg + 56 * au.arcmin + 27.400 * au.arcsec)).to(au.deg),
    +( 139.00) * au.m
)   # KVN Yonsei
name_KV, lat_KV, lon_KV, hgt_KV = (
    "KV",
    (+(36 * au.deg + 31 * au.arcmin + 21.800 * au.arcsec)).to(au.deg),
    (+(127 * au.deg + 18 * au.arcmin + 12.000 * au.arcsec)).to(au.deg),
    +( 194.62) * au.m
)   # Sejong (Sejong, South Korea)
name_KR, lat_KR, lon_KR, hgt_KR = (
    "KR",
    (+(36 * au.deg + 23 * au.arcmin + 51.162 * au.arcsec)).to(au.deg),
    (+(127 * au.deg + 22 * au.arcmin + 30.591 * au.arcsec)).to(au.deg),
    +( 144.00) * au.m
)   # TRAO (Daejeon, South Korea)
name_KS, lat_KS, lon_KS, hgt_KS = (
    "KS",
    (+(37 * au.deg + 27 * au.arcmin + 15.700 * au.arcsec)).to(au.deg),
    (+(126 * au.deg + 57 * au.arcmin + 19.000 * au.arcsec)).to(au.deg),
    +( 204.00) * au.m
)   # SRAO (Seoul, South Korea)

name_NY, lat_NY, lon_NY, hgt_NY = (
    "NY",
    (+(35 * au.deg + 56 * au.arcmin + 40.900 * au.arcsec)).to(au.deg),
    (+(138 * au.deg + 28 * au.arcmin + 21.200 * au.arcsec)).to(au.deg),
    +(1350.00) * au.m
)   # Nobeyama 45-m
name_TK, lat_TK, lon_TK, hgt_TK = (
    "TK",
    (+(36 * au.deg + 41 * au.arcmin + 54.500 * au.arcsec)).to(au.deg),
    (+(140 * au.deg + 41 * au.arcmin + 41.000 * au.arcsec)).to(au.deg),
    +( 117.10) * au.m
)   # Takahagi 32-m
name_HT, lat_HT, lon_HT, hgt_HT = (
    "HT",
    (+(36 * au.deg + 41 * au.arcmin + 50.800 * au.arcsec)).to(au.deg),
    (+(140 * au.deg + 41 * au.arcmin + 31.600 * au.arcsec)).to(au.deg),
    +( 120.20) * au.m
)   # Hitachi 32-m
name_YM, lat_YM, lon_YM, hgt_YM = (
    "YM",
    (+(34 * au.deg + 12 * au.arcmin + 57.700 * au.arcsec)).to(au.deg),
    (+(131 * au.deg + 33 * au.arcmin + 25.500 * au.arcsec)).to(au.deg),
    +( 133.00) * au.m
)   # Yamaguchi 32-m
name_T6, lat_T6, lon_T6, hgt_T6 = (
    "T6",
    (+(31 * au.deg +  5 * au.arcmin + 31.600 * au.arcsec)).to(au.deg),
    (+(121 * au.deg +  8 * au.arcmin +  9.400 * au.arcsec)).to(au.deg),
    +(  49.20) * au.m
)   # Tianma 65-m
name_SH, lat_SH, lon_SH, hgt_SH = (
    "SH",
    (+(31 * au.deg +  5 * au.arcmin + 57.000 * au.arcsec)).to(au.deg),
    (+(121 * au.deg + 11 * au.arcmin + 58.800 * au.arcsec)).to(au.deg),
    +(  29.40) * au.m
)   # Sheshan 25-m
name_UR, lat_UR, lon_UR, hgt_UR = (
    "UR",
    (+(43 * au.deg + 28 * au.arcmin + 15.600 * au.arcsec)).to(au.deg),
    (+( 87 * au.deg + 10 * au.arcmin + 40.400 * au.arcsec)).to(au.deg),
    +(2029.40) * au.m
)   # Nanshan 25-m (Urumqi)
name_KM, lat_KM, lon_KM, hgt_KM = (
    "KM",
    (+(25 * au.deg +  1 * au.arcmin + 40.800 * au.arcsec)).to(au.deg),
    (+(102 * au.deg + 47 * au.arcmin + 45.600 * au.arcsec)).to(au.deg),
    +(1974.00) * au.m
)   # Kunming 40-m
name_VM, lat_VM, lon_VM, hgt_VM = (
    "VM",
    (+(39 * au.deg +  8 * au.arcmin +  0.700 * au.arcsec)).to(au.deg),
    (+(141 * au.deg +  7 * au.arcmin + 57.300 * au.arcsec)).to(au.deg),
    +( 116.60) * au.m
)   # VERA Mizusawa
name_VR, lat_VR, lon_VR, hgt_VR = (
    "VR",
    (+(31 * au.deg + 44 * au.arcmin + 52.400 * au.arcsec)).to(au.deg),
    (+(130 * au.deg + 26 * au.arcmin + 23.600 * au.arcsec)).to(au.deg),
    +( 573.60) * au.m
)   # VERA Iriki
name_VO, lat_VO, lon_VO, hgt_VO = (
    "VO",
    (+(27 * au.deg +  5 * au.arcmin + 30.500 * au.arcsec)).to(au.deg),
    (+(142 * au.deg + 12 * au.arcmin + 59.800 * au.arcsec)).to(au.deg),
    +( 273.10) * au.m
)   # VERA Ogasawara
name_VS, lat_VS, lon_VS, hgt_VS = (
    "VS",
    (+(24 * au.deg + 24 * au.arcmin + 43.800 * au.arcsec)).to(au.deg),
    (+(124 * au.deg + 10 * au.arcmin + 15.600 * au.arcsec)).to(au.deg),
    +(  65.10) * au.m
)   # VERA Ishigakijima

"""
EVN
"""
name_JB, lat_JB, lon_JB, hgt_JB = (
    "JB",
    (+(53 * au.deg + 14 * au.arcmin +  2.281 * au.arcsec)).to(au.deg),
    (-(  2 * au.deg + 18 * au.arcmin + 14.031 * au.arcsec)).to(au.deg),
    +( 143.77) * au.m
)   # Jodrell Bank
name_EF, lat_EF, lon_EF, hgt_EF = (
    "EF",
    (+(50 * au.deg + 31 * au.arcmin + 29.390 * au.arcsec)).to(au.deg),
    (+(  6 * au.deg + 53 * au.arcmin +  1.022 * au.arcsec)).to(au.deg),
    +( 416.72) * au.m
)   # Effelsberg 100-m
name_MC, lat_MC, lon_MC, hgt_MC = (
    "MC",
    (+(44 * au.deg + 31 * au.arcmin + 13.788 * au.arcsec)).to(au.deg),
    (+( 11 * au.deg + 38 * au.arcmin + 48.984 * au.arcsec)).to(au.deg),
    +(  67.14) * au.m
)   # Medicina 32-m
name_NT, lat_NT, lon_NT, hgt_NT = (
    "NT",
    (+(36 * au.deg + 52 * au.arcmin + 33.796 * au.arcsec)).to(au.deg),
    (+( 14 * au.deg + 59 * au.arcmin + 20.594 * au.arcsec)).to(au.deg),
    +( 143.21) * au.m
)   # Noto 32-m
name_O6, lat_O6, lon_O6, hgt_O6 = (
    "O6",
    (+(57 * au.deg + 23 * au.arcmin + 45.023 * au.arcsec)).to(au.deg),
    (+( 11 * au.deg + 55 * au.arcmin + 34.902 * au.arcsec)).to(au.deg),
    +(  59.35) * au.m
)   # Onsala60 20-m
name_TR, lat_TR, lon_TR, hgt_TR = (
    "TR",
    (+(53 * au.deg +  5 * au.arcmin + 43.673 * au.arcsec)).to(au.deg),
    (+( 18 * au.deg + 33 * au.arcmin + 50.635 * au.arcsec)).to(au.deg),
    +( 133.61) * au.m
)   # Torun 32-m
name_MH, lat_MH, lon_MH, hgt_MH = (
    "MH",
    (+(60 * au.deg + 13 * au.arcmin +  4.122 * au.arcsec)).to(au.deg),
    (+( 24 * au.deg + 23 * au.arcmin + 35.227 * au.arcsec)).to(au.deg),
    +(  80.02) * au.m
)   # Medicina 32-m
name_YS, lat_YS, lon_YS, hgt_YS = (
    "YS",
    (+(40 * au.deg + 31 * au.arcmin + 28.810 * au.arcsec)).to(au.deg),
    (-(  3 * au.deg +  5 * au.arcmin + 12.683 * au.arcsec)).to(au.deg),
    +( 988.96) * au.m
)   # Yebes 40-m
name_HH, lat_HH, lon_HH, hgt_HH = (
    "HH",
    (-(25 * au.deg + 53 * au.arcmin + 23.091 * au.arcsec)).to(au.deg),
    (+( 27 * au.deg + 41 * au.arcmin +  7.430 * au.arcsec)).to(au.deg),
    +(1415.72) * au.m
)   # Hartebeesthoek 26-m
name_SR, lat_SR, lon_SR, hgt_SR = (
    "SR",
    (+(39 * au.deg + 29 * au.arcmin + 35.028 * au.arcsec)).to(au.deg),
    (+(  9 * au.deg + 14 * au.arcmin + 42.543 * au.arcsec)).to(au.deg),
    +( 671.47) * au.m
)   # Sardinia 65-m
name_RO, lat_RO, lon_RO, hgt_RO = (
    "RO",
    (+(40 * au.deg + 25 * au.arcmin + 52.362 * au.arcsec)).to(au.deg),
    (-(  4 * au.deg + 14 * au.arcmin + 52.817 * au.arcsec)).to(au.deg),
    +( 864.90) * au.m
)   # Robledo 70-m
name_NN, lat_NN, lon_NN, hgt_NN = (
    "NN",
    (+( 44 * au.deg + 38 * au.arcmin +  2.000 * au.arcsec)).to(au.deg),
    (+( 5 * au.deg + 54 * au.arcmin + 28.500 * au.arcsec)).to(au.deg),
    +(2550.00) * au.m
)   # NOEMA
name_PV, lat_PV, lon_PV, hgt_PV = (
    "PV",
    (+( 37 * au.deg +  3 * au.arcmin + 58.300 * au.arcsec)).to(au.deg),
    (-( 3 * au.deg + 23 * au.arcmin + 33.700 * au.arcsec)).to(au.deg),
    +(2850.00) * au.m
)   # Pico Veleta

"""
NRAO
"""
name_BR, lat_BR, lon_BR, hgt_BR = (
    "BR",
    (+(48 * au.deg +  7 * au.arcmin + 52.401 * au.arcsec)).to(au.deg),
    (-(119 * au.deg + 40 * au.arcmin + 59.815 * au.arcsec)).to(au.deg),
    +( 250.47) * au.m
)   # VLBA Brewster
name_FD, lat_FD, lon_FD, hgt_FD = (
    "FD",
    (+(30 * au.deg + 38 * au.arcmin +  6.102 * au.arcsec)).to(au.deg),
    (-(103 * au.deg + 56 * au.arcmin + 41.366 * au.arcsec)).to(au.deg),
    +(1606.42) * au.m
)   # VLBA Fort Davis
name_HN, lat_HN, lon_HN, hgt_HN = (
    "HN",
    (+(42 * au.deg + 56 * au.arcmin +  0.999 * au.arcsec)).to(au.deg),
    (-( 71 * au.deg + 59 * au.arcmin + 11.704 * au.arcsec)).to(au.deg),
    +( 295.57) * au.m
)   # VLBA Hancock
name_KP, lat_KP, lon_KP, hgt_KP = (
    "KP",
    (+(31 * au.deg + 57 * au.arcmin + 22.689 * au.arcsec)).to(au.deg),
    (-(111 * au.deg + 36 * au.arcmin + 44.733 * au.arcsec)).to(au.deg),
    +(1901.99) * au.m
)   # VLBA Kitt Peak
name_LA, lat_LA, lon_LA, hgt_LA = (
    "LA",
    (+(35 * au.deg + 46 * au.arcmin + 30.439 * au.arcsec)).to(au.deg),
    (-(106 * au.deg + 14 * au.arcmin + 44.158 * au.arcsec)).to(au.deg),
    +(1962.43) * au.m
)   # VLBA Los Alamos
name_MK, lat_MK, lon_MK, hgt_MK = (
    "MK",
    (+(19 * au.deg + 48 * au.arcmin +  5.000 * au.arcsec)).to(au.deg),
    (-(155 * au.deg + 27 * au.arcmin + 19.864 * au.arcsec)).to(au.deg),
    +(3762.99) * au.m
)   # VLBA Mauna Kea
name_NL, lat_NL, lon_NL, hgt_NL = (
    "NL",
    (+(41 * au.deg + 46 * au.arcmin + 17.128 * au.arcsec)).to(au.deg),
    (-( 91 * au.deg + 34 * au.arcmin + 26.911 * au.arcsec)).to(au.deg),
    +( 222.21) * au.m
)   # VLBA North Liberty
name_OV, lat_OV, lon_OV, hgt_OV = (
    "OV",
    (+(37 * au.deg + 13 * au.arcmin + 53.938 * au.arcsec)).to(au.deg),
    (-(118 * au.deg + 16 * au.arcmin + 37.414 * au.arcsec)).to(au.deg),
    +(1196.31) * au.m
)   # VLBA Owens Valley
name_PT, lat_PT, lon_PT, hgt_PT = (
    "PT",
    (+(34 * au.deg + 18 * au.arcmin +  3.657 * au.arcsec)).to(au.deg),
    (-(108 * au.deg +  7 * au.arcmin +  9.095 * au.arcsec)).to(au.deg),
    +(2364.68) * au.m
)   # VLBA Pie Town
name_SC, lat_SC, lon_SC, hgt_SC = (
    "SC",
    (+(17 * au.deg + 45 * au.arcmin + 23.703 * au.arcsec)).to(au.deg),
    (-( 64 * au.deg + 35 * au.arcmin +  1.069 * au.arcsec)).to(au.deg),
    -(  15.02) * au.m
)   # VLBA St. Croix
name_YY, lat_YY, lon_YY, hgt_YY = (
    "YY",
    (+(34 * au.deg +  4 * au.arcmin + 43.725 * au.arcsec)).to(au.deg),
    (-(107 * au.deg + 37 * au.arcmin +  6.013 * au.arcsec)).to(au.deg),
    +(2114.48) * au.m
)   # Phased-VLA
name_GB, lat_GB, lon_GB, hgt_GB = (
    "GB",
    (+(38 * au.deg + 25 * au.arcmin + 59.267 * au.arcsec)).to(au.deg),
    (-( 79 * au.deg + 50 * au.arcmin + 23.433 * au.arcsec)).to(au.deg),
    +( 823.66) * au.m
)   # Green Bank Telescope
name_GL, lat_GL, lon_GL, hgt_GL = (
    "GL",
    (+(76 * au.deg + 32 * au.arcmin + 06.000 * au.arcsec)).to(au.deg),
    (-( 68 * au.deg + 41 * au.arcmin + 09.000 * au.arcsec)).to(au.deg),
    +(3210.00) * au.m
)   # Green Bank Telescope
name_AL, lat_AL, lon_AL, hgt_AL = (
    "AL",
    -23.029 * au.deg,
    -67.755 * au.deg,
    5058.7 * au.m
)   # ALMA

"""
LBA
"""
name_AT, lat_AT, lon_AT, hgt_AT = (
    "AT",
    (-(30 * au.deg + 18 * au.arcmin + 46.333 * au.arcsec)).to(au.deg),
    (+(149 * au.deg + 33 * au.arcmin + 53.158 * au.arcsec)).to(au.deg),
    +( 252.02) * au.m
)   # ATCA
name_CD, lat_CD, lon_CD, hgt_CD = (
    "CD",
    (-(31 * au.deg + 52 * au.arcmin +  3.642 * au.arcsec)).to(au.deg),
    (+(133 * au.deg + 48 * au.arcmin + 35.420 * au.arcsec)).to(au.deg),
    +( 164.62) * au.m
)   # Ceduna 30-m
name_HO, lat_HO, lon_HO, hgt_HO = (
    "HO",
    (-(42 * au.deg + 48 * au.arcmin + 12.864 * au.arcsec)).to(au.deg),
    (+(147 * au.deg + 26 * au.arcmin + 25.880 * au.arcsec)).to(au.deg),
    +(  65.08) * au.m
)   # Hobart 26-m
name_MP, lat_MP, lon_MP, hgt_MP = (
    "MP",
    (-(31 * au.deg + 16 * au.arcmin +  4.071 * au.arcsec)).to(au.deg),
    (+(149 * au.deg +  5 * au.arcmin + 58.743 * au.arcsec)).to(au.deg),
    +( 867.32) * au.m
)   # Mopra 22-m
name_PA, lat_PA, lon_PA, hgt_PA = (
    "PA",
    (-(32 * au.deg + 59 * au.arcmin + 54.263 * au.arcsec)).to(au.deg),
    (+(148 * au.deg + 15 * au.arcmin + 48.636 * au.arcsec)).to(au.deg),
    +( 410.80) * au.m
)   # Parkes 64-m
name_TD, lat_TD, lon_TD, hgt_TD = (
    "TD",
    (-(35 * au.deg + 24 * au.arcmin +  8.730 * au.arcsec)).to(au.deg),
    (+(148 * au.deg + 58 * au.arcmin + 52.560 * au.arcsec)).to(au.deg),
    +( 688.80) * au.m
)   # Tidbinbilla 70-m

"""
JCMT
"""
name_JC, lat_JC, lon_JC, hgt_JC = (
    "JC",
    (+(19 * au.deg + 49 * au.arcmin + 22.000 * au.arcsec)).to(au.deg),
    (-(155 * au.deg + 28 * au.arcmin + 37.000 * au.arcsec)).to(au.deg),
    +(4092.00) * au.m
)   # JCMT

"""
CARMA
"""
name_CARMA, lat_CARMA, lon_CARMA, hgt_CARMA = (
    "CARMA",
    (+(37 * au.deg + 16 * au.arcmin + 49.000 * au.arcsec)).to(au.deg),
    (-(118 * au.deg + 8 * au.arcmin + 31.000 * au.arcsec)).to(au.deg),
    +(2196.00) * au.m
)   # CARMA

"""
SMT
"""
name_SMT, lat_SMT, lon_SMT, hgt_SMT = (
    "SMT",
    (+(32 * au.deg + 42 * au.arcmin + 6.000 * au.arcsec)).to(au.deg),
    (-(109 * au.deg + 53 * au.arcmin + 28.000 * au.arcsec)).to(au.deg),
    +(3185.00) * au.m
)   # SMT

"""
APEX
"""
name_AX, lat_AX, lon_AX, hgt_AX = (
    "AX",
    (-(23 * au.deg + 00 * au.arcmin + 21.000 * au.arcsec)).to(au.deg),
    (-(67 * au.deg + 45 * au.arcmin + 33.000 * au.arcsec)).to(au.deg),
    +(5050.00) * au.m
)   # SMT


dict_lat = {
    "KC":lat_KC, "KT":lat_KT, "KU":lat_KU, "KY":lat_KY,
    "KV":lat_KV, "KR":lat_KR, "KS":lat_KS,
    "NY":lat_NY, "TK":lat_TK, "HT":lat_HT, "YM":lat_YM, "T6":lat_T6,
    "SH":lat_SH, "UR":lat_UR, "KM":lat_KM, "VM":lat_VM, "VR":lat_VR,
    "VO":lat_VO, "VS":lat_VS,
    "JB":lat_JB, "EF":lat_EF, "MC":lat_MC, "NT":lat_NT, "O6":lat_O6,
    "TR":lat_TR, "MH":lat_MH, "YS":lat_YS, "HH":lat_HH, "SR":lat_SR,
    "RO":lat_RO, "NN":lat_NN, "PV":lat_PV,
    "BR":lat_BR, "FD":lat_FD, "HN":lat_HN, "KP":lat_KP, "LA":lat_LA,
    "MK":lat_MK, "NL":lat_NL, "OV":lat_OV, "PT":lat_PT, "SC":lat_SC,
    "AT":lat_AT, "CD":lat_CD, "HO":lat_HO, "MP":lat_MP, "PA":lat_PA,
    "TD":lat_TD,
    "YY":lat_YY, "GB":lat_GB, "GL":lat_GL, "AL":lat_AL,
    "JC":lat_JC,
    "CARMA":lat_CARMA, "SMT":lat_SMT, "AX":lat_AX
}

dict_lon = {
    "KC":lon_KC, "KT":lon_KT, "KU":lon_KU, "KY":lon_KY,
    "KV":lon_KV, "KR":lon_KR, "KS":lon_KS,
    "NY":lon_NY, "TK":lon_TK, "HT":lon_HT, "YM":lon_YM, "T6":lon_T6,
    "SH":lon_SH, "UR":lon_UR, "KM":lon_KM, "VM":lon_VM, "VR":lon_VR,
    "VO":lon_VO, "VS":lon_VS,
    "JB":lon_JB, "EF":lon_EF, "MC":lon_MC, "NT":lon_NT, "O6":lon_O6,
    "TR":lon_TR, "MH":lon_MH, "YS":lon_YS, "HH":lon_HH, "SR":lon_SR,
    "RO":lon_RO, "NN":lon_NN, "PV":lon_PV,
    "BR":lon_BR, "FD":lon_FD, "HN":lon_HN, "KP":lon_KP, "LA":lon_LA,
    "MK":lon_MK, "NL":lon_NL, "OV":lon_OV, "PT":lon_PT, "SC":lon_SC,
    "AT":lon_AT, "CD":lon_CD, "HO":lon_HO, "MP":lon_MP, "PA":lon_PA,
    "TD":lon_TD,
    "YY":lon_YY, "GB":lon_GB, "GL":lon_GL, "AL":lon_AL,
    "JC":lon_JC,
    "CARMA":lon_CARMA, "SMT":lon_SMT, "AX":lon_AX
}

dict_hgt = {
    "KC":hgt_KC, "KT":hgt_KT, "KU":hgt_KU, "KY":hgt_KY,
    "KV":hgt_KV, "KR":hgt_KR, "KS":hgt_KS,
    "NY":hgt_NY, "TK":hgt_TK, "HT":hgt_HT, "YM":hgt_YM, "T6":hgt_T6,
    "SH":hgt_SH, "UR":hgt_UR, "KM":hgt_KM, "VM":hgt_VM, "VR":hgt_VR,
    "VO":hgt_VO, "VS":hgt_VS,
    "JB":hgt_JB, "EF":hgt_EF, "MC":hgt_MC, "NT":hgt_NT, "O6":hgt_O6,
    "TR":hgt_TR, "MH":hgt_MH, "YS":hgt_YS, "HH":hgt_HH, "SR":hgt_SR,
    "RO":hgt_RO, "NN":hgt_NN, "PV":hgt_PV,
    "BR":hgt_BR, "FD":hgt_FD, "HN":hgt_HN, "KP":hgt_KP, "LA":hgt_LA,
    "MK":hgt_MK, "NL":hgt_NL, "OV":hgt_OV, "PT":hgt_PT, "SC":hgt_SC,
    "AT":hgt_AT, "CD":hgt_CD, "HO":hgt_HO, "MP":hgt_MP, "PA":hgt_PA,
    "TD":hgt_TD,
    "YY":hgt_YY, "GB":hgt_GB, "GL":hgt_GL, "AL":hgt_AL,
    "JC":hgt_JC,
    "CARMA":hgt_CARMA, "SMT":hgt_SMT, "AX":hgt_AX
}


def get_station(name):
    name_ = name.upper()
    ant = EarthLocation(
        lat=dict_lat[name_],
        lon=dict_lon[name_],
        height=dict_hgt[name_]
    )

    return ant
