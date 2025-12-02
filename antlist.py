
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
name_KC, lat_KC, lon_KC, hgt_KC =\
    (
        "KC",
        (+(37 * u.deg + 32 * u.arcmin + 00.100 * u.arcsec)).to(u.deg),
        (+(128 * u.deg + 26 * u.arcmin + 55.100 * u.arcsec)).to(u.deg),
        +( 557.00) * u.m
    )   # KVN Pyeongchang
name_KT, lat_KT, lon_KT, hgt_KT =\
    (
        "KT",
        (+(33 * u.deg + 17 * u.arcmin + 20.900 * u.arcsec)).to(u.deg),
        (+(126 * u.deg + 27 * u.arcmin + 34.400 * u.arcsec)).to(u.deg),
        +( 452.00) * u.m
    )   # KVN Tamna
name_KU, lat_KU, lon_KU, hgt_KU =\
    (
        "KU",
        (+(35 * u.deg + 32 * u.arcmin + 44.200 * u.arcsec)).to(u.deg),
        (+(129 * u.deg + 14 * u.arcmin + 59.300 * u.arcsec)).to(u.deg),
        +( 170.00) * u.m
    )   # KVN Ulsan
name_KY, lat_KY, lon_KY, hgt_KY =\
    (
        "KY",
        (+(37 * u.deg + 33 * u.arcmin + 54.900 * u.arcsec)).to(u.deg),
        (+(126 * u.deg + 56 * u.arcmin + 27.400 * u.arcsec)).to(u.deg),
        +( 139.00) * u.m
    )   # KVN Yonsei
name_KV, lat_KV, lon_KV, hgt_KV =\
    (
        "KV",
        (+(36 * u.deg + 31 * u.arcmin + 21.800 * u.arcsec)).to(u.deg),
        (+(127 * u.deg + 18 * u.arcmin + 12.000 * u.arcsec)).to(u.deg),
        +( 194.62) * u.m
    )   # Sejong (Sejong, South Korea)
name_KR, lat_KR, lon_KR, hgt_KR =\
    (
        "KR",
        (+(36 * u.deg + 23 * u.arcmin + 51.162 * u.arcsec)).to(u.deg),
        (+(127 * u.deg + 22 * u.arcmin + 30.591 * u.arcsec)).to(u.deg),
        +( 144.00) * u.m
    )   # TARO (Daejeon, South Korea)
name_KS, lat_KS, lon_KS, hgt_KS =\
    (
        "KS",
        (+(37 * u.deg + 27 * u.arcmin + 15.700 * u.arcsec)).to(u.deg),
        (+(126 * u.deg + 57 * u.arcmin + 19.000 * u.arcsec)).to(u.deg),
        +( 204.00) * u.m
    )   # SRAO (Seoul, South Korea)

name_NY, lat_NY, lon_NY, hgt_NY =\
    (
        "NY",
        (+(35 * u.deg + 56 * u.arcmin + 40.900 * u.arcsec)).to(u.deg),
        (+(138 * u.deg + 28 * u.arcmin + 21.200 * u.arcsec)).to(u.deg),
        +(1350.00) * u.m
    )   # Nobeyama 45-m
name_TK, lat_TK, lon_TK, hgt_TK =\
    (
        "TK",
        (+(36 * u.deg + 41 * u.arcmin + 54.500 * u.arcsec)).to(u.deg),
        (+(140 * u.deg + 41 * u.arcmin + 41.000 * u.arcsec)).to(u.deg),
        +( 117.10) * u.m
    )   # Takahagi 32-m
name_HT, lat_HT, lon_HT, hgt_HT =\
    (
        "HT",
        (+(36 * u.deg + 41 * u.arcmin + 50.800 * u.arcsec)).to(u.deg),
        (+(140 * u.deg + 41 * u.arcmin + 31.600 * u.arcsec)).to(u.deg),
        +( 120.20) * u.m
    )   # Hitachi 32-m
name_YM, lat_YM, lon_YM, hgt_YM =\
    (
        "YM",
        (+(34 * u.deg + 12 * u.arcmin + 57.700 * u.arcsec)).to(u.deg),
        (+(131 * u.deg + 33 * u.arcmin + 25.500 * u.arcsec)).to(u.deg),
        +( 133.00) * u.m
    )   # Yamaguchi 32-m
name_T6, lat_T6, lon_T6, hgt_T6 =\
    (
        "T6",
        (+(31 * u.deg +  5 * u.arcmin + 31.600 * u.arcsec)).to(u.deg),
        (+(121 * u.deg +  8 * u.arcmin +  9.400 * u.arcsec)).to(u.deg),
        +(  49.20) * u.m
    )   # Tianma 65-m
name_SH, lat_SH, lon_SH, hgt_SH =\
    (
        "SH",
        (+(31 * u.deg +  5 * u.arcmin + 57.000 * u.arcsec)).to(u.deg),
        (+(121 * u.deg + 11 * u.arcmin + 58.800 * u.arcsec)).to(u.deg),
        +(  29.40) * u.m
    )   # Sheshan 25-m
name_UR, lat_UR, lon_UR, hgt_UR =\
    (
        "UR",
        (+(43 * u.deg + 28 * u.arcmin + 15.600 * u.arcsec)).to(u.deg),
        (+( 87 * u.deg + 10 * u.arcmin + 40.400 * u.arcsec)).to(u.deg),
        +(2029.40) * u.m
    )   # Nanshan 25-m (Urumqi)
name_KM, lat_KM, lon_KM, hgt_KM =\
    (
        "KM",
        (+(25 * u.deg +  1 * u.arcmin + 40.800 * u.arcsec)).to(u.deg),
        (+(102 * u.deg + 47 * u.arcmin + 45.600 * u.arcsec)).to(u.deg),
        +(1974.00) * u.m
    )   # Kunming 40-m
name_VM, lat_VM, lon_VM, hgt_VM =\
    (
        "VM",
        (+(39 * u.deg +  8 * u.arcmin +  0.700 * u.arcsec)).to(u.deg),
        (+(141 * u.deg +  7 * u.arcmin + 57.300 * u.arcsec)).to(u.deg),
        +( 116.60) * u.m
    )   # VERA Mizusawa
name_VR, lat_VR, lon_VR, hgt_VR =\
    (
        "VR",
        (+(31 * u.deg + 44 * u.arcmin + 52.400 * u.arcsec)).to(u.deg),
        (+(130 * u.deg + 26 * u.arcmin + 23.600 * u.arcsec)).to(u.deg),
        +( 573.60) * u.m
    )   # VERA Iriki
name_VO, lat_VO, lon_VO, hgt_VO =\
    (
        "VO",
        (+(27 * u.deg +  5 * u.arcmin + 30.500 * u.arcsec)).to(u.deg),
        (+(142 * u.deg + 12 * u.arcmin + 59.800 * u.arcsec)).to(u.deg),
        +( 273.10) * u.m
    )   # VERA Ogasawara
name_VS, lat_VS, lon_VS, hgt_VS =\
    (
        "VS",
        (+(24 * u.deg + 24 * u.arcmin + 43.800 * u.arcsec)).to(u.deg),
        (+(124 * u.deg + 10 * u.arcmin + 15.600 * u.arcsec)).to(u.deg),
        +(  65.10) * u.m
    )   # VERA Ishigakijima

"""
EVN
"""
name_JB, lat_JB, lon_JB, hgt_JB =\
    (
        "JB",
        (+(53 * u.deg + 14 * u.arcmin +  2.281 * u.arcsec)).to(u.deg),
        (-(  2 * u.deg + 18 * u.arcmin + 14.031 * u.arcsec)).to(u.deg),
        +( 143.77) * u.m
    )   # Jodrell Bank
name_EF, lat_EF, lon_EF, hgt_EF =\
    (
        "EF",
        (+(50 * u.deg + 31 * u.arcmin + 29.410 * u.arcsec)).to(u.deg),
        (+(  6 * u.deg + 53 * u.arcmin +  1.022 * u.arcsec)).to(u.deg),
        +( 416.72) * u.m
    )   # Effelsberg 100-m
name_MC, lat_MC, lon_MC, hgt_MC =\
    (
        "MC",
        (+(44 * u.deg + 31 * u.arcmin + 13.788 * u.arcsec)).to(u.deg),
        (+( 11 * u.deg + 38 * u.arcmin + 48.984 * u.arcsec)).to(u.deg),
        +(  67.14) * u.m
    )   # Medicina 32-m
name_NT, lat_NT, lon_NT, hgt_NT =\
    (
        "NT",
        (+(36 * u.deg + 52 * u.arcmin + 33.796 * u.arcsec)).to(u.deg),
        (+( 14 * u.deg + 59 * u.arcmin + 20.594 * u.arcsec)).to(u.deg),
        +( 143.21) * u.m
    )   # Noto 32-m
name_O6, lat_O6, lon_O6, hgt_O6 =\
    (
        "O6",
        (+(57 * u.deg + 23 * u.arcmin + 45.023 * u.arcsec)).to(u.deg),
        (+( 11 * u.deg + 55 * u.arcmin + 34.902 * u.arcsec)).to(u.deg),
        +(  59.35) * u.m
    )   # Onsala60 20-m
name_TR, lat_TR, lon_TR, hgt_TR =\
    (
        "TR",
        (+(53 * u.deg +  5 * u.arcmin + 43.673 * u.arcsec)).to(u.deg),
        (+( 18 * u.deg + 33 * u.arcmin + 50.635 * u.arcsec)).to(u.deg),
        +( 133.61) * u.m
    )   # Torun 32-m
name_MH, lat_MH, lon_MH, hgt_MH =\
    (
        "MH",
        (+(60 * u.deg + 13 * u.arcmin +  4.122 * u.arcsec)).to(u.deg),
        (+( 24 * u.deg + 23 * u.arcmin + 35.227 * u.arcsec)).to(u.deg),
        +(  80.02) * u.m
    )   # Medicina 32-m
name_YS, lat_YS, lon_YS, hgt_YS =\
    (
        "YS",
        (+(40 * u.deg + 31 * u.arcmin + 28.810 * u.arcsec)).to(u.deg),
        (-(  3 * u.deg +  5 * u.arcmin + 12.683 * u.arcsec)).to(u.deg),
        +( 988.96) * u.m
    )   # Yebes 40-m
name_HH, lat_HH, lon_HH, hgt_HH =\
    (
        "HH",
        (-(25 * u.deg + 53 * u.arcmin + 23.091 * u.arcsec)).to(u.deg),
        (+( 27 * u.deg + 41 * u.arcmin +  7.430 * u.arcsec)).to(u.deg),
        +(1415.72) * u.m
    )   # Hartebeesthoek 26-m
name_SR, lat_SR, lon_SR, hgt_SR =\
    (
        "SR",
        (+(39 * u.deg + 29 * u.arcmin + 35.028 * u.arcsec)).to(u.deg),
        (+(  9 * u.deg + 14 * u.arcmin + 42.543 * u.arcsec)).to(u.deg),
        +( 671.47) * u.m
    )   # Sardinia 65-m
name_RO, lat_RO, lon_RO, hgt_RO =\
    (
        "RO",
        (+(40 * u.deg + 25 * u.arcmin + 52.362 * u.arcsec)).to(u.deg),
        (-(  4 * u.deg + 14 * u.arcmin + 52.817 * u.arcsec)).to(u.deg),
        +( 864.90) * u.m
    )   # Robledo 70-m

"""
NRAO
"""
name_BR, lat_BR, lon_BR, hgt_BR =\
    (
        "BR",
        (+(48 * u.deg +  7 * u.arcmin + 52.401 * u.arcsec)).to(u.deg),
        (-(119 * u.deg + 40 * u.arcmin + 59.815 * u.arcsec)).to(u.deg),
        +( 250.47) * u.m
    )   # VLBA Brewster
name_FD, lat_FD, lon_FD, hgt_FD =\
    (
        "FD",
        (+(30 * u.deg + 38 * u.arcmin +  6.102 * u.arcsec)).to(u.deg),
        (-(103 * u.deg + 56 * u.arcmin + 41.366 * u.arcsec)).to(u.deg),
        +(1606.42) * u.m
    )   # VLBA Fort Davis
name_HN, lat_HN, lon_HN, hgt_HN =\
    (
        "HN",
        (+(42 * u.deg + 56 * u.arcmin +  0.999 * u.arcsec)).to(u.deg),
        (-( 71 * u.deg + 59 * u.arcmin + 11.704 * u.arcsec)).to(u.deg),
        +( 295.57) * u.m
    )   # VLBA Hancock
name_KP, lat_KP, lon_KP, hgt_KP =\
    (
        "KP",
        (+(31 * u.deg + 57 * u.arcmin + 22.689 * u.arcsec)).to(u.deg),
        (-(111 * u.deg + 36 * u.arcmin + 44.733 * u.arcsec)).to(u.deg),
        +(1901.99) * u.m
    )   # VLBA Kitt Peak
name_LA, lat_LA, lon_LA, hgt_LA =\
    (
        "LA",
        (+(35 * u.deg + 46 * u.arcmin + 30.439 * u.arcsec)).to(u.deg),
        (-(106 * u.deg + 14 * u.arcmin + 44.158 * u.arcsec)).to(u.deg),
        +(1962.43) * u.m
    )   # VLBA Los Alamos
name_MK, lat_MK, lon_MK, hgt_MK =\
    (
        "MK",
        (+(19 * u.deg + 48 * u.arcmin +  5.000 * u.arcsec)).to(u.deg),
        (-(155 * u.deg + 27 * u.arcmin + 19.864 * u.arcsec)).to(u.deg),
        +(3762.99) * u.m
    )   # VLBA Mauna Kea
name_NL, lat_NL, lon_NL, hgt_NL =\
    (
        "NL",
        (+(41 * u.deg + 46 * u.arcmin + 17.128 * u.arcsec)).to(u.deg),
        (-( 91 * u.deg + 34 * u.arcmin + 26.911 * u.arcsec)).to(u.deg),
        +( 222.21) * u.m
    )   # VLBA North Liberty
name_OV, lat_OV, lon_OV, hgt_OV =\
    (
        "OV",
        (+(37 * u.deg + 13 * u.arcmin + 53.938 * u.arcsec)).to(u.deg),
        (-(118 * u.deg + 16 * u.arcmin + 37.414 * u.arcsec)).to(u.deg),
        +(1196.31) * u.m
    )   # VLBA Owens Valley
name_PT, lat_PT, lon_PT, hgt_PT =\
    (
        "PT",
        (+(34 * u.deg + 18 * u.arcmin +  3.657 * u.arcsec)).to(u.deg),
        (-(108 * u.deg +  7 * u.arcmin +  9.095 * u.arcsec)).to(u.deg),
        +(2364.68) * u.m
    )   # VLBA Pie Town
name_SC, lat_SC, lon_SC, hgt_SC =\
    (
        "SC",
        (+(17 * u.deg + 45 * u.arcmin + 23.703 * u.arcsec)).to(u.deg),
        (-( 64 * u.deg + 35 * u.arcmin +  1.069 * u.arcsec)).to(u.deg),
        -(  15.02) * u.m
    )   # VLBA St. Croix
name_YY, lat_YY, lon_YY, hgt_YY =\
    (
        "YY",
        (+(34 * u.deg +  4 * u.arcmin + 43.725 * u.arcsec)).to(u.deg),
        (-(107 * u.deg + 37 * u.arcmin +  6.013 * u.arcsec)).to(u.deg),
        +(2114.48) * u.m
    )   # Phased-VLA
name_GB, lat_GB, lon_GB, hgt_GB =\
    (
        "GB",
        (+(38 * u.deg + 25 * u.arcmin + 59.267 * u.arcsec)).to(u.deg),
        (-( 79 * u.deg + 50 * u.arcmin + 23.433 * u.arcsec)).to(u.deg),
        +( 823.66) * u.m
    )   # Green Bank Telescope
name_AL, lat_AL, lon_AL, hgt_AL =\
    (
        "AL",
        -23.029 * u.deg,
        -67.755 * u.deg,
        5058.7 * u.m)
  # ALMA

"""
LBA
"""
name_AT, lat_AT, lon_AT, hgt_AT =\
    (
        "AT",
        (-(30 * u.deg + 18 * u.arcmin + 46.333 * u.arcsec)).to(u.deg),
        (+(149 * u.deg + 33 * u.arcmin + 53.158 * u.arcsec)).to(u.deg),
        +( 252.02) * u.m
    )   # ATCA
name_CD, lat_CD, lon_CD, hgt_CD =\
    (
        "CD",
        (-(31 * u.deg + 52 * u.arcmin +  3.642 * u.arcsec)).to(u.deg),
        (+(133 * u.deg + 48 * u.arcmin + 35.420 * u.arcsec)).to(u.deg),
        +( 164.62) * u.m
    )   # Ceduna 30-m
name_HO, lat_HO, lon_HO, hgt_HO =\
    (
        "HO",
        (-(42 * u.deg + 48 * u.arcmin + 12.864 * u.arcsec)).to(u.deg),
        (+(147 * u.deg + 26 * u.arcmin + 25.880 * u.arcsec)).to(u.deg),
        +(  65.08) * u.m
    )   # Hobart 26-m
name_MP, lat_MP, lon_MP, hgt_MP =\
    (
        "MP",
        (-(31 * u.deg + 16 * u.arcmin +  4.071 * u.arcsec)).to(u.deg),
        (+(149 * u.deg +  5 * u.arcmin + 58.743 * u.arcsec)).to(u.deg),
        +( 867.32) * u.m
    )   # Mopra 22-m
name_PA, lat_PA, lon_PA, hgt_PA =\
    (
        "PA",
        (-(32 * u.deg + 59 * u.arcmin + 54.263 * u.arcsec)).to(u.deg),
        (+(148 * u.deg + 15 * u.arcmin + 48.636 * u.arcsec)).to(u.deg),
        +( 410.80) * u.m
    )   # Parkes 64-m
name_TD, lat_TD, lon_TD, hgt_TD =\
    (
        "TD",
        (-(35 * u.deg + 24 * u.arcmin +  8.730 * u.arcsec)).to(u.deg),
        (+(148 * u.deg + 58 * u.arcmin + 52.560 * u.arcsec)).to(u.deg),
        +( 688.80) * u.m
    )   # Tidbinbilla 70-m

"""
JCMT
"""
name_JC, lat_JC, lon_JC, hgt_JC =\
    (
        "JC",
        (+(19 * u.deg + 49 * u.arcmin + 22.000 * u.arcsec)).to(u.deg),
        (-(155 * u.deg + 28 * u.arcmin + 37.000 * u.arcsec)).to(u.deg),
        +(4092.00) * u.m
    )   # JCMT

"""
CARMA
"""
name_CARMA, lat_CARMA, lon_CARMA, hgt_CARMA =\
    (
        "CARMA",
        (+(37 * u.deg + 16 * u.arcmin + 49.000 * u.arcsec)).to(u.deg),
        (-(118 * u.deg + 8 * u.arcmin + 31.000 * u.arcsec)).to(u.deg),
        +(2196.00) * u.m
    )   # CARMA

"""
SMT
"""
name_SMT, lat_SMT, lon_SMT, hgt_SMT =\
    (
        "SMT",
        (+(32 * u.deg + 42 * u.arcmin + 6.000 * u.arcsec)).to(u.deg),
        (-(109 * u.deg + 53 * u.arcmin + 28.000 * u.arcsec)).to(u.deg),
        +(3185.00) * u.m
    )   # SMT


dict_lat =\
    {
        "KC":lat_KC, "KT":lat_KT, "KU":lat_KU, "KY":lat_KY,
        "KV":lat_KV, "KR":lat_KR, "KS":lat_KS,
        "NY":lat_NY, "TK":lat_TK, "HT":lat_HT, "YM":lat_YM, "T6":lat_T6,
        "SH":lat_SH, "UR":lat_UR, "KM":lat_KM, "VM":lat_VM, "VR":lat_VR,
        "VO":lat_VO, "VS":lat_VS,
        "JB":lat_JB, "EF":lat_EF, "MC":lat_MC, "NT":lat_NT, "O6":lat_O6,
        "TR":lat_TR, "MH":lat_MH, "YS":lat_YS, "HH":lat_HH, "SR":lat_SR,
        "RO":lat_RO,
        "BR":lat_BR, "FD":lat_FD, "HN":lat_HN, "KP":lat_KP, "LA":lat_LA,
        "MK":lat_MK, "NL":lat_NL, "OV":lat_OV, "PT":lat_PT, "SC":lat_SC,
        "AT":lat_AT, "CD":lat_CD, "HO":lat_HO, "MP":lat_MP, "PA":lat_PA,
        "TD":lat_TD,
        "YY":lat_YY, "GB":lat_GB, "AL":lat_AL,
        "JC":lat_JC,
        "CARMA":lat_CARMA, "SMT":lat_SMT
    }

dict_lon =\
    {
        "KC":lon_KC, "KT":lon_KT, "KU":lon_KU, "KY":lon_KY,
        "KV":lon_KV, "KR":lon_KR, "KS":lon_KS,
        "NY":lon_NY, "TK":lon_TK, "HT":lon_HT, "YM":lon_YM, "T6":lon_T6,
        "SH":lon_SH, "UR":lon_UR, "KM":lon_KM, "VM":lon_VM, "VR":lon_VR,
        "VO":lon_VO, "VS":lon_VS,
        "JB":lon_JB, "EF":lon_EF, "MC":lon_MC, "NT":lon_NT, "O6":lon_O6,
        "TR":lon_TR, "MH":lon_MH, "YS":lon_YS, "HH":lon_HH, "SR":lon_SR,
        "RO":lon_RO,
        "BR":lon_BR, "FD":lon_FD, "HN":lon_HN, "KP":lon_KP, "LA":lon_LA,
        "MK":lon_MK, "NL":lon_NL, "OV":lon_OV, "PT":lon_PT, "SC":lon_SC,
        "AT":lon_AT, "CD":lon_CD, "HO":lon_HO, "MP":lon_MP, "PA":lon_PA,
        "TD":lon_TD,
        "YY":lon_YY, "GB":lon_GB, "AL":lon_AL,
        "JC":lon_JC,
        "CARMA":lon_CARMA, "SMT":lon_SMT
    }

dict_hgt =\
    {
        "KC":hgt_KC, "KT":hgt_KT, "KU":hgt_KU, "KY":hgt_KY,
        "KV":hgt_KV, "KR":hgt_KR, "KS":hgt_KS,
        "NY":hgt_NY, "TK":hgt_TK, "HT":hgt_HT, "YM":hgt_YM, "T6":hgt_T6,
        "SH":hgt_SH, "UR":hgt_UR, "KM":hgt_KM, "VM":hgt_VM, "VR":hgt_VR,
        "VO":hgt_VO, "VS":hgt_VS,
        "JB":hgt_JB, "EF":hgt_EF, "MC":hgt_MC, "NT":hgt_NT, "O6":hgt_O6,
        "TR":hgt_TR, "MH":hgt_MH, "YS":hgt_YS, "HH":hgt_HH, "SR":hgt_SR,
        "RO":hgt_RO,
        "BR":hgt_BR, "FD":hgt_FD, "HN":hgt_HN, "KP":hgt_KP, "LA":hgt_LA,
        "MK":hgt_MK, "NL":hgt_NL, "OV":hgt_OV, "PT":hgt_PT, "SC":hgt_SC,
        "AT":hgt_AT, "CD":hgt_CD, "HO":hgt_HO, "MP":hgt_MP, "PA":hgt_PA,
        "TD":hgt_TD,
        "YY":hgt_YY, "GB":hgt_GB, "AL":hgt_AL,
        "JC":hgt_JC,
        "CARMA":hgt_CARMA, "SMT":hgt_SMT
    }


def get_station(name):
    name_ = name.upper()
    ant =\
        EarthLocation(
            lat=dict_lat[name_],
            lon=dict_lon[name_],
            height=dict_hgt[name_]
        )
    return ant

def get_vlbi(name):
    arr_xpo = np.array([])
    arr_ypo = np.array([])
    arr_zpo = np.array([])
    arr_lat = np.array([])
    arr_lon = np.array([])
    arr_hgt = np.array([])

    mask_list = isinstance(name, (list, tuple, np.ndarray))
    if mask_list:
        VLBIarr = name
    else:
        if not isinstance(name, str):
            raise TypeError("The name must be a string or a list of strings.")
        if name.upper() == "KVN":
            VLBIarr =\
                [
                    "KT", "KU", "KY"
                ]
        elif name.upper() == "EKVN":
            VLBIarr =\
                [
                    "KC", "KT", "KU", "KY"
                ]
        elif name.upper() == "EKVN+KR":
            VLBIarr =\
                [
                    "KC", "KT", "KU", "KY", "KR"
                ]
        elif name.upper() == "EKVN+KR+KV":
            VLBIarr =\
                [
                    "KC", "KT", "KU", "KY", "KV", "KR"
                ]
        elif name.upper() == "KVNMP":
            VLBIarr =\
                [
                    "KT", "KU", "KY", "MP"
                ]
        elif name.upper() == "EKVNMP":
            VLBIarr =\
                [
                    "KC", "KT", "KU", "KY", "MP"
                ]
        elif name.upper() == "VLBA":
            VLBIarr =\
                [
                    "BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC"
                ]
        elif name.upper() == "KAVA":
            VLBIarr =\
                [
                    "KC", "KT", "KU", "KY", "VM", "VR", "VO", "VS"
                ]
        elif name.upper() == "EAVN":
            VLBIarr =\
                [
                    "KC", "KT", "KU", "KY", "VM", "VR", "VO", "VS", "UR"
                ]
        elif name.upper() == "EVN":
            VLBIarr =\
                [
                    "JB", "EF", "MC", "NT", "O6", "T6", "UR", "KR", "MH", "YS",
                    "HH", "SR"
                ]
        elif name.upper() == "VLBA":
            VLBIarr =\
                [
                    "BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC"
                ]
        elif name.upper() == "NRAO":
            VLBIarr =\
                [
                    "BR", "FD", "HN", "KP", "LA", "MK", "NL", "OV", "PT", "SC",
                    "YY", "GB"
                ]
        elif name.upper() == "LBA":
            VLBIarr =\
                [
                    "AT", "CD", "HO", "MP", "PA", "TD"
                ]

    arr_nam = np.array(list(map(lambda x: x.upper(), VLBIarr)))
    for arr_ in VLBIarr:
        arr = arr_.upper()
        loc =\
            EarthLocation(
                lat=dict_lat[arr],
                lon=dict_lon[arr],
                height=dict_hgt[arr]
            )
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
