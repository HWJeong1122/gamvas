
from astropy import units as u
from astropy.coordinates import SkyCoord

source_positions = {

"3C66A"     : SkyCoord(( 2*u.hour+22*u.min+39.611*u.s).to(u.hour).value*15*u.deg, (+(43*u.deg+ 2*u.arcmin+ 7.799*u.arcsec)).to(u.deg)),
"0219+428"  : SkyCoord(( 2*u.hour+22*u.min+39.611*u.s).to(u.hour).value*15*u.deg, (+(43*u.deg+ 2*u.arcmin+ 7.799*u.arcsec)).to(u.deg)),

"0235+16"   : SkyCoord(( 2*u.hour+38*u.min+38.930*u.s).to(u.hour).value*15*u.deg, (+(16*u.deg+36*u.arcmin+59.275*u.arcsec)).to(u.deg)),
"0235+164"  : SkyCoord(( 2*u.hour+38*u.min+38.930*u.s).to(u.hour).value*15*u.deg, (+(16*u.deg+36*u.arcmin+59.275*u.arcsec)).to(u.deg)),

"3C84"      : SkyCoord(( 3*u.hour+19*u.min+48.160*u.s).to(u.hour).value*15*u.deg, (+(41*u.deg+30*u.arcmin+42.106*u.arcsec)).to(u.deg)),
"0316+413"  : SkyCoord(( 3*u.hour+19*u.min+48.160*u.s).to(u.hour).value*15*u.deg, (+(41*u.deg+30*u.arcmin+42.106*u.arcsec)).to(u.deg)),

"NRAO140"   : SkyCoord(( 3*u.hour+36*u.min+30.108*u.s).to(u.hour).value*15*u.deg, (+(32*u.deg+18*u.arcmin+29.342*u.arcsec)).to(u.deg)),
"0333+321"  : SkyCoord(( 3*u.hour+36*u.min+30.108*u.s).to(u.hour).value*15*u.deg, (+(32*u.deg+18*u.arcmin+29.342*u.arcsec)).to(u.deg)),

"NRAO150"   : SkyCoord(( 3*u.hour+59*u.min+29.747*u.s).to(u.hour).value*15*u.deg, (+(50*u.deg+57*u.arcmin+50.162*u.arcsec)).to(u.deg)),
"0355+508"  : SkyCoord(( 3*u.hour+59*u.min+29.747*u.s).to(u.hour).value*15*u.deg, (+(50*u.deg+57*u.arcmin+50.162*u.arcsec)).to(u.deg)),

"3C111"     : SkyCoord(( 4*u.hour+18*u.min+21.277*u.s).to(u.hour).value*15*u.deg, (+(38*u.deg+ 1*u.arcmin+35.800*u.arcsec)).to(u.deg)),
"0415+379"  : SkyCoord(( 4*u.hour+18*u.min+21.277*u.s).to(u.hour).value*15*u.deg, (+(38*u.deg+ 1*u.arcmin+35.800*u.arcsec)).to(u.deg)),

"0420-01"   : SkyCoord(( 4*u.hour+23*u.min+15.801*u.s).to(u.hour).value*15*u.deg, (-( 1*u.deg+20*u.arcmin+33.066*u.arcsec)).to(u.deg)),
"0420-014"  : SkyCoord(( 4*u.hour+23*u.min+15.801*u.s).to(u.hour).value*15*u.deg, (-( 1*u.deg+20*u.arcmin+33.066*u.arcsec)).to(u.deg)),

"3C120"     : SkyCoord(( 4*u.hour+33*u.min+11.096*u.s).to(u.hour).value*15*u.deg, (+( 5*u.deg+21*u.arcmin+15.619*u.arcsec)).to(u.deg)),
"0430+052"  : SkyCoord(( 4*u.hour+33*u.min+11.096*u.s).to(u.hour).value*15*u.deg, (+( 5*u.deg+21*u.arcmin+15.619*u.arcsec)).to(u.deg)),

"0446+112"  : SkyCoord(( 4*u.hour+49*u.min+ 7.671*u.s).to(u.hour).value*15*u.deg, (+(11*u.deg+21*u.arcmin+28.596*u.arcsec)).to(u.deg)),

"0506+056"  : SkyCoord(( 5*u.hour+ 9*u.min+25.964*u.s).to(u.hour).value*15*u.deg, (+( 5*u.deg+41*u.arcmin+35.334*u.arcsec)).to(u.deg)),

"0528+134"  : SkyCoord(( 5*u.hour+30*u.min+56.417*u.s).to(u.hour).value*15*u.deg, (+(13*u.deg+31*u.arcmin+55.150*u.arcsec)).to(u.deg)),

"0552+398"  : SkyCoord(( 5*u.hour+55*u.min+30.806*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+48*u.arcmin+49.165*u.arcsec)).to(u.deg)),
"DA193"     : SkyCoord(( 5*u.hour+55*u.min+30.806*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+48*u.arcmin+49.165*u.arcsec)).to(u.deg)),

"0716+714"  : SkyCoord(( 7*u.hour+21*u.min+53.448*u.s).to(u.hour).value*15*u.deg, (+(71*u.deg+20*u.arcmin+36.363*u.arcsec)).to(u.deg)),

"OI158"     : SkyCoord(( 7*u.hour+38*u.min+ 7.394*u.s).to(u.hour).value*15*u.deg, (+(17*u.deg+42*u.arcmin+18.998*u.arcsec)).to(u.deg)),
"0735+178"  : SkyCoord(( 7*u.hour+38*u.min+ 7.394*u.s).to(u.hour).value*15*u.deg, (+(17*u.deg+42*u.arcmin+18.998*u.arcsec)).to(u.deg)),

"0738+313"  : SkyCoord(( 7*u.hour+41*u.min+10.703*u.s).to(u.hour).value*15*u.deg, (+(31*u.deg+12*u.arcmin+ 0.229*u.arcsec)).to(u.deg)),

"4C71.07"   : SkyCoord(( 8*u.hour+41*u.min+24.365*u.s).to(u.hour).value*15*u.deg, (+(70*u.deg+53*u.arcmin+42.173*u.arcsec)).to(u.deg)),
"0836+710"  : SkyCoord(( 8*u.hour+41*u.min+24.365*u.s).to(u.hour).value*15*u.deg, (+(70*u.deg+53*u.arcmin+42.173*u.arcsec)).to(u.deg)),

"OJ287"     : SkyCoord(( 8*u.hour+54*u.min+48.875*u.s).to(u.hour).value*15*u.deg, (+(20*u.deg+ 6*u.arcmin+30.641*u.arcsec)).to(u.deg)),
"0851+202"  : SkyCoord(( 8*u.hour+54*u.min+48.875*u.s).to(u.hour).value*15*u.deg, (+(20*u.deg+ 6*u.arcmin+30.641*u.arcsec)).to(u.deg)),

"4C39.25"   : SkyCoord(( 9*u.hour+27*u.min+ 3.014*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+ 2*u.arcmin+20.852*u.arcsec)).to(u.deg)),
"0923+392"  : SkyCoord(( 9*u.hour+27*u.min+ 3.014*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+ 2*u.arcmin+20.852*u.arcsec)).to(u.deg)),

"0954+65"   : SkyCoord(( 9*u.hour+58*u.min+47.245*u.s).to(u.hour).value*15*u.deg, (+(65*u.deg+33*u.arcmin+54.818*u.arcsec)).to(u.deg)),
"0954+658"  : SkyCoord(( 9*u.hour+58*u.min+47.245*u.s).to(u.hour).value*15*u.deg, (+(65*u.deg+33*u.arcmin+54.818*u.arcsec)).to(u.deg)),

"1055+018"  : SkyCoord((10*u.hour+58*u.min+29.605*u.s).to(u.hour).value*15*u.deg, (+( 1*u.deg+33*u.arcmin+58.824*u.arcsec)).to(u.deg)),

"MRK421"    : SkyCoord((11*u.hour+ 4*u.min+27.314*u.s).to(u.hour).value*15*u.deg, (+(38*u.deg+12*u.arcmin+31.799*u.arcsec)).to(u.deg)),
"1101+384"  : SkyCoord((11*u.hour+ 4*u.min+27.314*u.s).to(u.hour).value*15*u.deg, (+(38*u.deg+12*u.arcmin+31.799*u.arcsec)).to(u.deg)),

"1127-145"  : SkyCoord((11*u.hour+30*u.min+ 7.053*u.s).to(u.hour).value*15*u.deg, (-(14*u.deg+49*u.arcmin+27.388*u.arcsec)).to(u.deg)),

"4C29.45"   : SkyCoord((11*u.hour+59*u.min+31.834*u.s).to(u.hour).value*15*u.deg, (+(29*u.deg+14*u.arcmin+43.827*u.arcsec)).to(u.deg)),
"TON599"    : SkyCoord((11*u.hour+59*u.min+31.834*u.s).to(u.hour).value*15*u.deg, (+(29*u.deg+14*u.arcmin+43.827*u.arcsec)).to(u.deg)),
"1156+295"  : SkyCoord((11*u.hour+59*u.min+31.834*u.s).to(u.hour).value*15*u.deg, (+(29*u.deg+14*u.arcmin+43.827*u.arcsec)).to(u.deg)),

"4C21.35"   : SkyCoord((12*u.hour+24*u.min+54.458*u.s).to(u.hour).value*15*u.deg, (+(21*u.deg+22*u.arcmin+46.388*u.arcsec)).to(u.deg)),
"1222+216"  : SkyCoord((12*u.hour+24*u.min+54.458*u.s).to(u.hour).value*15*u.deg, (+(21*u.deg+22*u.arcmin+46.388*u.arcsec)).to(u.deg)),

"3C273"     : SkyCoord((12*u.hour+29*u.min+ 6.700*u.s).to(u.hour).value*15*u.deg, (+( 2*u.deg+ 3*u.arcmin+ 8.598*u.arcsec)).to(u.deg)),
"1226+023"  : SkyCoord((12*u.hour+29*u.min+ 6.700*u.s).to(u.hour).value*15*u.deg, (+( 2*u.deg+ 3*u.arcmin+ 8.598*u.arcsec)).to(u.deg)),

"M87"       : SkyCoord((12*u.hour+30*u.min+49.423*u.s).to(u.hour).value*15*u.deg, (+(12*u.deg+23*u.arcmin+28.044*u.arcsec)).to(u.deg)),
"1228+126"  : SkyCoord((12*u.hour+30*u.min+49.423*u.s).to(u.hour).value*15*u.deg, (+(12*u.deg+23*u.arcmin+28.044*u.arcsec)).to(u.deg)),

"3C279"     : SkyCoord((12*u.hour+56*u.min+11.167*u.s).to(u.hour).value*15*u.deg, (-( 5*u.deg+47*u.arcmin+21.525*u.arcsec)).to(u.deg)),
"1253-055"  : SkyCoord((12*u.hour+56*u.min+11.167*u.s).to(u.hour).value*15*u.deg, (-( 5*u.deg+47*u.arcmin+21.525*u.arcsec)).to(u.deg)),

"OP313"     : SkyCoord((13*u.hour+10*u.min+28.664*u.s).to(u.hour).value*15*u.deg, (+(32*u.deg+20*u.arcmin+43.783*u.arcsec)).to(u.deg)),
"1308+326"  : SkyCoord((13*u.hour+10*u.min+28.664*u.s).to(u.hour).value*15*u.deg, (+(32*u.deg+20*u.arcmin+43.783*u.arcsec)).to(u.deg)),

"3C286"     : SkyCoord((13*u.hour+31*u.min+ 8.288*u.s).to(u.hour).value*15*u.deg, (+(30*u.deg+30*u.arcmin+32.959*u.arcsec)).to(u.deg)),
"1328+307"  : SkyCoord((13*u.hour+31*u.min+ 8.288*u.s).to(u.hour).value*15*u.deg, (+(30*u.deg+30*u.arcmin+32.959*u.arcsec)).to(u.deg)),

"1510-089"  : SkyCoord((15*u.hour+12*u.min+50.533*u.s).to(u.hour).value*15*u.deg, (-( 9*u.deg+ 5*u.arcmin+59.830*u.arcsec)).to(u.deg)),

"APLibrae"  : SkyCoord((15*u.hour+17*u.min+41.813*u.s).to(u.hour).value*15*u.deg, (-(24*u.deg+22*u.arcmin+19.476*u.arcsec)).to(u.deg)),
"1514-241"  : SkyCoord((15*u.hour+17*u.min+41.813*u.s).to(u.hour).value*15*u.deg, (-(24*u.deg+22*u.arcmin+19.476*u.arcsec)).to(u.deg)),

"1553+113"  : SkyCoord((15*u.hour+55*u.min+43.044*u.s).to(u.hour).value*15*u.deg, (+(11*u.deg+11*u.arcmin+24.366*u.arcsec)).to(u.deg)),

"4C38.41"   : SkyCoord((16*u.hour+35*u.min+15.493*u.s).to(u.hour).value*15*u.deg, (+(38*u.deg+ 8*u.arcmin+ 4.501*u.arcsec)).to(u.deg)),
"1633+382"  : SkyCoord((16*u.hour+35*u.min+15.493*u.s).to(u.hour).value*15*u.deg, (+(38*u.deg+ 8*u.arcmin+ 4.501*u.arcsec)).to(u.deg)),

"NGC6251"   : SkyCoord((16*u.hour+32*u.min+31.970*u.s).to(u.hour).value*15*u.deg, (+(82*u.deg+32*u.arcmin+16.400*u.arcsec)).to(u.deg)),

"NRAO512"   : SkyCoord((16*u.hour+40*u.min+29.633*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+46*u.arcmin+46.028*u.arcsec)).to(u.deg)),

"3C345"     : SkyCoord((16*u.hour+42*u.min+58.810*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+48*u.arcmin+36.994*u.arcsec)).to(u.deg)),
"1641+399"  : SkyCoord((16*u.hour+42*u.min+58.810*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+48*u.arcmin+36.994*u.arcsec)).to(u.deg)),

"MRK501"    : SkyCoord((16*u.hour+53*u.min+52.217*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+45*u.arcmin+36.609*u.arcsec)).to(u.deg)),
"1652+398"  : SkyCoord((16*u.hour+53*u.min+52.217*u.s).to(u.hour).value*15*u.deg, (+(39*u.deg+45*u.arcmin+36.609*u.arcsec)).to(u.deg)),

"NRAO530"   : SkyCoord((17*u.hour+33*u.min+ 2.706*u.s).to(u.hour).value*15*u.deg, (-(13*u.deg+ 4*u.arcmin+49.548*u.arcsec)).to(u.deg)),
"1730-130"  : SkyCoord((17*u.hour+33*u.min+ 2.706*u.s).to(u.hour).value*15*u.deg, (-(13*u.deg+ 4*u.arcmin+49.548*u.arcsec)).to(u.deg)),

"1741-038"  : SkyCoord((17*u.hour+43*u.min+58.856*u.s).to(u.hour).value*15*u.deg, (-( 3*u.deg+50*u.arcmin+ 4.617*u.arcsec)).to(u.deg)),

"OT081"     : SkyCoord((17*u.hour+51*u.min+32.819*u.s).to(u.hour).value*15*u.deg, (+( 9*u.deg+39*u.arcmin+ 0.728*u.arcsec)).to(u.deg)),
"1749+096"  : SkyCoord((17*u.hour+51*u.min+32.819*u.s).to(u.hour).value*15*u.deg, (+( 9*u.deg+39*u.arcmin+ 0.728*u.arcsec)).to(u.deg)),

"1803+784"  : SkyCoord((18*u.hour+ 0*u.min+45.684*u.s).to(u.hour).value*15*u.deg, (+(78*u.deg+28*u.arcmin+ 4.018*u.arcsec)).to(u.deg)),

"3C371"     : SkyCoord((18*u.hour+ 6*u.min+50.681*u.s).to(u.hour).value*15*u.deg, (+(69*u.deg+49*u.arcmin+28.109*u.arcsec)).to(u.deg)),
"1807+698"  : SkyCoord((18*u.hour+ 6*u.min+50.681*u.s).to(u.hour).value*15*u.deg, (+(69*u.deg+49*u.arcmin+28.109*u.arcsec)).to(u.deg)),

"1828+487"  : SkyCoord((18*u.hour+29*u.min+31.781*u.s).to(u.hour).value*15*u.deg, (+(48*u.deg+44*u.arcmin+46.161*u.arcsec)).to(u.deg)),
"3C380"     : SkyCoord((18*u.hour+29*u.min+31.781*u.s).to(u.hour).value*15*u.deg, (+(48*u.deg+44*u.arcmin+46.161*u.arcsec)).to(u.deg)),

"1845+797"  : SkyCoord((18*u.hour+42*u.min+ 8.990*u.s).to(u.hour).value*15*u.deg, (+(79*u.deg+46*u.arcmin+17.128*u.arcsec)).to(u.deg)),
"3C390.3"   : SkyCoord((18*u.hour+42*u.min+ 8.990*u.s).to(u.hour).value*15*u.deg, (+(79*u.deg+46*u.arcmin+17.128*u.arcsec)).to(u.deg)),

"3C395"     : SkyCoord((19*u.hour+ 2*u.min+55.939*u.s).to(u.hour).value*15*u.deg, (+(31*u.deg+59*u.arcmin+41.702*u.arcsec)).to(u.deg)),
"1901+319"  : SkyCoord((19*u.hour+ 2*u.min+55.939*u.s).to(u.hour).value*15*u.deg, (+(31*u.deg+59*u.arcmin+41.702*u.arcsec)).to(u.deg)),

"4C73.18"   : SkyCoord((19*u.hour+27*u.min+48.495*u.s).to(u.hour).value*15*u.deg, (+(73*u.deg+58*u.arcmin+ 1.570 *u.arcsec)).to(u.deg)),
"1928+738"  : SkyCoord((19*u.hour+27*u.min+48.495*u.s).to(u.hour).value*15*u.deg, (+(73*u.deg+58*u.arcmin+ 1.570 *u.arcsec)).to(u.deg)),

"CYGNUSA"   : SkyCoord((19*u.hour+59*u.min+28.357*u.s).to(u.hour).value*15*u.deg, (+(40*u.deg+44*u.arcmin+ 2.097 *u.arcsec)).to(u.deg)),
"1957+405"  : SkyCoord((19*u.hour+59*u.min+28.357*u.s).to(u.hour).value*15*u.deg, (+(40*u.deg+44*u.arcmin+ 2.097 *u.arcsec)).to(u.deg)),

"1959+650"  : SkyCoord((19*u.hour+59*u.min+59.852*u.s).to(u.hour).value*15*u.deg, (+(65*u.deg+ 8*u.arcmin+54.652*u.arcsec)).to(u.deg)),

"2134+004"  : SkyCoord((21*u.hour+36*u.min+38.586*u.s).to(u.hour).value*15*u.deg, (+( 0*u.deg+41*u.arcmin+54.213*u.arcsec)).to(u.deg)),

"4C06.69"   : SkyCoord((21*u.hour+48*u.min+ 5.459*u.s).to(u.hour).value*15*u.deg, (+( 6*u.deg+57*u.arcmin+38.604*u.arcsec)).to(u.deg)),
"2145+067"  : SkyCoord((21*u.hour+48*u.min+ 5.459*u.s).to(u.hour).value*15*u.deg, (+( 6*u.deg+57*u.arcmin+38.604*u.arcsec)).to(u.deg)),

"BLLAC"     : SkyCoord((22*u.hour+ 2*u.min+43.291*u.s).to(u.hour).value*15*u.deg, (+(42*u.deg+16*u.arcmin+39.980*u.arcsec)).to(u.deg)),
"2200+420"  : SkyCoord((22*u.hour+ 2*u.min+43.291*u.s).to(u.hour).value*15*u.deg, (+(42*u.deg+16*u.arcmin+39.980*u.arcsec)).to(u.deg)),

"4C31.63"   : SkyCoord((22*u.hour+ 3*u.min+14.976*u.s).to(u.hour).value*15*u.deg, (+(31*u.deg+45*u.arcmin+38.270*u.arcsec)).to(u.deg)),
"2201+315"  : SkyCoord((22*u.hour+ 3*u.min+14.976*u.s).to(u.hour).value*15*u.deg, (+(31*u.deg+45*u.arcmin+38.270*u.arcsec)).to(u.deg)),

"3C446"     : SkyCoord((22*u.hour+25*u.min+47.259*u.s).to(u.hour).value*15*u.deg, (-( 4*u.deg+57*u.arcmin+ 1.391*u.arcsec)).to(u.deg)),
"2223-052"  : SkyCoord((22*u.hour+25*u.min+47.259*u.s).to(u.hour).value*15*u.deg, (-( 4*u.deg+57*u.arcmin+ 1.391*u.arcsec)).to(u.deg)),

"2227-088"  : SkyCoord((22*u.hour+29*u.min+40.084*u.s).to(u.hour).value*15*u.deg, (-( 8*u.deg+32*u.arcmin+54.436*u.arcsec)).to(u.deg)),

"CTA102"    : SkyCoord((22*u.hour+32*u.min+36.409*u.s).to(u.hour).value*15*u.deg, (+(11*u.deg+43*u.arcmin+50.904*u.arcsec)).to(u.deg)),
"2230+114"  : SkyCoord((22*u.hour+32*u.min+36.409*u.s).to(u.hour).value*15*u.deg, (+(11*u.deg+43*u.arcmin+50.904*u.arcsec)).to(u.deg)),

"3C454.3"   : SkyCoord((22*u.hour+53*u.min+57.748*u.s).to(u.hour).value*15*u.deg, (+(16*u.deg+ 8*u.arcmin+53.561*u.arcsec)).to(u.deg)),
"2251+158"  : SkyCoord((22*u.hour+53*u.min+57.748*u.s).to(u.hour).value*15*u.deg, (+(16*u.deg+ 8*u.arcmin+53.561*u.arcsec)).to(u.deg)),

"4C45.51"   : SkyCoord((23*u.hour+54*u.min+21.680*u.s).to(u.hour).value*15*u.deg, (+(45*u.deg+53*u.arcmin+ 4.236*u.arcsec)).to(u.deg)),
"2351+456"  : SkyCoord((23*u.hour+54*u.min+21.680*u.s).to(u.hour).value*15*u.deg, (+(45*u.deg+53*u.arcmin+ 4.236*u.arcsec)).to(u.deg)),

"3C48"      : SkyCoord(( 1*u.hour+37*u.min+41.299*u.s).to(u.hour).value*15*u.deg, (+(33*u.deg+ 9*u.arcmin+35.134*u.arcsec)).to(u.deg)),
"3C147"     : SkyCoord(( 5*u.hour+42*u.min+36.138*u.s).to(u.hour).value*15*u.deg, (+(49*u.deg+51*u.arcmin+ 7.233*u.arcsec)).to(u.deg)),
"3C138"     : SkyCoord(( 5*u.hour+21*u.min+ 9.886*u.s).to(u.hour).value*15*u.deg, (+(16*u.deg+38*u.arcmin+22.052*u.arcsec)).to(u.deg)),
"SGRA"      : SkyCoord((17*u.hour+45*u.min+40.041*u.s).to(u.hour).value*15*u.deg, (-(29*u.deg+ 0*u.arcmin+28.118*u.arcsec)).to(u.deg)),


"R-LEO"     : SkyCoord(( 9*u.hour+47*u.min+33.490*u.s).to(u.hour).value*15*u.deg, (+(11*u.deg+25*u.arcmin+43.700*u.arcsec)).to(u.deg)),
"R-CAS"     : SkyCoord((23*u.hour+58*u.min+24.870*u.s).to(u.hour).value*15*u.deg, (+(51*u.deg+23*u.arcmin+19.700*u.arcsec)).to(u.deg)),
"IK-TAU"    : SkyCoord(( 3*u.hour+53*u.min+28.870*u.s).to(u.hour).value*15*u.deg, (+(11*u.deg+24*u.arcmin+21.700*u.arcsec)).to(u.deg)),
"U-HER"     : SkyCoord((16*u.hour+25*u.min+47.470*u.s).to(u.hour).value*15*u.deg, (+(18*u.deg+53*u.arcmin+32.900*u.arcsec)).to(u.deg)),
"CHI-CYG"   : SkyCoord((19*u.hour+50*u.min+33.920*u.s).to(u.hour).value*15*u.deg, (+(32*u.deg+54*u.arcmin+50.600*u.arcsec)).to(u.deg)),
"W-HYA"     : SkyCoord((13*u.hour+49*u.min+ 2.000*u.s).to(u.hour).value*15*u.deg, (-(28*u.deg+22*u.arcmin+ 3.500*u.arcsec)).to(u.deg)),
"VY-CMA"    : SkyCoord(( 7*u.hour+22*u.min+58.330*u.s).to(u.hour).value*15*u.deg, (-(25*u.deg+46*u.arcmin+ 3.200*u.arcsec)).to(u.deg)),
"ORION"     : SkyCoord(( 5*u.hour+35*u.min+14.160*u.s).to(u.hour).value*15*u.deg, (-( 5*u.deg+22*u.arcmin+30.630*u.arcsec)).to(u.deg)),
"TX-CAM"    : SkyCoord(( 5*u.hour+ 0*u.min+50.390*u.s).to(u.hour).value*15*u.deg, (+(56*u.deg+10*u.arcmin+52.600*u.arcsec)).to(u.deg)),

}
