Gaussian Multi-frequency VLBI Analyses for AGN jets (GaMVAs)
=
This is a Python package for modeling the sky brightness distribution of multi-frequency VLBI data by establishing Gaussian components and their spectra.
This module employs nested sampling, a powerful Bayesian technique for estimating model evidence and exploring complex and multimodal posterior distributions.
The nested sampling calculations are performed using the Python package ``dynesty``.
For further details on ``dynesty``, please refer to its documentation (https://dynesty.readthedocs.io/en/v2.1.5/) and associated publication (https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract).

Please note that this is an early release and may still contain some bugs or other issues.
If you have trouble applying this module to your work, please send me a report (email: hwjeongastro@gmail.com).



Installation
-
TBU (not uploaded on PyPi yet)



Tutorials
-
Some basic example scripts are available in the 'example' directory.


Updates
- (2026/07/10)
  - Flagging method is unified through ``flag_data(dotype, *kwargs)``. Available types are ['time', 'sigma', 'snr', 'antenna', 'nant', 'baseline', 'uvradius'].
  - Averaging UV data is performed through ``average(dotype, *kwargs)``. Available types are ['time', 'ifchan'].
  - If needed, you may want to modify the 'sigma' of complex visibilities through ``inflate_sigma_fractional()`` or `rescale_sigma()`.
  - Now the resulting image FITS (containing model parameters) & UVFITS files are generated.

- ~~(2025/06/13) - All the flagging methods ( e.g., ``flag_snr()`` ) are unified into ``flag_uvvis(type=type, value=value)``.~~

