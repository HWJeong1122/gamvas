Gaussian Multi-frequency VLBI Analyses for AGN jets (GaMVAs)
=
This is a Python package for modeling sky brightness distribution of multi-frequency VLBI data by establishing Gaussian components and their spectrum.
To converge the modeling, this module employs a nested-sampling method, implemented in ``dynesty``.
For more details on ``dynesty`` please refer to https://dynesty.readthedocs.io/en/v2.1.5/ (ADS: https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S/abstract).

Several Python modules are imported in this module (e.g., ``load``, ``modeling``, ``utils``, ``plotting``, ``functions``).

Please note that this module is early release, so might raise issues.
If you have issues in applying this module for your work, please send me a report for that (email: hwjeongastro@gmail.com).



Installation
-
TBU (not uploaded on PyPi yet)



Tutorials
-
TBU


Updates
-
(2025/06/13) - all the flagging methods ( e.g., ``flag_snr()`` ) are unified into ``flag_uvvis(type=type, value=value)``.
