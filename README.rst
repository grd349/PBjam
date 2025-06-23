
PBjam 2
============================

**Peakbagging made easy**

.. image:: https://img.shields.io/badge/GitHub-PBjam-green.svg
    :target: https://github.com/grd349/PBjam
.. image:: https://readthedocs.org/projects/pbjam/badge/?version=latest
    :target: https://pbjam.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/grd349/PBjam/blob/master/LICENSE
.. image:: https://img.shields.io/github/issues-closed/grd349/PBjam.svg
    :target: https://github.com/grd349/PBjam/issues
.. image:: https://badge.fury.io/py/pbjam.svg
    :target: https://badge.fury.io/py/pbjam
.. image:: http://img.shields.io/badge/arXiv-2012.00580-B31B1B.svg
    :target: https://arxiv.org/abs/2012.00580

PBjam is toolbox for analyzing the oscillation spectra of solar-like oscillators. This involves two main parts: identifying a set of modes of interest in a spectrum of oscillations, and accurately modeling those modes to measure their frequencies.

The mode identification works by fitting the asymptotic relation for p-modes to the l=2,0 pairs, which is followed by a applying selection of models for fitting the l=1 modes where each model is suitable for different stages of evolution.
The process relies on of large set of previous observations of the model parameters, which are then used to construct a prior distribution to inform the sampling. The observations have been gathered from the Kepler, K2 and TESS missions, and expanding it to improve accuracy is an on-going process. 

Modeling the modes, or 'peakbagging', is done using the a nested sampling or MCMC algorithm, where Lorentzian profiles are fit to each of the identified modes, with much fewer constraints than during the mode ID process. This allows for a more accurate model of the spectrum of frequencies than the heavily parameterized models like the asymptotic relations.

To get started with PBjam please see the docs at `pbjam.readthedocs.io <http://pbjam.readthedocs.io/>`_.

.. inclusion_marker0


Contributing
------------
If you want to raise an issue or contribute code to PBjam, see the `guidelines on contributing <https://github.com/grd349/PBjam/blob/master/CONTRIBUTING.rst>`_.

Authors
-------
There are different ways to contribute to PBjam, the Scientific Influencers help guide the scientific aspects of PBjam, the Chaos Engineers try to break the code or simply report bugs, while the Main Contributors submit Pull Requests with somewhat bigger additions to the code or documentation. 

===================================================== ================================================ ====================================================
Main Contributors                                     Chaos Engineers                                  Scientific Influencers
===================================================== ================================================ ====================================================
`Lindsey Carboneau <https://github.com/lmcarboneau>`_ `Warrick Ball <https://github.com/warrickball>`_ `Othman Benomar <https://github.com/OthmanB>`_
`Guy Davies <https://github.com/grd349>`_             `Rafa Garcia <https://github.com/rgarcibus>`_    Bill Chaplin 
`Oliver Hall <https://github.com/ojhall94>`_          `Tanda Li <https://github.com/litanda>`_	       `Enrico Corsaro <https://github.com/EnricoCorsaro>`_
`Alex Lyttle <https://github.com/alexlyttle>`_        Angharad Weeks                                   `Patrick Gaulme <https://github.com/gaulme>`_  
`Martin Nielsen <https://github.com/nielsenmb>`_      Jens Rersted Larsen                              `Mikkel Lund <https://github.com/Miklnl>`_
`Joel Ong <https://github.com/darthoctopus>`_         |                                                Benoit Mosser 
`George Hookway <https://github.com/George-Hookway>`_ |                                                Andy Moya
|                                                     |                                                Ian Roxburgh
===================================================== ================================================ ====================================================


Acknowledgements
----------------
If you use PBjam in your work please cite the one of the PBjam papers (`Paper I Nielsen et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021AJ....161...62N/abstract>`_,  `Paper II Nielsen et al. 2023 <https://ui.adsabs.harvard.edu/abs/2023A%26A...676A.117N/abstract>`_ ), and if possible provide links to the `GitHub repository <https://github.com/grd349/PBjam>`_. 

We encourage users to also cite the packages and publications that PBjam makes use of.  
