Teaching branch moved to master branch
============================
This branch was meant to be used for the tutorial during the Porto Summer School on Asteroseismology.

The code and tutorials from the school have been merged into the master branch now, so you should refer to that branch from now on.

This branch will likely be closed down in the near future.



PBjam
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
.. image:: https://travis-ci.com/grd349/PBjam.svg?branch=master
    :target: https://travis-ci.com/grd349/PBjam
.. image:: http://img.shields.io/badge/arXiv-2012.00580-B31B1B.svg
    :target: https://arxiv.org/abs/2012.00580

PBjam is toolbox for modeling the oscillation spectra of solar-like oscillators. This involves two main parts: identifying a set of modes of interest, and accurately modeling those modes to measure their frequencies.

Currently, the mode identification is based on fitting the asymptotic relation to the l=2,0 pairs, relying on the cumulative sum of prior knowledge gained from NASA's Kepler mission to inform the fitting process.

Modeling the modes, or 'peakbagging', is done using the HMC sampler from `pymc3 <https://docs.pymc.io/>`_, which fits a Lorentzian to each of the identified modes, with much fewer priors than during he mode ID process. This allows for a more accurate model of the spectrum of frequencies, than the heavily parameterized models like the asymptotic relation.


Read the docs at `pbjam.readthedocs.io <http://pbjam.readthedocs.io/>`_.

.. inclusion_marker0


Contributing
------------
If you want to raise and issue or contribute code to PBjam, see the `guidelines on contributing <https://github.com/grd349/PBjam/blob/master/CONTRIBUTING.rst>`_.

Authors
-------
There are different ways to contribute to PBjam, the Scientific Influencers help guide the scientific aspects of PBjam, the Chaos Engineers try to break the code or simply report bugs, while the Main Contributors submit Pull Requests with somewhat bigger additions to the code or documentation. 

===================================================== ================================================ ====================================================
Main Contributors                                     Chaos Engineers                                  Scientific Influencers
===================================================== ================================================ ====================================================
`Lindsey Carboneau <https://github.com/lmcarboneau>`_ `Warrick Ball <https://github.com/warrickball>`_ `Othman Benomar <https://github.com/OthmanB>`_
`Guy Davies <https://github.com/grd349>`_             `Rafa Garcia <https://github.com/rgarcibus>`_    Bill Chaplin 
`Oliver Hall <https://github.com/ojhall94>`_          `Tanda Li <https://github.com/litanda>`_	       `Enrico Corsaro <https://github.com/EnricoCorsaro>`_
`Alex Lyttle <https://github.com/alexlyttle>`_        `Joel Ong <https://github.com/darthoctopus>`_    `Patrick Gaulme <https://github.com/gaulme>`_  
`Martin Nielsen <https://github.com/nielsenmb>`_      |                                                `Mikkel Lund <https://github.com/Miklnl>`_
|                                                     |                                                Benoit Mosser 
|                                                     |                                                Andy Moya
|                                                     |                                                Ian Roxburgh
===================================================== ================================================ ====================================================


Acknowledgements
----------------
If you use PBjam in your work please cite the PBjam paper (forthcoming), and if possible provide links to the GitHub repository. 

We encourage PBjam users to also cite the packages and publications that PBjam makes use of. PBjam will automatically keep track of the publications that were used during a run, and can list them for you in bibtex format (see the `Referencing Notebook <https://github.com/grd349/PBjam/tree/master/Examples/Example-references.ipynb>`_).

