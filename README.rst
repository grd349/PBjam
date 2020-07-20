
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
================= =============== ====================
Main Contributors Chaos Engineers Scientific Oversight
================= =============== ====================
`Lindsey Carboneau <https://github.com/lmcarboneau>`_ `Warrick Ball <https://github.com/warrickball>`_ Othman Benomar
cell              cell            cell
================= =============== ====================
