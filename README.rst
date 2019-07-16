
PBjam: peakbagging made easy
============================
.. image:: https://img.shields.io/badge/GitHub-PBjam-green.svg
    :target: https://github.com/grd349/PBjam
.. image:: https://img.shields.io/github/issues-closed/grd349/PBjam.svg
    :target: https://github.com/grd349/PBjam/issues
.. image:: https://readthedocs.org/projects/pbjam/badge/?version=latest
    :target: https://pbjam.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/grd349/PBjam/blob/master/LICENSE

PBjam is toolbox for modeling the oscillation spectra of solar-like stars. This involves two main parts: identifying a set of modes of interest, and accurately modeling those modes to measure their frequencies. 

Currently, the mode identification is based on fitting the asymptotic relation to the l=2,0 pairs, relying on the cumulative sum of prior knowledge gained from NASA's Kepler mission to inform the fitting process. 

Modeling the modes, or 'peakbagging', is done using the HMC sampler from `pymc3 <https://docs.pymc.io/>`_, which fits a Lorentzian to each of the identified modes, with much fewer priors in place. This allows for a more accurate model of the spectrum of frequencies, than the heavily parameterized models like the asymptotic relation.


Read the docs at `pbjam.readthedocs.io <http://pbjam.readthedocs.io/>`_.

.. inclusion_marker0


Contributing
------------
PBjam is open source and we welcome contributions, large or small. 

If you spot any bugs, have ideas for optimizing the code, want new nifty features, feel free to submit issues on the `GitHub repo <https://github.com/grd349/PBjam/issues>`_. 

Pull requests are also welcome, these will be reviewed by the main authors of the code before merging. 

Authors
-------
Main Contributors
^^^^^^^^^^^^^^^^^
- `Guy Davies <https://github.com/grd349>`_ 
- `Oliver Hall <https://github.com/ojhall94>`_ 
- `Martin Nielsen <https://github.com/nielsenmb>`_ 

Chaos Engineers
^^^^^^^^^^^^^^^
- `Warrick Ball <https://github.com/warrickball>`_ 

Acknowledgments
---------------
PBjam relies heavily on many open source software packages to operate. 
