API
===

PBjam can be roughly divided into two parts, the mode ID and the peakbagging stages. PBjam currently supports mode ID with the :mod:`~pbjam.asy_peakbag` module, and peakbagging with the :mod:`~pbjam.peakbag` module. Additional methods for these steps may be added to PBjam in the future.

The :mod:`~pbjam.jar` and :mod:`~pbjam.star` modules are wrappers for many of the methods in :mod:`~pbjam.asy_peakbag` and :mod:`~pbjam.peakbag`, to help set up the analysis of one or more stars in a pipeline-like fashion.

Modules in PBjam
^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 0
   
   jar
   star
   asy_peakbag
   peakbag
   guess_epsilon
   mcmc








