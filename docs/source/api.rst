API
===

PBjam can be roughly divided into two parts, the mode ID and the peakbagging stages. PBjam currently supports mode ID methods for identifying the l=2,0 mode pairs and l=1 modes in the :mod:`~pbjam.modeID` module, and methods for detailed peakbagging with the :mod:`~pbjam.peakbagging` module. 

The :mod:`~pbjam.session` and :mod:`~pbjam.star` modules are wrappers for :mod:`~pbjam.modeID` and :mod:`~pbjam.peakbagging`, to help set up the analysis of one or more stars in a pipeline-like fashion. 

Modules in PBjam
^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1
   
   background
   core
   distributions
   DR
   IO
   jar
   l1models
   l20models
   modeID
   peakbagging
   plotting
   samplers
   
   
   







