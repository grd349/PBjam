User Guide
==========

PBjam is intended to be a user-friendly peakbagging tool. The most straightforward way of using PBjam is through the :class:`~pbjam.core.session` class. This class handles the organization of the inputs, the mode ID, peakbagging and output. The inputs to the session class can take a variety of forms which are shown in the examples below. 

Session
-------
The :class:`~pbjam.core.session` class is the most straightforward way to analyze one or more stars with PBjam. It can automatically download the data and compute the power density spectrum, and then go through all the steps in the moden ID and peakbagging process. The `Session notebook <Examples/example-session.ipynb>`_ provides an example of how to use the :class:`~pbjam.core.session` class. 
 

Star
----
It's also possible to use the :class:`~pbjam.core.star` class to analyze single stars, mainly for use in custom scripts. The :class:`~pbjam.core.session` class is really just a fancy wrapper for the :class:`~pbjam.core.star` class.  

The :class:`~pbjam.core.star` class is meant for more detailed control of the inputs for each star. The `Star notebook <Examples/example-star.ipynb>`_ shows a simple example of this. 
    

Advanced
--------
It is not strictly necessary to use either the :class:`~pbjam.core.session` or :class:`~pbjam.core.star` classes. The `mode ID <Examples/example-modeID.ipynb>`_ and `peakbag <Examples/example-peakbag.ipynb>`_ notebooks show a lower-level walkthrough of the steps that PBjam goes through for peakbagging.

.. note:: 
    For additional useful examples see the `Examples <https://github.com/grd349/PBjam/tree/master/Examples>`_ directory.

Papers
------
We have published a few papers on various bits. The `first paper <https://ui.adsabs.harvard.edu/abs/2021AJ....161...62N/abstract>`_ provides information mainly on the initial version of PBjam. The `second paper <https://ui.adsabs.harvard.edu/abs/2023A%26A...676A.117N/abstract>`_ discusses the method used to construct the prior probability densities that we now use in the latest version of PBjam. The latest paper (in prep.) focuses on how we construct the models for the l=1 modes.