User Guide
==========

PBjam is intended to be a user-friendly peakbagging tool. The most straightforward way of using PBjam is through the :class:`~pbjam.session.session` class. This class handles the organization of the inputs, the mode ID, peakbagging and output. The inputs to the session class can take a variety of forms which are shown in the examples below. 

It's also possible to use the :class:`~pbjam.star.star` class to analyze single stars, mainly for use in custom scripts.

Session
-------
The :class:`~pbjam.jar.session` class is the most straightforward way to analyze one or more stars with PBjam. The `Session notebook <example-session.ipynb>`_ provides a few simple examples of the forms of input that can be given to :class:`~pbjam.jar.session`. 

.. note:: 
    The :class:`~pbjam.session.session` class is really just a fancy wrapper for the :class:`~pbjam.star.star` class. After initializing the session class, the instance can be called to execute the entire peakbagging procedure automatically.


Star
----
The :class:`Star <~pbjam.star.star>` class is meant for more detailed control of the inputs for each star. The `Star notebook <example-star.ipynb>`_ shows a simple example of this. 
    

Advanced
--------
It is not strictly necessary to use either the :class:`~pbjam.session.session` or :class:`~pbjam.star.star` classes. The `Advanced notebook <example-advanced.ipynb>`_ shows an end-to-end walkthrough of the steps that PBjam goes through for peakbagging.

.. note:: 
    For additional useful examples see the `Examples <https://github.com/grd349/PBjam/tree/master/Examples>`_ directory.

