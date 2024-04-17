Installation
============

It is **highly** recommended that you set up a virtual environment before installing pbjam. To ensure the dependencies play well together we have fixed the dependency versions, which means installing PBjam may change the version of your local packages that it depends on. This is temporary and future updates will seek to rectify this.

Set up and activate a virtual environment 

.. code-block:: console
    $ python -m venv /path/to/virtual/evns/pbjam_env
    $ source /path/to/virtual/envs/pbjam_env/bin/activate
    
You can then install PBjam using pip

.. code-block:: console

    $ pip install pbjam --user

Or clone the GitHub repository

.. code-block:: console

    $ git clone https://github.com/grd349/PBjam.git
    $ pip install -e .


Requirements
------------
- numpy==1.22.1
- scipy==1.7.3
- matplotlib==3.8.4
- pandas==2.0.3
- statsmodels==0.14.1
- emcee==3.1.4
- astropy==6.0.1
- lightkurve==2.4.2
- corner==2.2.2
- nbsphinx==0.9.3
- cpnest==0.11.5
- pymc3==3.11.5














