Example Github workflow
^^^^^^^^^^^^^^^^^^^^^^^
These steps are the ones I use to work with git and GitHub. There are a lot of steps, but the idea is that steps 1, and 3 only have to be done once, and step 2 a few times as you work on your code. Also, the version control with git and GitHub makes it relatively safe to make changes to the code, so you won’t have to worry about messing things up.  

First off, make sure you have a GitHub account and that git is installed on your computer. 

#. In a web browser:

   #. Go to the `main PBjam repository <https://github.com/grd349/PBjam>`_ (repo) and press the Fork button in the top right corner. This copies the current state of the main repo to your GitHub account. If the main repo is updated, your copy will **not** change and vice versa.
   
   #. From your copy of the repo (called a Fork), press Clone, and then copy the repo link. It will look something like

**A. In a terminal:**

1. Fork the main **lightkurve** repository by logging into GitHub, browsing to
   ``https://github.com/KeplerGO/lightkurve`` and clicking on ``Fork`` in the top right corner.

2. Clone your fork to your computer:

.. code-block:: bash

    $ git clone https://github.com/YOUR-GITHUB-USERNAME/lightkurve.git
