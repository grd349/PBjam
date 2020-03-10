Example Github workflow
^^^^^^^^^^^^^^^^^^^^^^^
This is a simplified set of steps for working with git and GitHub. There are a lot of steps, but the idea is that parts A, and C only have to be done once, and step B a few times as you work on your code. Also, the version control with git and GitHub makes it relatively safe to make changes to the code, so you won’t have to worry about messing things up.  

If you haven't already,  `create a GitHub account <https://github.com/join?source=header-home>`_ and `install git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ on your computer.

**A. In a web browser:**

1. Go to the `main PBjam repository <https://github.com/grd349/PBjam>`_ (repo) and press the Fork button in the top right corner. This copies the current state of the main repo to your GitHub account. If the main repo is updated, your copy will **not** change and vice versa.
   
2. From your copy of the repo (called a fork), press Clone, and then copy the repo link. 

**B. In a terminal:**

1. To download a copy of your PBjam repository into a directory do:

.. code-block:: console

   $ git clone https://github.com/yourusername/PBjam.git
   
2. Now you can write/edit the code as you wish, with all the wonderful docstrings and unit tests (naturally!).
   
3. To see what files have been changed during your work, while in the PBjam directory, you can type

.. code-block:: console

   $ git status. 
       
4. Git allows you to bundle all the changes you have made into a 'commit'. This list of changes will eventually be uploaded to the your PBjam repository on GitHub. You can add multiple files to such a commit. After having checked which files have changed, start adding them to a commit by doing:

.. code-block:: console

   $ git add the/path/to/modified/file
      
5. Once all the relevant files have been added, you can wrap it all up with a brief description. Do this often, as it will be easier to roll back the code if something bad happens. This can be done by doing:

.. code-block:: console
   
   $ git commit -m 'A short message about the changes you have made'
   
6. Now you can upload, or 'push', the commit to the online GitHub repository. By repeating step 4 and 5, you can add several commits to a single push.  

.. code-block:: console
   
   $ git push origin master
   
7. If many people are working from the same repository, it is sometimes useful to make several branches or copies of the code. To push changes to a different branch simply replace :code:`master` with the name of the branch you are working on. 

**C. In a web browser**

1. The very last step is to request your code to be merged into the main PBjam repository. To do this, go to your copy of the PBjam repository (your fork).
   
2. Press the Pull Request button just below the green Clone button, and Create Pull Request on the following page. This will start the process of merging your changes into the main PBjam repository. 
   
3. Don’t panic. Any changes will be reviewed and tested, so you won’t break anything.
