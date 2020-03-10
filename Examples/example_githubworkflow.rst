Example Github workflow
^^^^^^^^^^^^^^^^^^^^^^^
These steps are the ones I use to work with git and GitHub. There are a lot of steps, but the idea is that steps 1, and 3 only have to be done once, and step 2 a few times as you work on your code. Also, the version control with git and GitHub makes it relatively safe to make changes to the code, so you won’t have to worry about messing things up.  

First off, make sure you have a GitHub account and that git is installed on your computer. 

**1. In a web browser:**
a. Go to the main PBjam repository (repo) and press the Fork button in the top right corner. This copies the current state of the main repo to your GitHub account. If the main repo is updated, your copy will not change and vice versa.
b. From your copy of the repo (called a Fork), press Clone, and then copy the repo link. 

**2. In a terminal:**
  a. Type git clone thelinkyoujustcopied. This downloads your copy of PBjam into the directory where you are currently sitting.
  b. Now you can write/edit the code as you wish, with all the wonderful docstrings and unit tests (naturally!).
  c. At any time, from the PBjam directory you can type git status. This will show you all the files that have been changed in your local PBjam directory. Any files that you have worked on should appear in the list(s).
  d. Now type git add the/path/to/modified/file. This adds the file to a bundle that git keeps track of, and that you will eventually upload to your online copy of the PBjam repository on GitHub. You can add multiple files to such a bundle.
  e. Now type git commit -m ‘A short message about the changes you have made’. This wraps up the changes you have made in a nice little bundle (called a commit) with a brief description. Think of it as a discrete unit of change to the code that git will keep track of from now on. Many smaller commits are better than 1 big one, so do this often. 
  f. When you are ready to send the changes to your GitHub repository, type git push origin master. This will upload the commit(s) to your online version of the PBjam repository on GitHub. Many commits can be pushed at the same time.

**3. In a web browser:**
  a. The very last step is to request your code to be merged into the main PBjam repository. To do this, go to your copy of the PBjam repository (your fork).
  b. Press the Pull Request button just below the green Clone button, and Create Pull Request on the following page. This will start the process of merging your changes into the main PBjam repository. 
  c. Don’t panic. Any changes will be reviewed and tested, so you won’t break anything.

