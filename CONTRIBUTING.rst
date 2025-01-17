PBjam is open source and we welcome contributions, large or small!

Raising issues
^^^^^^^^^^^^^^
If you're having problems with the code, either because you uncover a bug or there's something that's just totally unintuitive about using PBjam, feel free to `raise an issue <https://github.com/grd349/PBjam/issues>`_ (**do search the list of issues first though!**). When raising issues please explain what you tried to do, how to did it, and what PBjam threw back at you when you did.

We will try to address the issue as soon as possible.

Making pull requests
^^^^^^^^^^^^^^^^^^^^
If you spot an issue that you know the solution to, and can write a patch for it, feel free to make a pull request for it! 

One way to do this is:

- Go to `Branches <https://github.com/grd349/PBjam/branches>`_  
- Create a new branch **from the dev branch**
- Clone the repo to your local machine
- On your local machine do ``Checkout mynewbranch``, do the edits and push the changes to the new branch on Github. 
- You can then submit a pull request which will be reviewed. 

**For changes big or small, or new features please only create branches off of the dev branch, not master**

Also, please clearly explain what the pull request is meant to fix, and how you went about fixing it.

Please make sure to include docstrings in your submissions, along with the odd inline comment. This makes reviewing the submission that much easier. See the `example docstring <https://github.com/grd349/PBjam/blob/master/Examples/example_docstrings.rst>`_ for ideas on writing basic documentation.

Providing a few unit tests of your functions would be very helpful too. See the `example unittest <https://github.com/grd349/PBjam/blob/master/Examples/example_unittests.rst>`_ for a very simple example.