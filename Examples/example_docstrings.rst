Example Docstring
============

Docstrings are readable text associated with a function or unit of code that describes what it does, what inputs it takes and what outputs it provides. For PBjam we try to follow the Numpy format for docstring, which you can read about `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The following is an example:

.. code-block:: python

    def my_function(a, b=1):
        """ This line is a very short description of the function
        
        Followed by a somewhat longer and detailed description that explains what 
        the function does. 
    
        This should be followed by a list of input parameters and a list of 
        returned values. These lists should show the variable types, whether
        they are optional, as well say a few words about the parameter. 
    
        An example is also nice if it’s a really complicated function.
    
        Example
        ------------
        >>> c = my_function(1) # This is a silly example
        >>>
        >>> c = my_function(1, 2) 

        Parameters
        ----------
        a : TYPE 
            DESCRIPTION. 
        b : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        c : TYPE
            DESCRIPTION.
        
        """
        
        c = a + b

        return c
        
