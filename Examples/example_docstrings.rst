Example Docstring
============

Docstrings are readable text associated with a function or unit of code that describes what it does, what inputs it takes and what outputs it provides. For PBjam we try to follow the `Numpy format <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstring. 

The following is an example:

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
        >>> c = my_function(1) # This a basic usage example.
        >>>
        >>> c = my_function(1, 2) # This is an advanced usage example.

        Parameters
        ----------
        a : TYPE 
            DESCRIPTION. 
        b : TYPE, optional # If a parameter is optional it should be stated here.
            DESCRIPTION. The default is 1. # If practical the default value should be stated here.

        Returns
        -------
        c : TYPE
            DESCRIPTION.
        
        """
        
        c = a + b

        return c
        
