Example Docstring
============

Docstrings are readable text associated with a function or unit of code that describes what it does, what inputs it takes and what outputs it provides. In Python code the docstring is usually the bit enclosed in a set of """ (see below). For PBjam we try to follow the `Numpy format <https://numpydoc.readthedocs.io/en/latest/format.html>`_ for docstring. Inline comments are somewhat different and start with #. These are typically used to explain lines or bits of code that aren't completely intuitive, but not to provide longer more extended explanations. When contributing code, providing both is nice for the review.

The following is an example:

.. code-block:: python

    def my_function(a, b=1):
        """ Add two numbers
        
        The docstrings starts here. The first line is usually just 2-3 
        words describing the function.
        
        This is followed by a somewhat longer and detailed description that 
        explains what the function does. 
    
        This should be followed by a list of input parameters and a list of 
        returned values. These lists should show the variable types, whether
        they are optional, as well say a few words about the parameter. 
    
        An example is also nice, but usually only if it’s a really 
        complicated function.
    
        Example
        -------
        >>> c = my_function(1) # This is a silly case of inline commenting.
        >>>
        >>> c = my_function(1, 2) 

        Parameters
        ----------
        a : TYPE 
            DESCRIPTION. 
        b : TYPE, optional # If a parameter is optional it should be stated here.
            DESCRIPTION. The default is 1. # The default value should be stated here.

        Returns
        -------
        c : TYPE
            DESCRIPTION.
        
        """
        
        c = a + b

        return c
        
