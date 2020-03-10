Example Docstring
============

Docstrings are readable text associated with a function or unit of code that describes what it does, what inputs it takes and what outputs it provides. For PBjam we try to follow the Numpy format for docstring, which you can read about `here <https://numpydoc.readthedocs.io/en/latest/format.html>`_. The following is an example:

.. code-block:: python

    def my_function(a, b=1):
        """ This line is a very short description of the function
        """
        c = a + b

        return c
        
