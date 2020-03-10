Example Unit Tests
^^^^^^^^^^^^^^^^^^
Unit tests are small functions that test other functions by calling them, usually with a set of `dummy’ inputs, or inputs that you know will yield a consistent result. The purpose is to make sure the function does more or less what it’s supposed to do.

This can be done by inputting either reasonable values to check that the function returns something sensible, or something unreasonable to check that it returns something less sensible or breaks as it’s supposed to. They can also test the shape and type of outputs given some input. 

From the above example function a simple unit test could be:

.. code-block:: python

    def my_function(a, b=1):
        """ Add two numbers

        This should be a descriptive docstring, but we will
        skip it for now. 
        
        """

        c = a+b

        return c
