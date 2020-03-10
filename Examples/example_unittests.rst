Example Unit Tests
Unit tests are small functions that test other functions by calling them, usually with a set of ‘dummy’ inputs, or inputs that you know will yield a consistent result. The purpose is to make sure the function does more or less what it’s supposed to do.

This can be done by inputting either reasonable values to check that the function returns something sensible, or something unreasonable to check that it returns something less sensible or breaks as it’s supposed to. They can also test the shape and type of outputs given some shape or type of input. 

From the above example function a simple unit test could be:

def test_my_function():
    import pytest
    import numpy as np
    
    assert(my_function(1)==2)
    assert(my_function(1,2)==3)

    d = array([1,2])
    result =  my_function(d, 1)
    assert(np.shape(result) == np.shape(d))    

    with pytest.raises(TypeError):
        assert(my_function(1, 'adding a string to an integer should raise a TypeError'))

A single test function can contain multiple smaller tests. 

