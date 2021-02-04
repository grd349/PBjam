"""Tests for the jar module"""

from pbjam.jar import normal, to_log10, get_priorpath, get_percentiles, log_file, file_logger, debug
import pbjam.tests.pbjam_tests as pbt
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import logging, os

logger = logging.getLogger('pbjam.tests')

def test_normal():
    """Test for the log of a normal distribution"""
    
    # setup
    func = normal
    inp = [0, 0, 1]
    inp_arr = [np.linspace(-10,10,100), 0, 1]

    # simple tests    
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, float)
    pbt.right_shape(func, inp_arr, (len(inp_arr[0]),))
    
    # check height at mean is correct
    assert(10**func(*inp)==1)
    
def test_to_log10():
    """Test for the log of a normal distribution"""
    
    # setup
    func = to_log10
    inp = [10,1]
    
    # simple tests
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, list)
    
    # check some inputs
    assert(func(*inp)[0]==1)
    assert_almost_equal(func(*inp)[1], 0.043429, 5)
    
    inp = [1e5,1e4]
    assert(func(*inp)[0]==5)

def test_get_priorpath():
    """Tests the function for getting the default prior_data filepath"""
    
    # setup
    func = get_priorpath
    
    # simple tests
    pbt.does_it_run(func, None)
    pbt.does_it_return(func, None)
    pbt.right_type(func, None, str)
    
    # check some inputs
    assert('prior_data.csv' in func())
    assert('data' in func())
    
def test_get_percentiles():
    """Tests the function for getting the percentiless of a distribution"""
    
    # setup
    func = get_percentiles
    inp = [np.random.normal(size = 100), 3]
    print(func(*inp))

    #print(func(*inp)[])
    
    # simple tests
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, np.ndarray)
    pbt.right_shape(func, inp, (2*inp[1]+1,))
    
    # check some different inputs
    inp = [np.random.normal(size = 30000), 4]
    pbt.right_shape(func, inp, (2*inp[1]+1,))
    
    inp = [[0,0,0,1,1], 1]
    assert_array_equal(func(*inp), [0., 0., 1.])

def test_file_logger():
    """Tests subclassing ``jam`` to use the log file record decorator"""
    test_message = 'This should be logged in file.'

    class file_logger_test(file_logger):
        def __init__(self):
            super(file_logger_test, self).__init__(filename='test_jam.log')
            logger.debug('This should not be logged in file.')
            with self.log_file:
                # Records content in context to `log_file`
                logger.debug(test_message)
        
        @file_logger.listen  # records content of `example_method` to `log_file`
        def method(self):
            logger.debug(test_message)
    
    jt = file_logger_test()
    jt.method()

    filename = jt.log_file._filename
    with open(filename, 'r') as file_in:
        lines = file_in.read().splitlines()
        messages = [line.split('::')[-1].strip() for line in lines]
        assert(all([message == test_message for message in messages]))

    os.remove(filename)

def test_log_file():
    """Test ``file_logger`` context manager."""
    filename = 'test_file_logger.log'
    test_level = 'DEBUG'
    flog = log_file(filename, level=test_level)
    
    with flog:
        test_message = 'This should be logged in file.'
        logger.debug(test_message)
    logger.debug('This should not be logged in file')

    with open(filename, 'r') as file_in:
        lines = file_in.read().splitlines()
        assert(len(lines) == 1)

        record = lines.pop().split('::')
        level = record[1].strip()
        assert(level == test_level)

        message = record[-1].strip()
        assert(message == test_message)  
    
    os.remove(filename)

def test_debug_logger():
    """Tests ``log`` decorator debug messages"""
    test_message = 'Function in progress.'
    
    @debug(logger)
    def log_test():
        logger.debug(test_message)

    filename = 'test_log.log'
    flog = log_file(filename)

    with flog:
        log_test()

    with open(filename, 'r') as file_in:
        lines = file_in.read().splitlines()

        messages = [line.split('::')[-1].strip() for line in lines]
        
        end = log_test.__qualname__
        assert(messages[0].startswith('Entering') and messages[0].endswith(end))
        assert(messages[-1].startswith('Exiting') and messages[-1].endswith(end))
        assert(test_message in messages)

    os.remove(filename)

def test_debug_info():
    """Tests ``debug`` decorator with INFO level."""

    test_message = 'Function in progress.'
    
    @debug(logger)
    def log_test():
        logger.debug(test_message)
        logger.info(test_message)
        logger.warning(test_message)
        logger.error(test_message)
        logger.critical(test_message)

    filename = 'test_log.log'
    flog = log_file(filename, level='INFO')  # level='INFO' same as console_handler 

    with flog:
        log_test()

    with open(filename, 'r') as file_in:
        lines = file_in.read().splitlines()

        levels = [line.split('::')[0].strip() for line in lines]
        assert('DEBUG' not in levels)

    os.remove(filename)
