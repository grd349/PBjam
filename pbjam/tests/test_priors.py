
# from __future__ import division, print_function

# import pytest
# import numpy as np
# from numpy.testing import (assert_almost_equal, assert_array_equal,
#                            assert_allclose)
# from ..priors import kde

# solar_p = [np.log10(135.0), np.log10(3050.0), 1.25,
#            0.5, -2.5, 1.5, 2.2, 0.0, 3.77, 0.8]

# def test_prior_init():
#     ''' Test the kde init function '''
#     kde()

# def test_prior_select_prior_data():
#     ''' Check select prior data runs '''
#     prior = kde()
#     prior.select_prior_data(numax=[np.log10(30.0), 0.01])

# def test_prior_select_prior_data_toofew():
#     ''' Check a warning is raised is too few stars in prior '''
#     prior = kde()
#     with pytest.warns(UserWarning):
#         prior.select_prior_data(numax=[np.log10(30.0), 0.0001])

# def test_prior_select_prior_data_length():
#     ''' Checks that the prior data length is within tolerance '''
#     prior = kde()
#     with pytest.warns(UserWarning):
#         prior.select_prior_data(numax=[np.log10(30.0), 0.01])
#         assert(len(prior.prior_data) == 100)
#         prior.select_prior_data(numax=[np.log10(30.0), 0.001])
#         assert(len(prior.prior_data) == 100)
#         prior.select_prior_data(numax=[np.log10(30.0), 1.0])
#         assert(len(prior.prior_data) == 100)

# #@pytest.mark.slow
# #def test_make_kde():
# #    ''' Tests making the kde '''
# #    prior = kde()
# #    prior.log_obs = {'numax': [np.log10(30.0), 0.01]}
# #    prior.make_kde()
# #
# #def test_prior_prior():
# #    prior = kde()
# #    with pytest.warns(UserWarning):
# #        prior.log_obs = {'numax': [np.log10(3050.0), 0.01]}
# #        prior.make_kde()
# #        assert(np.exp(prior.prior(solar_p)) > 0)

