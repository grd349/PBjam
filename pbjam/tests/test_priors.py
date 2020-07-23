
from __future__ import division, print_function

import numpy as np
from numpy.testing import assert_almost_equal
from ..priors import kde
import pytest
from ..jar import to_log10
import pbjam.tests.pbjam_tests as pbt
import pandas as pd
import statsmodels.api as sm

cs = pbt.case('silly')

simp_kde = kde()
pdata = pd.read_csv(simp_kde.prior_file)

solar_p = np.array([np.log10(135.0), np.log10(3050.0), 1.25,
            0.5, -2.5, 1.5, 2.2, 0.0, 3.77, 0.8])


def test_kde_init():
    """ Test the kde init function """
    kde()

def test_select_prior_data():
    """Tests for kde.select_prior_data"""

    # Setup
    KDEsize = 87
    func = simp_kde.select_prior_data

    # Simplest run test. Given no numax it should return a sample that equals
    # the whole prior_data.csv
    pbt.does_it_run(func, None)
    assert(len(simp_kde.prior_data) == len(pdata))

    # With a given numax and KDEsize, should return KDEsize elements in the
    # prior_data attribute
    func(to_log10(30, 10), KDEsize)
    assert(len(simp_kde.prior_data) == KDEsize)


def test_prior_size_check():
    """Tests for kde.prior_size_check"""

    # Setup
    func = simp_kde._prior_size_check

    # Simple return test
    pbt.does_it_return(func, [pdata, to_log10(30, 10), 100])

    # These combinations should be OK
    for KDEsize in [10, 20]:
        for numax in [30, 200]:
            for sigma in [10, 100]:
                pdata_cut = func(pdata, to_log10(numax, sigma), KDEsize)
                assert((len(pdata_cut) > 0) & (len(pdata_cut) <= KDEsize))

    # These combinations should show warnings
    with pytest.warns(UserWarning):
        func(pdata, to_log10(300, 1), 500)

    # These combinations should raise errors
    with pytest.raises(ValueError):
        func(pdata, to_log10(300000, 1), 500)

def test_prior():
    """Tests for kde.prior"""

    # Setup
    simp_kde.kde = type('kde', (object,), {})()
    simp_kde.kde.samples = np.ones((2,10))
    data = np.array(solar_p.repeat(11).reshape((10,-1)))
    simp_kde.kde = sm.nonparametric.KDEMultivariate(data=data, var_type='cccccccccc', bw='scott')
    func = simp_kde.prior
    inp = [solar_p]

    # basic run/return/type/shape tests
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, float)

    # Should be nan given the dummy inputs above
    assert(np.isnan(func(*inp)))

def test_likelihood():
    """Tests for the kde.likelihood method"""

    # Setup
    simp_kde.kde = type('kde', (object,), {})()
    simp_kde.kde.samples = np.ones((2,10))
    simp_kde._obs = {'dnu': [solar_p[0],0.1], 'numax': [solar_p[1],0.1], 'teff': [solar_p[-2],0.1], 'bp_rp': [solar_p[-1],0.1]}
    simp_kde._log_obs = {x: to_log10(*simp_kde._obs[x]) for x in simp_kde._obs.keys() if x != 'bp_rp'}

    # This is a silly KDE example that should return nonsense
    data = np.array(solar_p.repeat(11).reshape((10,-1)))
    simp_kde.kde = sm.nonparametric.KDEMultivariate(data=data, var_type='cccccccccc', bw='scott')

    func = simp_kde.likelihood
    inp = [solar_p]

    # basic run/return/type/shape tests
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, float)

    # Should return real values < 0 for the dummy inputs above.
    assert(np.isreal(func(*inp)))
    assert(func(*inp) < 0)

def test_kde_predict():
    """Tests for the kde.kde_predict method"""

    # Test raise statement
    with pytest.raises(ValueError):
        kde().kde_predict(8)

    # Setup
    err = 0.01
    simp_kde.samples = np.vstack([np.random.normal(m, err*abs(m), 50) for m in solar_p]).T
    enns = range(15,25)
    func = simp_kde.kde_predict
    inp = [enns]

    # basic run/return/type/shape tests
    pbt.does_it_run(func, inp)
    pbt.does_it_return(func, inp)
    pbt.right_type(func, inp, tuple)
    pbt.right_shape(func, inp, (2, 10))

    # Test that errors propogate more or less correctly (probably not the most
    # robust test)
    out = simp_kde.kde_predict(enns)
    assert_almost_equal(out[1]/out[0], np.ones_like(out[1])*err, decimal = 1)


# The test functions below require longer runs and are not suitable for GitHub
# workflows. the mark.slow decorator doesn't seem to work with GitHub workflows.

#@pytest.mark.slow

#test_kde_sampler
#def test_make_kde():
#def test_kde_call():

#def test_make_kde():
#    """ Tests making the kde """
#    prior = kde()
#    prior.log_obs = {'numax': [np.log10(30.0), 0.01]}
#    prior.make_kde()
#
#def test_prior_prior():
#    prior = kde()
#    with pytest.warns(UserWarning):
#        prior.log_obs = {'numax': [np.log10(3050.0), 0.01]}
#        prior.make_kde()
#        assert(np.exp(prior.prior(solar_p)) > 0)
