"""

The priors module contains the :class:`~pbjam.priors.kde` class, which is used 
to construct the prior for `asy_peakbag' as well as providing the initial 
starting location .

"""

import numpy as np
import pandas as pd
from .mcmc import mcmc
import warnings
from .plotting import plotting
import statsmodels.api as sm
from .jar import get_priorpath, to_log10, normal

class kde(plotting):
    """ A class to produce prior for asy_peakbag and initial starting location.

    This class will take a sample of previously fit stars and compute a 
    multi-variate KDE which acts as a prior for asy_peakbag. 
    
    The class will also compute a posterior, using the computed KDE and the
    the input parameters, numax, dnu, Teff and Gbp-Grp as observational 
    constraints. This posterior is then sampled to estimate the most likely
    initial starting location for asy_peakbag.
    
    Examples
    --------
    Use KDE from the star class instance (recommended)
    
    >>> st = pbjam.star(ID='KIC4448777', pg=pg, numax=[220.0, 3.0], 
                           dnu=[16.97, 0.01], teff=[4750, 100],
                           bp_rp = [1.34, 0.01])
    >>> st.run_kde()
    
    Using KDE on it's own.
    
    >>> K = pbjam.prior.kde()
    >>> K(numax=[220.0, 3.0], dnu=[16.97, 0.01], teff=[4750, 100],
          bp_rp = [1.34, 0.01])

    
    Parameters
    ----------
    starinst : pbjam.star class instance, optional
        A star class instance to use for observational parameters. Default is
        None, which will then require observational parameters be provided at
        the class instance call.
    prior_file : str, optional
        File path to the csv file containing the previous fit values to be used
        to compute the KDE. Default is to use pbjam/data/prior_data.csv 

    """

    def __init__(self, starinst=None, prior_file=None):

        if starinst:
            self.f = starinst.f
            self.s = starinst.s
            self.pg = starinst.pg
            self._obs = starinst._obs
            self._log_obs = starinst._log_obs
            starinst.references._addRef(['statsmodels', 'pandas', 'emcee', 
                                         'numpy'])
            starinst.kde = self

        if prior_file:
            self.prior_file = prior_file
        elif starinst:
                self.prior_file = starinst.prior_file
        else:
            self.prior_file = get_priorpath()

        self.verbose = False

    def select_prior_data(self, numax=None, KDEsize = 100):
        """ Selects useful prior data based on proximity to estimated numax.

        Selects a subset of targets around input numax to use for computing the
        KDE. If the number of targets in the range considered for the prior is
        less than KDEsize, the range will be expanded until it ~KDEsize. This is
        to ensure that the KDE can be constructed. If the initial range includes
        more than KDEsize targets (e.g., in dense parts of the RGB) the 
        selection of targets will be downsampled randomly. The range is expanded
        to a maximum of $20\sigma$.

        Notes
        -----
        Does not ensure that the KDE is finite at the location of your target.

        Parameters
        ----------
        numax : list, optional
            The estimate of numax together with uncertainty in log space.
            Default is none, in which case no selection will be made - all data
            will be used (you probably don't want to do this).
        KDEsize : int, optional
            Number of targets to include in the KDE estimation. Default is 100.

        """

        pdata = pd.read_csv(self.prior_file)

        if numax is None:
            self.prior_data = pdata
        else:
            # Select numax range and expand if needed
            self.prior_data = self._prior_size_check(pdata, numax, KDEsize)

    def _prior_size_check(self, pdata, numax, KDEsize):
        """ Expand numax interval to reach sufficient KDE sample size

        If necessary, increases the range around starting numax, until the 
        sample contains at least KDEsize targets.
        
        Otherwise if the number of targets in the range around the input numax 
        is greater than KDEsize, KDEsize samples will be drawn from the 
        distribution within that range.

        Notes
        -----
        If downsampling is necessary it is done uniformly in numax. Multiplying 
        idx by a Gaussian can be done to change this to a normal distribution. 
        This hasn't been tested yet though.

        Parameters
        ----------
        numax : length 2 list [numax, numax_err]
            The estimate of numax and uncertainty in log-scale.
        KDEsize : int
            Number of targets to include in the KDE estimation.             

        Returns
        -------
        prior_data : panda
            Array of length equal to the total prior data sample that will be 
            used to compute the KDE. 1 for targets that are included in the 
            KDE estimation, and 0 otherwise.

        """

        nsigma = 1
        
        idx = np.abs(pdata.numax.values - numax[0]) < nsigma * numax[1]

        flag_warn = False
        while len(pdata[idx]) < KDEsize:
            idx = np.abs(pdata.numax.values - numax[0]) < nsigma * numax[1]

            if not flag_warn:
                warnings.warn(f'Only {len(pdata[idx])} star(s) near provided numax. ' +
                f'Trying to expand the range to include ~{KDEsize} stars.', stacklevel=2)
                flag_warn = True

            if nsigma >= KDEsize:
                break

            nsigma += 0.1
        
        ntgts = len(idx[idx==1])
        
        if ntgts == 0:
            raise ValueError('No prior targets found within range of target. This might mean no prior samples exist for stars like this, consider increasing the uncertainty on your numax input.')

        elif ntgts < KDEsize:
            warnings.warn(f'Sample for estimating KDE is less than the requested {KDEsize}.', stacklevel=2)
            KDEsize = ntgts
        
        return pdata.sample(KDEsize, weights=idx, replace=False)



    def make_kde(self, bw_fac=1.0):
        """ Takes the prior data and constructs a KDE function

        Computes the KDE based on the parameters of a previously fit sample of
        targets.
        
        The KDE bandwidth is by default computed automatically using the 
        cross-validated maximum liklihood method in the `statsmodels' package. 
        
        Notes
        -----
        If the bandwidth factor is != 1 the method for calculating the initial
        bandwidth is the 'scott' method. Otherwise it is the cross-validated
        maximum likelihood method, employed by the `statsmodels' package. 
        This is currently a limitation of imposed by the current version of 
        `statsmodels'.
        
        Parameters
        ----------
        bw_fac : float, optional
            Factor for expanding the bandwidth of the KDE. If float-like the 
            scaling will be the same for all paramaters. If array-like it must
            be of the same length as the number of fit parameters. Each scaling
            will then be applied individually to each parameter.
            
        """

        self.par_names = ['dnu', 'numax', 'eps', 'd02', 'alpha', 'env_height',
                          'env_width', 'mode_width', 'teff', 'bp_rp']

        self.select_prior_data(self._log_obs['numax'])

        if self.verbose:
                print(f'Selected data set length {len(self.prior_data)}')

        if bw_fac != 1:
            from statsmodels.nonparametric.bandwidths import select_bandwidth
            bw = select_bandwidth(self.prior_data[self.par_names].values,
                                  bw = 'scott', kernel=None) 
            bw *= bw_fac
            
        else:
            if self.verbose:
                print('Selecting sensible stars for kde')
                print(f'Full data set length {len(self.prior_data)}')
            bw = 'cv_ml'

        self.kde = sm.nonparametric.KDEMultivariate(
                            data=self.prior_data[self.par_names].values,
                            var_type='c'*len(self.par_names), bw=bw)



    def prior(self, p):
        """ Calculates the log prior for the initial guess fit.

        Evaluates the KDE for the parameters p. Additional hard/soft priors
        can be added here as needed to, e.g., apply boundaries to the fit.

        Hard constraints should be applied at the top so function exits early,
        if necessary.

        Parameters
        ----------
        p : ndarray
            Array of model parameters to evaluate the prior at.

        Returns
        -------
        lp : float
            The log likelihood evaluated at p.

        """

        # d02/dnu < 0.2  (np.log10(0.2) ~ -0.7)
        if p[3] - p[0] > -0.7:
            return -np.inf

        # Constraint from prior
        lp = np.log(self.kde.pdf(p))

        return lp

    def likelihood(self, p):
        """ Calculate likelihood for the initial guess fit

        Calculates the likelihood of the observed properties given the proposed
        parameters p.

        Parameters
        ----------
        p : ndarray
            Array of model parameters to evaluate the likelihood at.

        Returns
        -------
        lnlike : float
            The log likelihood evaluated at p.

        """

        lnlike = 0.0

        # Constraints from observational data
        lnlike += normal(p[0], *self._log_obs['dnu'])
        lnlike += normal(p[1], *self._log_obs['numax'])
        lnlike += normal(p[8], *self._log_obs['teff'])
        lnlike += normal(p[9], *self._obs['bp_rp'])

        return lnlike

    def kde_predict(self, n):
        """ Predict the l=0 mode frequencies from the KDE samples.

        Takes the samples drawn using the kde_sampler method and produces a
        distribution of where it thinks each radial mode should be.

        Parameters
        ----------
        n: numpy_array
            The radial order

        Returns
        -------
        freq_mean: ndarray
            A numpy array of length len(n) containing the mean frequency 
            estimates.
        freq_std: ndarray
            A numpy array of length len(n) containing the standard deviation
            of the frequency estimates.
        """

        if not hasattr(self, 'samples'):
            raise ValueError('Need to run the sampler first')
        
        dnu = 10**self.samples[:, 0]
        eps = self.samples[:, 2]
        nmax = 10**self.samples[:, 1] / dnu - eps
        alpha = 10**self.samples[:, 4]
        freq = np.array([(nn + eps + alpha/2.0 * (nn - nmax)**2) * dnu for nn in n])
        
        return freq.mean(axis=1), freq.std(axis=1)


    def kde_sampler(self, nwalkers=50):
        """ Samples the posterior distribution with the KDE prior

        Draws samples from the posterior given by
        
        $p(\theta|D) \propto p(\theta) p(D|\theta)$,

        where $p(\theta)$ is given by the KDE function of the prior data and
        $p(D|\theta)$ is given by the observable constraints

        Samples are drawn using the `emcee' package.
        
        Parameters
        ----------
        nwalkers : int
            Number of walkers to use to sample the posterior. This is passed
            to the `emcee' package. Default is 50.

        Returns
        -------
        flatchain : ndarray
            Flattened chains from the emcee sampling.

        """

        if self.verbose:
            print('Running KDE sampler')

        x0 = [self._log_obs['dnu'][0],  # log10 dnu
              self._log_obs['numax'][0],  # log10 numax
              1.0,  # eps
              np.log10(0.1 * self._obs['dnu'][0]),  # log10 d02
              -2.0,  # log10 alpha
              1.0,  # log10 env height
              1.0,  # log10 env width,
              -1.0,  # log10 mode width
              self._log_obs['teff'][0],
              self._obs['bp_rp'][0]]

        self.fit = mcmc(x0, self.likelihood, self.prior, nwalkers=nwalkers)

        flatchain = self.fit()
        
        return flatchain


    def __call__(self, dnu=[1, -1], numax=[1, -1], teff=[1, -1],
                 bp_rp=[1, -1], verbose=False, bw_fac=1.0):
        """ Compute and sample the KDE.

        Performs all the steps needed for using the KDE in the peakbagging
        process.

        Inputs
        ------
        dnu : list, optional
            Large frequency spacing and uncertainty    
        numax : list, optional
            Frequency of maximum power and uncertainty     
        teff : list, optional
            Stellar effective temperature and uncertainty            
        bp_rp : list, optional
            The Gaia Gbp - Grp color value and uncertainty
            (probably ~< 0.01)            
        verbose : bool, optional
            Should PBjam say something at this stage. Default is silence.            
        bw_fac : float, optional
            Factor for expanding the bandwidth of the KDE. If float-like the 
            scaling will be the same for all paramaters. If array-like it must
            be of the same length as the number of fit parameters. Each scaling
            will then be applied individually to each parameter.
            
        """
        
        self.verbose = verbose

        if not hasattr(self, '_obs'):
            self._obs = {'dnu': dnu, 'numax': numax, 'teff': teff, 'bp_rp': bp_rp}
            self._log_obs = {x: to_log10(*self._obs[x]) for x in self._obs.keys() if x != 'bp_rp'}
                
        self.make_kde(bw_fac)

        self.samples = self.kde_sampler()

        self.result = [self.samples[:, 2].mean(), self.samples[:, 2].std()]
