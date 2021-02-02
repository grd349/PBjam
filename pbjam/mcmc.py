""" 

PBjam uses MC sampling at several points during the peakbagging process. 
Samplers added to PBjam should be called from this module. 

"""

import emcee
import numpy as np
import scipy.stats as st
import cpnest.model
import pandas as pd
import os, logging

from .jar import debug

logger = logging.getLogger(__name__)


class mcmc():
    """ Class for MCMC sampling using `emcee'

    Uses `emcee' to sample the parameterspace of a provided spectrum model.

    Parameters
    ----------
    start : ndarray
        An array of starting position of the parameters.
    likelihood : function
        Function to call that returns the log likelihood when passed the
        parameters.
    prior : function
        Function to call that returns the log prior probability when passed
        the parameters.
    nwalkers : int, optional
        The number of walkers that `emcee' will use.
        
    Attributes
    ----------
    ndim : int
        Number of model parameters (length of start input).
    sampler : emcee.EnsembleSampler class instance
        A `emcee' sampler class instance initialized with the number of walkers,
        number of parameters, and the posterior comprised of the likelihood and
        prior input functions.
    chain : ndarray
        Sampled locations in parameters space of each walker at each step.
    lnlike : ndarray
        Likelihood at the sampled locations in parameter space.
    flatchain : ndarray
        Flattened chain.
    flatlnlike : ndarray
        Flattened likelihoods
    acceptance : ndarray
        Acceptance fraction at each step.
        
    """
    # @debug(logger)
    def __init__(self, start, likelihood, prior, nwalkers=50):

        self.start = start
        self.likelihood = likelihood
        self.prior = prior
        self.nwalkers = nwalkers
        
        self.ndim = len(start)
        self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                             self.ndim,
                                             self.logpost)

        self.chain = None
        self.lnlike = None
        self.flatchain = None
        self.flatlnlike = None
        self.acceptance = None

    def __repr__(self):
        return f'<pbjam.mcmc>'

    def logpost(self, p):
        """ Evaluate the likelihood and prior
        
        Returns the log posterior probability given parameters p. Evaluates
        first the prior function and then the likelihood function. In the 
        event that the prior returns -inf, the function exits.
        
        Parameters
        ----------
        p : list
            Fit parameters
        
        Returns
        -------
        log_posterior: float
            log posterior of the model given parameters p and the observed
            quantities.
        
        """
        
        logp = self.prior(p)
        if logp == -np.inf:
            return -np.inf

        loglike = self.likelihood(p)
        
        return logp + loglike

    def stationarity(self, nfactor=20):
        """ Tests to see if stationarity metrics are satified.
        
        Uses the autocorrelation timescale to estimate whether the MC chains
        have reached a stationary state. 
        
        Parameters
        ----------
        nfactor : int, optional
            Factor used to test stationary. If the number of steps in the
            MC chain exceeds nfactor*tau, where tau is the autocorrelation
            timescale of the chain, the sampling is considered stationary.
            
        """
        
        tau = self.sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * nfactor < self.sampler.iteration)
        return converged

    @debug(logger)
    def __call__(self, max_iter=20000, spread=1e-4, start_samples=[]):
        """ Initialize and run the EMCEE afine invariant sampler

        Parameters
        ----------
        max_iter: int, optional
            Maximum number of steps to take in the sampling. Stationarity is
            tested intermittently, so it might stop before this number is 
            reached.
        spread : float, optional
            Percent spread around the intial position of the walkers. Small
            value starts the walkers in a tight ball, large value fills out
            the range set by parameter bounds.
        start_samples: ndarray, optional
            An array that has samples from the distribution that you want to
            start the sampler at.

        Returns
        -------
        sampler.flatchain : array
            The chain of (nwalkers, niter, ndim) flattened to
            (nwalkers*niter, ndim).
            
        """
        
        nsteps = 1000

        # Start walkers in a tight random ball
        if len(start_samples) == 0:
            # Do this in the case of KDE
            pos = np.array([self.start + (np.random.randn(self.ndim) * spread) for i in range(self.nwalkers)])
        else:
            # Do this in the case of Asy_peakbag, should be replaced with the actual sample
            pos = np.random.randn(self.nwalkers, self.ndim)
            pos *= start_samples.std(axis=0)
            pos += start_samples.mean(axis=0)

        # Burn in
        pos, prob, state = self.sampler.run_mcmc(initial_state=pos, nsteps=nsteps)
        # Fold in low AR chains
        pos = self.fold(pos, spread=spread)
        # Reset sampler
        self.sampler.reset()

        # Run with burnt-in positions
        pos, prob, state = self.sampler.run_mcmc(initial_state=pos, nsteps=nsteps)
        while not self.stationarity():
            pos, prob, state = self.sampler.run_mcmc(initial_state=pos, nsteps=nsteps)
            logger.info(f'Steps taken: {self.sampler.iteration}')
            if self.sampler.iteration == max_iter:
                break
        if self.sampler.iteration < max_iter:
            logger.info(f'Chains reached stationary state after {self.sampler.iteration} iterations.')
        elif self.sampler.iteration == max_iter:
            logger.warning(f'Sampler stopped at {max_iter} (maximum). Chains did not necessarily reach a stationary state.')
        else:
            logger.error('Unhandled exception')

        # Fold in low AR chains and run a little bit to update emcee
        self.fold(pos, spread=spread)
        pos, prob, state = self.sampler.run_mcmc(initial_state=pos, nsteps=100, store=True)

        # Final acceptance
        self.acceptance = self.sampler.acceptance_fraction

        # Estimate autocorrelation time
        tau = self.sampler.get_autocorr_time(tol=0, discard = nsteps).mean()

        # 3D chains
        discard = int(tau*5)
        thin = int(tau/4)
        self.chain = self.sampler.get_chain(discard=discard, thin=thin, flat=False)
        self.lnlike = self.sampler.get_log_prob(discard=discard, thin=thin, flat=False)

        # 2D chains
        self.flatchain = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        self.flatlnlike = self.sampler.get_log_prob(discard=discard, thin=thin, flat=True)

        self.sampler.reset()  # This hopefully minimizes emcee memory leak
        
        return self.flatchain

    # @debug(logger)
    def fold(self, pos, accept_lim = 0.2, spread=0.1):
        """ Fold low acceptance walkers into main distribution

        At the end of the burn-in, some walkers appear stuck with low
        acceptance fraction. These can be selected using a threshold, and
        folded back into the main distribution, estimated based on the median
        of the walkers with an acceptance fraction above the threshold.

        The stuck walkers are redistributed with multivariate Gaussian, with
        mean equal to the median of the high acceptance walkers, and a standard
        deviation equal to the median absolute deviation of these.

        Parameters
        ----------
        pos : ndarray, optional
            The positions of the walkers after the burn-in phase.
        accept_lim: float, optional
            The value below which walkers will be labelled as bad and/or hence
            stuck.
        spread : float, optional
            Factor by which to scatter the folded walkers.
        
        Returns
        -------
        pos : ndarray
            The positions of the walkers after the low accepatance walkers have
            been folded into high acceptance distribution.
        
        """
        
        idx = self.sampler.acceptance_fraction < accept_lim
        nbad = np.shape(pos[idx, :])[0]
        if nbad > 0:
            flatchains = self.sampler.chain[~idx, :, :].reshape((-1, self.ndim))
            good_med = np.median(flatchains, axis = 0)
            good_mad = st.median_absolute_deviation(flatchains, axis = 0) * spread
            pos[idx, :] = np.array([np.random.randn(self.ndim) * good_mad + good_med for n in range(nbad)])
        return pos


class nested(cpnest.model.Model):
    """
    Runs CPnest to performed nested sampling from

    log P(theta | D) ~ likelihood + prior

    Note both likelihood and prior are in natural log.

    Attributes
    ----------

    names: list, strings
        A list of names of the model parameters

    bounds: list of tuples
        The bounds of the model parameters as [(0, 10), (-1, 1), ...]

    likelihood: func
        Function that will return the log likelihood when called as 
        likelihood(params)

    prior: func
        Function that will return the log prior when called as prior(params)

    """
    # @debug(logger)
    def __init__(self, names, bounds, likelihood, prior, path):
        self.names=names
        self.bounds=bounds
        self.likelihood = likelihood
        self.prior = prior
        
        self.path = os.path.join(*[path, 'cpnest'])
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        
    def __repr__(self):
        return f'<pbjam.nested>'

    def log_likelihood(self, param):
        """ Wrapper for log likelihood """
        return self.likelihood(param.values)

    def log_prior(self,p):
        """ Wrapper for log prior """
        if not self.in_bounds(p): return -np.inf
        return self.prior(p.values)

    @debug(logger)
    def __call__(self, nlive=100, nthreads=1, maxmcmc=100, poolsize=100):
        """
        Runs the nested sampling

        Parameters
        ----------
        nlive : int
            Number of live points to be used for the sampling. This is similar 
            to walkers in emcee. Default is 100.
        nthreads : int
            Number of parallel threads to run. More than one is currently slower
            since the likelihood is fairly quick to evaluate. Default is 1. 
        maxmcmc : int
            Maximum number of mcmc steps taken by the sampler. Default is 100.
        poolsize : int
            Number of objects for the affine invariant sampling. Default is 100.

        Returns
        -------
        df: pandas DataFrame
            A dataframe of the samples produced with the nested sampling.
        """
        
        self.nest = cpnest.CPNest(self, verbose=0, seed=53, nthreads=nthreads,
                                  nlive=nlive, maxmcmc=maxmcmc, 
                                  poolsize=poolsize, output=self.path)
        self.nest.run()
        self.samples = pd.DataFrame(self.nest.get_posterior_samples())[self.names]
        self.flatchain = self.samples.values
        self.acceptance = None
        return self.samples
