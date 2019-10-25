import emcee
import numpy as np
import scipy.stats as st

class mcmc():
    """ Class for MCMC sampling

    Use EMCEE to fit a provided model to a spectrum.

    Parameters
    ----------
    start : ndarray
        An array of size ndim for the starting position of the parameters.
    likelihood : function
        Function to call that returns the log likelihood when passed the
        parameters.
    lp : function
        Function to call that returns the log prior probability when passed
        the parameters.
    nwalkers : int
        The number of walkers that emcee will use.
    nthreads : int, optional
        Number of multiprocessing threads to use to perform the fit. For long
        cadence data 1 is best, more will just add parallelization overhead.

    """

    def __init__(self, start, likelihood, prior, nwalkers=50, nthreads=1):
        
        self.start = start
        self.ndim = len(start)
        self.likelihood = likelihood
        self.lp = prior

        self.nwalkers = nwalkers
        self.nthreads = nthreads

        self.sampler = emcee.EnsembleSampler(self.nwalkers,
                                             self.ndim,
                                             self.logpost,
                                             threads=self.nthreads)

        self.chain = None
        self.flatchain = None
        self.lnlike = None
        self.flatlnlike = None
        self.acceptance = None

    def logpost(self, p):
        '''
        Returns the log posterior probability given parameters p
        '''
        logp = self.lp(p)
        if logp == -np.inf:
            return -np.inf

        loglike = self.likelihood(p)
        return logp + loglike

    def converged(self, nfactor=20):
        """
        Tests to see if convergence metrics are satified.

        nfactor should be nearer 100 follwing the emcee docs
        but in PBjam useage I get better than reasonable results
        when using 20.
        """
        tau = self.sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * nfactor < self.sampler.iteration)
        return converged
        

    def __call__(self, max_iter=20000, spread=1e-4, start_samples=[]):
        """ Initialize and run the EMCEE afine invariant sampler

        Parameters
        ----------
        max_iter: int
            Don't run the sampler for longer than this.  The sampler will
            stop if convergence is deemed to have been achieved.
        spread : float
            Percent spread around the intial position of the walkers. Small
            value starts the walkers in a tight ball, large value fills out
            the range set by parameter bounds.
        start_samples: ndarray
            An array that has samples from the distribution that you want to
            start the sampler at.

        Returns
        -------
        sampler.flatchain : array
            The chain of (nwalkers, niter, ndim) flattened to
            (nwalkers*niter, ndim).
        """

        # Start walkers in a tight random ball
        if len(start_samples) == 0:
            p0 = np.array([self.start + (np.random.randn(self.ndim) * spread) for i in range(self.nwalkers)])
        else:
            p0 = np.random.randn(self.nwalkers, self.ndim)
            p0 *= start_samples.std(axis=0) * 0.1
            p0 += start_samples.mean(axis=0)

        pos, prob, state = self.sampler.run_mcmc(p0, 1000)
        pos = self.fold(pos, spread)
        self.sampler.reset()

        for sample in self.sampler.sample(pos, iterations=max_iter, progress=True):
            if self.sampler.iteration % 500:
                continue
            if self.converged():
                print(f'Converged after {self.sampler.iteration} iterations.')
                break

        self.chain = self.sampler.chain.copy()
        self.lnlike = self.sampler.lnprobability
        self.acceptance = self.sampler.acceptance_fraction

        tau = self.sampler.get_autocorr_time(tol=0).mean()
        self.flatchain = self.sampler.get_chain(discard=int(tau*5),
                                                          thin=int(tau/4),
                                                          flat=True)
        self.flatlnlike = self.sampler.get_log_prob(discard=int(tau*5),
                                                          thin=int(tau/4),
                                                          flat=True)
        
        self.sampler.reset()  # This hopefully minimizes emcee memory leak
        return self.flatchain


    def fold(self, pos, accept_lim = 0.2, spread=0.1):
        """ Fold low acceptance walkers into main distribution

        At the end of the burn-in, some walkers appear stuck with low
        acceptance fraction. These can be selected using a threshold, and
        folded back into the main distribution, estimated based on the median
        of the walkers with an acceptance fraction above the threshold.

        The stuck walkers are relocated with multivariate Gaussian, with mean
        equal to the median of the high acceptancew walkers, and a standard
        deviation equal to the median absolute deviation of these.

        Parameters
        ----------
        pos : array
            The final position of the walkers after the burn-in phase.
        accept_lim: float
            The value below which walkers will be labelled as bad and/or hence
            stuck.
        """
        idx = self.sampler.acceptance_fraction < accept_lim
        nbad = np.shape(pos[idx, :])[0]
        if nbad > 0:
            flatchains = self.sampler.chain[~idx, :, :].reshape((-1, self.ndim))
            good_med = np.median(flatchains, axis = 0)
            good_mad = st.median_absolute_deviation(flatchains, axis = 0) * spread
            pos[idx, :] = np.array([np.random.randn(self.ndim) * good_mad + good_med for n in range(nbad)])
        return pos
