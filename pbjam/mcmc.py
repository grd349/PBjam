import emcee
import numpy as np

class mcmc():
    """ Class for MCMC sampling

    Use EMCEE to fit a provided model to a spectrum.

    Parameters
    ----------
    start : ndarray
        TODO
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

    def __init__(self, start,
                 likelihood, lp,
                 nwalkers=50,
                 nthreads=1):
        self.start = start

        self.ndim = len(start)
        self.likelihood = likelihood
        self.lp = lp

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

    def __call__(self, max_iter=20000, spread=1e-4):
        """ Initialize and run the EMCEE afine invariant sampler

        Parameters
        ----------
        spread : float
            Percent spread around the intial position of the walkers. Small
            value starts the walkers in a tight ball, large value fills out
            the range set by parameter bounds.

        Returns
        -------
        sampler.flatchain : array
            The chain of (nwalkers, niter, ndim) flattened to
            (nwalkers*niter, ndim).
        """

        # Start walkers in a tight random ball
        p0 = np.array([self.start + (np.random.randn(self.ndim) * spread) for i in range(self.nwalkers)])

        for sample in self.sampler.sample(p0, iterations=max_iter, progress=True):
            if self.sampler.iteration % 100:
                continue
            if self.converged():
                break

        self.chain = self.sampler.chain.copy()
        self.flatchain = self.sampler.flatchain
        self.lnlike = self.sampler.lnprobability
        self.flatlnlike = self.sampler.flatlnprobability
        self.acceptance = self.sampler.acceptance_fraction

        self.sampler.reset()  # This hopefully minimizes emcee memory leak
        return self.flatchain


    def fold(self, sampler, pos, spread, accept_lim = 0.2):
        """ Fold low acceptance walkers into main distribution

        At the end of the burn-in, some walkers appear stuck with low
        acceptance fraction. These can be selected using a threshold, and
        folded back into the main distribution, estimated based on the median
        of the walkers with an acceptance fraction above the threshold.

        The stuck walkers are relocated with multivariate Gaussian, with mean
        equal to the median of the high acceptancew walkers, and a standard
        deviation equal to the median absolute deviation of these, with a
        small scaling factor.

        Parameters
        ----------
        sampler : emcee sampler object
            The sampler used in the fit
        pos : array
            The final position of the walkers after the burn-in phase
        spread : float
            The factor to apply to the walkers that are adjusted

        """
        idx = sampler.acceptance_fraction < accept_lim
        nbad = np.shape(pos[idx, :])[0]
        if nbad > 0:
            flatchains = sampler.chain[~idx, :, :].reshape((-1, self.ndim))
            good_med = np.median(flatchains, axis = 0)
            good_mad = mad(flatchains, axis = 0) * spread
            pos[idx, :] = np.array([[np.random.uniform(max(self.lp.bounds[j][0], good_med[j]-good_mad[j]),
                                                       min(self.lp.bounds[j][1], good_med[j]+good_mad[j])
                                                       ) for j in range(self.ndim)] for n in range(nbad)])
        return pos
