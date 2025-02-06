"""
The samplers modules contains a set of classes which are meant to be inherited by model classes. 
These classes contain many of the 'standard' methods for sampling with their respective algorithm.
"""

import dynesty, emcee, time, jax
from dynesty import utils as dyfunc
import scipy.stats as st
import jax.numpy as jnp
import numpy as np
from functools import partial

class EmceeSampling():
    """ Class used for handling MCMC sampling with Emcee

    This class is meant to be inherited by various model classes to perform
    affine invariant sampling using the Emcee package.

    """

    def __init__(self):
        pass
    
    def lnprior(self, theta):
        """
        Computes the log-probability of the prior for a given set of parameters.

        Assumes the prior distributions for the parameters are independent.

        If the computed log-prior is finite, it returns the value; otherwise, 
        it returns negative infinity (-jnp.inf).

        Parameters
        ----------
        theta : array-like
            A vector of parameter values.

        Returns
        -------
        lnp : float
            The log-probability of the prior.
        """

        lnp = 0
        
        for i, key in enumerate(self.priors.keys()):
            lnp += self.priors[key].logpdf(theta[i])

        return jax.lax.cond(jnp.isfinite(lnp), lambda : lnp, lambda : -jnp.inf)
    
    @partial(jax.jit, static_argnums=(0))
    def lnpost(self, theta):
        """
        Computes the log-probability of the posterior distribution for a given set of parameters.

        Parameters
        ----------
        theta : array-like
            A vector of parameter values.

        Returns
        -------
        lnp: float
            The log-probability of the posterior, which is the sum of the log-likelihood and the log-prior.
        """

        lnpr = self.lnprior(theta)
        
        lnlk = self.lnlikelihood(theta)
        
        return lnlk + lnpr
    
    def initSamples(self, nchains, spread=[0.45, 0.55]):
        """
        Initializes the starting samples for MCMC chains.

        Draws the samples from the respective parameter priors according the percentiles given by spread.

        Parameters
        ----------
        nchains : int
            The number of chains (starting points) to initialize.
        spread : list of float, optional
            The range of percentiles used to draw from the initial samples from the priors. Default is [0.45, 0.55].

        Returns
        -------
        ndarray
            An array of shape (nchains, ndims) containing the initial samples for each chain.

        Raises
        ------
        ValueError
            If any of the initial parameter draws result in a non-finite posterior evaluation.
        """

        p0 = np.zeros((nchains, self.ndims))

        for i in range(nchains):

            u = np.random.uniform(spread[0], spread[1], size=self.ndims)

            p0[i] = np.array([self.priors[key].ppf(u[i]) for i, key in enumerate(self.priors.keys())])

        # Check none of the starting points are bad        
        for i in range(nchains):
            if not np.isfinite(self.lnpost(p0[i, :])):
                thetaU = self.unpackParams(p0[i, :])
        
                mod = self.model(thetaU)
                 
                print('Parameters:', p0[i, :])
                print('lnlike:', self.lnlikelihood(mod))
                print('lnprior:', self.lnprior(p0[i, :]))
                print('add terms:', self.AddLikeTerms(thetaU))
 
                raise ValueError("One of the parameter draws returned nan in the posterior evaluation")
 
        return p0
    
    def _stopCheck(self, tau, avgDtau, DtauLimit, totalSteps, conservative=False):
        """
        Checks whether the burn-in criteria for an MCMC process are met.

        Checks if the average change in `tau` is below the specified `DtauLimit`.

        If `conservative` is True, it also ensures that the total number of steps is at least 50 times the current `tau` value.

        Parameters
        ----------
        tau : array-like
            An array representing the integrated autocorrelation time.
        avgDtau : float
            The average change in `tau` over recent iterations.
        DtauLimit : float
            The threshold for stopping based on the average change in `tau`.
        totalSteps : int
            The total number of steps taken in the MCMC process.
        conservative : bool, optional
            If True, applies more conservative stopping criteria based on `tau` and total steps. Default is False.

        Returns
        -------
        stop: bool
            True if the stopping criteria are met, False otherwise.
        """

        # Check if the average change in tau has become small
        print(f'Convergence >> 1: {np.round(avgDtau/DtauLimit, 1)}')

        stop = avgDtau < DtauLimit 

        if conservative:
            stop &= np.all(tau * 50 < totalSteps)

        # Add more stopping criteria if necessary.
         
        return stop
    
    def _getACmetrics(self, ctrlChain, oldTau, nsteps, ):
        """
        Computes the autocorrelation metrics for MCMC convergence diagnostics.

        The method calculates the new integrated autocorrelation time (`newTau`) 
        using `emcee.autocorr.integrated_time`.

        The fractional change in `tau` (`Dtau`) is computed as the relative 
        difference between `oldTau` and `newTau`, normalized by `nsteps` to get 
        the relative change per step.

        Parameters
        ----------
        ctrlChain : array-like
            The set of control chains from the MCMC sampling, used to estimate autocorrelation.
        oldTau : array-like
            The previous estimate of the integrated autocorrelation time.
        nsteps : int
            The number of steps in the current MCMC run.

        Returns
        -------
        newTau : array-like
            The updated estimate of the integrated autocorrelation time.
        avgDtau : float
            The average fractional change in `tau` per step.
        """

        newTau = emcee.autocorr.integrated_time(ctrlChain, tol=0)
            
        Dtau = np.abs(oldTau - newTau) / newTau / nsteps 
    
        avgDtau = np.mean(Dtau)

        return newTau, avgDtau
    
    def burnInSampler(self, nchains, DEfrac, earlyStop, walltime, nsteps, t0, ncontrol, DtauLimit):
        """
        Runs the burn-in phase of an MCMC sampler with convergence checking. The sampling is performed
        in sets of nsteps to prevent memory issues. Only ncontrol chains are retained for the entire run,
        otherwise only nsteps are retained. 

        Convergence is primarily monitored using the integrated autocorrelation time (`tau`), where the change
        in the AC time is the main requirement for convergence, stopping the burn-in when the change in tau
        drops below DtauLimit. 

        The chains can be run using a mixture of differential evolution moves and stretch moves (see Emcee documentation).

        Stops if run exceeds the specified walltime.
        
        Parameters
        ----------
        nchains : int
            The number of chains (walkers) to use in the MCMC sampler.
        DEfrac : float
            The fraction of differential evolution (DE) moves to use in the MCMC moves compared to stretch moves.
        earlyStop : bool
            Whether to stop the burn-in early if convergence is detected.
        walltime : float
            The maximum allowed runtime for the burn-in phase (in minutes).
        nsteps : int
            The number of steps to take in each iteration of the burn-in. ~1000 is usually a good choice.
        t0 : float
            The start time used to measure elapsed runtime.
        ncontrol : int
            The number of chains to monitor for autocorrelation convergence, 10-20 is 
            enough to get an OK average.
        DtauLimit : float
            The target limit for the fractional change in autocorrelation time per step. 
            Should be about 0.005%, setting it lower will make the sampler run longer.

        Returns
        -------
        sampler : emcee.EnsembleSampler
            The MCMC sampler after burn-in.
        ctrlChain : ndarray 
            The control chain used for autocorrelation diagnostics.

        Notes
        -----
        - You may run into memory issues if you set this nsteps >20,000 steps. 
        - It is recommended that nsteps is ~1000 to avoid running long when the chain has already converged.

        
        """
        
        p0 = self.initSamples(nchains)
  
        ctrlChain = np.array([]).reshape((0, ncontrol, self.ndims))
        
        old_tau = 1e-10

        stop = False

        sampler = False

        while not stop:

            if sampler:
                sampler.reset()

                del sampler

            sampler = emcee.EnsembleSampler(nchains, 
                                            self.ndims, 
                                            self.lnpost,
                                            moves=[(emcee.moves.StretchMove(), 1-DEfrac),
                                                   (emcee.moves.DEMove(), DEfrac),])
            
            walltimeFlag = self._iterateSampler(p0, sampler, nsteps, t0, walltime, progress=False)

            p0 = sampler.get_last_sample()
            
            chain = sampler.get_chain()
            
            ctrlChain = np.concatenate((ctrlChain, chain[:, :ncontrol, :]), axis=0)
             
            tau, avgDtau = self._getACmetrics(ctrlChain, old_tau, nsteps)
 
            totalSteps = ctrlChain.shape[0]

            converged = self._stopCheck(tau, 
                                       avgDtau,  
                                       DtauLimit,
                                       totalSteps)
            
            # runtime = (time.time() - t0)

            #print(f'{totalSteps} |  {np.round(totalSteps/runtime, 1)} | {np.round(np.mean(tau), 1)} | {np.round(avgDtau*100, 3)}')

            if converged and earlyStop:
                #print('Sampler has likely converged.')
                stop = True
            elif walltimeFlag:
                print('Burn-in stopped due to walltime.')            
                stop = True
            else:
                stop = False

            old_tau = tau 
 
        return sampler, ctrlChain
 
    def samplePosterior(self, pos, tau, DEfrac, nsamples, maxThin, walltime, t0, maxArrSize=5e8):
        """
        Sample the posterior distribution using MCMC, starting from a given position. 
        Assumes the chains have burnt in already. 
 
        Draws enough samples to ensure that, given the thinning, at least `nsamples` 
        independent samples are obtained. The thinning is derived from the AC time, tau.

        The sampling is done in sets corresponding to an integer number of the thinning 
        factor given by tau. This is done to conserve memory. If you are confident in the 
        amount of memory on your machine you can increase `maxArrSize`.

        Parameters
        ----------
        pos : ndarray
            The initial positions of the walkers.
        tau : array-like
            The integrated autocorrelation times for the parameters.
        DEfrac : float
            The fraction of differential evolution (DE) moves to use in the MCMC moves.
        nsamples : int
            The number of independent samples required.
        maxThin : int
            The maximum thinning interval to use, to ensure the sampler doesn't run 
            forever if given a bad tau estimate. Can be large though (~10,000 steps).
        walltime : float
            The maximum allowed runtime for sampling (in minutes).
        t0 : float
            The start time (typically `time.time()`) used to measure elapsed walltime.
        maxArrSize : float, optional
            The maximum size of the array to store the chain, in elements. Default is 5e8.
            If you run into memory issues you can safely decrease this.

        Returns
        ------- 
        samples : array-like
            The posterior samples, flattened across the thinned chains.
        chain : array-like 
            The full set of thinned MCMC chains.
        sampler : emcee.EnsembleSampler
            The MCMC sampler from the last sampling step.
        """
        
        # Get independent samples
        thin = min([int(np.mean(tau)*1.1), maxThin])
        
        # Number of steps needed to reach nsamples given thinning
        stepsNeeded = int(np.ceil(nsamples * thin / pos.shape[0])) + thin 

        # Maximum possible chunk size given memory constraints.
        maxPossibleSteps = int((maxArrSize/self.ndims/pos.shape[1])/thin)*thin

        # steps in a chunk
        nsteps = min([stepsNeeded, maxPossibleSteps]) 
        
        stepsTaken = 0

        chain = np.array([]).reshape((0, *pos.shape))
        
        sampler = None

        while stepsTaken < stepsNeeded:

            # ensures no memory isues
            if sampler:
                sampler.reset()

                del sampler
 
            sampler = emcee.EnsembleSampler(pos.shape[0], 
                                            self.ndims, 
                                            self.lnpost,
                                            moves=[(emcee.moves.StretchMove(), 1-DEfrac),
                                                (emcee.moves.DEMove(), DEfrac),])

            walltimeFlag = self._iterateSampler(pos, sampler, nsteps, t0, walltime, progress=False)
        
            if walltimeFlag:
                print('Posterior sampling stopped due to walltime.')
         
            chain = np.concatenate((chain, sampler.get_chain(thin=thin)), axis=0)

            stepsTaken += nsteps

            nsteps = np.min([stepsNeeded - stepsTaken, nsteps])

        samples = chain.reshape((-1, self.ndims))
        
        return samples, chain, sampler 
       
    def _fold(self, sampler, accept_lim=0.05, spread=0.1):
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
        pos = sampler.get_last_sample().coords

        acceptance = sampler.acceptance_fraction

        chain = sampler.get_chain()

        idx = acceptance < np.percentile(acceptance, accept_lim * 100)
        
        nbad = np.shape(pos[idx, :])[0]
        
        if nbad > 0:
            flatchains = chain[:, ~idx, :].reshape((-1, self.ndims))
        
            good_med = np.median(flatchains, axis = 0)
        
            good_mad = st.median_abs_deviation(flatchains, axis = 0) * spread
        
            pos[idx, :] = np.array([np.random.randn(self.ndims) * good_mad + good_med for n in range(nbad)])
        
        return pos

    def _checkWalltime(self, t0, walltime):
        """
        Checks if the walltime limit has been exceeded.

        Parameters
        ----------
        t0 : float
            The start time (typically `time.time()`) used to measure elapsed walltime.
        walltime : float
            The maximum allowed runtime in minutes.

        Returns
        -------
        bool
            True if the elapsed time exceeds the walltime limit, False otherwise.
        """
         
        return time.time()-t0 > 60*walltime
    
    def _iterateSampler(self, p0, sampler, nsteps, t0, walltime, progress=False, checkEvery=100):
        """
        Iterates the MCMC sampler while checking for walltime constraints.

        The method iterates the MCMC sampler for `nsteps` steps, checking the walltime 
        every `checkEvery` iterations.

        If the walltime is exceeded, the method stops sampling and returns `True` to 
        indicate that the walltime was exceeded.

        Parameters
        ----------
        p0 : array-like
            Initial positions of the walkers.
        sampler : emcee.EnsembleSampler
            The MCMC sampler to iterate.
        nsteps : int
            The number of steps to take in each iteration.
        t0 : float
            The start time (typically `time.time()`) used to measure elapsed walltime.
        walltime : float
            The maximum allowed runtime in minutes.
        progress : bool, optional
            Whether to display progress during sampling. Default is False.
        checkEvery : int, optional
            The number of iterations to allow overrunning the walltime check. Default is 100.

        Returns
        -------
        bool
            True if the walltime was exceeded during the sampling, False otherwise.
        """

        walltimeFlag=False

        for sample in sampler.sample(p0, iterations=nsteps, progress=progress):
        
            if sampler.iteration % checkEvery:
                    continue
            else:
                if self._checkWalltime(t0, walltime):
                    walltimeFlag=True
                    break

        return walltimeFlag
    
    def runSampler(self, nchains=None, DEfrac=1, earlyStop=True, walltime=99999, nsamples=5000, checkEvery=1000, maxThin=1000, ncontrol=10, DtauLimit=5e-5):
        """
        Runs the MCMC sampler, including burn-in and posterior sampling phases.

        Parameters
        ----------
        nchains : int, optional
            The number of chains (walkers) to use in the MCMC sampler. If None, defaults to 6 times the number of dimensions.
        DEfrac : float, optional
            The fraction of differential evolution (DE) moves to use in the MCMC moves. Default is 1.
        earlyStop : bool, optional
            Whether to stop the burn-in early if convergence is detected. Default is True.
        walltime : float, optional
            The maximum allowed runtime in minutes. Default is 60.
        nsamples : int, optional
            The number of independent samples desired. Default is 5000.
        checkEvery : int, optional
            The number of steps to take before checking for convergence. Default is 1000.
        maxThin : int, optional
            The maximum thinning interval to use. Default is 1000.
        ncontrol : int, optional
            The number of chains to monitor for autocorrelation convergence. Default is 10.
        DtauLimit : float, optional
            The target limit for the fractional change in autocorrelation time per step. Default is 0.005%.

        Returns
        -------
        ndarray
            The posterior samples, flattened across thinned chains.
        """

        if nchains is None:
            nchains = 6*self.ndims

        ncontrol = min([ncontrol, nchains])
 
        t0 = time.time()
        print('Burning in sampler')
        burnSampler, ctrlChain = self.burnInSampler(nchains, DEfrac, earlyStop, walltime, checkEvery, t0, ncontrol, DtauLimit)
 
        pos = self._fold(burnSampler)
        
        tau = emcee.autocorr.integrated_time(ctrlChain, tol=0)

        # ensures no memory isues
        if burnSampler:
            burnSampler.reset()

            del burnSampler


        print('Sampling posterior.')                                
        samples, chain, postSampler= self.samplePosterior(pos, tau, DEfrac, nsamples, maxThin, walltime, t0)

        self.samples = samples
        
        self.nsamples = samples.shape[0]

        self.chain = chain

        self.ctrlChain = ctrlChain

        # ensures no memory isues
        if postSampler:
            postSampler.reset()

            del postSampler

        return self.samples
 
class DynestySampling():
    """ Generic dynesty sampling methods to be inherited.
        
        The inheriting class must have a callable lnlikelihood function, a
        dictionary of callable prior ppf functions, and an integer ndims 
        attribute.
        
        """
    def __init__(self):        
        pass
    
    @partial(jax.jit, static_argnums=(0,)) # Must stay jitted.
    def ptform(self, u):
        """
        Transform a set of random variables from the unit hypercube to a set of 
        random variables distributed according to specified prior distributions.

        Parameters
        ----------
        u : jax device array
            Set of pionts distributed randomly in the unit hypercube.

        Returns
        -------
        theta : jax device array
            Set of random variables distributed according to specified prior 
            distributions.

        Notes
        -----
        This method uses the inverse probability integral transform 
        (also known as the quantile function or percent point function) to 
        transform each element of `u` using the corresponding prior 
        distribution. The resulting transformed variables are returned as a 
        JAX device array.

        Examples
        --------
        >>> from scipy.stats import uniform, norm
        >>> import jax.numpy as jnp
        >>> class MyModel:
        ...     def __init__(self):
        ...         self.priors = {'a': uniform(loc=0.0, scale=1.0), 'b': norm(loc=0.0, scale=1.0)}
        ...     def ptform(self, u):
        ...         theta = jnp.array([self.priors[key].ppf(u[i]) for i, key in enumerate(self.priors.keys())])
        ...         return theta
        ...
        >>> model = MyModel()
        >>> u = jnp.array([0.5, 0.8])
        >>> theta = model.ptform(u)
        >>> print(theta)
        [0.5        0.84162123]
        """

        theta = jnp.array([self.priors[key].ppf(u[i]) for i, key in enumerate(self.priors.keys())])

        return theta
 
    def initSamples(self, ndims, nlive, nliveMult=4, logl_kwargs={}, **kwargs):
        """
        Initializes live points for nested sampling.

        The method generates `nliveMult * nlive` points uniformly in the unit cube, 
        and transforms these points into the parameter space using the prior transform. 
        Only the finite log-likelihood values are kept, and the top `nlive` points are returned.

        Parameters
        ----------
        ndims : int
            The number of dimensions in the parameter space.
        nlive : int
            The number of live points to use in the nested sampling.
        nliveMult : int, optional
            The multiplier for the number of live points generated initially. Default is 4.
        logl_kwargs : dict, optional
            Additional keyword arguments to pass to the likelihood function. Default is an empty dictionary.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        list
            A list containing:
            - u (ndarray): The live points in the unit cube.
            - v (ndarray): The live points transformed to the parameter space.
            - L (ndarray): The log-likelihood values of the live points.
        """

        # TODO put in a check for output dims consistent with nlive.
         
        u = np.random.uniform(0, 1, size=(nliveMult * nlive, ndims))

        v = np.array([self.ptform(u[i, :]) for i in range(u.shape[0])])
         
        L = np.array([self.lnlikelihood(v[i, :], **logl_kwargs) for i in range(u.shape[0])])

        idx = np.isfinite(L)
                
        return [u[idx, :][:nlive, :], v[idx, :][:nlive, :], L[idx][:nlive]]
 
    def runSampler(self, dynamic=False, progress=False, minSamples=5000, logl_kwargs={}, 
                   sampler_kwargs={}, **kwargs):
        """
        Runs the nested sampling algorithm, either in static or dynamic mode, to sample the posterior distribution.

        By default assumes 50*ndim nlive points and sampling method is rwalk.

        Parameters
        ----------
        dynamic : bool, optional
            Whether to use dynamic nested sampling. Default is False (static nested sampling).
        progress : bool, optional
            Whether to display progress during sampling. Default is False.
        minSamples : int, optional
            The minimum number of samples to generate. Default is 5000.
        logl_kwargs : dict, optional
            Additional keyword arguments to pass to the likelihood function. Default is an empty dictionary.
        sampler_kwargs : dict, optional
            Additional keyword arguments to pass to the sampler. Default is an empty dictionary.

        Returns
        -------
        samples : array-like
            An array of posterior samples.
        """ 
         
        ndims = len(self.priors)
         
        skwargs = sampler_kwargs.copy()

        if 'nlive' not in skwargs.keys():
            skwargs['nlive'] = 50*self.ndims

        # rwalk seems to perform best out of all the sampling methods...
        if 'sample' not in skwargs.keys():
            skwargs['sample'] = 'rwalk'

        # Set the initial locations of live points based on the prior.
        if 'live_points' not in skwargs.keys() and not dynamic:
            skwargs['live_points'] = self.initSamples(ndims, logl_kwargs=logl_kwargs, **skwargs)
        
        if dynamic:
            sampler = dynesty.DynamicNestedSampler(self.lnlikelihood, 
                                                   self.ptform, 
                                                   ndims,  
                                                   **skwargs,
                                                   logl_kwargs=logl_kwargs,
                                                   )
            
            sampler.run_nested(print_progress=progress, 
                               wt_kwargs={'pfrac': 1.0}, 
                               dlogz_init=1e-3 * (skwargs['nlive'] - 1) + 0.01, 
                               nlive_init=skwargs['nlive'])  
            
            _nsamples = sampler.results.niter

            if _nsamples < minSamples:     
                missingSamples = minSamples-_nsamples

                sampler.run_nested(dlogz=1e-9, print_progress=progress, save_bounds=False, maxiter=missingSamples)

        else:
             
            sampler = dynesty.NestedSampler(self.lnlikelihood, 
                                            self.ptform, 
                                            ndims,  
                                            **skwargs,
                                            logl_kwargs=logl_kwargs,
                                            )
            
            sampler.run_nested(print_progress=progress, 
                               save_bounds=False, dlogz=0.1,)

            _nsamples = sampler.results.niter + sampler.results.nlive
            
            if _nsamples < minSamples:
                missingSamples = minSamples-_nsamples

                sampler.run_nested(dlogz=1e-9, print_progress=progress, save_bounds=False, maxiter=missingSamples)
 
        result = sampler.results

        self.unweighted_samples, self.weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        self.samples = dyfunc.resample_equal(self.unweighted_samples, self.weights)
 
        self.nsamples = self.samples.shape[0]

        self.logz = result.logz
        
        self.logwt = result.logwt

        sampler.reset()

        del sampler

        return self.samples
