"""

This module contains general purpose functions that are used throughout PBjam.

"""

from . import PACKAGEDIR
import os, jax, json, dynesty, emcee, time
import jax.numpy as jnp
import numpy as np
from scipy.special import erf
from functools import partial
import scipy.special as sc
import scipy.integrate as si
import scipy.stats as st
from dataclasses import dataclass
import pandas as pd
from dynesty import utils as dyfunc
import pbjam.distributions as dist

class generalModelFuncs():
    """
    A class containing general model functions for various models in PBjam.

    This class in inherited by the model classes.

    """

    def __init__(self):
        pass

    def getMedianModel(self, samplesU=None, rint=None, N=30):
        """
        Computes the median model from a set of N samples drawn from the posterior.

        Parameters
        ----------
        samplesU : dict, optional
            A dictionary of samples where each key corresponds to a parameter
            and each value is a list of sample values for that parameter.
            If None, it uses `self.samples` unpacked using `self.unpackSamples`.
        rint : array-like, optional
            Indices to select specific samples. If None, `N` indices are randomly
            chosen without replacement.
        N : int, optional
            The number of samples to use for computing the median model. Default is 30.

        Returns
        -------
        ndarray
            The median background model computed from the samples.
        """
                
        if samplesU is None:
            samplesU = self.unpackSamples(self.samples)

        mod = np.zeros((len(self.f), N))
        
        rkey = np.random.choice(list(samplesU.keys()))

        Nsamples = len(samplesU[rkey])

        if rint is None:
            rint = np.random.choice(np.arange(Nsamples), size=N, replace=False)
        
        for i, j in enumerate(rint):
            # Extract background parameters for the selected sample
            theta_u = {k: v[j] for k, v in samplesU.items()}
            
            # Compute the background model for the selected sample
            mod[:, i] = self.model(theta_u)
        
        # Compute the median background model across samples
        return np.median(mod, axis=1)
       
    @partial(jax.jit, static_argnums=(0,))
    def obsOnlylnlikelihood(self, theta):
        """
        Computes the log-likelihood using just the obs parameters.

        This ignores all spectrum information.

        Parameters
        ----------
        theta : array-like
            Parameter vector.

        Returns
        -------
        lnlike : float
            Log-likelihood value.
        """

        thetaU = self.unpackParams(theta)
    
        lnlike = self.addAddObsLike(thetaU)

        return lnlike

    def setAddObs(self, keys):
        """ 
        Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 

        Parameters
        ----------
        keys : list
            List of keys for additional observational data.
        """
        
        self.addObs = {}

        for key in keys:
            self.addObs[key] = dist.normal(loc=self.obs[key][0], 
                                           scale=self.obs[key][1])
 
    def chi_sqr(self, mod):
        """ Chi^2 2 dof likelihood

        Evaulates the likelihood of observing the data given the model.

        Parameters
        ----------
        mod : jax device array
            Spectrum model.

        Returns
        -------
        L : float
            Likelihood of the data given the model
        """

        L = -jnp.sum(jnp.log(mod) + self.s / mod)

        return L 
 
    @partial(jax.jit, static_argnums=(0))
    def lnlikelihood(self, theta):
        """
        Calculate the log likelihood of the model given parameters and data.
        
        Parameters
        ----------
        theta : numpy.ndarray
            Parameter values.
        nu : numpy.ndarray
            Array of frequency values.

        Returns
        -------
        float :
            Log-likelihood value.
        """

        thetaU = self.unpackParams(theta)
        
        lnlike = self.addAddObsLike(thetaU)  

        # Constraint from the periodogram 

        mod = self.model(thetaU)
      
        lnlike +=  self.chi_sqr(mod)  
         
        return lnlike 
    
    def addAddObsLike(self, thetaU):
        """ Add the additional probabilities to likelihood
        
        Adds the additional observational data likelihoods to the PSD likelihood.

        Parameters
        ----------
        p : list
            Sampling parameters.

        Returns
        -------
        lnp : float
            The likelihood of a sample given the parameter PDFs.
        """

        lnp = 0

        for key in self.addObs.keys():       
            lnp += self.addObs[key].logpdf(thetaU[key]) 
 
        return lnp
    
    def setLabels(self, addPriors, modelParLabels):
        """
        Set parameter labels and categorize them based on priors.

        Parameters
        ----------
        priors : dict
            Dictionary containing prior information for specific parameters.

        Notes
        -----
        - Initializes default PCA and additional parameter lists.
        - Checks if parameters are marked for PCA and not in priors; if so, 
            adds to PCA list.
        - Otherwise, adds parameters to the additional list.
        - Combines PCA and additional lists to create the final labels list.
        - Identifies parameters that use a logarithmic scale and adds them to 
            logpars list.
        """

        with open("pbjam/data/parameters.json", "r") as read_file:
            availableParams = json.load(read_file)
        
        self.variables = {key: availableParams[key] for key in modelParLabels}

        # Default PCA parameters       
        self.pcaLabels = []
        
        # Default additional parameters
        self.addLabels = []
        
        # If key appears in priors dict, override default and move it to add. 
        for key in self.variables.keys():
            if self.variables[key]['pca'] and (key not in addPriors.keys()):
                self.pcaLabels.append(key)
            else:
                self.addLabels.append(key)
 
        self.logpars = [key for key in self.variables.keys() if self.variables[key]['log10']]

    def unpackSamples(self, samples=None):
        """
        Unpack a set of parameter samples into a dictionary of arrays.

        Parameters
        ----------
        samples : array-like
            A 2D array of shape (n, m), where n is the number of samples and 
            m is the number of parameters.

        Returns
        -------
        S : dict
            A dictionary containing the parameter values for each parameter 
            label.

        Notes
        -----
        This method takes a 2D numpy array of parameter samples and unpacks each
        sample into a dictionary of parameter values. The keys of the dictionary 
        are the parameter labels and the values are 1D numpy arrays containing 
        the parameter values for each sample.

        Examples
        --------
        >>> class MyModel:
        ...     def __init__(self):
        ...         self.labels = ['a', 'b', 'c']
        ...     def unpackParams(self, theta):
        ...         return {'a': theta[0], 'b': theta[1], 'c': theta[2]}
        ...     def unpackSamples(self, samples):
        ...         S = {key: np.zeros(samples.shape[0]) for key in self.labels}
        ...         for i, theta in enumerate(samples):
        ...             theta_u = self.unpackParams(theta)
        ...             for key in self.labels:
        ...                 S[key][i] = theta_u[key]
        ...         return S
        ...
        >>> model = MyModel()
        >>> samples = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> S = model.unpackSamples(samples)
        >>> print(S)
        {'a': array([1., 4., 7.]), 'b': array([2., 5., 8.]), 'c': array([3., 6., 9.])}
        """

        if samples is None:
            samples = self.samples

        S = {key: np.zeros(samples.shape[0]) for key in self.pcaLabels + self.addLabels}
        
        jUnpack = jax.jit(self.unpackParams)

        for i, theta in enumerate(samples):
        
            thetaU = jUnpack(theta)
             
            for key in thetaU.keys():
                
                S[key][i] = thetaU[key]
            
        return S

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
        
                mod = self.model(*thetaU)
                 
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

        print('Warming up the sampler.')
        print(f'Steps | Steps/s | AC time | Delta AC / step (target: {DtauLimit*100}%)):')
        print('----------------------------------------------------------------------')
        while not stop:

            if sampler:
                sampler.reset()

                del sampler

            sampler = emcee.EnsembleSampler(nchains, 
                                            self.ndims, 
                                            self.lnpost,
                                            moves=[(emcee.moves.StretchMove(), 1-DEfrac),
                                                   (emcee.moves.DEMove(), DEfrac),])
            
            walltimeFlag = self._iterateSampler(p0, sampler, nsteps, t0, walltime)

            p0 = sampler.get_last_sample()
            
            chain = sampler.get_chain()
            
            ctrlChain = np.concatenate((ctrlChain, chain[:, :ncontrol, :]), axis=0)
             
            tau, avgDtau = self._getACmetrics(ctrlChain, old_tau, nsteps)
 
            totalSteps = ctrlChain.shape[0]

            converged = self._stopCheck(tau, 
                                       avgDtau,  
                                       DtauLimit,
                                       totalSteps)
            
            runtime = (time.time() - t0)

            print(f'{totalSteps} |  {np.round(totalSteps/runtime, 1)} | {np.round(np.mean(tau), 1)} | {np.round(avgDtau*100, 3)}')

            if converged and earlyStop:
                print('Sampler has likely converged.')
                stop=True
            elif walltimeFlag:
                print('Burn-in stopped due to walltime.')            
                stop=True
 
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
        
        # Number of steps needed to reach nsamples given thinning. 
        stepsNeeded = int(np.ceil(nsamples * thin / pos.shape[0])) + thin 

        maxPossibleSteps = int((maxArrSize/self.ndims/pos.shape[1])/thin)*thin

        nsteps = min([stepsNeeded, maxPossibleSteps]) 
        
        stepsTaken = 0

        chain = np.array([]).reshape((0, *pos.shape))
        
        print()
        print('Sampling from burnt-in chains. ')
        print('Steps | Independent samples')
        print('------------------------------')
        while stepsTaken < stepsNeeded:
        
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

            print(f'{stepsTaken}/{stepsNeeded} | {chain.shape[0]*chain.shape[1]}')
        
        samples = chain.reshape((-1, self.ndims))
        print(samples.shape)
        
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
    
    def runSampler(self, nchains=None, DEfrac=1, earlyStop=True, walltime=60, nsamples=5000, checkEvery=1000, maxThin=1000, ncontrol=10, DtauLimit=5e-5):
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
 
        t0 = time.time()

        sampler, ctrlChain = self.burnInSampler(nchains, DEfrac, earlyStop, walltime, checkEvery, t0, ncontrol, DtauLimit)
 
        pos = self._fold(sampler)
         
        tau = emcee.autocorr.integrated_time(ctrlChain, tol=0)
         
        samples, chain, sampler= self.samplePosterior(pos, tau, DEfrac, nsamples, maxThin, walltime, checkEvery, t0)
      
        print(f'Time taken {np.round((time.time() - t0)/60, 1)} minutes')

        self.samples = samples
        
        self.nsamples = samples.shape[0]

        self.chain = chain

        self.ctrlChain = ctrlChain

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
                   sampler_kwargs={}):
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

def envelope(nu, env_height, numax, env_width, **kwargs):
        """ Power of the seismic p-mode envelope
    
        Computes the power at frequency nu in the oscillation envelope from a 
        Gaussian distribution. Used for computing mode heights.
    
        Parameters
        ----------
        nu : float
            Frequency (in muHz).
        hmax : float
            Height of p-mode envelope (in SNR).
        numax : float
            Frequency of maximum power of the p-mode envelope (in muHz).
        width : float
            Width of the p-mode envelope (in muHz).
    
        Returns
        -------
        h : float
            Power at frequency nu (in SNR)   
        """
 
        return gaussian(nu, 2*env_height, numax, env_width)

def modeUpdoot(result, sample, key, Nmodes):
    """
    Updates the `result` dictionary with summary statistics and samples for a given key.

    Parameters
    ----------
    result : dict
        The result dictionary to be updated. It should contain 'summary' and 'samples' sub-dictionaries.
    sample : ndarray
        The sample data to be added to the result. It is an array of shape (Nsamples, Nmodes).
    key : str
        The key under which the summary statistics and samples are to be stored in the result dictionary.
    Nmodes : int
        The number of modes (columns) in the sample array.    
    """

    result['summary'][key] = np.hstack((result['summary'][key], np.array([smryStats(sample[:, j]) for j in range(Nmodes)]).T))

    result['samples'][key] = np.hstack((result['samples'][key], sample))

def visell1(emm, inc):
    """
    Computes the visibility for l=1 modes based on the azimuthal order (m) and inclination angle.

    Parameters
    ----------
    emm : int
        Absolute value of the azimuthal order (m).
    inc : float
        Inclination angle in radians.

    Returns
    -------
    float
        Visibility for the l=1 modes.
    """
    
    y = jax.lax.cond(emm == 0, 
                     lambda : jnp.cos(inc)**2, # m = 0
                     lambda : jax.lax.cond(emm == 1, 
                                           lambda : 0.5*jnp.sin(inc)**2, # m = 1
                                           lambda : jnp.nan # m > 1
                                           ))
                    
    return y

def visell2(emm, inc):
    """
    Computes the visibility for l=2 modes based on the azimuthal order (m) and inclination angle.

    Parameters
    ----------
    emm : int
        Absolute value of the azimuthal order (m).
    inc : float
        Inclination angle in radians.

    Returns
    -------
    float
        Visibility for the l=2 modes.
    """

    y = jax.lax.cond(emm == 0, 
                     lambda : 1/4 * (3 * jnp.cos(inc)**2 - 1)**2, # m = 0
                     lambda : jax.lax.cond(emm == 1,
                                           lambda : 3/8 * jnp.sin(2 * inc)**2, # m = 1
                                           lambda : jax.lax.cond(emm == 2, 
                                                                 lambda : 3/8 * jnp.sin(inc)**4, # m = 2
                                                                 lambda : jnp.nan # m > 2
                                                                 )))
    return y

def visell3(emm, inc):
    """
    Computes the visibility for l=3 modes based on the azimuthal order (m) and inclination angle.

    Parameters
    ----------
    emm : int
        Absolute value of the azimuthal order (m).
    inc : float
        Inclination angle in radians.

    Returns
    -------
    float
        Visibility for the l=3 modes.
    """

    y = jax.lax.cond(emm == 0, 
                     lambda : 1/64 * (5 * jnp.cos(3 * inc) + 3 * jnp.cos(inc))**2, # m = 0
                     lambda : jax.lax.cond(emm == 1,
                                           lambda : 3/64 * (5 * jnp.cos(2 * inc) + 3)**2 * jnp.sin(inc)**2, # m =1
                                           lambda : jax.lax.cond(emm == 2,
                                                                 lambda : 15/8 * jnp.cos(inc)**2 * jnp.sin(inc)**4, # m = 2
                                                                 lambda : jax.lax.cond(emm == 3, 
                                                                                       lambda : 5/16 * jnp.sin(inc)**6, # m = 3
                                                                                       lambda : np.nan # m > 3
                                                                                       ))))
    return y

def visibility(ell, m, inc):
    """
    Computes the visibility of a mode based on its degree (l), azimuthal order (m), and inclination angle.

    Parameters
    ----------
    ell : int
        The degree of the mode.
    m : int
        The azimuthal order of the mode.
    inc : float
        The inclination angle in radians.

    Returns
    -------
    float
        Visibility for the given mode.
    """

    emm = abs(m)

    y = jax.lax.cond(ell == 0, 
                     lambda : 1.,
                     lambda : jax.lax.cond(ell == 1,
                                           lambda : visell1(emm, inc),
                                           lambda : jax.lax.cond(ell == 2,
                                                                 lambda : visell2(emm, inc),
                                                                 lambda : jax.lax.cond(ell == 3,
                                                                                       lambda : visell3(emm, inc),
                                                                                       lambda : jnp.nan))))
    return y 

def updatePrior(ID, R, addObs):
    """
    Updates the prior data by adding a new entry based on the provided results and additional observations.

    Parameters
    ----------
    ID : str
        The identifier for the new entry.
    R : dict
        A dictionary containing the results. Keys should correspond to parameter names, and values are typically arrays or lists where the first element is used.
    addObs : dict
        A dictionary containing additional observational data, such as 'teff' and 'bp_rp'.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the updated prior data.

    Notes
    -----
    - The function reads the existing prior data from `pbjam/data/prior_data.csv`.
    - It filters out certain keys from the results that are not meant to be updated.
    - It applies a log10 transformation to certain parameters in the results before updating the prior data.
    - The new entry is then added to the prior DataFrame and returned.
    """

    prior = pd.read_csv('pbjam/data/prior_data.csv')

    badkeys = ['freq', 'height', 'width', 'teff', 'bp_rp', 'nurot_c', 'inc', 'H3_power', 'H3_nu', 'H3_exp', 'shot']

    r = {key: [R[key][0]] for key in R.keys() if key not in badkeys}

    for key in r.keys():
        if key in ['eps_p', 'eps_g', 'bp_rp', 'H1_exp', 'H2_exp']:
            continue
        else:
            r[key] = np.log10(r[key])

    r['ID'] = ID

    r['teff'] = np.log10(addObs['teff'][0])
    
    r['bp_rp'] = addObs['bp_rp'][0]

    row = pd.DataFrame.from_dict(r)
     
    prior = prior.append(row, ignore_index=True)

    return prior

@dataclass
class constants:
    """
    A dataclass for storing astrophysical constants and conversion factors.

    Attributes
    ----------
    nu_to_omega : float
        Conversion factor from frequency (muHz) to angular frequency (radians/muHz). Default is `2 * jnp.pi / 1e6`.
    """

    # Teff0: float = 5777 # K
    # TeffRed0: float = 8907 # K
    # numax0: float = 3090 # muHz
    # Delta_Teff: float = 1550 # K
    # Henv0: float = 0.1 # ppm^2/muHz
    nu_to_omega: float = 2 * jnp.pi / 1e6 # radians/muHz
    # dnu0: float = 135.9 # muHz
    # logg0 : float = 4.43775 # log10(2.74e4)

def smryStats(y):
    """
    Computes summary statistics (median and uncertainty) for a given array of samples.

    Parameters
    ----------
    y : array-like
        The input array of samples.

    Returns
    -------
    ndarray
        An array containing the median and the average absolute deviation.

    Notes
    -----
    - The function computes percentiles corresponding to the 16th, 50th, and 84th percentiles.
    - The uncertainty is the mean of the differences between the median and the 16th and 84th percentiles.
    """

    p = np.array([0.5 - sc.erf(n/np.sqrt(2))/2 for n in range(-1, 2)])[::-1]*100
     
    u = np.percentile(y, p)
    
    return np.array([u[1], np.mean(np.diff(u))])

def attenuation(f, nyq):
    """ The sampling attenuation

    Determine the attenuation of the PSD due to the discrete sampling of the
    variability.

    Parameters
    ----------
    f : np.array
        Frequency axis of the PSD.
    nyq : float
        The Nyquist frequency of the observations.

    Returns
    -------
    eta : np.array
        The attenuation at each frequency.
    """

    eta = jnp.sinc(0.5 * f/nyq)

    return eta

def lor(nu, nu0, h, w):
    """ Lorentzian to describe an oscillation mode.

    Parameters
    ----------
    nu0 : float
        Frequency of lorentzian (muHz).
    h : float
        Height of the lorentizan (SNR).
    w : float
        Full width of the lorentzian (muHz).

    Returns
    -------
    mode : ndarray
        The SNR as a function frequency for a lorentzian.
    """

    return h / (1.0 + 4.0/w**2*(nu - nu0)**2)

def getCurvePercentiles(x, y, cdf=None, percentiles=None):
    """ Compute percentiles of value along a curve

    Computes the cumulative sum of y, normalized to unit maximum. The returned
    percentiles values are where the cumulative sum exceeds the requested
    percentiles.

    Parameters
    ----------
    x : array
        Support for y.
    y : array
        Array
    percentiles: array

    Returns
    -------
    percs : array
        Values of y at the requested percentiles.
    """
    if percentiles is None:
        percentiles = [0.5 - sc.erf(n/np.sqrt(2))/2 for n in range(-2, 3)][::-1]

    y /= np.trapz(y, x)
  
    if cdf is None:
        cdf = si.cumulative_trapezoid(y, x, initial=0)
        cdf /= cdf.max()  
         
    percs = np.zeros(len(percentiles))
     
    for i, p in enumerate(percentiles):
        
        q = x[cdf >= p]
          
        percs[i] = q[0]

    return np.sort(percs)

class jaxInterp1D():
 
    def __init__(self, xp, fp, left=None, right=None, period=None):
        """ Replacement for scipy.interpolate.interp1d in jax
    
        Wraps the jax.numpy.interp in a callable class instance.

        Parameters
        ----------
        xp : jax device array 
            The x-coordinates of the data points, must be increasing if argument
             period is not specified. Otherwise, xp is internally sorted after 
             normalizing the periodic boundaries with xp = xp % period.

        fp : jax device array 
            The y-coordinates of the data points, same length as xp.

        left : float 
            Value to return for x < xp[0], default is fp[0].

        right: float 
            Value to return for x > xp[-1], default is fp[-1].

        period : float 
            A period for the x-coordinates. This parameter allows the proper 
            interpolation of angular x-coordinates. Parameters left and right 
            are ignored if period is specified.
        """

        self.xp = xp

        self.fp = fp
        
        self.left = left
        
        self.right = right
        
        self.period = period

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x):
        """ Interpolate onto new axis

        Parameters
        ----------
        x : jax device array 
            The x-coordinates at which to evaluate the interpolated values.

        Returns
        -------
        y : jax device array
            The interpolated values, same shape as x.
        """

        return jnp.interp(x, self.xp, self.fp, self.left, self.right, self.period)

class bibliography():
    """ A class for managing references used when running PBjam.

    This is inherited by session and star. 
    
    Attributes
    ----------
    bibfile : str
        The pathname to the pbjam references list.
    _reflist : list
        List of references that is updated when new functions are used.
    bibdict : dict
        Dictionary of bib items from the PBjam reference list.
    
    """
    
    def __init__(self):
        
        self.bibfile = os.path.join(*[PACKAGEDIR, 'data', 'pbjam_references.bib'])
        
        self._reflist = []
        
        self.bibdict = self._parseBibFile()

    def _findBlockEnd(self, string, idx):
        """ Find block of {}
        
        Go through string starting at idx, and find the index corresponding to 
        the curly bracket that closes the opening curly bracket.
        
        So { will be closed by } even if there are more curly brackets in 
        between.
        
        Note
        ----
        This also works in reverse, so opening with } will be closed by {.
        
        Parameters
        ----------
        string : str
            The string to parse.
        idx : int
            The index in string to start at.         
        """
        
        a = 0
        for i, char in enumerate(string[idx:]):
            if char == '{':
                a -= 1
            elif char == '}':
                a += 1
                
            if (i >= len(string[idx:])-1) and (a != 0):    
                print('Warning: Reached end of bibtex file with no closing curly bracket. Your .bib file may be formatted incorrectly. The reference list may be garbled.')
            if a ==0:
                break  
        
        if string[idx+i] == '{':
            print('Warning: Ended on an opening bracket. Your .bib file may be formatted incorrectly.')
            
        return idx+i
        
    def _parseBibFile(self):
        """ Put contents of a bibtex file into a dictionary.
        
        Takes the contents of the PBjam bib file and stores it as a dictionary
        of bib items.
        
        Article shorthand names (e.g., @Article{shorthand_name) become the
        dictionary key, similar to the way LaTeX handles citations.
        
        Returns
        -------
        bibdict : dict
            Dictionary of bib items from the PBjam reference list.
        """
        
        with open(self.bibfile, 'r') as bib:
            bib = bib.read()
            
            openers = ['@ARTICLE', '@article', '@Article'
                       '@MISC', '@misc',
                       '@BOOK', '@book',
                       '@SOFTWARE', '@software',
                       '@INPROCEEDINGS', '@inproceedings'] #Update this if other types of entries are added to the bib file.
            
            bibitems = []   
            safety = 0
            while any([x in bib for x in openers]):
                for opener in openers:
                    try:
                        start = bib.index(opener)
        
                        end = self._findBlockEnd(bib, start+len(opener))
         
                        bibitems.append([bib[start:end+1]])
        
                        bib = bib[:start] + bib[end+1:]
                            
                    except:
                        pass
                    safety += 1
                    
                    if safety > 1000:
                        break
                    
            bibitems = np.unique(bibitems)
            
            bibdict = {}
            for i, item in enumerate(bibitems):
                key = item[item.index('{')+1:item.index(',')]
                bibdict[key] = item
                
            return bibdict
            
    def _addRef(self, ref):
        """ Add reference from bibdict to active list
        
        The reference must be listed in the PBjam bibfile.
        
        Parameters
        ----------
        ref : str
            Bib entry to add to the list
        """
        if isinstance(ref, list):
            for r in ref:
                self._reflist.append(self.bibdict[r])
        else:
            self._reflist.append(self.bibdict[ref])
        
    def __call__(self, bibfile=None):
        """ Print the list of references used.
        
        Parameters
        ----------
        bibfile : str
            Filepath to print the list of bib items.
        """
        
        out = '\n\n'.join(np.unique(self._reflist))
        print('References used in this run.')
        print(out)
        
        if bibfile is not None:
            with open(bibfile, mode='w') as file_object: #robustify the filepath so it goes to the right place all the time.
                print(out, file=file_object)
                            
def getNormalPercentiles(X, nsigma=2, **kwargs):
    """ Get percentiles of an distribution
    
    Compute the percentiles corresponding to sigma=1,2,3.. including the 
    median (50th), of an array.
    
    Parameters
    ----------
    X : numpy.array()
        Array to find percentiles of
    sigma : int, optional.
        Sigma values to compute the percentiles of, e.g. 68% 95% are 1 and 2 
        sigma, etc. Default is 2.
    kwargs : dict
        Arguments to be passed to numpy.percentile
    
    returns
    -------
    percentiles : numpy.array()
        Numpy array of percentile values of X.
    
    """

    a = np.array([0.5*(1+erf(z/np.sqrt(2))) for z in range(nsigma+1)])
    
    percs = np.append((1-a[::-1][:-1]),a)*100

    return np.percentile(X, percs, **kwargs)

def to_log10(x, xerr):
    """ Transform to value to log10
    
    Takes a value and related uncertainty and converts them to logscale.

    Approximate.

    Parameters
    ----------
    x : float
        Value to transform to logscale
    xerr : float
        Value uncertainty

    Returns
    -------
    logval : list
        logscaled value and uncertainty
    """
    
    if xerr > 0:
        return [np.log10(x), xerr/x/np.log(10.0)]
    return [x, xerr]

def normal(x, mu, sigma):
    """ Evaluate logarithm of normal distribution (not normalized!!)

    Evaluates the logarithm of a normal distribution at x. 

    Inputs
    ------
    x : float
        Values to evaluate the normal distribution at.
    mu : float
        Distribution mean.
    sigma : float
        Distribution standard deviation.

    Returns
    -------
    y : float
        Logarithm of the normal distribution at x
    """

    return gaussian(x, 1/jnp.sqrt(2*jnp.pi*sigma**2), mu, sigma)

def gaussian(x, A, mu, sigma):
    """
    Computes the Gaussian function.

    Parameters
    ----------
    x : array-like
        Input array of x values.
    A : float
        Amplitude of the Gaussian.
    mu : float
        Mean (center) of the Gaussian.
    sigma : float
        Standard deviation (width) of the Gaussian.

    Returns
    -------
    array-like
        The computed Gaussian function values.
    """
        
    return A*jnp.exp(-(x-mu)**2/(2*sigma**2))
