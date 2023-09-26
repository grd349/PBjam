from functools import partial
import jax, dynesty, warnings, os
import jax.numpy as jnp
from dynesty import utils as dyfunc
import numpy as np
from pbjam import jar
from pbjam.DR import PCA
import pbjam.distributions as dist
from pbjam.mixedmodel import MixFreqModel
from pbjam.pairmodel import AsyFreqModel
from pbjam.background import bkgModel
from pbjam.plotting import plotting
import pandas as pd

class modeIDsampler(plotting):

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freq_limits=[1, 5000], 
                 vis={'V20': 0.71, 'V10': 1.22}, Npca=50, PCAdims=8, 
                 priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        self.set_labels(self.addPriors)

        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.setupDR()
  
        self.setPriors()
 
        self.AsyFreqModel = AsyFreqModel(self.N_p)

        self.ndims = len(self.latentLabels + self.addlabels)

        n_g_ppf, _, _, _ = self._makeTmpSample(['DPi1', 'eps_g'])
        
        self.MixFreqModel = MixFreqModel(self.N_p, self.obs, n_g_ppf)

        self.N_g = self.MixFreqModel.N_g
        
        self.background = bkgModel(self.Nyquist)

        self.sel = self.setFreqRange()

        self.setAddObs()

        self.trimVariables()

    def trimVariables(self):
    
        for i in range(self.N_g + self.N_p, 100, 1):
            del self.addlabels[self.addlabels.index(f'freqError{i}')]
            del self.labels[self.labels.index(f'freqError{i}')]
            del self.priors[f'freqError{i}']
            
        self.ndims = len(self.priors)   
 
    def _makeTmpSample(self, keys, N=1000):
        """
        Draw samples for specified keys.

        Parameters
        ----------
        keys : list
            List of parameter keys to be sampled.
        N : int, optional
            Number of samples to generate. Default is 1000.

        Returns
        -------
        tuple
            A tuple containing the quantile function, probability density 
            function, log probability density function, and cumulative 
            distribution function of the generated samples.

        Notes
        -----
        - Generates N random samples.
        - Transforms the samples using `ptform` and `unpackParams`.
        - Constructs arrays for specified keys.
        - Computes quantile function, probability density function,
        log probability density function, and cumulative distribution function.
        """

        K = np.zeros((len(keys), N))

        for i in range(N):
            u = np.random.uniform(0, 1, size=self.ndims)
        
            theta = self.ptform(u)

            theta_u = self.unpackParams(theta)

            K[:, i] = np.array([theta_u[key] for key in keys]) 
        
        ppf, pdf, logpdf, cdf = dist.getQuantileFuncs(K.T)
        
        return ppf, pdf, logpdf, cdf

    def set_labels(self, priors):
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

        # Default PCA parameters       
        self.pcalabels = []
        
        # Default additional parameters
        self.addlabels = []
        
        # If key appears in priors dict, override default and move it to add.
        for key in self.variables.keys():
            if self.variables[key]['pca'] and (key not in priors.keys()):
                self.pcalabels.append(key)

            else:
                if key == 'freqError':
                    for i in range(100):
                        self.addlabels.append(f'freqError{i}')
                else:
                    self.addlabels.append(key)
  
        self.labels = self.pcalabels + self.addlabels

        # Parameters that are in log10
        self.logpars = []
        for key in self.variables.keys():
            if self.variables[key]['log10']:
                self.logpars.append(key)
                # if key.startswith('p_L') or key.startswith('p_D'):
                #     self.logpars.append(key[:3])

    @partial(jax.jit, static_argnums=(0,))
    def unpackParams(self, theta): 
        """ Cast the parameters in a dictionary

        Parameters
        ----------
        theta : array
            Array of parameters drawn from the posterior distribution.

        Returns
        -------
        theta_u : dict
            The unpacked parameters.

        """
         
        theta_inv = self.DR.inverse_transform(theta[:self.DR.dims_R])

        theta_u = {key: theta_inv[i] for i, key in enumerate(self.pcalabels)}
         
        theta_u.update({key: theta[self.DR.dims_R:][i] for i, key in enumerate(self.addlabels)})

        theta_u['p_L'] = jnp.array([(theta_u['u1'] + theta_u['u2'])/jnp.sqrt(2)])

        theta_u['p_D'] = jnp.array([(theta_u['u1'] - theta_u['u2'])/jnp.sqrt(2)])

        # theta_u['p_L'] = jnp.array([theta_u[key] for key in theta_u.keys() if 'p_L' in key])

        # theta_u['p_D'] = jnp.array([theta_u[key] for key in theta_u.keys() if 'p_D' in key])

        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
    
    def setPriors(self):
        """ Set the prior distributions.

        The prior distributions are constructed from the projection of the 
        PCA sample onto the reduced dimensional space.

        """

        self.priors = {}

        for i, key in enumerate(self.latentLabels):
            self.priors[key] = dist.distribution(self.DR.ppf[i], 
                                                 self.DR.pdf[i], 
                                                 self.DR.logpdf[i], 
                                                 self.DR.cdf[i])

        orderedAddKeys = [k for k in self.variables if k in self.addPriors.keys()]
        self.priors.update({key : self.addPriors[key] for key in orderedAddKeys})
        
        # Core rotation prior
        self.priors['nurot_c'] = dist.uniform(loc=-2., scale=2.)

        self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        self.priors['inc'] = dist.truncsine()

        # The instrumental components are set based on the PSD, not Bayesian but...
        hi_idx = self.f > min([self.f[-1], self.Nyquist]) - 10
        shot_est = jnp.nanmean(self.s[hi_idx])

        lo_idx = abs(self.f - self.f[0]) < 10
        inst_est = jnp.nanmean(self.s[lo_idx])
        
        mu = jnp.array([1, inst_est - shot_est]).max()
        
        self.priors['H3_power'] = dist.normal(loc=jnp.log10(mu * self.f[0]), scale=1)  

        self.priors['H3_nu'] = dist.beta(a=1.2, b=1.2, loc=-1, scale=2)  
        
        self.priors['H3_exp'] = dist.beta(a=1.2, b=1.2, loc=1.5, scale=3.5)  
 
        self.priors['shot'] = dist.normal(loc=jnp.log10(shot_est), scale=0.1)

        for i in range(200):
            self.priors[f'freqError{i}'] = dist.normal(loc=0, scale=1/20 * self.obs['dnu'][0])

    @partial(jax.jit, static_argnums=(0,))
    def add20Pairs(self, modes, nu, d02, mode_width, nurot_e, inc, **kwargs):
 
        nu0_p, n_p = self.AsyFreqModel.asymptotic_nu_p(**kwargs)

        Hs0 = self.envelope(nu0_p, **kwargs)

        for n in range(self.N_p):

            modes += jar.lor(nu, nu0_p[n], Hs0[n], mode_width) 
            
            for m in [-2, -1, 0, 1, 2]:
                
                H = Hs0[n] * self.vis['V20'] * jar.visell2(abs(m), inc)
                
                f = nu0_p[n] - d02 + m * nurot_e

                modes += jar.lor(nu, f, H, mode_width)

        return modes, nu0_p, n_p
    
    @partial(jax.jit, static_argnums=(0,))
    def addl1(self, modes, nu, nu0_p, n_p, nurot_c, nurot_e, inc,  **kwargs):
        
        nu1s, zeta = self.MixFreqModel.mixed_nu1(nu0_p, n_p, **kwargs)
 
        Hs1 = self.envelope(nu1s, **kwargs)
        
        modewidth1s = self.l1_modewidths(zeta, **kwargs)
         
        nurot = zeta * nurot_c + (1 - zeta) * nurot_e
        
        for i in range(len(nu1s)):
            
            nul1 = nu1s[i] + kwargs[f'freqError{i}']

            modes += jar.lor(nu, nul1                     , Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.cos(inc)**2
        
            modes += jar.lor(nu, nul1 - zeta[i] * nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(inc)**2 / 2
        
            modes += jar.lor(nu, nul1 + zeta[i] * nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(inc)**2 / 2

        return modes

    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u, nu):
        
        # Background
        bkg = self.background(theta_u, nu)
 
        modes = jnp.ones_like(nu)

        # l=2,0
        modes, nu0_p, n_p = self.add20Pairs(modes, nu, **theta_u)
        
        # l=1
        modes = self.addl1(modes, nu, nu0_p, n_p, **theta_u)
         
        return modes * bkg

    def setupDR(self):
        """ Setup the latent parameters and projection functions

        Parameters
        ----------
        prior_file : str
            Full path name for the file containing the prior samples.
 
        """

        self.latentLabels = ['theta_%i' % (i) for i in range(self.PCAdims)]

        _obs = self.log_obs.copy()

        for key in ['bp_rp']:
            _obs[key] = self.obs[key]
         
        self.DR = PCA(_obs, self.pcalabels, self.priorpath, self.Npca, selectlabels=['numax', 'dnu', 'teff', 'bp_rp']) 

        self.DR.fit_weightedPCA(self.PCAdims)

        _Y = self.DR.transform(self.DR.data_F)

        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(_Y)

    def setFreqRange(self, envelope_only=False):
        """ Get frequency range around numax for model 

        Returns
        -------
        idx : jax device array
            Array of boolean values defining the interval of the frequency axis
            where the oscillation modes present.
        """

        if envelope_only:
            dnu =  self.obs['dnu'][0] if 'dnu' in self.obs.keys() else jnp.median(self.DR.data_F[:, 0])
    
            numax =  self.obs['numax'][0] if 'numax' in self.obs.keys() else jnp.median(self.DR.data_F[:, 1])
    
            eps_p =  self.obs['eps_p'][0] if 'eps_p' in self.obs.keys() else jnp.median(self.DR.data_F[:, 2])
        
            n_p_max = self.AsyFreqModel._get_n_p_max(dnu, numax, eps_p)

            n_p = self.AsyFreqModel._get_n_p(n_p_max)
            
            # The range is set +/- 25% of the upper and lower mode frequency 
            # estimate
            lfreq = (min(n_p) - 1.25 + eps_p) * dnu
            
            ufreq = (max(n_p) + 1.25 + eps_p) * dnu

        else:
            lfreq = self.freq_limits[0]
            
            ufreq = self.freq_limits[1]

        return (lfreq < self.f) & (self.f < ufreq)  
    
    @partial(jax.jit, static_argnums=(0,))
    def envelope(self, nu, env_height, numax, env_width, **kwargs):
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
    
        return jar.gaussian(nu, 2*env_height, numax, env_width)
    
    @partial(jax.jit, static_argnums=(0,))
    def l1_modewidths(self, zeta, mode_width, fac=1, **kwargs):
        """ Compute linewidths for mixed l1 modes

        Parameters
        ----------
        modewidth0 : jax device array
            Mode widths of l=0 modes.
        zeta : jax device array
            The mixing degree

        Returns
        -------
        modewidths : jax device array
            Mode widths of l1 modes.
        """
         
        return  fac * mode_width * jnp.maximum(0, 1. - zeta) 
      
    def setAddObs(self):
        """ Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 
        """
        
        self.addObs = {}

        self.addObs['teff'] = dist.normal(loc=self.obs['teff'][0], 
                                          scale=self.obs['teff'][1])
 
        self.addObs['bp_rp'] = dist.normal(loc=self.obs['bp_rp'][0], 
                                           scale=self.obs['bp_rp'][1])
        
    @partial(jax.jit, static_argnums=(0,))
    def addAddObsLike(self, theta_u):
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
               
        lnp = self.addObs['teff'].logpdf(theta_u['teff']) 

        lnp += self.addObs['bp_rp'].logpdf(theta_u['bp_rp']) 

        return lnp
    
    @partial(jax.jit, static_argnums=(0,))
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

        L = -jnp.sum(jnp.log(mod) + self.s[self.sel] / mod)

        return L      
         
    @partial(jax.jit, static_argnums=(0,))
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
     
    @partial(jax.jit, static_argnums=(0,))
    def lnlikelihood(self, theta, nu):
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
    
        theta_u = self.unpackParams(theta)
 
        # Constraint from input obs
        lnlike = self.addAddObsLike(theta_u)

        # Constraint from the periodogram 
        mod = self.model(theta_u, nu)
         
        lnlike += self.chi_sqr(mod)
         
        return lnlike
    
    def __call__(self, dynesty_kwargs={}):
        """
        Run the Dynesty sampler.

        Parameters
        ----------
        dynesty_kwargs : dict, optional
            Additional keyword arguments for the Dynesty sampler. 

        Returns
        -------
        samples : jax device array
            The generated samples
        result : dict
            Dictionary of parsed result. Contains summary statistics and samples
            in human-readable form.

        Notes
        -----
        - Calls the `runDynesty` method with provided keyword arguments.
        - Unpacks and parses the generated samples.
        - Stores the parsed result and returns both the samples and the result.
        """

        self.sampler, self.samples = self.runDynesty(**dynesty_kwargs)

        samples_u = self.unpackSamples()

        self.result = self.parseSamples(samples_u)

        return self.samples, self.result

    def getInitLive(self, nlive):
        """
        Generate initial live points for a Bayesian inference problem.

        Parameters
        ----------
        nlive : int
            The number of live points to generate.

        Returns
        -------
        list : list
            A list containing three arrays: [u, v, L].
                - u : ndarray
                    Initial live points in the unit hypercube [0, 1].
                    Shape: (nlive, ndims).
                - v : ndarray
                    Transformed live points obtained by applying the ptform method to each point in u.
                    Shape: (nlive, ndims).
                - L : ndarray
                    Log-likelihood values calculated for each point in v.
                    Shape: (nlive,).

        Notes
        -----
        This method generates initial live points for a Bayesian inference problem.
        It follows the following steps:
        1. Generate a 2D array u of shape (4*nlive, ndims) with values drawn from a uniform distribution in the range [0, 1].
        2. Apply the ptform method to each row of u to obtain a new 2D array v of the same shape.
        3. Calculate the log-likelihood values L for each point in v using the lnlikelihood method.
        4. Filter out invalid values of L (NaN or infinite) using a boolean mask.
        5. Select the first nlive rows from the filtered arrays to obtain the initial live points u, transformed points v, and log-likelihood values L.
        6. Return the list [u, v, L].
        """
        
        u = np.random.uniform(0, 1, size=(4*nlive, self.ndims))

        v = np.array([self.ptform(u[i, :]) for i in range(u.shape[0])])
        
        L = np.array([self.lnlikelihood(v[i, :], self.f[self.sel]) for i in range(u.shape[0])])

        idx = np.isfinite(L)

        return [u[idx, :][:nlive, :], v[idx, :][:nlive, :], L[idx][:nlive]]
    
    def runDynesty(self, dynamic=False, progress=True, nlive=100):
        """ Start nested sampling

        Initializes and runs the nested sampling with Dynesty. We use the 
        default settings for stopping criteria as per the Dynesty documentation.

        Parameters
        ----------
        dynamic : bool, optional
            Use dynamic sampling as opposed to static. Dynamic sampling achieves
            minutely higher likelihood levels compared to the static sampler. 
            From experience this is not usually worth the extra runtime. By 
            default False.
        progress : bool, optional
            Display the progress bar, turn off for commandline runs, by default 
            True
        nlive : int, optional
            Number of live points to use in the sampling. Conceptually similar 
            to MCMC walkers, by default 100.

        Returns
        -------
        sampler : Dynesty sampler object
            The sampler from the nested sampling run. Contains some diagnostics.
        samples : jax device array
            Array of samples from the nested sampling with shape (Nsamples, Ndim)
        """

        initLive = self.getInitLive(nlive)

        if dynamic:
            sampler = dynesty.DynamicNestedSampler(self.lnlikelihood, 
                                                   self.ptform, 
                                                   self.ndims, 
                                                   nlive=nlive, 
                                                   sample='rwalk',
                                                   live_points = initLive,
                                                   logl_args=[self.f[self.sel]])
            
            sampler.run_nested(print_progress=progress, 
                               wt_kwargs={'pfrac': 1.0}, 
                               dlogz_init=1e-3 * (nlive - 1) + 0.01, 
                               nlive_init=nlive)  
            
        else:           
            sampler = dynesty.NestedSampler(self.lnlikelihood, 
                                            self.ptform, 
                                            self.ndims, 
                                            nlive=nlive, 
                                            sample='rwalk',
                                            live_points=initLive,
                                            logl_args=[self.f[self.sel]])
            
            sampler.run_nested(print_progress=progress)
 
        result = sampler.results

        unweighted_samples, weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        samples = dyfunc.resample_equal(unweighted_samples, weights)

        return sampler, samples
    
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

        S = {key: np.zeros(samples.shape[0]) for key in self.labels + ['p_L', 'p_D']}
        
        for i, theta in enumerate(samples):
        
            theta_u = self.unpackParams(theta)
            
            for key in theta_u.keys():
                
                S[key][i] = theta_u[key]
            
        return S
    
    def meanBkg(self, nu, samples_u, N=30):
        """
        Compute median background model from samples.

        Parameters
        ----------
        nu : numpy.ndarray
            Array of frequency values at which to compute the background 
            model.
        samples_u : dict
            A dictionary containing samples of the background parameters.
        N : int, optional
            Number of samples to use for computing the median. Default is 
            30.

        Returns
        -------
        numpy.ndarray
            An array of shape (len(nu),) representing the median background 
            model.

        Notes
        -----
        - The function generates `N` random indices to select samples from 
          the provided `samples_u`.
        - For each selected sample, the `background` function is called to 
          compute the background model using the background parameters.
        - The median of these `N` background models is computed along each 
          frequency bin specified in `nu`, and the resulting median 
          background model is returned.
        """


        mod = np.zeros((len(nu), N))
        
        # Generate random indices for selecting samples
        idx = np.random.choice(np.arange(len(samples_u['dnu'])), size=N, replace=False)
        
        for i, j in enumerate(idx):
            # Extract background parameters for the selected sample
            theta_u = {k: v[j] for k, v in samples_u.items()}
            
            # Compute the background model for the selected sample
            mod[:, i] = self.background(theta_u, nu)
        
        # Compute the median background model across samples
        return np.median(mod, axis=1)

    variables = {'dnu'       : {'info': 'large frequency separation'               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                 'numax'     : {'info': 'frequency at maximum power'               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                 'eps_p'     : {'info': 'phase offset of the p-modes'              , 'log10': False, 'pca': True, 'unit': 'None'}, 
                 'd02'       : {'info': 'l=0,2 mean frequency difference'          , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                 'alpha_p'   : {'info': 'curvature of the p-modes'                 , 'log10': True , 'pca': True, 'unit': 'None'}, 
                 'env_width' : {'info': 'envelope width'                           , 'log10': True , 'pca': True, 'unit': 'muHz'},
                 'env_height': {'info': 'envelope height'                          , 'log10': True , 'pca': True, 'unit': 'ppm^2/muHz'}, 
                 'mode_width': {'info': 'mode width'                               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                 'teff'      : {'info': 'effective temperature'                    , 'log10': True , 'pca': True, 'unit': 'K'}, 
                 'bp_rp'     : {'info': 'Gaia Gbp-Grp color'                       , 'log10': False, 'pca': True, 'unit': 'mag'},
                 'H1_nu'     : {'info': 'Frequency of the high-frequency Harvey'   , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                 'H1_exp'    : {'info': 'Exponent of the high-frequency Harvey'    , 'log10': False, 'pca': True, 'unit': 'None'},
                 'H_power'   : {'info': 'Power of the Harvey law'                  , 'log10': True , 'pca': True, 'unit': 'ppm^2/muHz'}, 
                 'H2_nu'     : {'info': 'Frequency of the mid-frequency Harvey'    , 'log10': True , 'pca': True, 'unit': 'muHz'},
                 'H2_exp'    : {'info': 'Exponent of the mid-frequency Harvey'     , 'log10': False, 'pca': True, 'unit': 'None'},
                 'u1'        : {'info': 'Sum of p_L0 and p_D0'                     , 'log10': False, 'pca': True, 'unit': 'Angular frequency 1/muHz^2'},
                 'u2'        : {'info': 'Difference between p_L0 and p_D0'         , 'log10': False, 'pca': True, 'unit': 'Angular frequency 1/muHz^2'},
                 'DPi1'      : {'info': 'period spacing of the l=0 modes'          , 'log10': False, 'pca': True, 'unit': 's'}, 
                 'eps_g'     : {'info': 'phase offset of the g-modes'              , 'log10': False, 'pca': True, 'unit': 'None'}, 
                 'alpha_g'   : {'info': 'curvature of the g-modes'                 , 'log10': True , 'pca': True, 'unit': 'None'}, 
                 'd01'       : {'info': 'l=0,1 mean frequency difference'          , 'log10': False, 'pca': True, 'unit': 'muHz'},
                 'nurot_c'   : {'info': 'core rotation rate'                       , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
                 'nurot_e'   : {'info': 'envelope rotation rate'                   , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
                 'inc'       : {'info': 'stellar inclination axis'                 , 'log10': False, 'pca': False, 'unit': 'rad'},
                 'H3_power'  : {'info': 'Power of the low-frequency Harvey'        , 'log10': True , 'pca': False, 'unit': 'ppm^2/muHz'}, 
                 'H3_nu'     : {'info': 'Frequency of the low-frequency Harvey'    , 'log10': True , 'pca': False, 'unit': 'muHz'},
                 'H3_exp'    : {'info': 'Exponent of the low-frequency Harvey'     , 'log10': False, 'pca': False, 'unit': 'None'},
                 'shot'      : {'info': 'Shot noise level'                         , 'log10': True , 'pca': False, 'unit': 'ppm^2/muHz'},
                 'freqError' : {'info': 'Frequency error'                          , 'log10': False, 'pca': False, 'unit': 'muHz'}}

    def _modeUpdoot(self, result, sample, key, Nmodes):
        
        result['summary'][key] = np.hstack((result['summary'][key], 
                                           np.array([jar.smryStats(sample[:, j]) for j in range(Nmodes)]).T))
        result['samples'][key] = np.hstack((result['samples'][key], 
                                            sample))

    def parseSamples(self, smp, Nmax=10000):

        N = min([len(smp['dnu']), Nmax])
        
        for key in smp.keys():
            smp[key] = smp[key][:N]
        
        result = {'ell': np.array([]),
                  'enn': np.array([]),
                  'zeta': np.array([]),
                  'summary': {'freq': np.array([]).reshape((2, 0)), 
                              'height': np.array([]).reshape((2, 0)), 
                              'width': np.array([]).reshape((2, 0))
                             },
                  'samples': {'freq': np.array([]).reshape((N, 0)),
                              'height': np.array([]).reshape((N, 0)), 
                              'width': np.array([]).reshape((N, 0))
                             },
                }
        
        result['summary'].update({key: jar.smryStats(smp[key]) for key in smp.keys()})
        result['samples'].update(smp)
  
        # l=0
        asymptotic_samps = np.array([self.AsyFreqModel.asymptotic_nu_p(smp['numax'][i], smp['dnu'][i], smp['eps_p'][i], smp['alpha_p'][i]) for i in range(N)])
        n_p = np.median(asymptotic_samps[:, 1, :], axis=0).astype(int)
        
        result['ell'] = np.append(result['ell'], np.zeros(self.N_p))

        result['enn'] = np.append(result['enn'], n_p)

        result['zeta'] = np.append(result['zeta'], np.zeros(self.N_p))

        # # Frequencies
        nu0_samps = asymptotic_samps[:, 0, :]
        self._modeUpdoot(result, nu0_samps, 'freq', self.N_p)

        # # Heights
        H0_samps = np.array([self.envelope(nu0_samps[i, :], smp['env_height'][i], smp['numax'][i], smp['env_width'][i]) for i in range(N)])
        self._modeUpdoot(result, H0_samps, 'height', self.N_p)

        # # Widths
        W0_samps = np.tile(smp['mode_width'], self.N_p).reshape((self.N_p, N)).T
        self._modeUpdoot(result, W0_samps, 'width', self.N_p)
        
        

        # l=1
        A = np.array([self.MixFreqModel.mixed_nu1(nu0_samps[i, :], 
                                                  n_p, smp['d01'][i], 
                                                  smp['DPi1'][i], 
                                                  jnp.array([smp['p_L'][i]]),  
                                                  jnp.array([smp['p_D'][i]]), 
                                                  smp['eps_g'][i], 
                                                  smp['alpha_g'][i]) for i in range(N)])
        
        N_pg = self.MixFreqModel.N_p + self.MixFreqModel.N_g
        
        result['ell'] = np.append(result['ell'], np.zeros(N_pg) + 1)
        result['enn'] = np.append(result['enn'], np.zeros(N_pg) - 1)

        # # Frequencies 
        nu1_samps = A[:, 0, :]

        sigma_nul1 = np.array([smp[key] for key in smp.keys() if key.startswith('freqError')]).T
 
        self._modeUpdoot(result, nu1_samps + sigma_nul1, 'freq', N_pg)

        zeta_samps = A[:, 1, :]

        result['zeta'] = np.append(result['zeta'], np.median(zeta_samps, axis=0))
        
        # # Heights
        H1_samps = self.vis['V10'] * np.array([self.envelope(nu1_samps[i, :], 
                                                            smp['env_height'][i], 
                                                            smp['numax'][i], 
                                                            smp['env_width'][i]) for i in range(N)]) 
        self._modeUpdoot(result, H1_samps, 'height', N_pg)
        
        # # Widths
        W1_samps = np.array([self.l1_modewidths(zeta_samps[i, :], 
                                                smp['mode_width'][i]) for i in range(N)]) 
        self._modeUpdoot(result, W1_samps, 'width', N_pg)
        

        
        # l=2
        result['ell'] = np.append(result['ell'], np.zeros(self.N_p) + 2)
        result['enn'] = np.append(result['enn'], n_p-1)
        result['zeta'] = np.append(result['zeta'], np.zeros(self.N_p))

        # # Frequencies
        nu2_samps = np.array([nu0_samps[i, :] - smp['d02'][i] for i in range(N)])
        self._modeUpdoot(result, nu2_samps, 'freq', self.N_p)

        # # Heights
        H2_samps = self.vis['V20'] * np.array([self.envelope(nu2_samps[i, :],  
                                                            smp['env_height'][i], 
                                                            smp['numax'][i], 
                                                            smp['env_width'][i]) for i in range(N)])
        self._modeUpdoot(result, H2_samps, 'height', self.N_p)
        
        # # Widths
        W2_samps = np.tile(smp['mode_width'], np.shape(nu2_samps)[1]).reshape((nu2_samps.shape[1], nu2_samps.shape[0])).T
        self._modeUpdoot(result, W2_samps, 'width', self.N_p)

        # Background
        result['background'] = self.meanBkg(self.f, smp)  

        return result
    
    def storeResult(self, resultDict, ID=None):

        # TODO ID should come from star?
        if ID is not None:
            _ID = ID
        elif (ID is None) and hasattr(self, 'ID'):
            _ID = self.ID
        else:   
            _ID = f'unknown_tgt_{np.random.randint(0, 1e10)}'

            warnings.warn(f'Output stored under {_ID}. You should probably specify a target ID.')
        
        path = _ID
        
        basefilename = os.path.join(*[path, f'asymptotic_fit_summary_{_ID}'])
        
        if not os.path.exists(path):
            os.makedirs(path)

        # store everything
        np.savez(basefilename+'.npz', resultDict)

        # grab just model parameters and save
        _tmp = {key: self.result['summary'][key] for key in self.result['summary'].keys() if key not in ['freq', 'height', 'width']}

        df_data = [{'name': key, 'mean': value[0], 'error': value[1]} for key, value in _tmp.items()]

        # Create a DataFrame
        df = pd.DataFrame(df_data)
        
        df.to_csv(basefilename+'.csv', index=False)

    def testLikelihood(self):
    
        u = np.random.uniform(0, 1, self.ndims)
        
        theta = self.ptform(u)
        
        return self.lnlikelihood(theta, self.f[self.sel])

    def testModel(self):
        
        u = np.random.uniform(0, 1, self.ndims)
        
        theta = self.ptform(u)
        
        theta_u = self.unpackParams(theta)
        
        m = self.model(theta_u, self.f[self.sel])
        
        return self.f[self.sel], m
