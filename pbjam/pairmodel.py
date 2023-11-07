import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from pbjam import jar
from pbjam.background import bkgModel
from pbjam.DR import PCA
import pbjam.distributions as dist

jax.config.update('jax_enable_x64', True)

class AsyFreqModel(jar.DynestySamplingTools):

    def __init__(self, f, s, obs, addPriors, N_p, Npca, PCAdims,
                 vis={'V20': 0.71}, priorpath=None):
        
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
 
        self.Nyquist = self.f[-1]

        self.modelVars = {}

        self.modelVars.update(self.variables['l20'])
        
        self.modelVars.update(self.variables['background'])
        
        self.modelVars.update(self.variables['common'])

        self.set_labels(self.addPriors)

        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.setupDR()
  
        self.setPriors()
 
        self.background = bkgModel(self.Nyquist)
 
        self.ndims = len(self.latentLabels + self.addlabels)
 
        self.setAddObs()

    def setFreqRange(self,):
        """ Get frequency range around numax for model 

        Returns
        -------
        idx : jax device array
            Array of boolean values defining the interval of the frequency axis
            where the oscillation modes present.
        """
 
        lfreq = self.freq_limits[0]
        
        ufreq = self.freq_limits[1]

        return (lfreq < self.f) & (self.f < ufreq) 
      
    def setAddObs(self, ):
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

        AddKeys = [k for k in self.modelVars if k in self.addPriors.keys()]

        self.priors.update({key : self.addPriors[key] for key in AddKeys})
 
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

        # Core/envelope rotation prior
        self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        self.priors['inc'] = dist.truncsine()

    def setupDR(self):
        """ Setup the latent parameters and projection functions

        Parameters
        ----------
        prior_file : str
            Full path name for the file containing the prior samples.
 
        """
 
        _obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in ['numax', 'dnu', 'teff']}
         
        for key in ['bp_rp']:
            _obs[key] = self.obs[key]
         
        self.DR = PCA(_obs, self.pcalabels, self.priorpath, self.Npca, selectlabels=['numax', 'dnu', 'teff', 'bp_rp']) 

        self.DR.fit_weightedPCA(self.PCAdims)

        _Y = self.DR.transform(self.DR.data_F)

        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(_Y)
        
        self.latentLabels = ['theta_%i' % (i) for i in range(self.PCAdims)]
       
    def set_labels(self, addPriors):
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
        for key in self.modelVars.keys():
                
            if self.modelVars[key]['pca'] and (key not in addPriors.keys()):
                self.pcalabels.append(key)
            else:
                self.addlabels.append(key)

        self.labels = self.pcalabels + self.addlabels

        # Parameters that are in log10
        self.logpars = []
        for key in self.modelVars.keys():
            if self.modelVars[key]['log10']:
                self.logpars.append(key)

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

        lnp = 0

        for key in self.addObs.keys():       
            lnp += self.addObs[key].logpdf(theta_u[key]) 
 
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

        L = -jnp.sum(jnp.log(mod) + self.s / mod)

        return L      
    
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

    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u, nu):
        
        modes = jnp.ones_like(nu)
 
        # l=2,0
        modes, _, _ = self.add20Pairs(modes, nu, **theta_u)
        
        # Background
        bkg = self.background(theta_u, nu)
         
        return modes * bkg

    @partial(jax.jit, static_argnums=(0,))
    def add20Pairs(self, modes, nu, d02, mode_width, nurot_e, inc, **kwargs):
         
        nu0_p, n_p = self.asymptotic_nu_p(**kwargs)

        Hs0 = self.envelope(nu0_p, **kwargs)

        for n in range(self.N_p):

            # Adding l=0
            modes += jar.lor(nu, nu0_p[n], Hs0[n], mode_width) 
            
            # Adding l=2 multiplet
            for m in [-2, -1, 0, 1, 2]:
                
                H = Hs0[n] * self.vis['V20'] * jar.visell2(abs(m), inc)
                
                f = nu0_p[n] - d02 + m * nurot_e

                modes += jar.lor(nu, f, H, mode_width)

        return modes, nu0_p, n_p

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
 
        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u

    @partial(jax.jit, static_argnums=(0,))
    def _get_n_p_max(self, dnu, numax, eps):
        """Compute radial order at numax.
    
        Compute the radial order at numax, which in this implimentation of the
        asymptotic relation is not necessarily integer.
    
        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps : float
            Epsilon phase term in asymptotic relation (muHz).
    
        Returns
        -------
        nmax : float
            non-integer radial order of maximum power of the p-mode envelope      
        """
    
        return numax / dnu - eps

    @partial(jax.jit, static_argnums=(0,))
    def _get_n_p(self, nmax):
        """Compute radial order numbers.

        Get the enns that will be included in the asymptotic relation fit.
        These are all integer.

        Parameters
        ----------
        nmax : float
            Frequency of maximum power of the oscillation envelope.

        Returns
        -------
        enns : jax device array
            Array of norders radial orders (integers) around nu_max (nmax).
        """

        below = jnp.floor(nmax - jnp.floor(self.N_p/2)).astype(int)
         
        enns = jnp.arange(self.N_p) + below

        return enns 

    @partial(jax.jit, static_argnums=(0,))
    def asymptotic_nu_p(self, numax, dnu, eps_p, alpha_p, **kwargs):
        """ Compute the l=0 mode frequencies from the asymptotic relation for
        p-modes
    
        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope (muHz).
        dnu : float
            Large separation of l=0 modes (muHz).
        eps_p : float
            Epsilon phase term in asymptotic relation (unitless).
        alpha_p : float
            Curvature factor of l=0 ridge (second order term, unitless).
    
        Returns
        -------
        nu0s : ndarray
            Array of l=0 mode frequencies from the asymptotic relation (muHz).
            
        """
        
        n_p_max = self._get_n_p_max(dnu, numax, eps_p)

        n_p = self._get_n_p(n_p_max)

        return (n_p + eps_p + alpha_p/2*(n_p - n_p_max)**2) * dnu, n_p
 
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

        S = {key: np.zeros(samples.shape[0]) for key in self.labels}
        
        for i, theta in enumerate(samples):
        
            theta_u = self.unpackParams(theta)
             
            for key in theta_u.keys():
                
                S[key][i] = theta_u[key]
            
        return S

    def testModel(self):
        
        u = np.random.uniform(0, 1, self.ndims)
        
        theta = self.ptform(u)
        
        theta_u = self.unpackParams(theta)
        
        m = self.model(theta_u, self.f)
        
        return self.f, m
    
    def getMedianModel(self, nu, samples_u, N=30):
 
        mod = np.zeros((len(nu), N))
        
        # Generate random indices for selecting samples
        rkey = np.random.choice(list(samples_u.keys()))
        
        nsamples = len(samples_u[rkey])

        idx = np.random.choice(np.arange(nsamples), size=N, replace=False)
        
        for i, j in enumerate(idx):
            # Extract background parameters for the selected sample
            theta_u = {k: v[j] for k, v in samples_u.items()}
            
            # Compute the background model for the selected sample
            mod[:, i] = self.model(theta_u, nu)
        
        # Compute the median background model across samples
        return np.median(mod, axis=1)
    
    variables = {'l20':{'dnu'       : {'info': 'large frequency separation'               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                        'numax'     : {'info': 'frequency at maximum power'               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                        'eps_p'     : {'info': 'phase offset of the p-modes'              , 'log10': False, 'pca': True, 'unit': 'None'}, 
                        'd02'       : {'info': 'l=0,2 mean frequency difference'          , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                        'alpha_p'   : {'info': 'curvature of the p-modes'                 , 'log10': True , 'pca': True, 'unit': 'None'}, 
                        'env_width' : {'info': 'envelope width'                           , 'log10': True , 'pca': True, 'unit': 'muHz'},
                        'env_height': {'info': 'envelope height'                          , 'log10': True , 'pca': True, 'unit': 'ppm^2/muHz'}, 
                        'mode_width': {'info': 'mode width'                               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                        'teff'      : {'info': 'effective temperature'                    , 'log10': True , 'pca': True, 'unit': 'K'}, 
                        'bp_rp'     : {'info': 'Gaia Gbp-Grp color'                       , 'log10': False, 'pca': True, 'unit': 'mag'},
                        },
                'background' : {'H1_nu'     : {'info': 'Frequency of the high-frequency Harvey'   , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
                                'H1_exp'    : {'info': 'Exponent of the high-frequency Harvey'    , 'log10': False, 'pca': True, 'unit': 'None'},
                                'H_power'   : {'info': 'Power of the Harvey law'                  , 'log10': True , 'pca': True, 'unit': 'ppm^2/muHz'}, 
                                'H2_nu'     : {'info': 'Frequency of the mid-frequency Harvey'    , 'log10': True , 'pca': True, 'unit': 'muHz'},
                                'H2_exp'    : {'info': 'Exponent of the mid-frequency Harvey'     , 'log10': False, 'pca': True, 'unit': 'None'},
                                'H3_power'  : {'info': 'Power of the low-frequency Harvey'        , 'log10': True , 'pca': False, 'unit': 'ppm^2/muHz'}, 
                                'H3_nu'     : {'info': 'Frequency of the low-frequency Harvey'    , 'log10': True , 'pca': False, 'unit': 'muHz'},
                                'H3_exp'    : {'info': 'Exponent of the low-frequency Harvey'     , 'log10': False, 'pca': False, 'unit': 'None'},
                                'shot'      : {'info': 'Shot noise level'                         , 'log10': True , 'pca': False, 'unit': 'ppm^2/muHz'},
                                },
                 
                'common': {'nurot_e'   : {'info': 'envelope rotation rate'                   , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
                           'inc'       : {'info': 'stellar inclination axis'                 , 'log10': False, 'pca': False, 'unit': 'rad'},}
                }



 
    

