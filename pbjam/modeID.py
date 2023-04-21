from functools import partial
import jax, dynesty
import jax.numpy as jnp
from dynesty import utils as dyfunc
from pbjam.mixedmodel import MixFreqModel
from pbjam.pairmodel import AsyFreqModel
import pbjam.distributions as dist
from pbjam import jar
from pbjam.DR import PCA
import numpy as np
from pbjam.background import bkgModel

class modeIDsampler():

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freq_limits=[1, 5000], 
                 vis={'V20': 0.71, 'V10': 1.22}, envelope_only=False, 
                 Npca=50, PCAdims=8, priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        self.set_labels(self.addPriors)

        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.setupDR()
  
        self.setPriors()
 
        self.ndims = len(self.latent_labels+self.addlabels)

        self.AsyFreqModel = AsyFreqModel(self.N_p)

        n_g_ppf, _, _, _ = self._makeTmpSample(['DPi0', 'eps_g'])

        self.MixFreqModel = MixFreqModel(self.N_p, self.obs, n_g_ppf)

        self.background = bkgModel(self.Nyquist)

        self.sel = self.setFreqRange(self.envelope_only)

        self.setAddObs()
 

    def _makeTmpSample(self, keys, N=1000):

        K = np.zeros((len(keys), N))

        for i in range(N):
            u = np.random.uniform(0, 1, size=self.ndims)
        
            theta = self.ptform(u)

            theta_u = self.unpackParams(theta)

            K[:, i] = np.array([theta_u[key] for key in keys]) 
        
        ppf, pdf, logpdf, cdf = dist.getQuantileFuncs(K.T)
        
        return ppf, pdf, logpdf, cdf

    def set_labels(self, priors):

        # Default PCA parameters       
        self.pcalabels = []
        
        # Default additional parameters
        self.addlabels = []
        
        # If key appears in priors dict, override default and move it to add.
        for key in self.variables.keys():

            if self.variables[key]['pca'] and (key not in priors.keys()):
                self.pcalabels.append(key)

            else:
                self.addlabels.append(key)

        self.labels = self.pcalabels + self.addlabels

        # Parameters that are in log10
        self.logpars = [key for key in self.variables.keys() if self.variables[key]['log10']]

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
         
        theta_u['p_L'] = jnp.array([theta_u[key] for key in theta_u.keys() if 'p_L' in key])

        theta_u['p_D'] = jnp.array([theta_u[key] for key in theta_u.keys() if 'p_D' in key])

        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
    
    def setPriors(self):
        """ Set the prior distributions.

        The prior distributions are constructed from the projection of the 
        PCA sample onto the reduced dimensional space.

        """

        self.priors = {}

        for i, key in enumerate(self.latent_labels):
            self.priors[key] = dist.distribution(self.DR.ppf[i], 
                                                 self.DR.pdf[i], 
                                                 self.DR.logpdf[i], 
                                                 self.DR.cdf[i])

        self.priors.update((k, v) for k, v in self.addPriors.items())
        
        # Core rotation prior
        self.priors['nurot_c'] = dist.uniform(loc=-2., scale=1.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        self.priors['inc'] = dist.truncsine()

        # The instrumental components are set based on the PSD, not Bayesian but...
        hi_idx = self.f > min([self.f[-1], self.Nyquist]) - 10
        shot_est = jnp.nanmean(self.s[hi_idx])

        lo_idx = abs(self.f - self.f[0]) < 10
        inst_est = jnp.nanmean(self.s[lo_idx])
        
        mu = jnp.array([1, inst_est - shot_est]).max()
        
        self.priors['H3_power'] = dist.normal(loc=jnp.log10(mu * self.f[0]), scale=2)  

        self.priors['H3_nu'] = dist.beta(a=1.2, b=1.2, loc=-1, scale=2)  
        
        self.priors['H3_exp'] = dist.beta(a=1.2, b=1.2, loc=1.5, scale=3.5)  
 
        self.priors['shot'] = dist.normal(loc=jnp.log10(shot_est), scale=0.1)

    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u, nu):

        # # Background
        # H1 = self.harvey(nu, theta_u['H_power'], theta_u['H1_nu'], theta_u['H1_exp'],)

        # H2 = self.harvey(nu, theta_u['H_power'], theta_u['H2_nu'], theta_u['H2_exp'],)

        # H3 = self.harvey(nu, theta_u['H3_power'], theta_u['H3_nu'], theta_u['H3_exp'],)

         
        # eta = jar.attenuation(nu, self.Nyquist)**2

        # bkg = (H1 + H2 + H3) * eta + theta_u['shot']

        bkg = self.background(theta_u, nu)

        # l=2,0
        nu0_p, n_p = self.AsyFreqModel.asymptotic_nu_p(**theta_u)

        Hs0 = self.envelope(nu0_p, **theta_u)
        
        modes = jnp.zeros_like(nu)

        for n in range(self.N_p):
            modes += self.pair(nu, nu0_p[n], Hs0[n], **theta_u)
        
         
        # l=1
        nu1s, zeta = self.MixFreqModel.mixed_nu1(nu0_p, n_p, **theta_u)
       
        Hs1 = self.envelope(nu1s, **theta_u)
        
        modewidth1s = self.l1_modewidths(zeta, **theta_u)
    
        for i in range(len(nu1s)):
            modes += jar.lor(nu, nu1s[i]                               , Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.cos(theta_u['inc'])**2
        
            modes += jar.lor(nu, nu1s[i] - zeta[i] * theta_u['nurot_c'], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(theta_u['inc'])**2 / 2
        
            modes += jar.lor(nu, nu1s[i] + zeta[i] * theta_u['nurot_c'], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(theta_u['inc'])**2 / 2
 
        return (1 + modes) * bkg

    def setupDR(self):
        """ Setup the latent parameters and projection functions

        Parameters
        ----------
        prior_file : str
            Full path name for the file containing the prior samples.
 
        """

        self.latent_labels = ['theta_%i' % (i) for i in range(self.PCAdims)]

        log_obs = self.log_obs.copy()

        for key in ['bp_rp']:
            if key in log_obs.keys():
                log_obs[key] = self.obs[key]
         
        self.DR = PCA(log_obs, self.pcalabels, self.priorpath, self.Npca) 

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
    def l1_modewidths(self, zeta, mode_width, **kwargs):
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
         
        return  mode_width * jnp.maximum(0, 1. - zeta) 
    
    @partial(jax.jit, static_argnums=(0,))
    def pair(self, nu, nu0, h0, mode_width, d02, **kwargs):
        """Define a pair as the sum of two Lorentzians.

        A pair is assumed to consist of an l=0 and an l=2 mode. The widths are
        assumed to be identical, and the height of the l=2 mode is scaled
        relative to that of the l=0 mode. The frequency of the l=2 mode is the
        l=0 frequency minus the small separation.

        Parameters
        ----------
        nu : jax device array
            Frequency range to compute the pair on.
        nu0 : float
            Frequency of the l=0 (muHz).
        h0 : float
            Height of the l=0 (SNR).
        w0 : float
            The mode width (identical for l=2 and l=0) (log10(muHz)).
        d02 : float
            The small separation (muHz).

        Returns
        -------
        pair_model : array
            The SNR as a function of frequency of a mode pair.
            
        """
        
        pair_model = jar.lor(nu, nu0, h0, mode_width) + jar.lor(nu, nu0 - d02, h0 * self.vis['V20'], mode_width)

        return pair_model
    
    @partial(jax.jit, static_argnums=(0,))
    def harvey(self, nu, a, b, c):
        """ Harvey-profile

        Parameters
        ----------
        f : np.array
            Frequency axis of the PSD.
        a : float
            The amplitude (divided by 2 pi) of the Harvey-like profile.
        b : float
            The characeteristic frequency of the Harvey-like profile.
        c : float
            The exponent parameter of the Harvey-like profile.

        Returns
        -------
        H : np.array
            The Harvey-like profile given the relevant parameters.
        """
         
        H = a / b * 1 / (1 + (nu / b)**c)

        return H

    def setAddObs(self):
        """ Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 

        """
        
        self.addObs = {}

        self.addObs['teff'] = dist.normal(loc=self.obs['teff'][0], 
                                          scale=self.obs['teff'][1])

        # bp_rp is not logged in the prior file so should not be logged here.
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
        """ Likelihood function for set of model parameters

        Evaluates the likelihood function for a set of model parameters given
        the data. This includes the constraint from the observed variables.

        The samples l are drawn from the latent parameter priors and are first
        projected into the model space before the model is constructed and the
        likelihood is constructed.

        Parameters
        ----------
        l : list
            Array of latent parameters

        Returns
        -------
        lnlike : float
            The log likelihood evaluated at the model parameters p.
        """

        theta_u = self.unpackParams(theta)
 
        # Constraint from input obs
        lnlike = self.addAddObsLike(theta_u)
        
        # Constraint from the periodogram 
        mod = self.model(theta_u, nu)

        lnlike += self.chi_sqr(mod)
 
        T = (theta_u['H3_nu'] < theta_u['H2_nu']) & \
            (theta_u['H2_nu'] < theta_u['numax']) & \
            (theta_u['H2_nu'] < theta_u['H1_nu'])

        lnlike += jax.lax.cond(T, lambda : 0., lambda : -jnp.inf)
        
        return lnlike
    
    def __call__(self, dynamic=False, progress=True, nlive=100):
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

        if dynamic:
            sampler = dynesty.DynamicNestedSampler(self.lnlikelihood, 
                                                   self.ptform, 
                                                   self.ndims, 
                                                   nlive=nlive, 
                                                   sample='rwalk',
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
                                            logl_args=[self.f[self.sel]])
            
            sampler.run_nested(print_progress=progress)
 
        result = sampler.results

        unweighted_samples, weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        samples = dyfunc.resample_equal(unweighted_samples, weights)

        return sampler, samples
    
    def unpackSamples(self, samples):
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

        S = {key: np.zeros(samples.shape[0]) for key in self.labels}
        
        for i, theta in enumerate(samples):
        
            theta_u = self.unpackParams(theta)
            
            for key in self.labels:
                
                S[key][i] = theta_u[key]
            
        return S
    
    def meanBkg(self, nu, samples, N=30):

        idx =  np.random.choice(np.arange(samples.shape[0]), size=N, replace=False)

        mod = np.zeros((len(nu), N))
        
        for i,j in enumerate(idx):

            theta = samples[i, :]

            theta_u = self.unpackParams(theta)

            mod[:, j] = self.background(theta_u, nu)

        return np.median(mod, axis=1)

    variables = {'dnu'       : {'info': 'large frequency separation'               , 'log10': True , 'pca': True}, 
                 'numax'     : {'info': 'frequency at maximum power'               , 'log10': True , 'pca': True}, 
                 'eps_p'     : {'info': 'phase offset of the p-modes'              , 'log10': False, 'pca': True}, 
                 'd02'       : {'info': 'l=0,2 mean frequency difference'          , 'log10': True , 'pca': True}, 
                 'alpha_p'   : {'info': 'curvature of the p-modes'                 , 'log10': True , 'pca': True}, 
                 'env_width' : {'info': 'envelope width'                           , 'log10': True , 'pca': True},
                 'env_height': {'info': 'envelope height'                          , 'log10': True , 'pca': True}, 
                 'mode_width': {'info': 'mode width'                               , 'log10': True , 'pca': True}, 
                 'teff'      : {'info': 'effective temperature'                    , 'log10': True , 'pca': True}, 
                 'bp_rp'     : {'info': 'Gaia Gbp-Grp color'                       , 'log10': False, 'pca': True},
                 'H1_nu'     : {'info': 'Frequency of the high-frequency Harvey'   , 'log10': True , 'pca': True}, 
                 'H1_exp'    : {'info': 'Exponent of the high-frequency Harvey'    , 'log10': False, 'pca': True},
                 'H_power'   : {'info': 'Power of the Harvey law'                  , 'log10': True , 'pca': True}, 
                 'H2_nu'     : {'info': 'Frequency of the mid-frequency Harvey'    , 'log10': True , 'pca': True},
                 'H2_exp'    : {'info': 'Exponent of the mid-frequency Harvey'     , 'log10': False, 'pca': True},
                 'p_L0'      : {'info': 'First polynomial coefficient for L matrix', 'log10': True, 'pca': True},  
                 'p_D0'      : {'info': 'First polynomial coefficient for D matrix', 'log10': True, 'pca': True}, 
                 'DPi0'      : {'info': 'period spacing of the l=0 modes'          , 'log10': True, 'pca': True}, 
                 'eps_g'     : {'info': 'phase offset of the g-modes'              , 'log10': False, 'pca': True}, 
                 'alpha_g'   : {'info': 'curvature of the g-modes'                 , 'log10': True, 'pca': True}, 
                 'd01'       : {'info': 'l=0,1 mean frequency difference'          , 'log10': True, 'pca': True},
                 'nurot_c'   : {'info': 'core rotation rate'                       , 'log10': True , 'pca': False}, 
                 'inc'       : {'info': 'stellar inclination axis'                 , 'log10': False, 'pca': False},
                 'H3_power'  : {'info': 'Power of the low-frequency Harvey'        , 'log10': True , 'pca': False}, 
                 'H3_nu'     : {'info': 'Frequency of the low-frequency Harvey'    , 'log10': True , 'pca': False},
                 'H3_exp'    : {'info': 'Exponent of the low-frequency Harvey'     , 'log10': False, 'pca': False},
                 'shot'      : {'info': 'Shot noise level'                         , 'log10': True , 'pca': False}}


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
                                                smp['DPi0'][i], 
                                                jnp.array([smp['p_L0'][i]]),  
                                                jnp.array([smp['p_D0'][i]]), 
                                                smp['eps_g'][i], 
                                                smp['alpha_g'][i]) for i in range(N)])
        
        N_pg = self.MixFreqModel.N_p + self.MixFreqModel.N_g
        
        result['ell'] = np.append(result['ell'], np.zeros(N_pg) + 1)
        result['enn'] = np.append(result['enn'], np.zeros(N_pg) - 1)

        # # Frequencies 
        nu1_samps = A[:, 0, :]
        self._modeUpdoot(result, nu1_samps, 'freq', N_pg)

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
        result['ell'] = np.append(result['ell'], np.zeros(N_pg) + 2)
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
    
        return result

        
