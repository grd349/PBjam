import jax.numpy as jnp
from functools import partial
import jax, dynesty
import pbjam.distributions as dist
from pbjam import jar
from dynesty import utils as dyfunc
import numpy as np
from pbjam.plotting import plotting
from tinygp import GaussianProcess, kernels

class peakbag(plotting):
    
    def __init__(self, f, s, ell, zeta, background, freq, height, width, numax, 
                 dnu, d02, addPriors={}, freq_limits=[], **kwargs):
        
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        self.Nyquist = self.f[-1]

        self.bkg = self.getBkg() 

        self.snr = self.s / self.bkg(self.f)

        idx = self.pickFreqs(self.ell, self.freq, self.dnu)

        self.ell = self.ell[idx]

        self.N_p = len(self.ell[self.ell==0])

        self.freq = self.freq[:, idx]
        
        self.height = self.height[:, idx]
         
        self.width = self.width[:, idx]
         
        self.zeta = self.zeta[idx]

        self.Nmodes = len(self.ell)
        
        self.set_labels()
        
        self.setPriors()
              
        self.sel = self.setFreqRange()
        
        self.ndims = len(self.labels)

        self.setAddLikeTerms()

    def getBkg(self, a=0.66, b=0.88, skips=100):
        """ Estimate the background

        Takes an average of the power at linearly spaced points along the
        log(frequency) axis, where the width of the averaging window increases
        as a power law.

        The mean power values are interpolated back onto the full linear
        frequency axis to estimate the background noise level at all
        frequencies.

        Returns
        -------
        b : array
            Array of psd values approximating the background.
        """

        freq_skips = np.exp(np.linspace(np.log(self.f[0]), np.log(self.f[-1]), skips))

        m = np.array([np.median(self.s[np.abs(self.f-fi) < a*fi**b]) for fi in freq_skips])

        bkgModel = jar.jaxInterp1D(freq_skips, m/np.log(2))

        return bkgModel

    def pickFreqs(self, ell, freq, dnu, fac=1, modes='l201'):
        """
        Pick frequency indices that fall within +/- fac * dnu of the lowest
        and highest l=0 mode frequency in freq. 

        Parameters
        ----------
        ell : numpy.ndarray
            Array of angular degrees values.
        freq : numpy.ndarray
            Array of frequency values, contains all angular degrees.
        dnu : float
            Larage frequency spacing.
        fac : float
            Number of large separations away from lowest and highest l=0 freqs.
        all : bool, optional
            Flag to select all frequencies. Default is False.

        Returns
        -------
        numpy.ndarray
            Boolean array indicating selected frequency indices.

        Notes
        -----
        - If 'all' is False, selects indices within the range of the lowest ell=0 frequency.
        - If 'all' is True, selects all frequency indices.
        """
         
        if modes == 'all':
            idx = jnp.ones(freq.shape[1], dtype=bool)
        
        else:
            idx_ell0 = ell == 0

            nu0 = freq[0, idx_ell0]

            idx = (nu0.min() - fac * dnu[0] < freq[0, :]) & (freq[0,:] < nu0.max() + fac * dnu[0])

            if modes == 'l20':
                idx *= ell != 1

        return idx
    
    def setPriors(self, freq_err=0.03):
 
        self.priors = {}

        self.priors.update((k, v) for k, v in self.addPriors.items())
        
        for i in range(self.Nmodes):
            _key = f'freq{i}'
            if _key not in self.priors:
                freqScale = freq_err * self.dnu[0] # min([freq_err * self.dnu[0], self.freq[1, i]])
                # self.priors[_key] = dist.normal(loc=self.freq[0, i],  
                #                                 scale=freqScale)
                 
                self.priors[_key] = dist.normal(loc=self.freq[0, i],  
                                                scale=freqScale)
                 

        for i in range(self.Nmodes):
            _key = f'height{i}'
            if _key not in self.priors:
                mode_snr = self.height[0, i] / self.bkg(self.freq[0, i])
                self.priors[_key] = dist.normal(loc=jnp.log10(mode_snr), scale=0.1)

        
        # The GP expects a smooth function, so divide by 1-zeta and add it in later in unpack.
        for i in range(self.Nmodes):
            _key = f'width{i}'
            if _key not in self.priors:
                self.priors[_key] = dist.normal(loc=jnp.log10(self.width[0, i]/(1-self.zeta[i])),
                                                scale=0.1)
        
        # Envelope rotation prior
        if 'nurot_e' not in self.priors.keys():
            self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # Core rotation prior
        if 'nurot_c' not in self.priors.keys():
            self.priors['nurot_c'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        if 'inc' not in self.priors.keys():
            self.priors['inc'] = dist.truncsine()

        if 'shot' not in self.priors.keys():
            self.priors['shot'] = dist.normal(loc=0, scale=0.01)

        if not all([key in self.labels for key in self.priors.keys()]):
            raise ValueError('Length of labels doesnt match lenght of priors.')
     
    def set_labels(self):
 
        # Default additional parameters
        self.labels = []
        
        # If key appears in priors dict, override default and move it to add.
        for key in self.variables.keys():

            if key in ['freq', 'height', 'width']:
                self.labels += [f'{key}{i}' for i in range(self.Nmodes)]
            else:
                self.labels.append(key)
        
        # Parameters that are in log10
        self.logpars = [key for key in self.variables.keys() if self.variables[key]['log10']]
        
    def setFreqRange(self,):
        """ Get frequency range around numax for model 

        Returns
        -------
        idx : jax device array
            Array of boolean values defining the interval of the frequency axis
            where the oscillation modes present.
        """
        
        if len(self.freq_limits) == 2:
            lfreq = self.freq_limits[0]
            
            ufreq = self.freq_limits[1]
        
        elif (self.modeIDres is not None) and (len(self.freq_limits) == 0):
            pad = self.modeIDres['summary']['dnu'][0] / 2

            muFreqs = jnp.array([self.priors[key].ppf(0.5) for key in self.labels if 'freq' in key])
            
            lfreq = muFreqs.min() - pad

            ufreq = muFreqs.max() + pad
        else:
            raise ValueError('Supply a modeID result dictionary or freq_limits to set fit window.')

        return (lfreq < self.f) & (self.f < ufreq)  
    
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

        L = -jnp.sum(jnp.log(mod) + self.snr[self.sel] / mod)
 
        return L 
        
    variables = {'freq'   : {'info': 'mode frequency list'      , 'log10': False},
                 'height' : {'info': 'mode height list'         , 'log10': True},
                 'width'  : {'info': 'mode width list'          , 'log10': True},
                 'nurot_e': {'info': 'envelope rotation rate'   , 'log10': True},
                 'nurot_c': {'info': 'core otation rate'        , 'log10': True}, 
                 'inc'    : {'info': 'stellar inclination axis' , 'log10': False},
                 'shot'   : {'info': 'Shot noise level'         , 'log10': True }}
    
    
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
    def lnlikelihood(self, theta):
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
  
        # Constraint from the periodogram 
        mod = self.model(theta_u)

        lnlike = self.chi_sqr(mod)

        lnlike += self.AddLikeTerms(theta, theta_u)
        
        return lnlike

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
        
        theta_u = {'freq'   : theta[0: self.Nmodes],
                   'height' : theta[self.Nmodes: 2 * self.Nmodes],
                   'width'  : theta[2 * self.Nmodes: 3 * self.Nmodes],
                   'nurot_e': theta[self.labels.index('nurot_e')],
                   'nurot_c': theta[self.labels.index('nurot_c')],
                   'inc'    : theta[self.labels.index('inc')],
                   'shot'   : theta[self.labels.index('shot')],
                   }
        
        

        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]

        theta_u['width'] = theta_u['width']*(1-self.zeta)
        
        return theta_u
 
    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u):
         
        modes = jnp.zeros_like(self.f[self.sel]) + theta_u['shot']
        
        omega = self.zeta * theta_u['nurot_c'] + (1 - self.zeta) * theta_u['nurot_e']
 
        for i, l in enumerate(self.ell):
            
            for m in jnp.arange(2 * l + 1) - l:
                
                E = jar.visibility(l, m, theta_u['inc'])
                
                nu = theta_u['freq'][i] + omega[i] * m

                H = E * theta_u['height'][i]

                modes += jar.lor(self.f[self.sel], nu, H, theta_u['width'][i])
        
        return modes
    
    def __call__(self, dynesty_kwargs={}):

        self.runDynesty(**dynesty_kwargs)

        self.result = self.parseSamples(self.samples)
  
        return self.samples, self.result

    def parseSamples(self, samples, N=10000):
    
        theta_u = self.unpackSamples(samples)
        
        result = {'ell': np.array([self.ell]),
                  'zeta': np.array([self.zeta]),
                  'summary': {},
                  'samples': {}
                 }
        
        for key in self.variables:
            arr = np.array([theta_u[_key] for _key in theta_u.keys() if key in _key])
            
            result['summary'][key] = np.array([np.mean(arr, axis=1), 
                                            np.std(arr, axis=1)]).T 
            
            
            result['samples'][key] = arr[:, :N]
            
            for field in ['summary', 'samples']:
                if result[field][key].shape[0] == 1:
                    result[field][key] = result[field][key].flatten()
    
        return result

    def runDynesty(self, dynamic=False, progress=True, nlive=200, **dynesty_kwargs):
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
                                                   **dynesty_kwargs
                                                   )
            
            sampler.run_nested(print_progress=progress, 
                               wt_kwargs={'pfrac': 1.0}, 
                               dlogz_init=1e-3 * (nlive - 1) + 0.01, 
                               nlive_init=nlive)  
            
        else:           
            sampler = dynesty.NestedSampler(self.lnlikelihood, 
                                            self.ptform, 
                                            self.ndims, 
                                            nlive=nlive, 
                                            **dynesty_kwargs
                                            )
            
            sampler.run_nested(print_progress=progress)
 
        self.sampler = sampler

        result = self.sampler.results

        unweighted_samples, weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        self.samples = dyfunc.resample_equal(unweighted_samples, weights)

        return self.sampler, self.samples


    def unpackSamples(self, samples):
        S = {key: np.zeros(samples.shape[0]) for key in self.labels}
    
        for i, key in enumerate(self.labels):
            S[f'{key}'] = samples[:, i]
        
        for key in self.labels:
            if any([key.startswith(logkey) for logkey in self.logpars]):
                S[key] = 10**S[key]
    
        return S

    def setAddLikeTerms(self):
        """ Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 
        """

        self.addObs = {}

        # TODO this should be changed to something that can't go below 0
        self.addObs['d02'] = dist.normal(loc=self.d02[0], 
                                         scale=10 * self.d02[1])

        # Correlated Noise Regularisation for width
        wGPtheta={'amp': 1, 'scale': self.dnu[0]}

        wGPmuFunc = jar.jaxInterp1D(self.freq[0, :], jnp.log10(self.width[0, :]/(1-self.zeta)))
    
        wGP = self.build_gp(wGPtheta, self.freq[0, :], wGPmuFunc)

        self.addObs['widthGP'] = wGP.log_probability


        # # Correlated Noise Regularisation for amplitude
        # hGPtheta={'amp': 1, 'scale': self.dnu[0]}

        # hGPmuFunc = jar.jaxInterp1D(self.freq[0, :], jnp.log10(self.height[0, :]))
    
        # hGP = self.build_gp(hGPtheta, self.freq[0, :], hGPmuFunc)

        # self.addObs['heightGP'] = hGP.log_probability
 
    @partial(jax.jit, static_argnums=(0,))
    def AddLikeTerms(self, theta, theta_u):
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
        delta = theta_u['freq'][self.ell==0] - theta_u['freq'][self.ell==2]

        lnp = jnp.sum(self.addObs['d02'].logpdf(delta))
        
        #lnp += self.addObs['heightGP'](theta[self.Nmodes: 2 * self.Nmodes])

        lnp += self.addObs['widthGP'](theta[2 * self.Nmodes: 3 * self.Nmodes])

        return lnp


    def build_gp(self, theta, X, muFunc, muKwargs={}):

        kernel = theta["amp"] * kernels.ExpSquared(theta["scale"])

        GP = GaussianProcess(kernel, X, diag=1e-6, mean=partial(muFunc, **muKwargs))

        return GP

