import jax.numpy as jnp
from functools import partial
import jax, dynesty
import pbjam.distributions as dist
from pbjam import jar
from dynesty import utils as dyfunc
import numpy as np
from pbjam.plotting import plotting

class peakbag(plotting):
    
    def __init__(self, f, s, ell, zeta, background, freq, height, width, numax, 
                 dnu, d02, addPriors={}, freq_limits=[], **kwargs):
        
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        self.Nyquist = self.f[-1]

        self.bkg = self.getBkg() #jar.jaxInterp1D(self.f, self.bkg)

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

        self.setAddObs()

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

    def pickFreqs(self, ell, freq, dnu, fac=1, all=False):
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
         
        if not all:
            idx_ell0 = ell == 0

            nu0 = freq[0, idx_ell0]

            idx = (nu0.min() - fac * dnu[0] < freq[0, :]) & (freq[0,:] < nu0.max() + fac * dnu[0])
        else:
            idx = jnp.ones(len(freq), dtype=bool)

        return idx
    
    def setPriors(self, freq_err=0.03):
 

        self.priors = {}

        self.priors.update((k, v) for k, v in self.addPriors.items())
        
        for i in range(self.Nmodes):
            _key = f'freq{i}'
            if _key not in self.priors:
                self.priors[_key] = dist.normal(loc=self.freq[0,i], # self.modeIDres['summary']['freq'][0, i],
                                                scale=freq_err * self.dnu[0])
        for i in range(self.Nmodes):
            _key = f'height{i}'
            if _key not in self.priors:
                mode_snr = self.height[0, i] / self.bkg(self.freq[0, i])
                self.priors[_key] = dist.normal(loc=jnp.log10(mode_snr), scale=0.1)

        for i in range(self.Nmodes):
            _key = f'width{i}'
            if _key not in self.priors:
                self.priors[_key] = dist.normal(loc=jnp.log10(self.width[0, i]),
                                                scale=0.5)
        
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
            self.priors['shot'] = dist.normal(loc=0, scale=0.1)

        if not all([key in self.labels for key in self.priors.keys()]):
            raise ValueError('Length of labels doesnt match lenght of priors.')

    # def _checkInputs(self):

    #     if (self.ell is None) and (self.modeIDres is not None):
    #         self.ell = self.modeIDres['ell']
            
#         reqs = [self.__dict__[key] is None for key in ['ell', 'modeIDres']]

#         for key in ['freqs', 'heights', 'widths', 'ell']:
#             if (self.__dict__[key] is None) and (self.modeIDres is not None):
#                 if key == 'ell':
#                     self.ell = self.modeIDres[key]
#                 else:
#                     self.__dict__[key] = self.modeIDres['summary'][key]
     
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

        lnlike += self.addAddObsLike(theta_u)
 
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
        
        theta_u = {'freq'   : theta[0:self.Nmodes],
                   'height' : theta[self.Nmodes: 2 * self.Nmodes],
                   'width'  : theta[2 * self.Nmodes: 3 * self.Nmodes],
                   'nurot_e': theta[self.labels.index('nurot_e')],
                   'nurot_c': theta[self.labels.index('nurot_c')],
                   'inc'    : theta[self.labels.index('inc')],
                   'shot'   : theta[self.labels.index('shot')],
                   }

        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
 
    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u):
         
        modes = jnp.zeros_like(self.f[self.sel]) + theta_u['shot']
        
        omega = self.zeta * theta_u['nurot_c'] + (1 - self.zeta) * theta_u['nurot_e']
 
        for i, l in enumerate(self.ell):
            
            for m in jnp.arange(2 * l + 1) - l:
                
                e = jar.visibility(l, m, theta_u['inc'])
                
                modes += jar.lor(self.f[self.sel], 
                                 theta_u['freq'][i] + omega[i] * m, 
                                 e * theta_u['height'][i],
                                 theta_u['width'][i]
                                )
        
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

    def setAddObs(self):
        """ Set attribute containing additional observational data

        Additional observational data other than the power spectrum goes here. 

        Can be Teff or bp_rp color, but may also be additional constraints on
        e.g., numax, dnu. 
        """
        
        self.addObs = {}

        # TODO this should be changed to something that can't go below 0
        self.addObs['d02'] = dist.normal(loc=self.d02[0], 
                                         scale=10 * self.d02[1])
 
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
        delta = theta_u['freq'][self.ell==0] - theta_u['freq'][self.ell==2]

        lnp = jnp.sum(self.addObs['d02'].logpdf(delta))

        return lnp

# """

# This module is used for constructing a `PyMC3' model of the power spectrum using
# the outputs from asy_peakbag as priors. 

# """

# import numpy as np
# import pymc3 as pm
# import warnings
# from .plotting import plotting

# class peakbag(plotting):
#     """ Class for the final peakbagging.

#     This class is used after getting the frequency intervals from asy_peakbag,
#     that include the $l=0,2$ mode pairs. 
    
#     The

#     Examples
#     --------
#     Using peakbag from the star class instance (recommended)    
    
#     >>> st = pbjam.star(ID='KIC4448777', pg=pg, numax=[220.0, 3.0], 
#                            dnu=[16.97, 0.01], teff=[4750, 100],
#                            bp_rp = [1.34, 0.01])
#     >>> st.run_kde()
#     >>> st.run_asy_peakbag(norders=7)
#     >>> st.run_peakbag()

#     Using peakbag on it's own. Requires output from `asy_peakbag'.
    
#     >>> pb = pbjam.peakbag(st.asy_fit)
#     >>> pb()
    
#     Parameters
#     ----------
#     f : float, array
#         Array of frequency bins of the spectrum (muHz). Truncated to the range
#         around numax.
#     snr : float, array
#         Array of SNR values for the frequency bins in f (dimensionless).
#     asy_fit : asy_fit
#         The result from the asy_peakbag method.
#     init : bool
#         If true runs make_start and trim_ladder to prepare starting
#         guesses for sampling and transforms the data onto a ladder.

#     Attributes
#     ----------
#     f : float, ndarray
#         Array of frequency bins of the spectrum (muHz). Truncated to the range
#         around numax.
#     snr : float, ndarray
#         Array of SNR values for the frequency bins in f (dimensionless).
#     asy_fit : asy_fit
#         The result from the asy_peakbag method.
#         This is a dictionary of 'modeID' and 'summary'.
#         'modeID' is a DataFrame with a list of modes and basic properties.
#         'summary' are summary statistics from the asymptotic_fit.
#         See asy_peakbag asymptotic_fit for more details.

#     """

#     def __init__(self, starinst, init=True, path=None,  verbose=False):

#         self.pg = starinst.pg
#         self.f = starinst.f
#         self.s = starinst.s
#         self.asy_fit = starinst.asy_fit
#         self.norders = self.asy_fit.norders
#         if init:
#             self.make_start()
#             self.trim_ladder(verbose=verbose)
#         self.gp0 = [] # Used for gp linewidth info.

#         starinst.references._addRef('pymc3')
        
#         starinst.peakbag = self


#     def make_start(self):
#         """ Set the starting model for peakbag
        
#         Function uses the result of the asymptotic peakbagging and builds a 
#         dictionary of starting values for the peakbagging methods.

#         """

#         idxl0 = self.asy_fit.modeID.ell == 0
#         idxl2 = self.asy_fit.modeID.ell == 2

#         l0 = self.asy_fit.modeID.loc[idxl0, 'nu_med'].values.flatten()
#         l2 = self.asy_fit.modeID.loc[idxl2, 'nu_med'].values.flatten()

#         l0, l2 = self.remove_outsiders(l0, l2)

#         width = 10**(np.ones(len(l0)) * self.asy_fit.summary.loc['mode_width', 'mean']).flatten()
#         height =  (10**self.asy_fit.summary.loc['env_height', 'mean'] * \
#                  np.exp(-0.5 * (l0 - 10**self.asy_fit.summary.loc['numax', 'mean'])**2 /
#                  (10**self.asy_fit.summary.loc['env_width', 'mean'])**2)).flatten()
#         back = np.ones(len(l0))

#         self.parnames = ['l0', 'l2', 'width0', 'width2', 'height0', 'height2',
#                          'back']

#         pars = [l0, l2, width, width, height, 0.7*height, back]

#         self.start ={x:y for x,y in zip(self.parnames, pars)}

#         self.n = np.linspace(0.0, 1.0, len(self.start['l0']))[:, None]

#     def remove_outsiders(self, l0, l2):
#         """ Drop outliers

#         Drops modes where the guess frequency is outside of the supplied
#         frequency range.
        
#         Parameters
#         ----------
        
#         l0 : ndarray
#             Array of l0 mode frequencies
#         l2 : ndarray
#             Array of l2 mode frequencies

#         """

#         sel = np.where(np.logical_and(l0 < self.f.max(), l0 > self.f.min()))
#         return l0[sel], l2[sel]

#     def trim_ladder(self, lw_fac=10, extra=0.01, verbose=False):
#         """ Turns mode frequencies into list of pairs
        
#         This function turns the list of mode frequencies into pairs and then 
#         selects only the pairs in the ladder that have modes that are to be fit.

#         Each pair is constructed so that the central frequency is
#         the mid point between the l=0 and l=2 modes as determined by the
#         information in the asy_fit dictionary.

#         Parameters
#         ----------
#         lw_fac: float
#             The factor by which the mode line width is multiplied in order
#             to contribute to the pair width.
#         extra: float
#             The factor by which dnu is multiplied in order to contribute to
#             the pair width.

#         """

#         d02 = 10**self.asy_fit.summary.loc['d02', 'mean']
#         d02_lw = d02 + lw_fac * 10**self.asy_fit.summary.loc['mode_width', 'mean']
#         w = d02_lw + (extra * 10**self.asy_fit.summary.loc['dnu', 'mean'])
#         bw = self.f[1] - self.f[0]
#         w /= bw
#         if verbose:
#             print(f'w = {int(w)}')
#             print(f'bw = {bw}')
#         ladder_trim_f = np.zeros([len(self.start['l0']), int(w)])
#         ladder_trim_s = np.zeros([len(self.start['l0']), int(w)])
#         for idx, freq in enumerate(self.start['l0']):
#             loc_mid_02 = np.argmin(np.abs(self.f - (freq - d02/2.0)))
#             if loc_mid_02 == 0:
#                 warnings.warn('Did not find optimal pair location')
#             if verbose:
#                 print(f'loc_mid_02 = {loc_mid_02}')
#                 print(f'w/2 = {int(w/2)}')
#             ladder_trim_f[idx, :] = \
#                 self.f[loc_mid_02 - int(w/2): loc_mid_02 - int(w/2) + int(w)]
#             ladder_trim_s[idx, :] = \
#                 self.s[loc_mid_02 - int(w/2): loc_mid_02 - int(w/2) + int(w) ]
#         self.ladder_f = ladder_trim_f
#         self.ladder_s = ladder_trim_s

#     def lor(self, freq, w, h):
#         """ Simple Lorentzian profile
        
#         Calculates N Lorentzian profiles, where N is the number of pairs in the
#         frequency list. 
        
#         Parameters
#         ----------
#         freq : float, ndarray
#             Central frequencies the N Lorentzians
#         w : float, ndarray
#             Widths of the N Lorentzians
#         h : float, ndarray
#             Heights of the N Lorentzians   
         
#         Returns
#         -------
#         lors : ndarray
#            A list containing one Lorentzian per pair.

#         """

#         norm = 1.0 + 4.0 / w**2 * (self.ladder_f.T - freq)**2
        
#         return h / norm

#     def model(self, l0, l2, width0, width2, height0, height2, back):
#         """
#         Calcuates a simple model of a flat backgroud plus two lorentzians
#         for each of the N pairs in the list of frequencies under consideration.

#         Parameters
#         ----------
#         l0 : ndarray
#             Array of length N, of the l=0 mode frequencies.
#         l2 : ndarray
#             Array of length N, of the l=2 mode frequencies.
#         width0 : ndarray
#             Array of length N, of the l=0 mode widths.
#         width2 : ndarray
#             Array of length N, of the l=2 mode widths.
#         height0 : ndarray
#             Array of length N, of the l=0 mode heights.
#         height2 : ndarray
#             Array of length N, of the l=2 mode heights.
#         back : ndarray
#             Array of length N, of the background levels.

#         Returns
#         -------
#         mod : ndarray
#             A 2D array (or 'ladder') containing the calculated models for each
#             of the N pairs.

#         """

#         mod = np.ones(self.ladder_f.shape).T * back
#         mod += self.lor(l0, width0, height0)
#         mod += self.lor(l2, width2, height2)
#         return mod.T

#     def init_model(self, model_type):
#         """ Initialize the pymc3 model for peakbag
        
#         Sets up the pymc3 model to sample, to perform the final peakbagging. 
        
#         Two treatements of the mode widths are available, the default
#         independent mode widths for each pair, or modeling the mode widths as
#         a function of freqeuency as a Gaussian Process. 

#         Parameters
#         ----------
#         model_type : str
#             Model choice for the mode widths. The default is to treat the all
#             mode widths independently. Alternatively they can be modeled as
#             a GP.

#         """

#         self.pm_model = pm.Model()

#         dnu = 10**self.asy_fit.summary.loc['dnu', 'mean']
#         dnu_fac = 0.03 # Prior on mode frequency has width 3% of Dnu.
#         height_fac = 0.4 # Lognorrmal prior on height has std=0.4.
#         width_fac = 1.0 # Lognorrmal prior on width has std=1.0.
#         back_fac = 0.5 # Lognorrmal prior on back has std=0.5.
#         N = len(self.start['l2'])

#         with self.pm_model:

#             if model_type != 'model_gp':
#                 if model_type != 'simple': # defaults to simple if bad input
#                     warnings.warn('Model not defined - using simple model')
#                 width0 = pm.Lognormal('width0', mu=np.log(self.start['width0']),
#                                   sigma=width_fac, shape=N)
#                 width2 = pm.Lognormal('width2', mu=np.log(self.start['width2']),
#                                   sigma=width_fac, shape=N)

#                 self.init_sampler = 'adapt_diag'
#                 self.target_accept = 0.9

#             elif model_type == 'model_gp':
#                 warnings.warn('This model is developmental - use carefully')
#                 # Place a GP over the l=0 mode widths ...
#                 m0 = pm.Normal('gradient0', 0, 10)
#                 c0 = pm.Normal('intercept0', 0, 10)
#                 sigma0 = pm.Lognormal('sigma0', np.log(1.0), 1.0)
#                 ls = pm.Lognormal('ls', np.log(0.3), 1.0)
#                 mean_func0 = pm.gp.mean.Linear(coeffs=m0, intercept=c0)
#                 cov_func0 = sigma0 * pm.gp.cov.ExpQuad(1, ls=ls)
#                 self.gp0 = pm.gp.Latent(cov_func=cov_func0, mean_func=mean_func0)
#                 ln_width0 = self.gp0.prior('ln_width0', X=self.n)
#                 width0 = pm.Deterministic('width0', pm.math.exp(ln_width0))
#                 # and on the l=2 mode widths
#                 m2 = pm.Normal('gradient2', 0, 10)
#                 c2 = pm.Normal('intercept2', 0, 10)
#                 sigma2 = pm.Lognormal('sigma2', np.log(1.0), 1.0)
#                 mean_func2 = pm.gp.mean.Linear(coeffs=m2, intercept=c2)
#                 cov_func2 = sigma2 * pm.gp.cov.ExpQuad(1, ls=ls)
#                 self.gp2 = pm.gp.Latent(cov_func=cov_func2, mean_func=mean_func2)
#                 ln_width2 = self.gp2.prior('ln_width2', X=self.n)
#                 width2 = pm.Deterministic('width2', pm.math.exp(ln_width2))

#                 self.init_sampler = 'advi+adapt_diag'
#                 self.target_accept = 0.99


#             l0 = pm.Normal('l0', self.start['l0'], dnu*dnu_fac, shape=N)

#             l2 = pm.Normal('l2', self.start['l2'], dnu*dnu_fac, shape=N)

#             height0 = pm.Lognormal('height0', mu=np.log(self.start['height0']),
#                                     sigma=height_fac, shape=N)
#             height2 = pm.Lognormal('height2', mu=np.log(self.start['height2']),
#                                     sigma=height_fac, shape=N)
#             back = pm.Lognormal('back', mu=np.log(1.0), sigma=back_fac, shape=N)

#             limit = self.model(l0, l2, width0, width2, height0, height2, back)
            
#             pm.Gamma('yobs', alpha=1, beta=1.0/limit, observed=self.ladder_s)


#     def _addPPRatio(self):
#         """ Add the prior/posterior width ratio to summary
        
#         Computes the ratio of the prior width and the posterior width. This is a
#         quantity which indicates which probability predominantly informs the 
#         resulting mode frequency. If the ratio is < 1 the prior dominates, and
#         vice versa for ratios > 1. 
        
#         No cut-off is made based on this ratio, it is merely to inform the user.
                
#         """
        
#         dnu = 10**self.asy_fit.summary.loc['dnu', 'mean']
        
#         log_ppr = np.log10((0.03*dnu))-np.log10(self.summary['sd'])
        
#         idx = np.array(['l' in name for name in self.summary.index], dtype = 'bool')
        
#         self.summary['log_ppr'] = np.nan
        
#         self.summary.at[idx, 'log_ppr'] = log_ppr[idx]


#     def __call__(self, model_type='simple', tune=1500, nthreads=1, maxiter=4,
#                      advi=False):
#         """ Perform all the steps in peakbag.
        
#         Initializes and samples the `PyMC3' model that is set up using the
#         outputs from asy_peakbag as priors. 

#         Parameters
#         ----------
#         model_type : str
#             Defaults to 'simple'.
#             Can be either 'simple' or 'model_gp' which sets the type of model
#             to be fitted to the data.
#         tune : int, optional
#             Numer of tuning steps passed to pym3.sample. Default is 1500.
#         nthreads : int, optional
#             Number of cores to use - passed to pym3.sample. Default is 1.
#         maxiter : int, optional
#             Number of times to attempt to reach convergence. Default is 4.
#         advi : bool, optional
#             Whether or not to fit using the fullrank_advi option in pymc3. 
#             Default is False.

#         """

#         self.init_model(model_type=model_type)
               
#         # REMOVE THIS WHEN pymc3 v3.8 is a bit older. 
#         try:
#             rhatfunc = pm.diagnostics.gelman_rubin
#             warnings.warn('pymc3.diagnostics.gelman_rubin is depcrecated; upgrade pymc3 to v3.8 or newer.', DeprecationWarning)
#         except:
#             rhatfunc = pm.stats.rhat
        

#         if advi:
#             with self.pm_model:
#                 cb = pm.callbacks.CheckParametersConvergence(every=1000,
#                                                              diff='absolute',
#                                                              tolerance=0.01)

#                 mean_field = pm.fit(n=200000, method='fullrank_advi',
#                                     start=self.start,
#                                     callbacks=[cb])
#                 self.traces = mean_field.sample(1000)
#         else:
#             Rhat_max = 10
#             niter = 1
#             while Rhat_max > 1.05:
#                 if niter > maxiter:
#                     warnings.warn('Did not converge!')
#                     break
#                 with self.pm_model:
#                     self.traces = pm.sample(tune=tune * niter, cores=nthreads,
#                                              start=self.start,
#                                              init=self.init_sampler,
#                                              target_accept=self.target_accept,
#                                              progressbar=False)
#                 Rhat_max = np.max([v.max() for k, v in rhatfunc(self.traces).items()])
#                 niter += 1
        
#         # REMOVE THIS WHEN pymc3 v3.8 is a bit older
#         try:
#             self.summary = pm.summary(self.traces)
#         except:
#             self.summary = pm.stats.summary(self.traces)
        
#         self.par_names = self.summary.index
        
#         samps = np.array([self.traces[x] for x in self.traces.varnames if not x.endswith('_log__')])
#         self.samples = np.array([]).reshape((samps.shape[1], 0))
#         for i in range(samps.shape[0]):   
#             self.samples = np.concatenate((self.samples, samps[i, :, :]), axis =1)
        
