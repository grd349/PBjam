import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from pbjam import jar
from pbjam.background import bkgModel
from pbjam.DR import PCA
import pbjam.distributions as dist
from pbjam.jar import generalModelFuncs
jax.config.update('jax_enable_x64', True)

class Asyl20model(jar.DynestySamplingTools, generalModelFuncs):

    def __init__(self, f, s, obs, addPriors, N_p, Npca, PCAdims,
                 vis={'V20': 0.71}, priorpath=None):
        
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
 
        self.Nyquist = self.f[-1]

        modelParLabels = ['dnu', 'numax', 
                          'eps_p', 'd02', 
                          'alpha_p', 'env_width',
                          'env_height', 'mode_width', 
                          'teff', 'bp_rp', 
                          'H1_nu',  'H1_exp', 
                          'H_power', 'H2_nu', 
                          'H2_exp', 'H3_power', 
                          'H3_nu', 'H3_exp', 
                          'shot', 'nurot_e',  
                          'inc',
                         ]
        self.setLabels(self.addPriors, modelParLabels)
         
        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.setupDR()
  
        self.setPriors()
 
        self.background = bkgModel(self.f, self.Nyquist)
 
        self.ndims = len(self.priors.keys())
 
        self.setAddObs(keys=['teff', 'bp_rp'])

        self.makeEmpties()

    def makeEmpties(self):
        """ Make a bunch of static matrices so we don't need to make them during
        sampling
        """
        
        self.N_p_range = jnp.arange(self.N_p)
 
        self.N_p_mid = jnp.floor(self.N_p/2)
 
        self.ones_nu = jnp.ones_like(self.f)
       
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

        AddKeys = [k for k in self.variables if k in self.addPriors.keys()]

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
         
        self.DR = PCA(_obs, self.pcalabels, self.priorpath, self.Npca, selectLabels=['numax', 'dnu', 'teff', 'bp_rp']) 

        self.DR.fit_weightedPCA(self.PCAdims)

        _Y = self.DR.transform(self.DR.dataF)

        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(_Y)
        
        self.latentLabels = ['theta_%i' % (i) for i in range(self.PCAdims)]
       
    def model(self, theta_u):
        
        # l=2,0
        modes, _, _ = self.add20Pairs(self.ones_nu, **theta_u)
        
        # Background
        bkg = self.background(theta_u)
         
        return modes * bkg
    
    def add20Pairs(self, modes, d02, mode_width, nurot_e, inc, **kwargs):
         
        nu0_p, n_p = self.asymptotic_nu_p(**kwargs)

        Hs0 = self.envelope(nu0_p, **kwargs)

        for n in range(self.N_p):

            # Adding l=0
            modes += jar.lor(self.f, nu0_p[n], Hs0[n], mode_width) 
            
            # Adding l=2 multiplet
            for m in [-2, -1, 0, 1, 2]:
                
                H = Hs0[n] * self.vis['V20'] * jar.visell2(abs(m), inc)
                
                f = nu0_p[n] - d02 + m * nurot_e

                modes += jar.lor(self.f, f, H, mode_width)

        return modes, nu0_p, n_p
    
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
         
        theta_inv = self.DR.inverse_transform(theta[:self.DR.dimsR])
         
        theta_u = {key: theta_inv[i] for i, key in enumerate(self.pcalabels)}
         
        theta_u.update({key: theta[self.DR.dimsR:][i] for i, key in enumerate(self.addlabels)})
 
        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
    
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

        below = jnp.floor(nmax - self.N_p_mid).astype(int)
         
        enns = self.N_p_range + below

        return enns 

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
 
    def getMedianModel(self, samples_u, N=30):
 
        mod = np.zeros((len(self.f), N))
        
        # Generate random indices for selecting samples
        rkey = np.random.choice(list(samples_u.keys()))
        
        nsamples = len(samples_u[rkey])

        idx = np.random.choice(np.arange(nsamples), size=N, replace=False)
        
        for i, j in enumerate(idx):
            # Extract background parameters for the selected sample
            theta_u = {k: v[j] for k, v in samples_u.items()}
            
            # Compute the background model for the selected sample
            mod[:, i] = self.model(theta_u)
        
        # Compute the median background model across samples
        return np.median(mod, axis=1)
    
    def parseSamples(self, smp, Nmax=10000):

        N = min([len(list(smp.values())[0]), 
                 Nmax])
        
        for key in smp.keys():
            smp[key] = smp[key][:N]
        
        result = {'ell': np.array([]),
                  'enn': np.array([]),
                  'zeta': np.array([]),
                  'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                              'height': np.array([]).reshape((2, 0)), 
                              'width' : np.array([]).reshape((2, 0))
                             },
                  'samples': {'freq'  : np.array([]).reshape((N, 0)),
                              'height': np.array([]).reshape((N, 0)), 
                              'width' : np.array([]).reshape((N, 0))
                             },
                }
        
        result['summary'].update({key: jar.smryStats(smp[key]) for key in smp.keys()})
        result['samples'].update(smp)
  
        # l=0
        asymptotic_samps = np.array([self.asymptotic_nu_p(smp['numax'][i], smp['dnu'][i], smp['eps_p'][i], smp['alpha_p'][i]) for i in range(N)])
        n_p = np.median(asymptotic_samps[:, 1, :], axis=0).astype(int)
        
        result['ell'] = np.append(result['ell'], np.zeros(self.N_p))

        result['enn'] = np.append(result['enn'], n_p)

        result['zeta'] = np.append(result['zeta'], np.zeros(self.N_p))

        # # Frequencies
        nu0_samps = asymptotic_samps[:, 0, :]
        jar.modeUpdoot(result, nu0_samps, 'freq', self.N_p)

        # # Heights
        H0_samps = np.array([self.envelope(nu0_samps[i, :], smp['env_height'][i], smp['numax'][i], smp['env_width'][i]) for i in range(N)])
        jar.modeUpdoot(result, H0_samps, 'height', self.N_p)

        # # Widths
        W0_samps = np.tile(smp['mode_width'], self.N_p).reshape((self.N_p, N)).T
        jar.modeUpdoot(result, W0_samps, 'width', self.N_p)
        
        # l=2
        result['ell'] = np.append(result['ell'], np.zeros(self.N_p) + 2)
        result['enn'] = np.append(result['enn'], n_p-1)
        result['zeta'] = np.append(result['zeta'], np.zeros(self.N_p))

        # # Frequencies
        nu2_samps = np.array([nu0_samps[i, :] - smp['d02'][i] for i in range(N)])
        jar.modeUpdoot(result, nu2_samps, 'freq', self.N_p)

        # # Heights
        H2_samps = self.vis['V20'] * np.array([self.envelope(nu2_samps[i, :],  
                                                            smp['env_height'][i], 
                                                            smp['numax'][i], 
                                                            smp['env_width'][i]) for i in range(N)])
        jar.modeUpdoot(result, H2_samps, 'height', self.N_p)
        
        # # Widths
        W2_samps = np.tile(smp['mode_width'], np.shape(nu2_samps)[1]).reshape((nu2_samps.shape[1], nu2_samps.shape[0])).T
        jar.modeUpdoot(result, W2_samps, 'width', self.N_p)

        #Background
        #result['background'] = self.meanBkg(self.f, smp)  
  
        return result

# def setAddObs(self, ):
#         """ Set attribute containing additional observational data

#         Additional observational data other than the power spectrum goes here. 

#         Can be Teff or bp_rp color, but may also be additional constraints on
#         e.g., numax, dnu. 
#         """
        
#         self.addObs = {}

#         self.addObs['teff'] = dist.normal(loc=self.obs['teff'][0], 
#                                           scale=self.obs['teff'][1])

#         self.addObs['bp_rp'] = dist.normal(loc=self.obs['bp_rp'][0], 
#                                            scale=self.obs['bp_rp'][1])
        
 # def addAddObsLike(self, theta_u):
    #     """ Add the additional probabilities to likelihood
        
    #     Adds the additional observational data likelihoods to the PSD likelihood.

    #     Parameters
    #     ----------
    #     p : list
    #         Sampling parameters.

    #     Returns
    #     -------
    #     lnp : float
    #         The likelihood of a sample given the parameter PDFs.
    #     """

    #     lnp = 0

    #     for key in self.addObs.keys():       
    #         lnp += self.addObs[key].logpdf(theta_u[key]) 
 
    #     return lnp

    # def chi_sqr(self, mod):
    #     """ Chi^2 2 dof likelihood

    #     Evaulates the likelihood of observing the data given the model.

    #     Parameters
    #     ----------
    #     mod : jax device array
    #         Spectrum model.

    #     Returns
    #     -------
    #     L : float
    #         Likelihood of the data given the model
    #     """

    #     L = -jnp.sum(jnp.log(mod) + self.s / mod)

    #     return L      
    
    # @partial(jax.jit, static_argnums=(0,))
    # def lnlikelihood(self, theta):
    #     """
    #     Calculate the log likelihood of the model given parameters and data.
        
    #     Parameters
    #     ----------
    #     theta : numpy.ndarray
    #         Parameter values.
    #     nu : numpy.ndarray
    #         Array of frequency values.

    #     Returns
    #     -------
    #     float :
    #         Log-likelihood value.
    #     """
    
    #     theta_u = self.unpackParams(theta)
 
    #     # Constraint from input obs
    #     lnlike = self.addAddObsLike(theta_u)
         
    #     # Constraint from the periodogram 
    #     mod = self.model(theta_u)
         
    #     lnlike += self.chi_sqr(mod)
         
    #     return lnlike

# def envelope(self, nu, env_height, numax, env_width, **kwargs):
    #     """ Power of the seismic p-mode envelope
    
    #     Computes the power at frequency nu in the oscillation envelope from a 
    #     Gaussian distribution. Used for computing mode heights.
    
    #     Parameters
    #     ----------
    #     nu : float
    #         Frequency (in muHz).
    #     hmax : float
    #         Height of p-mode envelope (in SNR).
    #     numax : float
    #         Frequency of maximum power of the p-mode envelope (in muHz).
    #     width : float
    #         Width of the p-mode envelope (in muHz).
    
    #     Returns
    #     -------
    #     h : float
    #         Power at frequency nu (in SNR)   
    #     """
    
    #     return jar.gaussian(nu, 2*env_height, numax, env_width)

    # variables = {'l20':{'dnu'       : {'info': 'large frequency separation'               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
    #                     'numax'     : {'info': 'frequency at maximum power'               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
    #                     'eps_p'     : {'info': 'phase offset of the p-modes'              , 'log10': False, 'pca': True, 'unit': 'None'}, 
    #                     'd02'       : {'info': 'l=0,2 mean frequency difference'          , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
    #                     'alpha_p'   : {'info': 'curvature of the p-modes'                 , 'log10': True , 'pca': True, 'unit': 'None'}, 
    #                     'env_width' : {'info': 'envelope width'                           , 'log10': True , 'pca': True, 'unit': 'muHz'},
    #                     'env_height': {'info': 'envelope height'                          , 'log10': True , 'pca': True, 'unit': 'ppm^2/muHz'}, 
    #                     'mode_width': {'info': 'mode width'                               , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
    #                     'teff'      : {'info': 'effective temperature'                    , 'log10': True , 'pca': True, 'unit': 'K'}, 
    #                     'bp_rp'     : {'info': 'Gaia Gbp-Grp color'                       , 'log10': False, 'pca': True, 'unit': 'mag'},
    #                     },
    #             'background' : {'H1_nu'     : {'info': 'Frequency of the high-frequency Harvey'   , 'log10': True , 'pca': True, 'unit': 'muHz'}, 
    #                             'H1_exp'    : {'info': 'Exponent of the high-frequency Harvey'    , 'log10': False, 'pca': True, 'unit': 'None'},
    #                             'H_power'   : {'info': 'Power of the Harvey law'                  , 'log10': True , 'pca': True, 'unit': 'ppm^2/muHz'}, 
    #                             'H2_nu'     : {'info': 'Frequency of the mid-frequency Harvey'    , 'log10': True , 'pca': True, 'unit': 'muHz'},
    #                             'H2_exp'    : {'info': 'Exponent of the mid-frequency Harvey'     , 'log10': False, 'pca': True, 'unit': 'None'},
    #                             'H3_power'  : {'info': 'Power of the low-frequency Harvey'        , 'log10': True , 'pca': False, 'unit': 'ppm^2/muHz'}, 
    #                             'H3_nu'     : {'info': 'Frequency of the low-frequency Harvey'    , 'log10': True , 'pca': False, 'unit': 'muHz'},
    #                             'H3_exp'    : {'info': 'Exponent of the low-frequency Harvey'     , 'log10': False, 'pca': False, 'unit': 'None'},
    #                             'shot'      : {'info': 'Shot noise level'                         , 'log10': True , 'pca': False, 'unit': 'ppm^2/muHz'},
    #                             },
                 
    #             'common': {'nurot_e'   : {'info': 'envelope rotation rate'                   , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
    #                        'inc'       : {'info': 'stellar inclination axis'                 , 'log10': False, 'pca': False, 'unit': 'rad'},}
    #             }



 # def setFreqRange(self,):
    #     """ Get frequency range around numax for model 

    #     Returns
    #     -------
    #     idx : jax device array
    #         Array of boolean values defining the interval of the frequency axis
    #         where the oscillation modes present.
    #     """
 
    #     lfreq = self.freq_limits[0]
        
    #     ufreq = self.freq_limits[1]

    #     return (lfreq < self.f) & (self.f < ufreq) 
    

# def setLabels(self, addPriors, modelParLabels):
    #     """
    #     Set parameter labels and categorize them based on priors.

    #     Parameters
    #     ----------
    #     priors : dict
    #         Dictionary containing prior information for specific parameters.

    #     Notes
    #     -----
    #     - Initializes default PCA and additional parameter lists.
    #     - Checks if parameters are marked for PCA and not in priors; if so, 
    #       adds to PCA list.
    #     - Otherwise, adds parameters to the additional list.
    #     - Combines PCA and additional lists to create the final labels list.
    #     - Identifies parameters that use a logarithmic scale and adds them to 
    #       logpars list.
    #     """
 
    #     with open("pbjam/data/parameters.json", "r") as read_file:
    #         availableParams = json.load(read_file)
        
    #     self.variables = {key: availableParams[key] for key in modelParLabels}

    #     # Default PCA parameters       
    #     self.pcalabels = []
        
    #     # Default additional parameters
    #     self.addlabels = []
        
    #     # If key appears in priors dict, override default and move it to add. 
    #     for key in self.variables.keys():
                
    #         if self.variables[key]['pca'] and (key not in addPriors.keys()):
    #             self.pcalabels.append(key)
    #         else:
    #             self.addlabels.append(key)

    #     #self.labels = self.pcalabels + self.addlabels

    #     # Parameters that are in log10
    #     self.logpars = []
    #     for key in self.variables.keys():
    #         if self.variables[key]['log10']:
    #             self.logpars.append(key)

    #     return variables, pcalabels, addlabels, logpars