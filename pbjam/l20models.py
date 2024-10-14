import jax
import jax.numpy as jnp
import numpy as np
from pbjam import jar, samplers
from pbjam.background import bkgModel
from pbjam.DR import PCA
import pbjam.distributions as dist 
jax.config.update('jax_enable_x64', True)

class Asyl20model(samplers.DynestySampling, jar.generalModelFuncs):
    """
    A class for constructing the l20 model using the asymptotic relation for p-modes.

    Parameters
    ----------
    f : array-like
        The frequency array of the spectrum.
    s : array-like
        The values of the power density spectrum.
    obs : dict
        Dictionary of observational inputs.
    addPriors : dict, optional
        Additional priors to be added. Default is an empty dictionary.
    N_p : int, optional
        Number of radial orders to use for mode identification. Default is 7.
    PCAsamples : int
        Number of samples for PCA.
    PCAdims : int
        Number of dimensions for PCA.
    vis : dict, optional
        Dictionary of visibility ratios for mode power given as V_{l}/V_{l=0}. Default is {'V20': 0.71}.
    priorPath : str, optional
        Path to prior information. If None, assumes it is in the pbjam/data directory.

    Attributes
    ----------
    Nyquist : float
        Nyquist frequency, defined as the highest frequency in `f`.
    modelParLabels : list
        Labels for model parameters.
    log_obs : dict
        Logarithms of the observational data.
    background : bkgModel
        Background model class instance for the frequency data.
    ndims : int
        Number of parameters in the model.
    ell : ndarray
        Array of angular degrees of the modes (l=0 and l=2).
    emm : ndarray
        Array of azimuthal orders of modes (fixed at m=0 but this may change in the future).
    """

    def __init__(self, f, s, obs, addPriors, N_p, PCAsamples, PCAdims, vis={'V20': 0.71}, priorPath=None):
        
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

        self.ell = np.append(np.zeros(self.N_p), np.zeros(self.N_p) + 2)
        
        self.emm = np.zeros(2*self.N_p)

        self._makeEmpties()

    def _makeEmpties(self):
        """ 
        Make a bunch of static matrices so we don't need to make them during
        sampling. Just for keeping things a bit neater. 
        """
        
        self.N_p_range = jnp.arange(self.N_p)
 
        self.N_p_mid = jnp.floor(self.N_p/2)
 
        self.ones_nu = jnp.ones_like(self.f)
       
    def setPriors(self):
        """
        Assigns the priors attribute, which is a dictionary containing class instances of 
        prior distributions for the different model parameters. Each must have ppf, pdf, 
        logpdf and cdf as callable methods. 

        The latent parameters are assigned initially followed by any priors that might 
        be manually specified in addPriors.     
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
        
        if 'H3_power' not in self.addPriors.keys():
            self.priors['H3_power'] = dist.normal(loc=jnp.log10(mu * self.f[0]), scale=1)  

        if 'H3_nu' not in self.addPriors.keys():
            self.priors['H3_nu'] = dist.beta(a=1.2, b=1.2, loc=-1, scale=2)  
        
        if 'H3_exp' not in self.addPriors.keys():
            self.priors['H3_exp'] = dist.beta(a=1.2, b=1.2, loc=1.5, scale=3.5)  

        if 'shot' not in self.addPriors.keys():
            self.priors['shot'] = dist.normal(loc=jnp.log10(shot_est), scale=0.1)

        # Envelope rotation prior. Envelope rotation is not included in this model.
        if 'nurot_e' not in self.addPriors.keys():
            self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        if 'inc' not in self.addPriors.keys():
            self.priors['inc'] = dist.truncsine()

    def setupDR(self):
        """ 
        Sets up the latent parameters as distribution class instances with 
        callable ppf, pdf, logpdf and cdf methods.
         
        Also assigns the projection functions for the dimensionality reduction 
        to transform between the latent and model parameter spaces.

        Parameters included in the addPriors argument are not included in the 
        PCA dimensionality reduction.

        Each latent parameter is assigned a parameter label `theta_i`. 
        """
 
        _obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in ['numax', 'dnu', 'teff']}
         
        for key in ['bp_rp']:
            _obs[key] = self.obs[key]
         
        self.DR = PCA(_obs, self.pcaLabels, self.priorPath, self.PCAsamples, selectLabels=['numax', 'dnu', 'teff', 'bp_rp']) 

        self.DR.fit_weightedPCA(self.PCAdims)

        _Y = self.DR.transform(self.DR.dataF)

        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(_Y)
        
        self.latentLabels = ['theta_%i' % (i) for i in range(self.PCAdims)]
       
    def model(self, thetaU):
        """
        Computes the model spectrum by combining the l20 mode pairs and background components.

        Notes
        -----
        The l20 model is defined based on the SNR ratio of the modes, and so is multiplied onto the
        background model (instead of being added as is usually the case.)

        Parameters
        ----------
        thetaU : dict
            A dictionary of model parameters.

        Returns
        -------
        mod : ndarray
            The computed model spectrum.
        """
        
        # l=2,0
        modes, _, _ = self.add20Pairs(**thetaU)
        
        # Background
        bkg = self.background(thetaU)

        mod = modes * bkg
         
        return mod
    
    def add20Pairs(self, d02, mode_width, nurot_e, inc, **kwargs):
        """
        Adds l=2,0 mode pairs to the spectrum.

        The mode heights are defined in terms of the SNR of the modes. The
        resulting model should therefore be multipled onto a background to get
        a correct spectrum model.

        Parameters
        ----------
        d02 : float
            The frequency separation between l=0 and l=2 modes.
        mode_width : float
            The width of the modes.
        nurot_e : float
            The envelope rotation frequency.
        inc : float
            The inclination angle.
        **kwargs : dict
            Additional keyword arguments for the asymptotic frequency calculation and envelope functions.

        Returns
        -------
        modes : ndarray
            Spectrum model of the combined l=2,0 mode pairs.
        nu0_p : ndarray
            The frequencies of the l=0 modes.
        n_p : ndarray
            The radial order of the modes.
        """
 
        nu0_p, n_p = self.asymptotic_nu_p(**kwargs)

        Hs0 = jar.envelope(nu0_p, **kwargs)

        modes = self.ones_nu

        for n in range(self.N_p):

            # Adding l=0
            modes += jar.lor(self.f, nu0_p[n], Hs0[n], mode_width) 
            
            # Adding l=2 multiplet
            for m in [-2, -1, 0, 1, 2]:
                
                H = Hs0[n] * self.vis['V20'] * jar.visell2(abs(m), inc)
                
                f = nu0_p[n] - d02 + m * nurot_e

                modes += jar.lor(self.f, f, H, mode_width)

        return modes, nu0_p, n_p
    
    def unpackParams(self, theta): 
        """ Put the parameters in theta in a dictionary.

        Parameters
        ----------
        theta : array
            Array of parameters drawn from the posterior distribution.

        Returns
        -------
        thetaU : dict
            The unpacked parameters.
        """
         
        theta_inv = self.DR.inverse_transform(theta[:self.DR.dimsR])
         
        thetaU = {key: theta_inv[i] for i, key in enumerate(self.pcaLabels)}
         
        thetaU.update({key: theta[self.DR.dimsR:][i] for i, key in enumerate(self.addLabels)})
 
        for key in self.logpars:
            thetaU[key] = 10**thetaU[key]
 
        return thetaU
    
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

        Get the enns that will be included in the asymptotic relation model.
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
    
    def parseSamples(self, smp, Nmax=5000):
        """
        Parses the samples to extract and organize the model parameters.

        Attempts to include at most N samples from the model, but will default
        to the actual number of samples of the model parameters if it's less than N.

        The resulting dictionary contains some global parameters, ell, enn, emm etc. and 
        two dictionaries, one containing the samples drawn and one with their summary 
        statistics.

        Samples and summary statistics are also included for the mode height, width and 
        frequencies.

        Parameters
        ----------
        smp : dict
            A dictionary of sampled parameters.
        Nmax : int, optional
            Maximum number of samples to include. Default is 5000.

        Returns
        -------
        result : dict
            A dictionary containing parsed and organized model parameters.
        """

        N = min([len(list(smp.values())[0]), 
                 Nmax])
        
        for key in smp.keys():
            smp[key] = smp[key][:N]
        
        result = {'ell': self.ell,
                  'enn': np.array([]),
                  'emm': self.emm,
                  'zeta': np.array([]),
                  'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                              'height': np.array([]).reshape((2, 0)), 
                              'width' : np.array([]).reshape((2, 0)),
                              'rotAsym': np.zeros((2, 2*self.N_p))
                             },
                  'samples': {'freq'  : np.array([]).reshape((N, 0)),
                              'height': np.array([]).reshape((N, 0)), 
                              'width' : np.array([]).reshape((N, 0)), 
                              'rotAsym' : np.zeros((N, 2*self.N_p))
                             },
                }
        
        result['summary'].update({key: jar.smryStats(smp[key]) for key in smp.keys()})
        result['samples'].update(smp)
  
        # l=0
        jasymptotic_nu_p = jax.jit(self.asymptotic_nu_p)
        asymptotic_samps = np.array([jasymptotic_nu_p(smp['numax'][i], smp['dnu'][i], smp['eps_p'][i], smp['alpha_p'][i]) for i in range(N)])
        n_p = np.median(asymptotic_samps[:, 1, :], axis=0).astype(int)
        
        result['enn'] = np.append(result['enn'], n_p)

        result['zeta'] = np.append(result['zeta'], np.zeros(self.N_p))

        # Frequencies
        nu0_samps = asymptotic_samps[:, 0, :]
        jar.modeUpdoot(result, nu0_samps, 'freq', self.N_p)

        # Heights
        jenvelope = jax.jit(jar.envelope)
        H0_samps = np.array([jenvelope(nu0_samps[i, :], smp['env_height'][i], smp['numax'][i], smp['env_width'][i]) for i in range(N)])
        jar.modeUpdoot(result, H0_samps, 'height', self.N_p)

        # Widths
        W0_samps = np.tile(smp['mode_width'], self.N_p).reshape((self.N_p, N)).T
        jar.modeUpdoot(result, W0_samps, 'width', self.N_p)
        
        # l=2
        result['enn'] = np.append(result['enn'], n_p-1)
        result['zeta'] = np.append(result['zeta'], np.zeros(self.N_p))

        # Frequencies
        nu2_samps = np.array([nu0_samps[i, :] - smp['d02'][i] for i in range(N)])
        jar.modeUpdoot(result, nu2_samps, 'freq', self.N_p)

        # Heights
        H2_samps = self.vis['V20'] * np.array([jenvelope(nu2_samps[i, :],  
                                                            smp['env_height'][i], 
                                                            smp['numax'][i], 
                                                            smp['env_width'][i]) for i in range(N)])
        jar.modeUpdoot(result, H2_samps, 'height', self.N_p)
        
        # Widths
        W2_samps = np.tile(smp['mode_width'], np.shape(nu2_samps)[1]).reshape((nu2_samps.shape[1], nu2_samps.shape[0])).T
        jar.modeUpdoot(result, W2_samps, 'width', self.N_p)
  
        return result

