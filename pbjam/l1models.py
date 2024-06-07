import jax.numpy as jnp
import numpy as np
import jax, warnings
from pbjam import jar
from functools import partial
from pbjam.jar import constants as c
from pbjam.DR import PCA
import pbjam.distributions as dist
from pbjam.jar import generalModelFuncs, DynestySamplingTools
import dynesty
from dynesty import utils as dyfunc

jax.config.update('jax_enable_x64', True)

 
class Asyl1Model(DynestySamplingTools, generalModelFuncs):
    def __init__(self, f, s, obs, addPriors, N_p, NPriorSamples, vis={'V10': 1.22}, priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
 
        modelParLabels = ['d01', 'nurot_e',  'inc',]

        self.setLabels(self.addPriors, modelParLabels)

        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.setPriors()

        self.setAddObs(keys=[])

        self.ndims = len(self.priors)

        self.ones_nu = jnp.ones_like(self.f)
 
    def setPriors(self,):
        """ Set the prior distributions.

        The prior distributions are constructed from the projection of the 
        PCA sample onto the reduced dimensional space.

        """

        self.priors = {}

        _obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in ['numax', 'dnu', 'teff']}
         
        for key in ['bp_rp']:
            _obs[key] = self.obs[key]

        self.DR = PCA(_obs, ['d01'], self.priorpath, self.NPriorSamples, selectLabels=['numax', 'dnu', 'teff'], dropNansIn='Not all') 
        
        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(self.DR.dataF)
 
        self.priors['d01'] = dist.distribution(self.DR.ppf[0], 
                                               self.DR.pdf[0], 
                                               self.DR.logpdf[0], 
                                               self.DR.cdf[0])

        # Core rotation prior
        self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        self.priors['inc'] = dist.truncsine()            
 
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

        theta_u = {key: theta[i] for i, key in enumerate(self.labels)}
 
        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
    
    def nu1_frequencies(self, theta_u):
        return self.obs['nu0_p'] + theta_u['d01']
    
    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u,):
        
        nu1s = self.nu1_frequencies(theta_u)
         
        Hs1 = self.envelope(nu1s, self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0])
        
        modewidth1s = self.obs['mode_width'][0] 
         
        nurot = theta_u['nurot_e']
        
        modes = self.ones_nu

        for i in range(len(nu1s)):
 
            modes += jar.lor(self.f, nu1s[i]        , Hs1[i] * self.vis['V10'], modewidth1s) * jnp.cos(theta_u['inc'])**2
        
            modes += jar.lor(self.f, nu1s[i] - nurot, Hs1[i] * self.vis['V10'], modewidth1s) * jnp.sin(theta_u['inc'])**2 / 2
        
            modes += jar.lor(self.f, nu1s[i] + nurot, Hs1[i] * self.vis['V10'], modewidth1s) * jnp.sin(theta_u['inc'])**2 / 2

        return modes
 
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
   
    def parseSamples(self, smp, Nmax=10000):

        N = min([len(list(smp.values())[0]), Nmax])

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

        # l=1
        nu1_samps = np.array([self.nu1_frequencies({key: smp[key][i] for key in ['d01']}) for i in range(N)])

        result['ell'] = np.append(result['ell'], np.zeros(self.N_p) + 1)
        result['enn'] = np.append(result['enn'], np.zeros(self.N_p) - 1)
            
        jar.modeUpdoot(result, nu1_samps, 'freq', self.N_p)
    
        result['zeta'] = np.append(result['zeta'], np.zeros(result['summary']['freq'].shape[1]))

        # # Heights
        H1_samps = self.vis['V10'] * np.array([self.envelope(nu1_samps[i, :], self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0]) for i in range(N)]) 
        jar.modeUpdoot(result, H1_samps, 'height', self.N_p)

        # # Widths
        W1_samps = np.array([self.obs['mode_width'][0]*np.ones(result['summary']['freq'].shape[1]) for i in range(N)]) 
        jar.modeUpdoot(result, W1_samps, 'width', self.N_p)
        result['summary']['width'][1, :] = self.obs['mode_width'][1]*np.ones(result['summary']['freq'].shape[1])

        return result
    
     
class Mixl1Model(DynestySamplingTools, generalModelFuncs):

    def __init__(self, f, s, obs, addPriors, N_p, Npca, PCAdims,
                 vis={'V10': 1.22}, priorpath=None):
   
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        modelParLabels = ['u1', 'u2', 'DPi1', 'eps_g',
                          'd01', 'dnu', 'numax', 'nurot_c', 
                          'nurot_e', 'inc', 'teff'
                          ]
                    
        self.setLabels(self.addPriors, modelParLabels)
 
        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.badPrior = False

        self.setupDR()
 
        self.setAddObs(keys=['teff', 'dnu', 'numax'])

        if not self.badPrior: 
 
            self.setPriors()
 
            self.n_g = self.select_n_g()

            self.N_g = len(self.n_g)

            for i in range(self.N_g + self.N_p):
                self.addlabels.append(f'freqError{i}')

                self.priors[f'freqError{i}'] = dist.normal(loc=0, scale=0.03 * self.obs['dnu'][0])

            self.ndims = len(self.priors)

            # self.trimVariables()

            self.makeEmpties()
 
    def makeEmpties(self):
        """ Make a bunch of static matrices so we don't need to make them during
        sampling
        """

        self.ones_nu = jnp.ones_like(self.f)

        self.ones_block = jnp.ones((self.N_p, self.N_g))

        self.zeros_block = jnp.zeros((self.N_p, self.N_g))

        self.eye_N_p = jnp.eye(self.N_p)

        self.eye_N_g = jnp.eye(self.N_g)

        self.D_gamma = jnp.vstack((jnp.zeros((self.N_p, self.N_p + self.N_g)), 
                                jnp.hstack((self.zeros_block.T, self.eye_N_g))))
        
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
        
        # TODO maybe in future put bp_rp back in, when we aren't using models anymore
        self.DR = PCA(_obs, self.pcalabels, self.priorpath, self.Npca, selectLabels=['numax', 'dnu', 'teff'], dropNansIn='Not all') 

        self.badPrior = False
        
        # If no prior samples are returned, flag bad prior
        if len(self.DR.selectedSubset) == 0:
            self.badPrior = True
        # Else cycle through the selection labels and compare with obs, if too far away the prior is also labeled as bad
        else:
            for i, key in enumerate(self.DR.selectLabels):
                
                S = self.DR.selectedSubset[key].values

                if (min(S) - self.DR.obs[key][0] > 0.1) or (self.DR.obs[key][0]- max(S) > 0.1):
                    
                    self.badPrior = True
                    
                    warnings.warn(f'Target {key} more than 10 percent beyond limits of the viable prior sample. Prior is not reliable.', stacklevel=2)
 
        self.DR.fit_weightedPCA(self.PCAdims)
 
        if len(self.pcalabels) > 0 and not self.badPrior:

            _Y = self.DR.transform(self.DR.dataF)

            self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(_Y)

            self.latentLabels = ['theta_%i' % (i) for i in range(self.PCAdims)]

        else:
            self.latentLabels = []

            self.DR.inverse_transform = lambda x: []

            self.DR.dimsR = 0


    
    def select_n_g(self, fac=1):
        """ Select and initial range for n_g

        Computes the number of g-modes that are relevant near the oscillation
        envelope. This is based on the expected range for DPi1 and eps_g and 
        numax.

        This is used to set the number of g-modes at the start of the run, and
        sets the number of g-modes at or near the p-mode envelope. The range is
        significantly wider than the actual power distribution of the envelope
        so there is room for DPi1 and eps_g to change.

        Returns
        -------
        n_g_ppf : list
            The quauntile functions for DPi1 and eps_g. 
        fac : float
            g-modes are considered if they fall within +/- fac * envelope_width
            of numax. A larger may(??) increase precision at the cost of time
            to perform eigendecomposition.
        """
 
        _sampler = dynesty.NestedSampler(self.obsOnlylnlikelihood, 
                                        self.ptform, 
                                        len(self.priors),
                                        sample='rwalk'
                                        )
            
        _sampler.run_nested(print_progress=False, save_bounds=False,)

        _samples = dyfunc.resample_equal(_sampler.results.samples, 
                                         jnp.exp(_sampler.results.logwt - _sampler.results.logz[-1]))
       
        _sampler.reset()

        del _sampler

        _samplesU = self.unpackSamples(_samples)
 
        DPi1 = np.median(_samplesU['DPi1'])
        
        eps_g = np.median(_samplesU['eps_g'])
 
        freq_lims = (min(self.obs['nu0_p']) - 5*self.obs['dnu'][0],  max(self.obs['nu0_p']) + 5*self.obs['dnu'][0])
 
        # Start with an exagerated number of g-modes.
        init_n_g = jnp.arange(10000)[::-1] + 1

        min_n_g_c = init_n_g.max()
 
        max_n_g_c = init_n_g.min()
 
        def update_limit(limit, idx, init_n_g, crit):
            """ Update min/max frequency limits

            Revise the frequency range min/max g-mode radial order, based on the
            previous min/max frequency limits for the g-modes to include, and 
            the current set of frequencies that fall near the
            envelope.
            
            """

            if crit == 'min':
                t = jnp.where(idx, init_n_g, 0 * init_n_g + jnp.inf).min()
                 
                limit = jnp.minimum(limit, t)

            elif crit == 'max':
                t = jnp.where(idx, init_n_g, 0 * init_n_g - 1).max()
                 
                limit = jnp.maximum(limit, t)

            return limit
                
        nu_g = self.asymptotic_nu_g(init_n_g, DPi1, eps_g)
        
        idx_c = (freq_lims[0] < nu_g) & (nu_g < freq_lims[1])
             
        min_n_g_c = update_limit(min_n_g_c, idx_c, init_n_g, 'min')  
        
        max_n_g_c = update_limit(max_n_g_c, idx_c, init_n_g, 'max')   
 
        
        n_g = jnp.arange(min_n_g_c, max_n_g_c, dtype=int)[::-1]
        
       
        if len(n_g) > 100:
            warnings.warn(f'{len(n_g)} g-modes in the coupling matrix.')

        # Force a minimum of 1 g-mode to be included as a test
        if len(n_g) == 0:
            n_g = jnp.array([1])

        return n_g

    def trimVariables(self):
        
        N = self.N_p + self.N_g  

        for i in range(N, 200, 1):
            del self.addlabels[self.addlabels.index(f'freqError{i}')]
            #del self.labels[self.labels.index(f'freqError{i}')]
            del self.priors[f'freqError{i}']
            
        self.ndims = len(self.priors)    
   
    def setPriors(self,):
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
 
        # Core rotation prior
        self.priors['nurot_c'] = dist.uniform(loc=-2., scale=2.)

        self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        self.priors['inc'] = dist.truncsine() 

        
             
    
 
    def l1_modewidths(self, zeta, fac=1, **kwargs):
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
         
        return  fac * self.obs['mode_width'][0] * jnp.maximum(1e-6, 1. - zeta) 
 
    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u,):

        theta_u['p_L'] = (theta_u['u1'] + theta_u['u2'])/jnp.sqrt(2)

        theta_u['p_D'] = (theta_u['u1'] - theta_u['u2'])/jnp.sqrt(2)
        
        nu1s, zeta = self.nu1_frequencies(theta_u)
         
        Hs1 = self.envelope(nu1s, self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0])
        
        modewidth1s = self.l1_modewidths(zeta,)
         
        nurot = zeta * theta_u['nurot_c'] + (1 - zeta) * theta_u['nurot_e']
        
        modes = self.ones_nu

        for i in range(len(nu1s)):
             
            nul1 = nu1s[i] + theta_u[f'freqError{i}']  

            modes += jar.lor(self.f, nul1                     , Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.cos(theta_u['inc'])**2
        
            modes += jar.lor(self.f, nul1 - nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(theta_u['inc'])**2 / 2
        
            modes += jar.lor(self.f, nul1 + nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(theta_u['inc'])**2 / 2

        return modes
    
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
        
        # for k, v in {key: theta[self.DR.dimsR:][i] for i, key in enumerate(self.addlabels)}.items():
        #     print(k, v)

        theta_u.update({key: theta[self.DR.dimsR:][i] for i, key in enumerate(self.addlabels)})

        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
 
    def asymptotic_nu_g(self, n_g, DPi1, eps_g):
        """Asymptotic relation for g-modes

        Asymptotic relation for the g-mode frequencies in terms of a fundamental
        period offset (defined by the maximum Brunt-Vaisala frequency), the 
        asymptotic g-mode period spacing, the g-mode phase offset, and an 
        optional curvature term.

        Parameters
        ----------
        n_g : jax device array
            Array of radial orders for the g-modes.
        DPi1 : float
            Period spacing for l=1 in seconds).
        eps_g : float
            Phase offset of the g-modes.
        alpha_g : float
            Curvature scale of the g-modes.
        max_N2 : float
            Maximum of the Brunt-Vaisala frequency.
        Returns
        -------
        jax device array
            Frequencies of the notionally pure g-modes of degree l.
        """
 
        DPi1 *= 1e-6 # DPi1 in s to Ms.  
 
        P = DPi1 * (n_g + eps_g)
        
        return 1/P

    @partial(jax.jit, static_argnums=(0,))
    def nu1_frequencies(self, theta_u):
        """
        Calculate mixed nu1 values and associated zeta values.
        
        Parameters
        ----------
        nu0_p : float
            Initial nu0 value.
        n_p : int
            Number of n values.
        d01 : float
            The d01 frequency separation
        DPi1 : float
            Period spacing for l=1.
        p_L : jax device array
            Polynomial coefficients for the L coupling strength matrix.
        p_D : jax device array
            Polynomial coefficients for the D coupling strength matrix.
        eps_g : float
            Phase offset of the g-modes.
        alpha_g : float
            Curvature scale of the g-modes.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        nu : jax device array
            Array of frequencies of the mixed l=1 modes. 
        zeta : jax device array
            Array of mixing degrees for the modes.
        """

        nu1_p = self.obs['nu0_p'] + theta_u['d01'] 
        
        nu_g = self.asymptotic_nu_g(self.n_g, theta_u['DPi1'], theta_u['eps_g'])

        L, D = self.generate_matrices(nu1_p, nu_g, theta_u['p_L'], theta_u['p_D'])
         
        nu, zeta = self.new_modes(L, D)
 
        return nu, zeta 
 
    def generate_matrices(self, nu_p, nu_g, p_L, p_D):
        """Generate coupling strength matrices

        Computes the coupling strength matrices based on the asymptotic p- and
        g-mode frequencies and the polynomial representation of the coupling
        strengths.

        Parameters
        ----------
        n_p : jax device array
            Array containing p-mode radial orders.
        n_g : jax device array
            Array containing g-mode radial orders.
        nu_p : jax device array
            Array containing asymptotic l=1 p-mode frequencies.
        nu_g : jax device array
            Array containing asymptotic l=1 g-mode frequencies.
        p_L : jax device array
            Parameter vector describing 2D polynomial coefficients for coupling 
            strengths.
        p_D : jax device array
            Parameter vector describing 2D polynomial coefficients for overlap 
            integrals.

        Returns
        -------
        L : jax device array
            Matrix of coupling strengths.
        D : jax device array
            Matrix of overlap integrals.
        """
 
        L_cross = self.ones_block * p_L * (nu_g * c.nu_to_omega)**2

        D_cross = p_D * nu_g[jnp.newaxis, :] / nu_p[:, jnp.newaxis]

        L = jnp.hstack((jnp.vstack((jnp.diag(-(nu_p * c.nu_to_omega)**2), L_cross.T)),
                        jnp.vstack((L_cross, jnp.diag( -(nu_g * c.nu_to_omega)**2 )))
                        ))

        D = jnp.hstack((jnp.vstack((self.eye_N_p, D_cross.T)),
                        jnp.vstack((D_cross, self.eye_N_g))
                        ))

        return L, D
     
    def new_modes(self, L, D):
        """ Solve for mixed mode frequencies

        Given the matrices L and D such that we have eigenvectors

        L cᵢ = -ωᵢ² D cᵢ,

        with ω in Hz, we solve for the frequencies ν (μHz), mode mixing 
        coefficient zeta.

        Parameters
        ----------
        L : jax device array
            The coupling strength matrix.
        D : jax device array
            The overlap integral.

        Returns
        -------
        nu_mixed : jax device array
            Array of mixed mode frequencies.
        zeta : jax device array
            The mixing degree for each of the mixed modes.
        """

        Lambda, V = self.generalized_eig(L, D)
        
        new_omega2 = -Lambda
        
        zeta = jnp.diag(V.T @ self.D_gamma @ V)

        sidx = jnp.argsort(new_omega2)

        return jnp.sqrt(new_omega2)[sidx] / c.nu_to_omega, zeta[sidx]  

    def generalized_eig(self, A, B):
        
        B_inv = jnp.linalg.inv(B)
        
        U, V = jnp.linalg.eig(B_inv @ A)
        
        return U.real, V.real
    
    def generalized_eigh(self, A, B):
        
        B_inv = jnp.linalg.inv(B)
        
        U, V = jnp.linalg.eigh(B_inv @ A)
        
        return U.real, V.real
 
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

 
        smp['p_L'] = (smp['u1'] + smp['u2'])/jnp.sqrt(2)

        smp['p_D'] = (smp['u1'] - smp['u2'])/jnp.sqrt(2)

        result['samples'].update(smp)


        A = np.array([self.nu1_frequencies({key: smp[key][i] for key in ['d01', 'DPi1', 'p_L', 'p_D', 'eps_g']}) for i in range(N)])
 
        N_pg = self.N_p + self.N_g
        
        result['ell'] = np.append(result['ell'], np.zeros(N_pg) + 1)
        result['enn'] = np.append(result['enn'], np.zeros(N_pg) - 1)

        # Frequencies 
        nu1_samps = A[:, 0, :]
        
        sigma_nul1 = np.array([smp[key] for key in smp.keys() if key.startswith('freqError')]).T
        
        if len(sigma_nul1) == 0:        
            jar.modeUpdoot(result, nu1_samps, 'freq', N_pg)
        else:
            jar.modeUpdoot(result, nu1_samps + sigma_nul1, 'freq', N_pg)

        zeta_samps = A[:, 1, :]

        result['zeta'] = np.append(result['zeta'], np.median(zeta_samps, axis=0))
        
        # Heights
        H1_samps = self.vis['V10'] * np.array([self.envelope(nu1_samps[i, :], self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0]) for i in range(N)]) 
        jar.modeUpdoot(result, H1_samps, 'height', N_pg)
        
        # Widths
        W1_samps = np.array([self.l1_modewidths(zeta_samps[i, :], ) for i in range(N)]) 
        jar.modeUpdoot(result, W1_samps, 'width', N_pg)

        return result

    # def _makeTmpSample(self, keys, N=1000):
    #     """
    #     Draw samples for specified keys.

    #     Parameters
    #     ----------
    #     keys : list
    #         List of parameter keys to be sampled.
    #     N : int, optional
    #         Number of samples to generate. Default is 1000.

    #     Returns
    #     -------
    #     tuple
    #         A tuple containing the quantile function, probability density 
    #         function, log probability density function, and cumulative 
    #         distribution function of the generated samples.

    #     Notes
    #     -----
    #     - Generates N random samples.
    #     - Transforms the samples using `ptform` and `unpackParams`.
    #     - Constructs arrays for specified keys.
    #     - Computes quantile function, probability density function,
    #     log probability density function, and cumulative distribution function.
    #     """

    #     K = np.zeros((len(keys), N))

    #     for i in range(N):
    #         u = np.random.uniform(0, 1, size=self.ndims)
        
    #         theta = self.ptform(u)

    #         theta_u = self.unpackParams(theta)

    #         K[:, i] = np.array([theta_u[key] for key in keys]) 
        
    #     ppf, pdf, logpdf, cdf = dist.getQuantileFuncs(K.T)
        
    #     return ppf, pdf, logpdf, cdf
     
class RGBl1Model(DynestySamplingTools, generalModelFuncs):

    def __init__(self, f, s, obs, addPriors, N_g, NPriorSamples, maxiter=15, vis={'V10': 1.22}, priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
  
        modelParLabels = ['d01', 'DPi1', 'teff', 'bp_rp', 'eps_g', 'q',
                          'nurot_c', 'nurot_e', 'inc', 'dnu',
                          'numax']
                    
        self.setLabels(self.addPriors, modelParLabels)
        
        self.setPriors()
 
        self.setAddObs(keys=['teff', 'bp_rp', 'dnu', 'numax'])

        self.ndims = len(self.priors)

        self.n_g_min = 1
        
        self.makeEmpties()
 
    def setPriors(self,):
        """ Set the prior distributions.

        The prior distributions are constructed from the projection of the 
        PCA sample onto the reduced dimensional space.

        """

        self.priors = {}

        _obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in ['numax', 'dnu', 'teff']}
         
        for key in ['bp_rp']:
            _obs[key] = self.obs[key]

        self.DR = PCA(_obs, ['d01', 'DPi1', 'teff', 'bp_rp', 'dnu', 'numax'], self.priorpath, self.NPriorSamples, selectLabels=['numax', 'dnu', 'teff'], dropNansIn='Not all') 
        
        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(self.DR.dataF)
 
        for i, key in enumerate(['d01', 'DPi1', 'teff', 'bp_rp', 'dnu', 'numax']):

            self.priors[key] = dist.distribution(self.DR.ppf[i], 
                                                 self.DR.pdf[i], 
                                                 self.DR.logpdf[i], 
                                                 self.DR.cdf[i])
 
        self.priors['eps_g'] = dist.normal(loc=0.8, scale=0.1)

        self.priors['q'] = dist.uniform(loc=0.01, scale=0.6)

        # Core rotation prior
        self.priors['nurot_c'] = dist.uniform(loc=-2., scale=2.)

        self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        self.priors['inc'] = dist.truncsine() 
 
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

        theta_u = {key: theta[i] for i, key in enumerate(self.priors.keys())}
 
        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
    
    def makeEmpties(self):
        """ Compile the coupling function ahead of runtime
        """
        _nu_p = jnp.arange(len(self.obs['nu0_p']), dtype=np.float64)
    
        _nu_g = jnp.arange(self.N_g, dtype=np.float64)

        # Also precompute some things
        self.arange_nup = jnp.arange(len(_nu_p), dtype=np.float64)

        self.arange_nug = jnp.arange(len(_nu_g), dtype=np.float64)

        self.ones_nu = jnp.ones_like(self.f)

        # q_dummy = 0.1

        # DPi1_dummy = 100.
 
        #self.couple = jax.jit(self.couple_halley).lower(_nu_p, _nu_g, q_dummy, q_dummy, DPi1_dummy).compile()
     
    def model(self, thetaU):
 
        nu1_g = 1 / (thetaU['DPi1']/ 1e6 * (jnp.arange(self.n_g_min, self.n_g_min+self.N_g) + thetaU['eps_g']))[::-1]

        nu1_p = self.obs['nu0_p'] + thetaU['d01'] 

        num_p, num_g = self.couple(nu1_p, nu1_g, thetaU['q'], thetaU['q'], thetaU['DPi1'] / 1e6)

        nu1s = jnp.append(num_p, num_g) # TODO is this correct? Should all num_p and num_g be included in the model?

        zeta = self.zeta_p(nu1s, thetaU['q'], thetaU['DPi1']/1e6, thetaU['dnu'], nu1_p) # jnp.zeros_like(nu1s) # TODO how should zeta be calculated?

        Hs1 = self.envelope(nu1s, self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0])
        
        modewidth1s = self.l1_modewidths(zeta,)
         
        nurot = zeta * thetaU['nurot_c'] + (1 - zeta) * thetaU['nurot_e']
        
        modes = self.ones_nu
 
        for i in range(len(nu1s)):
            
            modes += jar.lor(self.f, nu1s[i]           , Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.cos(thetaU['inc'])**2
        
            modes += jar.lor(self.f, nu1s[i] - nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(thetaU['inc'])**2 / 2
        
            modes += jar.lor(self.f, nu1s[i] + nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(thetaU['inc'])**2 / 2

        return modes
     
    def l1_modewidths(self, zeta, fac=1, **kwargs):
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
         
        return  fac * self.obs['mode_width'][0] * jnp.maximum(1e-6, 1. - zeta)   

    def nearest(self, nu, nu_target):
        """
        Utility function: given 1d arrays nu and nu_target, return a 1d array with the 
        same shape as nu, containing the nearest elements of nu_target to each element of nu.
        """
 
        return nu_target[jnp.argmin(jnp.abs(nu[:, None] - nu_target[None, :]), axis=1)]

    def Theta_p(self, nu, Dnu, nu_p):
        """
        p-mode phase function Theta_p. Provide a list of p-mode frequencies nu_p.
        """
        return jnp.pi * jnp.where((nu <= jnp.max(nu_p)) & (nu >= jnp.min(nu_p)),
                                 jnp.interp(nu, nu_p, self.arange_nup),
                                 (nu - self.nearest(nu, nu_p)) / Dnu + jnp.round((self.nearest(nu, nu_p) - nu_p[0]) / Dnu)
                                )

    def Theta_g(self, nu, DPi1, nu_g):
        """
        g-mode phase function Theta_g. Provide a list of g-mode frequencies nu_g.
        """
        return jnp.pi * jnp.where((nu <= jnp.max(nu_g)) & (nu >= jnp.min(nu_g)),
                                  -jnp.interp(1 / nu, jnp.sort(1 / nu_g), self.arange_nug),
                                  (1 / self.nearest(nu, nu_g) - 1 / nu) / DPi1
                                 )

    def zeta(self, nu, q, DPi1, Dnu, nu_p, nu_g):
        """
        zeta, the approximate local mixing fraction.
        """
        Theta_p = self.Theta_p(nu, Dnu, nu_p)
        
        Theta_g = self.Theta_g(nu, DPi1, nu_g)
        
        return 1 / (1 + DPi1 / Dnu * nu**2 / q * jnp.sin(Theta_g)**2 / jnp.cos(Theta_p)**2)

    def zeta_p(self, nu, q, DPi1, Dnu, nu_p):
        """
        zeta as defined using only the p-mode phase function. Agrees with zeta only at the 
        eigenvalues (i.e. roots of the characteristic equation F(nu) = 0).
        """
        Theta = self.Theta_p(nu, Dnu, nu_p)
        
        return 1 / (1 + DPi1 / Dnu * nu**2 / (q * jnp.cos(Theta)**2 + jnp.sin(Theta)**2/q))

    def zeta_g(self, nu, q, DPi1, Dnu, nu_g):
        """
        zeta as defined using only the g-mode phase function. Agrees with zeta only at the
        eigenvalues (i.e. roots of the characteristic equation F(nu) = 0).
        """
        Theta = self.Theta_g(nu, DPi1, nu_g)

        return 1 / (1 + DPi1 / Dnu * nu**2 * (q * jnp.cos(Theta)**2 + jnp.sin(Theta)**2/q))

    def F(self, nu, nu_p, nu_g, Dnu, DPi1, q):
        """
        Characteristic function F such that F(nu) = 0 yields eigenvalues.
        """
        return jnp.tan(self.Theta_p(nu, Dnu, nu_p)) * jnp.tan(self.Theta_g(nu, DPi1, nu_g)) - q

    def Fp(self, nu, nu_p, nu_g, Dnu, DPi1, qp=0):
        """
        First derivative dF/dnu. Required for some numerical methods.
        """
        return (jnp.tan(self.Theta_g(nu, DPi1, nu_g)) / jnp.cos(self.Theta_p(nu, Dnu, nu_p))**2 * jnp.pi / Dnu
              + jnp.tan(self.Theta_p(nu, Dnu, nu_p)) / jnp.cos(self.Theta_g(nu, DPi1, nu_g))**2 * jnp.pi / DPi1 / nu**2
              - qp)

    def Fpp(self, nu, nu_p, nu_g, Dnu, DPi1, qpp=0):
        """
        Second derivative d²F / dnu². Required for some numerical methods.
        """
        return (2 * self.F(nu, nu_p, nu_g, Dnu, DPi1, 0) / jnp.cos(self.Theta_p(nu, Dnu, nu_p))**2 * (jnp.pi / Dnu)**2
              + 2 * self.F(nu, nu_p, nu_g, Dnu, DPi1, 0) / jnp.cos(self.Theta_g(nu, DPi1, nu_g))**2 * (jnp.pi / DPi1 / nu**2)**2
              - 2 * jnp.tan(self.Theta_p(nu, Dnu, nu_p)) / jnp.cos(self.Theta_g(nu, DPi1, nu_g))**2 * jnp.pi / DPi1 / nu**3
              + 2 / jnp.cos(self.Theta_p(nu, Dnu, nu_p))**2 * jnp.pi / Dnu / jnp.cos(self.Theta_g(nu, DPi1, nu_g))**2 * jnp.pi / DPi1 / nu**2
              - qpp)
    
    def halley_iteration(self, x, y, yp, ypp, lmbda=1.):
        """
        Halley's method (2nd order Householder):
        x_{n+1} = x_n = 2 f f' / (2 f'² - f f''),
        again with damping.
        """
        return x - lmbda * 2 * y * yp / (2 * yp * yp - y * ypp)

    def couple(self, nu_p, nu_g, q_p, q_g, DPi1, lmbda=.5):
        """
        Solve the characteristic equation with Halley's method.
        This converges even faster than Newton's method and is capable
        of handling quite numerically difficult scenarios with not
        very much damping.
        """
 
        num_p = jnp.copy(nu_p)
        
        num_g = jnp.copy(nu_g)

        for _ in range(self.maxiter):
            num_p = self.halley_iteration(num_p,
                                          self.F(num_p, nu_p, nu_g, self.obs['dnu'][0], DPi1, q_p),
                                          self.Fp(num_p, nu_p, nu_g, self.obs['dnu'][0], DPi1),
                                          self.Fpp(num_p, nu_p, nu_g, self.obs['dnu'][0], DPi1), lmbda=lmbda)
            num_g = self.halley_iteration(num_g,
                                          self.F(num_g, nu_p, nu_g, self.obs['dnu'][0], DPi1, q_g),
                                          self.Fp(num_g, nu_p, nu_g, self.obs['dnu'][0], DPi1),
                                          self.Fpp(num_g, nu_p, nu_g, self.obs['dnu'][0], DPi1), lmbda=lmbda)

        return num_p, num_g
    