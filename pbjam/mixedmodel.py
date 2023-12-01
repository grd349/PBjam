import jax.numpy as jnp
import numpy as np
import jax, warnings
from pbjam import jar
from functools import partial
from pbjam.jar import constants as c
from pbjam.DR import PCA
import pbjam.distributions as dist
import jax.scipy.linalg as jsl

jax.config.update('jax_enable_x64', True)
 
class MixFreqModel(jar.DynestySamplingTools):

    def __init__(self, f, s, obs, addPriors, N_p, Npca, PCAdims,
                 vis={'V10': 1.22}, priorpath=None):
   
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
 
        self.modelVars = {}

        self.modelVars.update(self.variables['l1'])
        
        self.modelVars.update(self.variables['common'])

        self.set_labels(self.addPriors) 

        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.setupDR()
  
        self.setPriors()

        self.ndims = len(self.priors)

        n_g_ppf, _, _, _ = self._makeTmpSample(['DPi1', 'eps_g'])
 
        self.n_g = self.select_n_g(n_g_ppf)

        self.N_g = len(self.n_g)

        self.trimVariables()

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
         
        self.DR = PCA(_obs, self.pcalabels, self.priorpath, self.Npca, selectlabels=['numax', 'dnu', 'teff', 'bp_rp']) 

        self.DR.fit_weightedPCA(self.PCAdims)

        _Y = self.DR.transform(self.DR.data_F)

        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(_Y)
        
        if len(self.pcalabels) > 0:
            self.latentLabels = ['theta_%i' % (i) for i in range(self.PCAdims)]
        else:
            self.latentLabels = []
            self.DR.inverse_transform = lambda x: []
            self.DR.dims_R = 0

    

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

    def trimVariables(self):
 
        for i in range(self.mixed_to_fit, 200, 1):
            del self.addlabels[self.addlabels.index(f'freqError{i}')]
            del self.labels[self.labels.index(f'freqError{i}')]
            del self.priors[f'freqError{i}']
            
        self.ndims = len(self.priors)    

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
                if key == 'freqError':
                    for i in range(200):
                        self.addlabels.append(f'freqError{i}')
                else:
                    self.addlabels.append(key)

        self.labels = self.pcalabels + self.addlabels

        # Parameters that are in log10
        self.logpars = []
        for key in self.modelVars.keys():
            if self.modelVars[key]['log10']:
                self.logpars.append(key)

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

        AddKeys = [k for k in self.modelVars if k in self.addPriors.keys()]

        self.priors.update({key : self.addPriors[key] for key in AddKeys})
 
        for i in range(200):
            self.priors[f'freqError{i}'] = dist.normal(loc=0, scale=1/20 * self.obs['dnu'][0])
  
        # Core rotation prior
        self.priors['nurot_c'] = dist.uniform(loc=-2., scale=2.)

        self.priors['nurot_e'] = dist.uniform(loc=-2., scale=2.)

        # The inclination prior is a sine truncated between 0, and pi/2.
        self.priors['inc'] = dist.truncsine()            
 
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
           
        # Constraint from the periodogram 
        mod = self.model(theta_u, nu)
        
        lnlike = self.chi_sqr(mod)
         
        return lnlike

    @partial(jax.jit, static_argnums=(0,))
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
         
        return  fac * self.obs['mode_width'][0] * jnp.maximum(0, 1. - zeta) 
    
    @partial(jax.jit, static_argnums=(0,))
    def envelope(self, nu,):
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
 
        return jar.gaussian(nu, 2*self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0])
    
    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta_u, nu,):

        theta_u['p_L'] = (theta_u['u1'] + theta_u['u2'])/jnp.sqrt(2)

        theta_u['p_D'] = (theta_u['u1'] - theta_u['u2'])/jnp.sqrt(2)

        nu1s, zeta = self.mixed_nu1(self.obs['nu0_p'], **theta_u)
         
        Hs1 = self.envelope(nu1s, )
        
        modewidth1s = self.l1_modewidths(zeta,)
         
        nurot = zeta * theta_u['nurot_c'] + (1 - zeta) * theta_u['nurot_e']
        
        modes = jnp.ones_like(nu)

        for i in range(len(nu1s)):
            
            nul1 = nu1s[i] + theta_u[f'freqError{i}']

            modes += jar.lor(nu, nul1                     , Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.cos(theta_u['inc'])**2
        
            modes += jar.lor(nu, nul1 - zeta[i] * nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(theta_u['inc'])**2 / 2
        
            modes += jar.lor(nu, nul1 + zeta[i] * nurot[i], Hs1[i] * self.vis['V10'], modewidth1s[i]) * jnp.sin(theta_u['inc'])**2 / 2

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
         
        # theta_inv = self.DR.inverse_transform(theta[:self.DR.dims_R])
         
        # theta_u = {key: theta_inv[i] for i, key in enumerate(self.pcalabels)}
         
        theta_u = {key: theta[i] for i, key in enumerate(self.addlabels)}
 
        for key in self.logpars:
            theta_u[key] = 10**theta_u[key]
 
        return theta_u
 
    def select_n_g(self, n_g_ppf, fac=1):
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
  
        n = self.N_p // 2 + 1
 
        width = max((n + 1) * self.obs['dnu'][0], fac * jar.scalingRelations.envWidth(self.obs['numax'][0]))
        #width = max((n + 1) * self.obs['dnu'][0], fac * jar.scalingRelations.envWidth(self.obs['numax'][0]))

        # freq_lims = (self.obs['numax'][0] - width, 
        #              self.obs['numax'][0] + width)

        freq_lims = {'coupling': (min(self.obs['nu0_p']) - 2*self.obs['dnu'][0], 
                                  max(self.obs['nu0_p']) + 2*self.obs['dnu'][0]),
                     'fit':      (min(self.obs['nu0_p']) - 0.5*self.obs['dnu'][0], 
                                  max(self.obs['nu0_p']) + 0.5*self.obs['dnu'][0])
                    }
         
        # Start with an exagerated number of g-modes.
        init_n_g = jnp.arange(10000)[::-1] + 1

        min_n_g_c = init_n_g.max()
        min_n_g_f = init_n_g.max()

        max_n_g_c = init_n_g.min()
        max_n_g_f = init_n_g.min()


        def update_limit(limit, idx, init_n_g, crit):
            if crit == 'min':
                t = jnp.where(idx, init_n_g, 0 * init_n_g + jnp.inf).min()
                 
                limit = jnp.minimum(limit, t)

            elif crit == 'max':
                t = jnp.where(idx, init_n_g, 0 * init_n_g - 1).max()
                 
                limit = jnp.maximum(limit, t)

            return limit
                
        # Loop over combinations of DPi1 and eps_g as drawn from the respective PDFs.       
        for DPi1 in jnp.linspace(n_g_ppf[0](0.05), n_g_ppf[0](0.95), 3):
            
            for eps_g in jnp.linspace(n_g_ppf[1](0.05), n_g_ppf[1](0.95), 3):
                
                nu_g = self.asymptotic_nu_g(init_n_g, DPi1, eps_g)
                
                idx_c = (freq_lims['coupling'][0] < nu_g) & (nu_g < freq_lims['coupling'][1])
                 
                #t = jnp.where(idx, init_n_g, 0 * init_n_g + jnp.inf).min()
                 
                min_n_g_c = update_limit(min_n_g_c, idx_c, init_n_g, 'min') #jnp.minimum(min_n_g, t)
                
                #t = jnp.where(idx, init_n_g, 0 * init_n_g - 1).max()
                max_n_g_c = update_limit(max_n_g_c, idx_c, init_n_g, 'max')  #jnp.maximum(max_n_g, t)

                idx_f = (freq_lims['fit'][0] < nu_g) & (nu_g < freq_lims['fit'][1])
                 
                #t = jnp.where(idx, init_n_g, 0 * init_n_g + jnp.inf).min()
                 
                min_n_g_f = update_limit(min_n_g_f, idx_f, init_n_g, 'min') #jnp.minimum(min_n_g, t)
                
                #t = jnp.where(idx, init_n_g, 0 * init_n_g - 1).max()
                max_n_g_f = update_limit(max_n_g_f, idx_f, init_n_g, 'max')  #jnp.maximum(max_n_g, t)

        
        n_g = jnp.arange(min_n_g_c, max_n_g_c, dtype=int)[::-1]
        n_g_to_fit = jnp.arange(min_n_g_f, max_n_g_f, dtype=int)[::-1]
        
        print(min_n_g_c, len(n_g))
        print(min_n_g_f, len(n_g_to_fit))
        
        self.mixed_to_fit = self.N_p + len(n_g_to_fit)

        if len(n_g) > 100:
            warnings.warn(f'{len(n_g)} g-modes in the p-mode envelope area.')

        return n_g
 
    @partial(jax.jit, static_argnums=(0,))
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
         
        #P0 = 0 # DPi1*1e-6 # 1 / (jnp.sqrt(max_N2) / c.nu_to_omega)

        #DPi0 = DPi1 / jnp.sqrt(2)  
        DPi1 *= 1e-6 # DPi1 in s to Ms.  

        #P_max = 1/self.obs['numax'][0]

        #n_gmax = (P_max - P0) / DPi1 - eps_g

        #P = P0 + DPi1 * (n_g + eps_g)  + alpha_g * (n_g - n_gmax)**2)
        P = DPi1 * (n_g + eps_g)
        
        return 1/P

    @partial(jax.jit, static_argnums=(0,))
    def mixed_nu1(self, nu0_p, d01, DPi1, p_L, p_D, eps_g, **kwargs):
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

        nu1_p = nu0_p + d01  
    
        nu_g = self.asymptotic_nu_g(self.n_g, DPi1, eps_g)

        L, D = self.generate_matrices(nu1_p, nu_g, p_L, p_D)
         
        nu, zeta = self.new_modes(L, D)

        idx = jnp.argpartition(abs(nu-self.obs['numax'][0]), self.mixed_to_fit-1)
 
        return nu[idx[:self.mixed_to_fit]], zeta[idx[:self.mixed_to_fit]] 

    @partial(jax.jit, static_argnums=(0,))
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
 
    @partial(jax.jit, static_argnums=(0,))
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
    
    @partial(jax.jit, static_argnums=(0,))
    def generalized_eig(self, A, B):
 
        B_inv = jnp.linalg.inv(B)
            
        U, Vprime = jnp.linalg.eig(A @ B_inv)

        V = B_inv @ Vprime
    
        Vnorm = V / jnp.linalg.norm(V, axis=0)
        
        return U.real, Vnorm.real
    
    @partial(jax.jit, static_argnums=(0,))
    def generalized_eigh(self, A, B):

        L = jnp.linalg.cholesky(B)

        L_inv = jnp.linalg.inv(L)

        C = L_inv @ A @ L_inv.T

        eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)

        eigenvectors_transformed = L_inv.T @ eigenvectors_transformed

        eigenvectors = eigenvectors_transformed / jnp.linalg.norm(eigenvectors_transformed, axis=0)

        return eigenvalues, eigenvectors
 
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
         
        # l=1
        smp['p_L'] = (smp['u1'] + smp['u2'])/jnp.sqrt(2)

        smp['p_D'] = (smp['u1'] - smp['u2'])/jnp.sqrt(2)

         
        A = np.array([self.mixed_nu1(self.obs['nu0_p'],                                       
                                     smp['d01'][i], 
                                     smp['DPi1'][i], 
                                     smp['p_L'][i],  
                                     smp['p_D'][i], 
                                     smp['eps_g'][i], 
                                     ) for i in range(N)])
                                    #self.obs['n_p'],
                                    #smp['alpha_g'][i],

        N_pg = self.mixed_to_fit #self.N_p + self.N_g
        
        result['ell'] = np.append(result['ell'], np.zeros(N_pg) + 1)
        result['enn'] = np.append(result['enn'], np.zeros(N_pg) - 1)

        # # Frequencies 
        nu1_samps = A[:, 0, :]

        sigma_nul1 = np.array([smp[key] for key in smp.keys() if key.startswith('freqError')]).T
 
        jar.modeUpdoot(result, nu1_samps + sigma_nul1, 'freq', N_pg)

        zeta_samps = A[:, 1, :]

        result['zeta'] = np.append(result['zeta'], np.median(zeta_samps, axis=0))
        
        # # Heights
        H1_samps = self.vis['V10'] * np.array([self.envelope(nu1_samps[i, :]) for i in range(N)]) 
        jar.modeUpdoot(result, H1_samps, 'height', N_pg)
        
        # # Widths
        W1_samps = np.array([self.l1_modewidths(zeta_samps[i, :], 
                                                self.obs['mode_width'][0]) for i in range(N)]) 
        jar.modeUpdoot(result, W1_samps, 'width', N_pg)

        return result

    variables = {'l1': {'u1'        : {'info': 'Sum of p_L0 and p_D0 over sqrt(2)'        , 'log10': False, 'pca': True, 'unit': 'Angular frequency 1/muHz^2'},
                       'u2'        : {'info': 'Difference of p_L0 and p_D0 over sqrt(2)' , 'log10': False, 'pca': True, 'unit': 'Angular frequency 1/muHz^2'},
                       'DPi1'      : {'info': 'period spacing of the l=0 modes'          , 'log10': False, 'pca': True, 'unit': 's'}, 
                       'eps_g'     : {'info': 'phase offset of the g-modes'              , 'log10': False, 'pca': True, 'unit': 'None'}, 
                       #'alpha_g'   : {'info': 'curvature of the g-modes'                 , 'log10': True , 'pca': True, 'unit': 'None'}, 
                       'd01'       : {'info': 'l=0,1 mean frequency difference'          , 'log10': False,  'pca': True, 'unit': 'muHz'},
                       'freqError' : {'info': 'Frequency error'                          , 'log10': False, 'pca': False, 'unit': 'muHz'},
                       },
                'common': {'nurot_c'   : {'info': 'core rotation rate'                       , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
                           'nurot_e'   : {'info': 'envelope rotation rate'                   , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
                           'inc'       : {'info': 'stellar inclination axis'                 , 'log10': False, 'pca': False, 'unit': 'rad'},}
                }


# @partial(jax.jit, static_argnums=(0,))
    # def _wrap_polyval2d(self, x, y, p):
    #     """Evaluate 2D polynomial

    #     Evaluates 2D polynomial for the the top right or bottom left blocks of 
    #     the L and D matrices. 

    #     Parameters
    #     ----------
    #     x : jax device array
    #         Array of radial orders for the pi modes.
    #     y : jax device array
    #         Array of radial orders for the gamma modes.
    #     p : jax device array
    #         Polynomial coeffecients.

    #     Returns
    #     -------
    #     jax device array
    #         Block of shape (len(x), len(y) to be inserted in L or D coupling 
    #         strength matrices.
        
    #     Notes
    #     -----
    #     This is legacy code which is not currently in use. May be used in the 
    #     future if it's necessary to compute more complicated 2D poly's.

    #     """
      
    #     xx = (x + 0 * y)

    #     yy = (y + 0 * x)

    #     shape = xx.shape

    #     P = self.poly_params_2d(p)

    #     block = self.polyval2d(xx.flatten(), yy.flatten(), P)

    #     return block.reshape(shape)

    # @partial(jax.jit, static_argnums=(0,))
    # def poly_params_2d(self, p):
    #     """Reshape polynomial coefficients

    #     Turn a parameter vector into an upper triangular coefficient matrix.

    #     For a list of polynomial coefficients [1,2,3] the result is of the
    #     form 
    #     [1 3 0]
    #     [2 0 0]
    #     [0 0 0]

    #     Parameters
    #     ----------
    #     p : jax device array
    #         Array of polynomial coefficients

    #     Returns
    #     -------
    #     jax device array
    #         Square matrix of reshaped polynomial coefficients.
        
    #     Notes
    #     -----
    #     This is legacy code which is not currently in use. May be used in the 
    #     future if it's necessary to compute more complicated 2D poly's.

    #     """
 
    #     # Check that len(p) a triangular number
    #     if len(p) == 1:
    #         n = 1
    #     elif len(p) == 3:
    #         n = 2
    #     elif len(p) == 6:
    #         n = 3
    #     else:
    #         raise ValueError('The length of p must be 1, 3, or 6.')

    #     P = jnp.zeros((n, n))

    #     # TODO: I think this part is slow in jax, can be speeded up?
    #     for i in range(n):
    #         # ith triangular number as offset
    #         n_i = i * (i + 1) // 2

    #         for j in range(i + 1):

    #             P = P.at[i - j, j].set(p[n_i + j])

    #     return P

    # @partial(jax.jit, static_argnums=(0,))
    # def polyval2d(self, x, y, P, increasing=True):
    #     """Evaluate a 2D polynomial

    #     Jaxxable replacement for Numpy's polyval2d. Doesn't seem to exist in 
    #     jax.numpy yet.

    #     Parameters
    #     ----------
    #     x : jax device array
    #         x-coordinates to evaluate the polynomial at. In this case n_g or 
    #         n_p
    #     y : jax device array
    #         y-coordinates to evaluate the polynomial at. In this case n_g or 
    #         n_p
    #     P : jax device array
    #         Upper triangular matrix of polynomial coefficients.
    #     increasing : bool, optional
    #         Ordering of the powers of the columns of the Vandermonde matrices. 
    #         If True, the powers increase from left to right, if False 
    #         (the default) they are reversed.

    #     Returns
    #     -------
    #     jax device array
    #         Matrix of the polymial values.

    #     Notes
    #     -----
    #     This is legacy code which is not currently in use. May be used in the 
    #     future if it's necessary to compute more complicated 2D poly's.
    #     """

    #     X = jnp.vander(x, P.shape[0], increasing=increasing)

    #     Y = jnp.vander(y, P.shape[1], increasing=increasing)

    #     return jnp.sum(X[:, :, None] * Y[:, None, :]*P, axis=(2, 1))

# def generalized_eig(A, B=None):
 
    #     B_inv = jnp.linalg.inv(B)

    #     eigenvalues, eigenvectors = jnp.linalg.eig(B_inv @ A)

    
    #     eigenvectors = B_inv @ eigenvectors
    
    #     #for i in range(eigenvectors.shape[1]):
    #     #    eigenvectors[:, i] /= jnp.linalg.norm(eigenvectors[:, i])
    
    #     return eigenvalues, eigenvectors
    #     
# @partial(jax.jit, static_argnums=(0,))
    # def standardize_angle(self, w, b):
    #     """ Helper for eigh solver

    #     Parameters
    #     ----------
    #     w : jax device array
    #         Matrix
    #     b : jax device array
    #         Matrix

    #     Returns
    #     -------
    #     w : jax device array
    #         Modified w.
    #     """

    #     return w * jnp.sign(w[0, :])

    # @partial(jax.jit, static_argnums=(0,))
    # def eigh(self, a, b):
        # """ Jaxxable replacement for numpy.eigh.

        # From https://jackd.github.io/posts/generalized-eig-jvp/

        # Compute the solution to the symmetrized generalized eigenvalue problem.

        # a_s @ w = b_s @ w @ np.diag(v)

        # where a_s = (a + a.H) / 2, b_s = (b + b.H) / 2 are the symmetrized 
        # versions of the inputs and H is the Hermitian (conjugate transpose) 
        # operator.

        # For self-adjoint inputs the solution should be consistent with 
        # `scipy.linalg.eigh` i.e.

        # v, w = eigh(a, b)
        # v_sp, w_sp = scipy.linalg.eigh(a, b)

        # np.testing.assert_allclose(v, v_sp)
        # np.testing.assert_allclose(w, standardize_angle(w_sp))

        # Note this currently uses `jax.linalg.eig(jax.linalg.solve(b, a))`, which
        # will be slow because there is no GPU implementation of `eig` and it's 
        # just a generally inefficient way of doing it. Future implementations 
        # should wrap cuda primitives. This implementation is provided primarily 
        # as a means to test `eigh_jvp_rule`.

        # Parameters
        # ----------
        # a : jax device array
        #     [n, n] float self-adjoint matrix (i.e. conj(transpose(a)) == a)
        # b : jax device array
        #     [n, n] float self-adjoint matrix (i.e. conj(transpose(b)) == b)

        # Returns
        # -------
        # v : jax device array
        #     Eigenvalues of the generalized problem in ascending order.
        # w : jax device array
        #     Eigenvectors of the generalized problem, normalized such that
        #     w.H @ b @ w = I.
        # """
        
        # b_inv_a = jax.scipy.linalg.cho_solve(jax.scipy.linalg.cho_factor(b), a)

        # v, w = jax.jit(jax.numpy.linalg.eig, backend="cpu")(b_inv_a)

        # v = v.real
 
        # w = w.real

        # # reorder as ascending in w
        # order = jnp.argsort(v)

        # v = v.take(order, axis=0)

        # w = w.take(order, axis=1)

        # # renormalize so v.H @ b @ H == 1
        # norm2 = jax.vmap(lambda wi: (wi.conj() @ b @ wi).real, in_axes=1)(w)

        # w = w / jnp.sqrt(norm2)

        # w = self.standardize_angle(w, b)

        # return v, w