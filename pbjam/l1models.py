import jax.numpy as jnp
import numpy as np
import jax, warnings, dynesty
from pbjam import jar
from pbjam.jar import constants as c
from pbjam.DR import PCA
import pbjam.distributions as dist
from dynesty import utils as dyfunc
jax.config.update('jax_enable_x64', True)

class commonFuncs(jar.generalModelFuncs):
    def __init__(self):
        pass

    def modewidths(self, Gamma, zeta, fac=1):
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
         
        return  fac * Gamma * jnp.maximum(1e-6, 1. - zeta)
    
    def heights(self, nu1s):
        return self.vis['V10'] * jar.envelope(nu1s, self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0])

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
         
        Returns
        -------
        jax device array
            Frequencies of the notionally pure g-modes of degree l.
        """
 
        DPi1 *= 1e-6 # DPi1 in s to Ms.  
 
        P = DPi1 * (n_g + eps_g)
        
        return 1/P
    
    def asymptotic_nu_p(self, d01):
        return self.obs['nu0_p'] + d01

    def select_n_g(self, fac=7):
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
 
        freq_lims = (min(self.obs['nu0_p']) - fac*self.obs['dnu'][0],  max(self.obs['nu0_p']) + fac*self.obs['dnu'][0])
        
        # Start with an exagerated number of g-modes.
        init_n_g = jnp.arange(10000)[::-1] + 1
                
        nu_g = self.asymptotic_nu_g(init_n_g, DPi1, eps_g)
        
        idx_c = (freq_lims[0] < nu_g) & (nu_g < freq_lims[1])
  
        n_g = jnp.arange(init_n_g[idx_c].min(), 
                         init_n_g[idx_c].max(), 
                         dtype=int)[::-1]
              
        if len(n_g) > 100:
            warnings.warn(f'{len(n_g)} g-modes in the coupling matrix.')

        # Force a minimum of 1 g-mode to be included as a test
        if len(n_g) == 0:
            n_g = jnp.array([1])

        return n_g

    def asymmetry(self, nulm, m=1):
        """ Compute the asymmetry of the m=-1 and m=1 modes relative to m=0 for an l=1 multiplet.
        
        The multiplet must be a 1D array of length 3, ordered such that m=[-1,0,1].
        
        Notes
        -----
        Using m=1 or m=-1 as optional arguments should return the same answer.
        
        Parameters
        ----------
        nulm: array-like
            The frequencies of an l=1 multiplet.
        m: int, optional
            Azimuthal order to use for the calculation. m=-1 and m=1 returns the same value.
        
        Returns
        -------
        asym: float
            Asymmetry for a single l=1 triplet.
        """
        
        if m == -1:
            idx = 0
        elif m == 1:
            idx = 2
        else:
            raise ValueError('m must be either -1 or 1')

        asym = (nulm[idx] - nulm[1]) / (m**2 * (nulm[2] - nulm[0])/2) - 1/m
        
        return asym


class Asyl1model(jar.DynestySamplingTools, commonFuncs):
    def __init__(self, f, s, obs, addPriors, Npca, vis={'V10': 1.22}, priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
 
        modelParLabels = ['d01', 'nurot_e',  'inc',]

        self.N_p = len(self.obs['nu0_p'])
 
        self.N_g = 0

        self.N_pg = self.N_p + self.N_g

        self.setLabels(self.addPriors, modelParLabels)

        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.setPriors()

        self.ell = np.ones(self.N_pg)

        self.emm = np.zeros_like(self.ell)
        
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

        self.DR = PCA(_obs, ['d01'], self.priorpath, self.Npca, selectLabels=['numax', 'dnu', 'teff'], dropNansIn='Not all') 
        
        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(self.DR.dataF)
 
        self.priors['d01'] = dist.distribution(self.DR.ppf[0], 
                                               self.DR.pdf[0], 
                                               self.DR.logpdf[0], 
                                               self.DR.cdf[0])

        # Core rotation prior
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
        thetaU: dict
            The unpacked parameters.

        """

        thetaU= {key: theta[i] for i, key in enumerate(self.priors.keys())}
 
        for key in self.logpars:
            thetaU[key] = 10**thetaU[key]
 
        return thetaU
    
    def nu1_frequencies(self, thetaU):
        nu = self.asymptotic_nu_p(thetaU['d01'])
        
        zeta = jnp.zeros_like(nu)
        
        return nu, zeta
    
    def model(self, thetaU,):
        
        nu1s, zeta = self.nu1_frequencies(thetaU)
         
        Hs1 = self.heights(nu1s)
        
        modewidth1s = self.modewidths(self.obs['mode_width'][0], zeta,)  
         
        nurot = self.rotation(**thetaU)
        
        modes = self.ones_nu

        for i in range(len(nu1s)):
 
            modes += jar.lor(self.f, nu1s[i]        , Hs1[i], modewidth1s[i]) * jnp.cos(thetaU['inc'])**2
        
            modes += jar.lor(self.f, nu1s[i] - nurot, Hs1[i], modewidth1s[i]) * jnp.sin(thetaU['inc'])**2 / 2
        
            modes += jar.lor(self.f, nu1s[i] + nurot, Hs1[i], modewidth1s[i]) * jnp.sin(thetaU['inc'])**2 / 2

        return modes

    def rotation(self, nurot_e, **kwargs):
        return nurot_e
   
    def parseSamples(self, smp, Nmax=5000):

        N = min([len(list(smp.values())[0]), Nmax])

        for key in smp.keys():
            smp[key] = smp[key][:N]

        result = {'ell': self.ell,
                  'enn': np.zeros_like(self.ell) - 1,
                  'emm': self.emm,
                  'zeta': np.zeros(self.N_pg),
                  'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                            'height': np.array([]).reshape((2, 0)), 
                            'width' : np.array([]).reshape((2, 0)),
                            'rotAsym' : np.array([]).reshape((2, 0))
                            },
                  'samples': {'freq'  : np.array([]).reshape((N, 0)),
                            'height': np.array([]).reshape((N, 0)), 
                            'width' : np.array([]).reshape((N, 0)),
                            'rotAsym' : np.array([]).reshape((N, 0))
                            },
                }

        result['summary'].update({key: jar.smryStats(smp[key]) for key in smp.keys()})
        result['samples'].update(smp)

        # l=1
        A = np.array([self.nu1_frequencies({key: smp[key][i] for key in ['d01']}) for i in range(N)])
        
        nu1_samps = A[:, 0, :]
        
        jar.modeUpdoot(result, nu1_samps, 'freq', self.N_p)
    
        result['zeta'] = np.append(result['zeta'], np.zeros(result['summary']['freq'].shape[1]))

        # # Heights
        H1_samps = np.array([self.heights(nu1_samps[i, :]) for i in range(N)]) #self.vis['V10'] * np.array([jar.envelope(nu1_samps[i, :], self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0]) for i in range(N)]) 
        jar.modeUpdoot(result, H1_samps, 'height', self.N_p)

        # # Widths
        W1_samps = np.array([self.obs['mode_width'][0]*np.ones(result['summary']['freq'].shape[1]) for i in range(N)]) 
        jar.modeUpdoot(result, W1_samps, 'width', self.N_p)
        result['summary']['width'][1, :] = self.obs['mode_width'][1]*np.ones(result['summary']['freq'].shape[1])

        # Mode asymmetry
        nurot = self.rotation(result['samples']['nurot_e'])
         
        nulm = np.array([result['samples']['freq'].T - nurot, 
                         result['samples']['freq'].T, 
                         result['samples']['freq'].T + nurot])
        
        asym_samps = np.array([[self.asymmetry(nulm[:, i, j]) for i in range(self.N_p)] for j in range(N)])
 
        jar.modeUpdoot(result, asym_samps, 'rotAsym', self.N_p)

        return result
        
class Mixl1model(jar.DynestySamplingTools, commonFuncs):

    def __init__(self, f, s, obs, addPriors, Npca, PCAdims, vis={'V10': 1.22}, priorpath=None):
   
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        modelParLabels = ['p_L', 'p_D', 'DPi1', 'eps_g',
                          'd01', 'dnu', 'numax', 'nurot_c', 
                          'nurot_e', 'inc', 'teff'
                          ]
        
        self.N_p = len(self.obs['nu0_p'])
            
        self.setLabels(self.addPriors, modelParLabels)
 
        self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

        self.badPrior = False

        self.setupDR()
 
        self.setAddObs(keys=['teff', 'dnu', 'numax'])

        if not self.badPrior: 
 
            self.setPriors()
 
            self.n_g = self.select_n_g(fac=5)

            self.N_g = len(self.n_g)

            self.N_pg = self.N_p + self.N_g
            
            self.ell = np.ones(self.N_pg)
            
            self.emm = np.zeros_like(self.ell)

            for i in range(self.N_g + self.N_p):
                self.addlabels.append(f'freqError{i}')

                self.priors[f'freqError{i}'] = dist.normal(loc=0, scale=0.03 * self.obs['dnu'][0])

            self.ndims = len(self.priors)
 
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
 
    def model(self, thetaU,):

        nu1s, zeta = self.nu1_frequencies(thetaU)
         
        Hs1 = self.heights(nu1s) #self.vis['V10'] * jar.envelope(nu1s, self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0])
         
        modewidth1s = self.modewidths(self.obs['mode_width'][0], zeta,)
         
        nurot = self.rotation(zeta, thetaU['nurot_c'], thetaU['nurot_e']) #zeta * thetaU['nurot_c'] + (1 - zeta) * thetaU['nurot_e']
        
        modes = self.ones_nu

        for i in range(len(nu1s)):
             
            nul1 = nu1s[i] + thetaU[f'freqError{i}']  

            modes += jar.lor(self.f, nul1           , Hs1[i], modewidth1s[i]) * jnp.cos(thetaU['inc'])**2
        
            modes += jar.lor(self.f, nul1 - nurot[i], Hs1[i], modewidth1s[i]) * jnp.sin(thetaU['inc'])**2 / 2
        
            modes += jar.lor(self.f, nul1 + nurot[i], Hs1[i], modewidth1s[i]) * jnp.sin(thetaU['inc'])**2 / 2

        return modes
    
    def unpackParams(self, theta): 
        """ Cast the parameters in a dictionary

        Parameters
        ----------
        theta : array
            Array of parameters drawn from the posterior distribution.

        Returns
        -------
        thetaU: dict
            The unpacked parameters.

        """
  
        theta_inv = self.DR.inverse_transform(theta[:self.DR.dimsR])

        thetaU = {key: theta_inv[i] for i, key in enumerate(self.pcalabels)}
   
        thetaU.update({key: theta[self.DR.dimsR:][i] for i, key in enumerate(self.addlabels)})

        for key in self.logpars:
            thetaU[key] = 10**thetaU[key]
 
        return thetaU
 
    def nu1_frequencies(self, thetaU):
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

        nu_p = self.asymptotic_nu_p(thetaU['d01'])
        
        nu_g = self.asymptotic_nu_g(self.n_g, thetaU['DPi1'], thetaU['eps_g'])

        L, D = self.generate_matrices(nu_p, nu_g, thetaU['p_L'], thetaU['p_D'])
         
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

    def rotation(self, zeta, nurot_c, nurot_e):
        return zeta * nurot_c + (1 - zeta) * nurot_e
      
    def parseSamples(self, smp, Nmax=5000):

        N = min([len(list(smp.values())[0]), Nmax])
        
        for key in smp.keys():
            smp[key] = smp[key][:N]
        
        result = {'ell': self.ell,
                  'enn': np.zeros_like(self.ell) - 1,
                  'emm': self.emm,
                  'zeta': np.array([]),
                  'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                              'height': np.array([]).reshape((2, 0)), 
                              'width' : np.array([]).reshape((2, 0)),
                              'rotAsym'  : np.array([]).reshape((2, 0))
                             },
                  'samples': {'freq'  : np.array([]).reshape((N, 0)),
                              'height': np.array([]).reshape((N, 0)), 
                              'width' : np.array([]).reshape((N, 0)),
                              'rotAsym'  : np.array([]).reshape((N, 0))
                             },
                }
        
        result['summary'].update({key: jar.smryStats(smp[key]) for key in smp.keys()})
 
        result['samples'].update(smp)


        A = np.array([self.nu1_frequencies({key: smp[key][i] for key in ['d01', 'DPi1', 'p_L', 'p_D', 'eps_g', 'p_L', 'p_D']}) for i in range(N)])
  
        # Frequencies 
        nu1_samps = A[:, 0, :]
        
        sigma_nul1 = np.array([smp[key] for key in smp.keys() if key.startswith('freqError')]).T
        
        if len(sigma_nul1) == 0:        
            jar.modeUpdoot(result, nu1_samps, 'freq', self.N_pg)
        else:
            jar.modeUpdoot(result, nu1_samps + sigma_nul1, 'freq', self.N_pg)

        zeta_samps = A[:, 1, :]

        result['zeta'] = np.append(result['zeta'], np.median(zeta_samps, axis=0))
        
        # Heights
        H1_samps = self.vis['V10'] * np.array([jar.envelope(nu1_samps[i, :], self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0]) for i in range(N)]) 
        jar.modeUpdoot(result, H1_samps, 'height', self.N_pg)
        
        # Widths
        W1_samps = np.array([self.modewidths(self.obs['mode_width'][0], zeta_samps[i, :], ) for i in range(N)]) 
        jar.modeUpdoot(result, W1_samps, 'width', self.N_pg)

        # Mode asymmetry
        nurot = np.array([self.rotation(zeta_samps[i], 
                                        result['samples']['nurot_c'][i], 
                                        result['samples']['nurot_e'][i]) for i in range(N)]) #zeta_samps * result['samples']['nurot_c'] + (1 - zeta_samps) * result['samples']['nurot_e'] 
        
        nulm = np.array([result['samples']['freq'] - nurot, 
                         result['samples']['freq'], 
                         result['samples']['freq'] + nurot])
        
        asym_samps = np.array([[self.asymmetry(nulm[:, j, i]) for i in range(self.N_p)] for j in range(N)])
 
        jar.modeUpdoot(result, asym_samps, 'rotAsym', self.N_p)

        return result
    
class RGBl1model(jar.DynestySamplingTools, commonFuncs):

    def __init__(self, f, s, obs, addPriors, Npca, rootiter=15, vis={'V10': 1.22}, priorpath=None, modelChoice='simple'):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
  
        modelParLabels = ['d01', 'DPi1', 'teff', 'eps_g', 'q',
                          'nurot_c', 'nurot_e', 'inc', 'dnu',
                          'numax']
                    
        self.setLabels(self.addPriors, modelParLabels)
        
        self.setPriors()
 
        self.setAddObs(keys=['teff', 'dnu', 'numax'])

        self.ndims = len(self.priors)
 
        self.n_g = self.select_n_g()  

        self.N_g = len(self.n_g)

        self.N_p = len(self.obs['nu0_p'])

        self.N_pg = self.N_p + self.N_g

        self.initRotationModel(self.modelChoice)

        self.ones_nu = jnp.ones_like(self.f)

        self.ell = jnp.ones(self.N_pg) # m=-1,0,1
 
    def initRotationModel(self, modelChoice):
        """Setup the chosen frequency model"""
 
        self.arange_nup = jnp.arange(self.N_p, dtype=np.float64)

        self.arange_nug = jnp.arange(self.N_g, dtype=np.float64)
        
        self.emm = jnp.array([jnp.zeros(self.N_pg)])

        if modelChoice == 'simple':
             
            self.nu1_frequencies = self.simpleFrequencies

            
        else:  
            self.nu1_frequencies = self.rotationCoupledFrequencies
  
    def setPriors(self,):
        """ Set the prior distributions.

        The prior distributions are constructed from the projection of the 
        PCA sample onto the reduced dimensional space.

        """

        self.priors = {}

        _obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in ['numax', 'dnu', 'teff']}
         
        for key in ['bp_rp']:
            _obs[key] = self.obs[key]
         
        self.DR = PCA(_obs, ['d01', 'DPi1', 'teff', 'dnu', 'numax'], self.priorpath, self.Npca, selectLabels=['numax', 'dnu', 'teff'], dropNansIn='Not all') 
         
        self.DR.ppf, self.DR.pdf, self.DR.logpdf, self.DR.cdf = dist.getQuantileFuncs(self.DR.dataF)
 
        for i, key in enumerate(['d01', 'DPi1', 'teff', 'dnu', 'numax']):

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
        thetaU: dict
            The unpacked parameters.

        """

        thetaU= {key: theta[i] for i, key in enumerate(self.priors.keys())}
 
        for key in self.logpars:
            thetaU[key] = 10**thetaU[key]
 
        return thetaU

    def simpleFrequencies(self, thetaU):

        # Compute unmixed modes
        nu1_g = self.asymptotic_nu_g(self.n_g, thetaU['DPi1'], thetaU['eps_g'])[::-1]
         
        nu1_p = self.asymptotic_nu_p(thetaU['d01'])

        # Compute coupling
        num_pg = self.couple(nu1_p, nu1_g, thetaU['q'], thetaU['q'], thetaU['DPi1'] * 1e-6)
 
        # Split by linear combination of rotation
        zeta = self.zeta_p(num_pg, thetaU['q'], thetaU['DPi1'] * 1e-6, thetaU['dnu'], nu1_p)

        nurot = zeta * thetaU['nurot_c']/2 + (1 - zeta) * thetaU['nurot_e']

        num = jnp.array([num_pg - nurot, # m=-1
                         num_pg, # m=0
                         num_pg + nurot]) # m=1
        
        zeta = jnp.array([zeta, # m=-1
                          zeta, # m=0
                          zeta]) # m=1
        
        return num, zeta
    
    def rotationCoupledFrequencies(self, thetaU):

        # Compute unmixed modes
        _nu1_g = self.asymptotic_nu_g(self.n_g, thetaU['DPi1'], thetaU['eps_g'])[::-1]

        _nu1_p = self.asymptotic_nu_p(thetaU['d01'])

        # Split by relevant rotation rates
        nu1_g = jnp.array([_nu1_g - thetaU['nurot_c']/2,
                           _nu1_g, 
                           _nu1_g + thetaU['nurot_c']/2])
 
        nu1_p = jnp.array([_nu1_p - thetaU['nurot_e'], 
                           _nu1_p, 
                           _nu1_p + thetaU['nurot_e']])

        # Compute coupling. 
        num = jnp.array([self.couple(nu1_p[i, :], nu1_g[i, :], thetaU['q'], thetaU['q'], thetaU['DPi1'] * 1e-6) for i in range(3)])
        
        zeta = jnp.array([self.zeta_p(num[i, :], thetaU['q'], thetaU['DPi1'] * 1e-6, thetaU['dnu'], nu1_p[i, :]) for i in range(3)])

        return num, zeta
            
    def model(self, thetaU):
        """_summary_

        Follows Li et al 2024 Approach Two

        Parameters
        ----------
        thetaU : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        nus, zeta = self.nu1_frequencies(thetaU)

        Hs = self.heights(nus[1, :]) #self.vis['V10']*jar.envelope(nu1s, self.obs['env_height'][0], self.obs['numax'][0], self.obs['env_width'][0])
        
        Ws = self.modewidths(self.obs['mode_width'][0], zeta[1, :],)
         
        modes = self.ones_nu
 
        for i in range(self.N_pg):

            modes += jar.lor(self.f, nus[0, i], Hs[i], Ws[i]) * jnp.cos(thetaU['inc'])**2
        
            modes += jar.lor(self.f, nus[1, i], Hs[i], Ws[i]) * jnp.sin(thetaU['inc'])**2 / 2
        
            modes += jar.lor(self.f, nus[2, i], Hs[i], Ws[i]) * jnp.sin(thetaU['inc'])**2 / 2

        # for i in range(self.N_pg):
        #     for j, emm in range([-1, 0, 1])
        #         modes += jar.lor(self.f, nus[j, i], Hs[i], Ws[i]) * jar.visell1(abs(emm), thetaU['inc'])
         
        return modes
 
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

        for _ in range(self.rootiter):
            num_p = self.halley_iteration(num_p,
                                          self.F(num_p, nu_p, nu_g, self.obs['dnu'][0], DPi1, q_p),
                                          self.Fp(num_p, nu_p, nu_g, self.obs['dnu'][0], DPi1),
                                          self.Fpp(num_p, nu_p, nu_g, self.obs['dnu'][0], DPi1), lmbda=lmbda)
            num_g = self.halley_iteration(num_g,
                                          self.F(num_g, nu_p, nu_g, self.obs['dnu'][0], DPi1, q_g),
                                          self.Fp(num_g, nu_p, nu_g, self.obs['dnu'][0], DPi1),
                                          self.Fpp(num_g, nu_p, nu_g, self.obs['dnu'][0], DPi1), lmbda=lmbda)

        return jnp.append(num_p, num_g)
 
    def parseSamples(self, smp, Nmax=5000):

        N = min([len(list(smp.values())[0]), Nmax])
        
        for key in smp.keys():
            smp[key] = smp[key][:N]
        
        result = {'ell': self.ell,
                  'enn': jnp.zeros_like(self.ell) - 1,
                  'emm': self.emm,
                  'zeta': np.array([]),
                  'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                              'height': np.array([]).reshape((2, 0)), 
                              'width' : np.array([]).reshape((2, 0)),
                              'rotAsym': np.array([]).reshape((2, 0))
                             },
                  'samples': {'freq'  : np.array([]).reshape((N, 0)),
                              'height': np.array([]).reshape((N, 0)), 
                              'width' : np.array([]).reshape((N, 0)),
                              'rotAsym': np.array([]).reshape((N, 0)),
                             },
                }
        
        result['summary'].update({key: jar.smryStats(smp[key]) for key in smp.keys()})
 
        result['samples'].update(smp)
 
        nu_samples = np.zeros((N, self.N_pg))

        zeta_samples = np.zeros((N, self.N_pg))
        
        height_samples = np.zeros((N, self.N_pg))
        
        width_samples = np.zeros((N, self.N_pg))

        asym_samples = np.zeros((N, self.N_pg))
        
        jfreqs = jax.jit(self.nu1_frequencies)
 
        for i in range(N):
             
            thetaU = {key: smp[key][i] for key in smp.keys()}  

            nus, zetas = jfreqs(thetaU) 
        
            nu_samples[i, :] = nus[1, :]
         
            zeta_samples[i, :] = zetas[1, :]
 
            height_samples[i, :] = self.heights(nus[1, :]) 
            
            width_samples[i, :] = self.modewidths(self.obs['mode_width'][0], zetas[1, :],)

            asym_samples[i, :] = self.asymmetry(nus)
        
        jar.modeUpdoot(result, nu_samples, 'freq', self.N_pg)

        jar.modeUpdoot(result, height_samples, 'height', self.N_pg)

        jar.modeUpdoot(result, width_samples, 'width', self.N_pg)

        result['zeta'] = np.append(result['zeta'], np.median(zeta_samples, axis=0))

        jar.modeUpdoot(result, asym_samples, 'rotAsym', self.N_pg)
 
        return result

 