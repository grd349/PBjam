from functools import partial
import jax, warnings, os
import jax.numpy as jnp
import numpy as np
from pbjam import jar
import pbjam.distributions as dist
from pbjam.mixedmodel import MixFreqModel
from pbjam.pairmodel import AsyFreqModel
from pbjam.plotting import plotting
import pandas as pd

class modeIDsampler(plotting, ):

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freq_limits=[1, 5000], 
                 vis={'V20': 0.71, 'V10': 1.22}, Npca=50, PCAdims=8, 
                 priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        self.sel = self.setFreqRange()

    def setFreqRange(self, ):
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

    def __call__(self, sampler_kwargs={}, logl_kwargs={}):
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
        logl_kwargs['nu'] = self.f[self.sel]    
        
        self.AsyFreqModel = AsyFreqModel(self.f[self.sel], 
                                         self.s[self.sel], 
                                         self.obs, 
                                         self.addPriors, 
                                         self.N_p, 
                                         self.Npca, 
                                         self.PCAdims,
                                         priorpath=self.priorpath)
        
         
        self.l20samples = self.AsyFreqModel.runDynesty(logl_kwargs=logl_kwargs, 
                                                       sampler_kwargs=sampler_kwargs)
        
        l20samples_u = self.AsyFreqModel.unpackSamples(self.l20samples)
        
        asymptotic_samps = np.array([self.AsyFreqModel.asymptotic_nu_p(l20samples_u['numax'][i], 
                                                                       l20samples_u['dnu'][i], 
                                                                       l20samples_u['eps_p'][i], 
                                                                       l20samples_u['alpha_p'][i]) for i in range(50)])
        self.summary = {}

        self.summary['n_p'] = np.median(asymptotic_samps[:, 1, :], axis=0).astype(int)

        self.summary['nu0_p'] = np.median(asymptotic_samps[:, 0, :], axis=0)

        for key in ['numax', 'dnu', 'env_height', 'env_width', 'mode_width', 'teff', 'bp_rp']:
            self.summary[key] = jar.smryStats(l20samples_u[key])

        l20model = self.AsyFreqModel.getMedianModel(self.f, l20samples_u)

        self.l20residual = self.s/l20model

        self.MixFreqModel = MixFreqModel(self.f[self.sel], 
                                         self.l20residual[self.sel], 
                                         self.summary, 
                                         self.addPriors,
                                         self.N_p, 
                                         self.Npca, 
                                         self.PCAdims,
                                         priorpath=self.priorpath)

         
        self.l1samples = self.MixFreqModel.runDynesty(logl_kwargs=logl_kwargs, 
                                                      sampler_kwargs=sampler_kwargs)

        l1samples_u = self.MixFreqModel.unpackSamples(self.l1samples)

        # Get results from both
        self.result = self.parseSamples(l20samples_u, l1samples_u)

        
    
    # def defineModel(self, spectrum=None, modelType='l20'):  
         
    #     if spectrum is None:
    #         spectrum = self.s

    #     self._spec = spectrum

    #     self.modelVars = {}

    #     self.modelVars.update(self.variables[modelType])
        
    #     if modelType == 'l20':
    #         self.modelVars.update(self.variables['background'])
        
    #     self.modelVars.update(self.variables['common'])

    #     self.set_labels(self.addPriors)

    #     self.log_obs = {x: jar.to_log10(*self.obs[x]) for x in self.obs.keys() if x in self.logpars}

    #     self.setupDR()
  
    #     self.setPriors(modelType)
 
    #     self.background = bkgModel(self.Nyquist)

    #     self.AsyFreqModel = AsyFreqModel(self.N_p)

    #     self.ndims = len(self.latentLabels + self.addlabels)

    #     if modelType == 'l1':
    #         n_g_ppf, _, _, _ = self._makeTmpSample(['DPi1', 'eps_g'])
            
    #         self.MixFreqModel = MixFreqModel(self.N_p, self.obs, n_g_ppf)

    #         self.N_g = self.MixFreqModel.N_g
            
    #         self.trimVariables()

    #     self.sel = self.setFreqRange()

    #     self.setAddObs(modelType)

    #     self.model = self._pickmodelFunc(modelType)

      



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
    #             'l1': {'u1'        : {'info': 'Sum of p_L0 and p_D0 over sqrt(2)'        , 'log10': False, 'pca': True, 'unit': 'Angular frequency 1/muHz^2'},
    #                    'u2'        : {'info': 'Difference of p_L0 and p_D0 over sqrt(2)' , 'log10': False, 'pca': True, 'unit': 'Angular frequency 1/muHz^2'},
    #                    'DPi1'      : {'info': 'period spacing of the l=0 modes'          , 'log10': False, 'pca': True, 'unit': 's'}, 
    #                    'eps_g'     : {'info': 'phase offset of the g-modes'              , 'log10': False, 'pca': True, 'unit': 'None'}, 
    #                    'alpha_g'   : {'info': 'curvature of the g-modes'                 , 'log10': True , 'pca': True, 'unit': 'None'}, 
    #                    'd01'       : {'info': 'l=0,1 mean frequency difference'          , 'log10': False,  'pca': True, 'unit': 'muHz'},
    #                    'freqError' : {'info': 'Frequency error'                          , 'log10': False, 'pca': False, 'unit': 'muHz'},
    #                    },
    #             'common': {'nurot_c'   : {'info': 'core rotation rate'                       , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
    #                        'nurot_e'   : {'info': 'envelope rotation rate'                   , 'log10': True , 'pca': False, 'unit': 'muHz'}, 
    #                        'inc'       : {'info': 'stellar inclination axis'                 , 'log10': False, 'pca': False, 'unit': 'rad'},}
    #             }

    

    def _modeUpdoot(self, result, sample, key, Nmodes):
        
        result['summary'][key] = np.hstack((result['summary'][key], 
                                           np.array([jar.smryStats(sample[:, j]) for j in range(Nmodes)]).T))
        result['samples'][key] = np.hstack((result['samples'][key], 
                                            sample))

    def parseSamples(self, l20samples_u, l1samples_u, Nmax=10000):

        N = min([len(list(l20samples_u.values())[0]), 
                 len(list(l1samples_u.values())[0]),
                 Nmax])
        
        smp = l20samples_u | l1samples_u

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
        H0_samps = np.array([self.AsyFreqModel.envelope(nu0_samps[i, :], smp['env_height'][i], smp['numax'][i], smp['env_width'][i]) for i in range(N)])
        self._modeUpdoot(result, H0_samps, 'height', self.N_p)

        # # Widths
        W0_samps = np.tile(smp['mode_width'], self.N_p).reshape((self.N_p, N)).T
        self._modeUpdoot(result, W0_samps, 'width', self.N_p)
        
        # l=2
        result['ell'] = np.append(result['ell'], np.zeros(self.N_p) + 2)
        result['enn'] = np.append(result['enn'], n_p-1)
        result['zeta'] = np.append(result['zeta'], np.zeros(self.N_p))

        # # Frequencies
        nu2_samps = np.array([nu0_samps[i, :] - smp['d02'][i] for i in range(N)])
        self._modeUpdoot(result, nu2_samps, 'freq', self.N_p)

        # # Heights
        H2_samps = self.vis['V20'] * np.array([self.AsyFreqModel.envelope(nu2_samps[i, :],  
                                                            smp['env_height'][i], 
                                                            smp['numax'][i], 
                                                            smp['env_width'][i]) for i in range(N)])
        self._modeUpdoot(result, H2_samps, 'height', self.N_p)
        
        # # Widths
        W2_samps = np.tile(smp['mode_width'], np.shape(nu2_samps)[1]).reshape((nu2_samps.shape[1], nu2_samps.shape[0])).T
        self._modeUpdoot(result, W2_samps, 'width', self.N_p)

        # Background
        #result['background'] = self.meanBkg(self.f, smp)  


         
        # l=1
        smp['p_L'] = (smp['u1'] + smp['u2'])/jnp.sqrt(2)

        smp['p_D'] = (smp['u1'] - smp['u2'])/jnp.sqrt(2)

         
        A = np.array([self.MixFreqModel.mixed_nu1(nu0_samps[i, :], 
                                                  n_p, 
                                                  smp['d01'][i], 
                                                  smp['DPi1'][i], 
                                                  smp['p_L'][i],  
                                                  smp['p_D'][i], 
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
        H1_samps = self.vis['V10'] * np.array([self.MixFreqModel.envelope(nu1_samps[i, :]) for i in range(N)]) 
        self._modeUpdoot(result, H1_samps, 'height', N_pg)
        
        # # Widths
        W1_samps = np.array([self.MixFreqModel.l1_modewidths(zeta_samps[i, :], 
                                                smp['mode_width'][i]) for i in range(N)]) 
        self._modeUpdoot(result, W1_samps, 'width', N_pg)

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
 
    


        

    # def getInitLive(self, nlive):
    #     """
    #     Generate initial live points for a Bayesian inference problem.

    #     Parameters
    #     ----------
    #     nlive : int
    #         The number of live points to generate.

    #     Returns
    #     -------
    #     list : list
    #         A list containing three arrays: [u, v, L].
    #             - u : ndarray
    #                 Initial live points in the unit hypercube [0, 1].
    #                 Shape: (nlive, ndims).
    #             - v : ndarray
    #                 Transformed live points obtained by applying the ptform method to each point in u.
    #                 Shape: (nlive, ndims).
    #             - L : ndarray
    #                 Log-likelihood values calculated for each point in v.
    #                 Shape: (nlive,).

    #     Notes
    #     -----
    #     This method generates initial live points for a Bayesian inference problem.
    #     It follows the following steps:
    #     1. Generate a 2D array u of shape (4*nlive, ndims) with values drawn from a uniform distribution in the range [0, 1].
    #     2. Apply the ptform method to each row of u to obtain a new 2D array v of the same shape.
    #     3. Calculate the log-likelihood values L for each point in v using the lnlikelihood method.
    #     4. Filter out invalid values of L (NaN or infinite) using a boolean mask.
    #     5. Select the first nlive rows from the filtered arrays to obtain the initial live points u, transformed points v, and log-likelihood values L.
    #     6. Return the list [u, v, L].
    #     """
        
    #     u = np.random.uniform(0, 1, size=(4*nlive, self.ndims))

    #     v = np.array([self.ptform(u[i, :]) for i in range(u.shape[0])])
        
    #     L = np.array([self.lnlikelihood(v[i, :], self.f[self.sel]) for i in range(u.shape[0])])

    #     idx = np.isfinite(L)

    #     return [u[idx, :][:nlive, :], v[idx, :][:nlive, :], L[idx][:nlive]]
    
    # def runDynesty(self, dynamic=False, progress=True, nlive=100, logl_kwargs={}):
    #     """ Start nested sampling

    #     Initializes and runs the nested sampling with Dynesty. We use the 
    #     default settings for stopping criteria as per the Dynesty documentation.

    #     Parameters
    #     ----------
    #     dynamic : bool, optional
    #         Use dynamic sampling as opposed to static. Dynamic sampling achieves
    #         minutely higher likelihood levels compared to the static sampler. 
    #         From experience this is not usually worth the extra runtime. By 
    #         default False.
    #     progress : bool, optional
    #         Display the progress bar, turn off for commandline runs, by default 
    #         True
    #     nlive : int, optional
    #         Number of live points to use in the sampling. Conceptually similar 
    #         to MCMC walkers, by default 100.

    #     Returns
    #     -------
    #     sampler : Dynesty sampler object
    #         The sampler from the nested sampling run. Contains some diagnostics.
    #     samples : jax device array
    #         Array of samples from the nested sampling with shape (Nsamples, Ndim)
    #     """

    #     initLive = self.getInitLive(nlive)

    #     if dynamic:
    #         sampler = dynesty.DynamicNestedSampler(self.lnlikelihood, 
    #                                                self.ptform, 
    #                                                self.ndims, 
    #                                                nlive=nlive, 
    #                                                sample='rwalk',
    #                                                live_points=initLive,
    #                                                logl_args=[self.f[self.sel]])
            
    #         sampler.run_nested(print_progress=progress, 
    #                            wt_kwargs={'pfrac': 1.0}, 
    #                            dlogz_init=1e-3 * (nlive - 1) + 0.01, 
    #                            nlive_init=nlive)  
            
    #     else:           
    #         sampler = dynesty.NestedSampler(self.lnlikelihood, 
    #                                         self.ptform, 
    #                                         self.ndims, 
    #                                         nlive=nlive, 
    #                                         sample='rwalk',
    #                                         live_points=initLive,
    #                                         logl_args=[self.f[self.sel]])
            
    #         sampler.run_nested(print_progress=progress)
 
    #     result = sampler.results

    #     unweighted_samples, weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

    #     samples = dyfunc.resample_equal(unweighted_samples, weights)

    #     return sampler, samples
