"""

This module contains the mode identification class for PBjam.

The general mode ID strategy is to model the background + l=2,0 components 
separately from the l=1 modes, since the latter are often more computationally 
difficult.

A usage example could be:

M = modeID(...)
M.runl20model(...)
M.runl1model(...)

This provides the inputs for the detailed peakbagging stage which is provided as 
a separate module.

Several plotting options are available to display the results, including `echelle`, 
`spectrum` and `corner`.

"""

import warnings, os, pickle
import jax.numpy as jnp
import numpy as np
from pbjam.l1models import Asyl1model, Mixl1model, RGBl1model
from pbjam.l20models import Asyl20model
from pbjam.plotting import plotting
from pbjam import IO
import pandas as pd

class modeID(plotting, ):  
    """
    Class for identifying modes in solar-like oscillators.

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
    freqLimits : list, optional
        Frequency limits for mode identification. If None, it is calculated based on 'numax' and 'dnu'.
    priorPath : str, optional
        Path to prior sample csv. If None, a default path is used.
    """

    def __init__(self, f, s, obs, addPriors={}, N_p=7, freqLimits=None, priorPath=None, **kwargs):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.f = jnp.array(self.f)

        self.s = jnp.array(self.s)
      
        self.Nyquist = self.f[-1]

        # Set frequency range to compute the model on. Default is one radial order above/below the requested number.
        if self.freqLimits is None:
            self.freqLimits = [self.obs['numax'][0] - self.obs['dnu'][0]*(self.N_p//2+1), 
                               self.obs['numax'][0] + self.obs['dnu'][0]*(self.N_p//2+1),]
            
        self.sel = (np.array(self.freqLimits).min() < self.f) & (self.f < np.array(self.freqLimits).max())   

        if self.priorPath is None:
            self.priorPath = IO.getPriorPath()
 
    def runl20model(self, progress=True, dynamic=False, minSamples=5000, sampler_kwargs={}, logl_kwargs={}, PCAsamples=50, PCAdims=6, **kwargs):
        """
        Runs the l20 model on the selected spectrum.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress during the model run. Default is True.
        dynamic : bool, optional
            Whether to use dynamic nested sampling. Default is False (static nested sampling).
        minSamples : int, optional
            The minimum number of samples to generate. Default is 5000.
        logl_kwargs : dict, optional
            Additional keyword arguments for the log-likelihood function. Default is an empty dictionary.
        PCAsamples : int, optional
            Number of samples for PCA. Default is 50.
        PCAdims : int, optional
            Number of dimensions for PCA. Default is 6.

        Returns
        -------
        result : dict
            Parsed results from the l20 model.
        """

        f = self.f[self.sel]

        s = self.s[self.sel]

        self.l20model = Asyl20model(f, s, 
                                    self.obs, 
                                    self.addPriors, 
                                    self.N_p, 
                                    PCAsamples, 
                                    PCAdims,
                                    priorPath=self.priorPath)
        
        self.l20Samples = self.l20model.runSampler(progress=progress,
                                                   dynamic=dynamic,
                                                   minSamples=minSamples, 
                                                   logl_kwargs=logl_kwargs, 
                                                   sampler_kwargs=sampler_kwargs)

        l20samples_u = self.l20model.unpackSamples(self.l20Samples)

        self.l20result = self.l20model.parseSamples(l20samples_u)

        self.result = self.mergeResults(l20result=self.l20result)
 
        return self.l20result

    def runl1model(self, progress=True, dynamic=False, minSamples=5000, sampler_kwargs={}, logl_kwargs={}, model='MS', PCAsamples=500, PCAdims=7, **kwargs):
        """
        Runs the l1 model on the selected spectrum.

        Should follow the l20 model run.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress during the model run. Default is True.
        dynamic : bool, optional
            Whether to use dynamic nested sampling. Default is False (static nested sampling).
        minSamples : int, optional
            The minimum number of samples to generate. Default is 5000.
        sampler_kwargs : dict, optional
            Additional keyword arguments for the sampler. Default is an empty dictionary.
        logl_kwargs : dict, optional
            Additional keyword arguments for the log-likelihood function. Default is an empty dictionary.
        model : str
            Choice of which model to use for estimating the l=1 mode locations. Choices are MS, SG, RGB models.
        PCAsamples : int, optional
            Number of samples for PCA. Default is 100.
        PCAdims : int, optional
            Number of dimensions for PCA. Default is 5.

        Returns
        -------
        result : dict
            Parsed results from the l1 model.
        """

        # Compute the l=2,0 model residual. 
        self.l20residual = self.s[self.sel] / self.l20model.getMedianModel()
      
        f = self.f[self.sel]

        s = self.l20residual

        summary = {'n_p': self.l20result['enn'][self.l20result['ell']==0],
                   'nu0_p': self.l20result['summary']['freq'][0, self.l20result['ell']==0]}
 
        for key in ['numax', 'dnu', 'env_height', 'env_width', 'mode_width', 'teff', 'bp_rp']:
            summary[key] = self.l20result['summary'][key]

        if model.lower() == 'ms':
            self.l1model = Asyl1model(f, s, 
                                      summary, 
                                      self.addPriors,
                                      PCAsamples, 
                                      priorPath=self.priorPath)

        elif model.lower() == 'sg':    
            self.l1model = Mixl1model(f, s, 
                                      summary, 
                                      self.addPriors,
                                      PCAsamples, 
                                      PCAdims,
                                      priorPath=self.priorPath)
            
        elif model.lower() == 'rgb':
            self.l1model = RGBl1model(f, s,  
                                      summary, 
                                      self.addPriors, 
                                      PCAsamples,
                                      rootiter=15,
                                      priorPath=self.priorPath,
                                      modelChoice='simple')
        else:
            raise ValueError(f'Model {model} is invalid. Please use either MS, SG or RGB.')
         
        self.l1Samples  = self.l1model.runSampler(progress=progress,
                                                  dynamic=dynamic,
                                                  minSamples=minSamples, 
                                                  logl_kwargs=logl_kwargs, 
                                                  sampler_kwargs=sampler_kwargs)
        
        l1SamplesU = self.l1model.unpackSamples(self.l1Samples)

        self.l1result = self.l1model.parseSamples(l1SamplesU)

        self.result = self.mergeResults(l20result=self.l20result, l1result=self.l1result)

        return self.l1result

    def __call__(self, model='MS', progress=True, dynamic=False, sampler_kwargs={}, logl_kwargs={}, **kwargs):
        """
        Run both the l20 and l1 models.

        The results are stored in the modeID.result dictionary after each step is 
        completed.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress during the model run. Default is True.
        sampler_kwargs : dict, optional
            Additional keyword arguments for the sampler. Default is an empty dictionary.
        logl_kwargs : dict, optional
            Additional keyword arguments for the log-likelihood function. Default is an empty dictionary.
        """
         
        self.runl20model(progress, dynamic, sampler_kwargs=sampler_kwargs, logl_kwargs=logl_kwargs)
        
        self.runl1model(progress, dynamic, model=model, sampler_kwargs=sampler_kwargs, logl_kwargs=logl_kwargs)
 
    def mergeResults(self, l20result=None, l1result=None, N=5000):
        """
        Merges results from l20 and l1 models into a single result dictionary. 

        Attempts to include N samples from both models, but will use the lowest common
        value in case one of the models returns less than N samples.

        Note that if l20 and l1 share any parameters (like numax and dnu) the results 
        from the l20 model are used since they are probably a bit more reliable.

        Parameters
        ----------
        l20result : dict, optional
            The result dictionary from the l20 model. 
        l1result : dict, optional
            The result dictionary from the L1 model. 
        N : int, optional
            The number of samples to include in the merged results. Default is 5000.

        Returns
        -------
        R : dict
            A dictionary containing merged results from the l20 and l1 models.
        """

        # Initialize an empty result dictionary
        R = {'ell': np.array([]),
             'enn': np.array([]),
             'emm': np.array([]),
             'zeta': np.array([]),
             'summary': {'freq'  : np.array([]).reshape((2, 0)), 
                         'height': np.array([]).reshape((2, 0)), 
                         'width' : np.array([]).reshape((2, 0)),
                         'rotAsym': np.array([]).reshape((2, 0)),
                        },
             'samples': {'freq'  : np.array([]).reshape((N, 0)),
                         'height': np.array([]).reshape((N, 0)), 
                         'width' : np.array([]).reshape((N, 0)),
                         'rotAsym' : np.array([]).reshape((N, 0))
                        },
            }
        
        # Use self.l20result if l20result is not provided
        if l20result is None and hasattr(self, 'l20result'):
            l20result = self.l20result

        _N = np.append(N, l20result['samples']['freq'].shape[0])
         
        # Use self.l1result if l1result is not provided
        if l1result is None and hasattr(self, 'l1model'):
            l1result = self.l1result
        
        if l1result is not None:
            _N = np.append(_N, l1result['samples']['freq'].shape[0])
 
        # Use the minimum number of samples.
        N = np.min(_N)
        
        # This order overrides the numax, dnu and teff from l1result in the output since the l20result is more reliable. 
        resList = [l1result, l20result] 

        # Merge top level dictionary keys.
        for rootkey in ['ell', 'enn', 'emm', 'zeta']:
            for D in resList:
                if D is not None:
                    R[rootkey] = np.append(R[rootkey], D[rootkey])
        
        # Merge summary and samples level keys.
        for rootkey in ['summary', 'samples']:
            for D in resList: 
                if D is not None:
                    for subkey in list(D[rootkey].keys()):
                        if subkey not in ['freq', 'height', 'width', 'rotAsym']:
                            R[rootkey][subkey] = D[rootkey][subkey]

                    for subkey in ['freq', 'height', 'width', 'rotAsym']:
                        R[rootkey][subkey] = np.hstack((R[rootkey][subkey][:N, :], 
                                                        D[rootkey][subkey][:N, :]))

                else:
                    continue

        return R

    def storeResult(self, resultDict, path=None, ID=None):
        """
        Stores the results in a specified directory with identifier. The results are stored as a 
        pickled Python dictionary file and a CSV file. The pickle file contains all results,
        while the CSV file contains only the summary model parameters.

        Parameters
        ----------
        resultDict : dict
            The dictionary containing the results to be stored.
        path : str, optional
            The directory path where the results should be stored. If None, it defaults to the current directory.
        ID : str, optional
            A unique identifier for the stored results. If None, a random identifier is generated.
        """

        # If no path is specified use cwd.
        if path is None:
            path = os.getcwd()
        else:
            path = str(path)
        
        # Make the dir path if it doesn't exist
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(path)

        # Assign random number identifier to the save
        if ID is not None:
            _ID = str(ID) 
        else:   
            _ID = f'unknown_tgt_{np.random.randint(0, 1e10)}'

            warnings.warn(f'Output stored under {_ID}. You should probably specify a target ID.')
        
        basefilename = os.path.join(*[path, f'{_ID}_modeIDresult'])
        
        # Store everything in pickled dict
        pickle.dump(resultDict, open(basefilename+'.pkl', 'wb'))

        # Grab just model parameters and save
        _tmp = {key: self.result['summary'][key] for key in self.result['summary'].keys() if key not in ['freq', 'height', 'width']}

        df_data = [{'name': key, 'mean': value[0], 'error': value[1]} for key, value in _tmp.items()]

        # Create a DataFrame
        df = pd.DataFrame(df_data)
        
        df.to_csv(basefilename+'.csv', index=False)
 
  
