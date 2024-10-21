import numpy as np
import copy
from collections.abc import Iterable
import jax.numpy as jnp
from pbjam.plotting import plotting
from pbjam import IO
from pbjam.modeID import modeID
from pbjam.peakbagging import peakbag

def _convertToList(arg):
    """
    Convert the input argument to a list.

    Parameters
    ----------
    arg : str, tuple, np.ndarray, list
        The input argument to be converted. Can be a string, tuple, NumPy array, or list.

    Returns
    -------
    list
        The input argument converted to a list. If the input is already a list, it is returned as is.
        If the input is a string, it is wrapped in a list.

    Raises
    ------
    TypeError
        If the input argument is not of a supported type (str, tuple, list, or np.ndarray).
    """

    # If the argument is a string, return it wrapped in a list
    if isinstance(arg, str):
        return [arg]
    
    # If it's an iterable but not a list, convert it to a list
    elif isinstance(arg, (tuple, np.ndarray)):
        return list(arg)
    
    # If it's already a list, return as is
    elif isinstance(arg, list):
        return arg
    
    # Otherwise raise an error if the type is not supported
    else:
        raise TypeError("Unsupported type")
    
def _validateObs(obs, name):
    """
    Validate the observational data for a given target, ensuring required keys are present and values are in the form (value, error).

    Parameters
    ----------
    obs : dict
        Dictionary containing observational data with each key having a tuple of (value, error).
    name : str
        Name or identifier of the target being validated.

    Raises
    ------
    ValueError
        If any of the required keys ('numax', 'dnu', 'teff') are missing from the `obs` dictionary.
    AssertionError
        If any value in `obs` is not iterable or is not in the form of a tuple with two elements (value, error).
    """

    for key in ['numax', 'dnu', 'teff']:
        if key not in obs.keys():
            raise ValueError(f'Missing {key} in obs for target {name}')
        
    for key, val in obs.items():
        assert isinstance(val, Iterable), 'Entries in obs must be of the form (value, error)'
    
        assert len(val) == 2, 'Entries in obs must be of the form (value, error)'

class session():
    """ Main class used to initiate peakbagging for several stars.

    Use this class to initialize a star class instance for one or more targets.
    Once initialized, calling the session class instance will execute a complete
    peakbagging run.

    The observational constraints, such numax, dnu, teff, bp_rp, must be provided 
    through keyword entries in a dictionary, which is then passed via the obs 
    argument when initalizing the session class.
        
    Unless you provide the time series or spectrum, PBjam will download it. In
    which case it will do some rudimentary reduction, like removing outliers,
    removing NaN values and running a median filter through the light curve,
    with a width appropriate for the provided numax.
 
    Parameters
    ----------
    name : str
        Target name, most commonly used identifiers can be used if you 
        want PBjam to download the data (KIC, TIC, HD, Bayer etc.). If you 
        provide data yourself the name can be any string.
    obs : dict
        Dictionary of observational inputs: numax, dnu, teff, bp_rp. 
    timeseries : object, optional
        Timeseries input. Leave as None for PBjam to download it automatically.
        Otherwise, arrays of shape (2,N).
    spectrum : object, optional
        Spectrum input. Leave as None for PBjam to use Timeseries to compute
        it for you. Otherwise, arrays of shape (2,N).
    lk_kwargs : dict, optional
        Arguments passed to lightkurve to download the time series.
    outpath : str, optional
        Path to store the plots and results for the various stages of the 
        peakbagging process.    
    downloadDir : str, optional
        Directory to cache lightkurve downloads. Lightkurve will place the fits
        files in the default lightkurve cache path in your home directory.            
    """

    def __init__(self, name, obs, timeseries=None, spectrum=None, lk_kwargs={}, outpath=None, downloadDir=None):
      
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        # Expect names is a list, 

        # Setup inputs dictionary
        self.inputs = {}

        for nm in _convertToList(name):
            self.inputs[nm] = {}
 
        # Handle obs
        assert isinstance(obs, dict), 'The obs argument must be a dictionary.'

        # If keys don't match, assume it applies to all targets.
        for key in self.inputs.keys():
            if key in obs.keys():
                _obs = obs[key]
            else:
                _obs = obs

            _validateObs(_obs, key)

            self.inputs[key]['obs']= _obs


        # # Handle time series and spectrum

        # spectrum can be a dictionary with keys corresponding to names, 
        if isinstance(spectrum, dict):
            
            assert spectrum.keys() == self.inputs.keys(), 'The targets in spectrum must match those in names.'
            
            # The values for each key must be iterable of shape (2, N)
            for key in self.inputs.keys():
                 
                assert spectrum[key].shape[0] == 2, f'Shape of spectrum for {key} must be (2, N)'
            
                self.inputs[key]['f'] = spectrum[key][0]

                self.inputs[key]['s'] = spectrum[key][1]
                
        # Spectrum can be a iterable of shape (2, N)
        elif isinstance(spectrum, (type(np.array([])), type(jnp.array([])))):
            assert spectrum.shape[0] == 2, f'Shape of spectrum for must be (2, N)'
            
            for key in self.inputs.keys():
                self.inputs[key]['f'] = spectrum[0]

                self.inputs[key]['s'] = spectrum[1]

        # Spectrum can also be None, in which case we first look for a corresponding item in timeseries
        elif spectrum is None:

            if isinstance(timeseries, dict):

                assert timeseries.keys() == self.inputs.keys(), 'The targets in timeseries must match those in names.'

                for key in self.inputs.keys():    
                    if timeseries[key].shape[0] == 3:
                        psd = IO.psd(key, time=timeseries[key][0], flux=timeseries[key][1], flux_err=timeseries[key][2], useWeighted=True)

                    elif timeseries[key].shape[0] == 2:
                        psd = IO.psd(key, time=timeseries[key][0], flux=timeseries[key][1])

                    else:
                        raise ValueError(f'Unhandled timeseries shape for computing psd for {key}')
                    
                    psd()
                    
                    self.inputs[key]['f'] = psd.freq

                    self.inputs[key]['s'] = psd.powerdensity
                    
            elif isinstance(timeseries, (type(np.array([])), type(jnp.array([])))):
                if timeseries.shape[0] == 3:
                    psd = IO.psd(key, time=timeseries[0], flux=timeseries[1], flux_err=timeseries[2], useWeighted=True)

                elif timeseries.shape[0] == 2:
                    psd = IO.psd(key, time=timeseries[0], flux=timeseries[1])

                else:
                    raise ValueError(f'Unhandled timeseries shape for computing psd for {key}')
                
                for key in self.inputs.keys():
                    psd()
                    
                    self.inputs[key]['f'] = psd.freq

                    self.inputs[key]['s'] = psd.powerdensity

            elif timeseries is None:

                # Make sure lk_kwargs is not None
                assert isinstance(lk_kwargs, dict), 'To download data lk_kwargs must be a dict.'
                assert len(list(lk_kwargs.keys())) > 0

                # If keys are the same as input, loop through them and assign to input[key]
                for key in self.inputs.keys():

                    if key in lk_kwargs.keys():
                        _lk_kwargs = lk_kwargs[key]
                    else:
                        _lk_kwargs = lk_kwargs
                        
                    psd = IO.psd(key, lk_kwargs=_lk_kwargs, downloadDir=downloadDir)

                    psd()
                    
                    self.inputs[key]['f'] = psd.freq

                    self.inputs[key]['s'] = psd.powerdensity
            else:
                raise ValueError('Timeseries must be a (2, N) array-like or dictionary with entries like the name argument.')
        else:   
            raise ValueError('Spectrum must be a (2, N) array-like or dictionary with entries like the name argument.')

 
        self.stars = []

        for nm, inpt in self.inputs.items():
            
            self.stars.append(star(nm, outpath=outpath, **inpt))


    def __call__(self,  modeID_kwargs={}, peakbag_kwargs={}):
        """ Sequentially call all the star class instances

        Calling the session class instance will loop through all the stars that it contains, and call each one. 
        
        This performs a full peakbagging run on each star in the session.

        Parameters
        ----------
        modeID_kwargs : dict
            Arguments passed to the modeID stage of PBjam.
        peakbag_kwargs : dict
            Arguments passed to the peakbag stage of PBjam
        """
 
        # Assume top-level keys correspond to names
        if not (self.inputs.keys() == modeID_kwargs.keys()):
            _modeID_kwargs = {f'{st.name}': modeID_kwargs for st in self.stars}
            
        if not (self.inputs.keys() == peakbag_kwargs.keys()):
            _peakbag_kwargs = {f'{st.name}': peakbag_kwargs for st in self.stars}

        # Otherwise leave it to the user to make sure the keys are correct.
        for i, st in enumerate(self.stars):
            st(_modeID_kwargs[f'{st.name}'], _peakbag_kwargs[f'{st.name}'])
  
 
class star(plotting):
    """ Main class used to initiate peakbagging for a single stars.

    Use this class to initialize a star class instance for one target. Once 
    initialized, calling the star class instance will execute a complete
    peakbagging run.

    The observational constraints, such numax, dnu, teff, bp_rp, must be provided 
    through keyword entries in a dictionary, which is then passed via the obs 
    argument when initalizing the session class.
        
    The star class only accepts a power density spectrum in the form of a list of
    frequency bins 'f' and power density 's'.

    Parameters
    ----------
    name : str
        Target name, most commonly used identifiers can be used if you 
        want PBjam to download the data (KIC, TIC, HD, Bayer etc.). If you 
        provide data yourself the name can be any string.
    f : array-like
        Frequency bins of the power density spectrum.
    s : array-like
        Power density spectrum with the same shape as 'f'.
    obs : dict
        Dictionary of observational inputs: numax, dnu, teff, bp_rp. 
    outpath : str, optional
        Path to store the plots and results for the various stages of the 
        peakbagging process. Default is to output to the working directory.
    kwargs : dict
        Dictionary of additional keyword arguments for either modeID or peakbag.    
    """

    def __init__(self, name, f, s, obs, outpath=None, **kwargs):
                
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        self.__dict__.update(kwargs)
        
        del self.__dict__['kwargs']
  
        self.outpath = IO._setOutpath(self.name, self.outpath)

        for key, val in self.obs.items():
            assert isinstance(val, Iterable), 'Entries in obs must be of the form (value, error)'
            assert len(val) == 2, 'Entries in obs must be of the form (value, error)'
            
    def runModeID(self, modeID_kwargs={}):
        """ Run the mode identification process using the provided or default keyword arguments.

        This method creates a `modeID` instance and executes it with the arguments provided in 
        `modeID_kwargs` or from the current object's attributes. If `priorpath` is not specified, 
        it fetches the path to the prior file.

        Parameters
        ----------
        modeID_kwargs : dict, optional
            Dictionary of additional keyword arguments to update or override the current object's attributes 
            when initializing the `modeID` instance. Default is an empty dictionary.

        Raises
        ------
        KeyError
            If required parameters for mode identification are missing.
        """
            
        _modeID_kwargs = copy.deepcopy(self.__dict__)
        
        _modeID_kwargs.update(modeID_kwargs)
         
        if not 'priorpath' in _modeID_kwargs:
            self.priorpath = IO.getPriorPath()
            
            _modeID_kwargs['priorpath'] = self.priorpath
        
        self.modeID = modeID(**_modeID_kwargs)

        self.modeID(**_modeID_kwargs)
        
    def runPeakbag(self, peakbag_kwargs={}):
        """ Run the peakbagging process using the provided or default keyword arguments.

        This method creates a `peakbag` instance and executes it with the arguments provided in 
        `peakbag_kwargs` or from the current object's attributes. It uses mode identification results 
        to set missing parameters if necessary.

        Parameters
        ----------
        peakbag_kwargs : dict, optional
            Dictionary of additional keyword arguments to update or override the current object's attributes 
            when initializing the `peakbag` instance. Default is an empty dictionary.
        """ 
        
        _peakbag_kwargs = copy.deepcopy(self.__dict__)
                
        _peakbag_kwargs.update(peakbag_kwargs)
        
        if not 'ell' in _peakbag_kwargs.keys():
            _peakbag_kwargs.update({'ell': self.modeID.result['ell']})
        
        _peakbag_kwargs.update(self.modeID.result['summary'])
        
        self.peakbag = peakbag(**_peakbag_kwargs)

        self.peakbag(**_peakbag_kwargs)
 
    def __call__(self, modeID_kwargs={}, peakbag_kwargs={}):
        """ Execute the modeID and peakbag stages.

        Will pass the relevant output from modeID to peakbag. 

        Passing arguments that have already been given when initializing 
        the star class will override that parameter. This can be used to, 
        for example, change the resolution of the spectrum from the modeID 
        to the peakbag stage, which may sometimes save time since the 
        modeID doesn't require such high resolution.

        Parameters
        ----------
        modeID_kwargs : dict, optional
            Arguments to be passed to the modeID module. 
        peakbag_kwargs : dict, optional
            Arguments to be passed to the peakbag module. 

        Returns
        -------
        modeID_result: dict
            Dictionary of results from the modeID stage.
        peakbag_result: dict
            Dictionary of results from the peakbag stage.
        """
        # Run the mode ID stage
        self.runModeID(modeID_kwargs)

        # Run the peakbag stage
        self.runPeakbag(peakbag_kwargs)
        
        return self.modeID.result, self.peakbag.result
