from . import PACKAGEDIR
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simps
from pbjam.jar import scalingRelations
import os, pickle, re, time
import lightkurve as lk
from lightkurve.periodogram import Periodogram
from datetime import datetime
from astroquery.mast import ObservationsClass as AsqMastObsCl
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astropy.timeseries import LombScargle
from astropy import units
 
 
class psd(scalingRelations):
    
    def __init__(self, ID, lk_kwargs={}, time=None, flux=None, flux_err=None, 
                 downloadDir=None, fit_mean=False, timeConversion=86400, 
                 use_cached=False):
    
        """ Asteroseismology wrapper for Astropy Lomb-Scargle

        Uses the Astropy.LombScargle class to compute the power spectrum of a given
        time series. A variety of choices for computing the spectrum are available.
        The recommended methods are either `fast' or `Cython'.

        Notes
        -----
        The Cython implemenation is very slow for time series longer than about
        1 month (array size of ~1e5). The Fast implementation is similar to the an
        FFT, but at a very slight loss of accuracy. There appears to be a slight
        increasing slope with frequency toward the Nyquist frequency.

        The adjustments to the frequency resolution, due to gaps, performed in the
        KASOC filter may not be beneficial the statistics we use in the detection
        algorithm.  This has not been thuroughly tested yet though. So recommend
        leaving it in, but with a switch to turn it off for testing.

        Parameters
        ----------
        time : array
            Time stamps of the time series.
        flux : array
            Flux values of the time series.
        flux_error : array
            Flux value errors of the time series.
        fit_mean : bool, optional
            Keyword for Astropy.LombScargle. If True, uses the generalized
            Lomb-Scargle approach and fits with a floating mean. Default is 
            False.
        timeConversion : float
            Factor to convert the time series such that it is in seconds. Note, 
            all stored time values, e.g. cadence or duration, are kept in the 
            input units. Default is 86400 to convert from days to seconds.

        Attributes
        ----------
        dt : float
            Cadence of the time series.
        dT : float
            Total length of the time series.
        NT : int
            Number of data points in the time series.
        dutyCycle : float
            Duty cycle of the time series.
        Nyquist : float
            Nyquist frequency in Hz.
        df : float
            Fundamental frequency spacing in Hz.
        ls : astropy.timeseries.LombScargle object:
            Astropy Lomb-Scargle class instance used in computing the power
            spectrum.
        indx : array, bool
            Mask array for removing nan and/or -inf values from the time series.
        freqHz : array, float
            Frequency range in Hz.
        freq : array, float
            Freqeuency range in muHz.
        normfactor : float
            Normalization factor to ensure the power conforms with Parseval.
        power : array, float
            Power spectrum of the time series in ppm^2.
        powerdensity : array float
            Power density spectrum of the time series in ppm^2/muHz
        amplitude : array, float
            Amplitude spectrum of the time series in ppm.
        """


        self.ID = ID

        self.downloadDir = downloadDir

        self.TS = timeSeries(ID, self.downloadDir, lk_kwargs)

        if (time is None) and (flux is None):
            
            time, flux = self.TS._getTS(use_cached=use_cached)

            flux = (flux/np.nanmedian(flux) - 1) * 1e6
         
        self._time = time
        
        self._flux = flux

        self._getBadIndex(time, flux)
 
        self.time, self.flux = time[self.indx], flux[self.indx]
 
        self.fit_mean = fit_mean

        self.timeConversion = timeConversion

        self.dt = self._getSampling()

        self.dT = self.time.max() - self.time.min()

        self.NT = len(self.time)

        self.dutyCycle = self._getDutyCycle()

        if flux_err is None:
            # Init Astropy LS class without weights
            self.ls = LombScargle(self.time * self.timeConversion,
                                  self.flux, center_data=True,
                                  fit_mean=self.fit_mean)

        else:
            # Init Astropy LS class with weights
            self.ls = LombScargle(self.time * self.timeConversion,
                                  self.flux, center_data=True,
                                  fit_mean=self.fit_mean,
                                  dy=self.flux_err,)

        self.Nyquist = 1/(2*self.timeConversion*self.dt) # Hz

        self.df = self._fundamental_spacing_integral()
 


    def __call__(self, oversampling=1, nyquist_factor=1.0, method='fast'):
        """ Compute power spectrum

        Computes the power spectrum and normalizes it to conform with Parseval's
        theorem. The output is available as the power in ppm^2, powerdensity in
        ppm^2/muHz and the amplitude spectrum in ppm.

        The frequency range is transformed to muHz as this is customarily used
        in asteroseismology of main sequence stars.

        Parameters
        ----------
        oversampling : int
            The number of times the frequency range should be oversampled. This
            equates to zero-padding when using the FFT.
        nyquist_factor : float
            Factor by which to extend the spectrum past the Nyquist frequency.
            The default is 10% greater than the true Nyquist frequency. We use
            this to get a better handle on the background level at high
            frequency.
        method : str
            The recommended methods are either `fast' or `Cython'. Cython is
            a bit more accurate, but significantly slower.
        """

        self.freqHz = np.arange(self.df/oversampling, nyquist_factor*self.Nyquist, 
                                self.df/oversampling, dtype='float64')

        self.freq = jnp.array(self.freqHz*1e6) # muHz is usually used in seismology

        # Calculate power at frequencies using fast Lomb-Scargle periodogram:
        power = self.ls.power(self.freqHz, normalization='psd', method=method, 
                              assume_regular_frequency=True)

        # Due to numerical errors, the "fast implementation" can return power < 0.
        # Replace with random exponential values instead of 0?
        power = np.clip(power, 0, None)

        self._getNorm(power)

        self.power = jnp.array(power * self.normfactor * 2)

        self.powerdensity = jnp.array(power * self.normfactor / (self.df * 1e6))
 
        self.amplitude = jnp.array(power * np.sqrt(power * self.normfactor * 2))        

        pg = Periodogram(self.freq * units.uHz, units.Quantity(self.powerdensity))
 
        self.pg = pg
 
    def _getBadIndex(self, time, flux):
        """ Identify indices with nan/inf values

        Flags array indices where either the timestamps, flux values, or flux errors
        are nan or inf.

        """

        self.indx = np.invert(np.isnan(time) | np.isnan(flux) | np.isinf(time) | np.isinf(flux))

    def getTSWindowFunction(self, tmin=None, tmax=None, cadenceMargin=1.01):

        if tmin is None:
            tmin = min(self.time)

        if tmax is None:
            tmax = max(self.time)

        t = self.time.copy()[self.indx]

        w = np.ones_like(t)

        break_counter = 0
        epsilon = 0.0001 # this is a tiny scaling of dt to avoid numerical issues

        while any(np.diff(t) > cadenceMargin*self.dt):

            idx = np.where(np.diff(t)>cadenceMargin*self.dt)[0][0]

            t_gap_fill = np.arange(t[idx], t[idx+1]-epsilon*self.dt, self.dt)

            w_gap_fill = np.zeros(len(t_gap_fill))
            w_gap_fill[0] = 1

            t = np.concatenate((t[:idx], t_gap_fill, t[idx+1:]))

            w = np.concatenate((w[:idx], w_gap_fill, w[idx+1:]))

            break_counter +=1
            if break_counter == 100:
                break

        if (tmin is not None) and (tmin < t[0]):
            padLow = np.arange(tmin, t[0], self.dt)

            t = np.append(padLow, t)
            
            w = np.append(np.zeros_like(padLow), w)

        if (tmax is not None) and (t[0] < tmax):
            padHi = np.arange(t[-1], tmax, self.dt)
            
            t = np.append(t, padHi)
            
            w = np.append(w, np.zeros_like(padHi))

        return t, w

    def _getDutyCycle(self, cadence=None):
        """ Compute the duty cycle

        If cadence is not provided, it is assumed to be the median difference
        of the time stamps in the time series.

        Parameters
        ----------
        cadence : float
            Nominal cadence of the time series. Units should be the
            same as t.

        Returns
        -------
        dutyCycle : float
            Duty cycle of the time series
        """

        if cadence is None:
            cadence = self._getSampling()

        nomLen = np.ceil((np.nanmax(self.time) - np.nanmin(self.time)) / cadence)

        idx = np.invert(np.isnan(self.time) | np.isinf(self.time))

        dutyCycle = len(self.time[idx]) / nomLen

        return dutyCycle

    def _getSampling(self):
        """ Compute sampling rate

        Computes the average sampling rate in the time series.

        This should approximate the nominal sampling rate,
        even with gaps in the time series.

        Returns
        ----------
        dt : float
            Cadence of the time stamps.
        """
        idx = np.invert(np.isnan(self.time) | np.isinf(self.time))

        dt = np.median(np.diff(self.time[idx]))

        return dt

    def _getNorm(self, power):
        """ Parseval normalization

        Computes the normalization factor for the power spectrum such that it
        conforms with Parseval's theorem.

        power : array
            Unnormalized array of power.
        """

        N = len(self.ls.t)

        if self.ls.dy is None:
            tot_MS = np.sum((self.ls.y - np.nanmean(self.ls.y))**2)/N
        else:
            tot_MS = np.sum(((self.ls.y - np.nanmean(self.ls.y))/self.ls.dy)**2)/np.sum((1/self.ls.dy)**2)

        self.normfactor = tot_MS/np.sum(power)

    def _fundamental_spacing_integral(self):
        """ Estimate fundamental frequency bin spacing

        Computes the frequency bin spacing using the integral of the spectral
        window function.

        For uniformly sampled data this is given by df=1/T. Which under ideal
        circumstances ensures that power in neighbouring frequency bins is
        independent. However, this fails when there are gaps in the time series.
        The integral of the spectral window function is a better approximation
        for ensuring the bins are less correlated.

        """

        # The nominal frequency resolution
        df = 1/(self.timeConversion*(np.nanmax(self.time) - np.nanmin(self.time))) # Hz

        # Compute the window function
        freq, window = self.windowfunction(df, width=100*df, oversampling=5) # oversampling for integral accuracy

        # Integrate the windowfunction to get the corrected frequency resolution
        df = simps(window, freq)

        return df*1e-6

    def windowfunction(self, df, width=None, oversampling=10):
        """ Spectral window function.

        Parameters
        ----------
		 width : float, optional
            The width in Hz on either side of zero to calculate spectral window.
            Default is None.
        oversampling : float, optional
            Oversampling factor. Default is 10.
        """

        if width is None:
            width = 100*df

        freq_cen = 0.5*self.Nyquist

        Nfreq = int(oversampling*width/df)

        freq = freq_cen + (df/oversampling) * np.arange(-Nfreq, Nfreq, 1)

        x = 0.5*np.sin(2*np.pi*freq_cen*self.ls.t) + 0.5*np.cos(2*np.pi*freq_cen*self.ls.t)

        # Calculate power spectrum for the given frequency range:
        ls = LombScargle(self.ls.t, x, center_data=True, fit_mean=self.fit_mean)

        power = ls.power(freq, method='fast', normalization='psd', assume_regular_frequency=True)

        power /= power[int(len(power)/2)] # Normalize to have maximum of one

        freq -= freq_cen

        freq *= 1e6

        return freq, power

class timeSeries():

    def __init__(self, ID, downloadDir=None, lk_kwargs={}):
        

        self.constants = {'kplr_lc_exptime': 1800, 
                          'kplr_sc_exptime': 60,
                          'tess_120_exptime': 120,
                          'tess_20_exptime': 20, 
                          'tess_1800_exptime': 1800,
                          'tess_200_exptime': 200
                          }
        
        self.constants['kplr_lc_nyquist'] = nyquist(self.constants['kplr_lc_exptime'])

        self.ID = self._set_ID(ID)

        self.downloadDir = downloadDir

        self.lk_kwargs = lk_kwargs

        self._set_mission()

        self._set_author()

        self._set_exptime(lk_kwargs)
 
    def _set_mission(self, ):
        """ Set mission keyword.
        
        If no mission is selected will attempt to figure it out based on any
        prefixes in the ID string, and add this to the LightKurve keywords 
        arguments dictionary.
        
        Parameters
        ----------
        ID : str
            ID string of the target
        lkwargs : dict
            Dictionary to be passed to LightKurve
            
        """

        if not 'mission' in self.lk_kwargs:
            if ('kic' in self.ID.lower()):
                self.lk_kwargs['mission'] = 'Kepler'
            elif ('epic' in self.ID.lower()) :
                self.lk_kwargs['mission'] = 'K2'
            elif ('tic' in self.ID.lower()):
                self.lk_kwargs['mission'] = 'TESS'
            else:
                self.lk_kwargs['mission'] = ('Kepler', 'K2', 'TESS')

    def _set_author(self,):
        if not 'author' in self.lk_kwargs.keys():
            if ('KIC' in self.ID) or (self.lk_kwargs['mission']=='Kepler'):
                self.lk_kwargs['author'] = 'Kepler'
            if ('TIC' in self.ID) or (self.lk_kwargs['mission']=='TESS'):
                self.lk_kwargs['author'] = 'SPOC'

    def _set_exptime(self, lk_kwargs):

        if not 'exptime' in lk_kwargs.keys():
            if ('KIC' in self.ID) and (lk_kwargs['exptime'] < 1500):
                lk_kwargs['exptime'] = self.constants['kplr_sc_exptime']
                
                self.lc_type = 'Kepler 60s'
            
            else:
                lk_kwargs['exptime'] = self.constants['kplr_lc_exptime']
                
                self.lc_type = 'Kepler 1800s'

            if 'TIC' in self.ID:
                lk_kwargs['exptime'] = 120
            
                self.lc_type = 'TESS 120s'

    def _set_ID(self, ID):
        """ Format input ID
        
        Users tend to be inconsistent in naming targets, which is an issue for 
        looking stuff up on, e.g., Simbad. This function formats the ID so that 
        Simbad doesn't throw a fit.
        
        If the name doesn't look like anything in the variant list it will only be 
        changed to a lower-case string.
        
        Parameters
        ----------
        ID : str
            Name to be formatted.
        
        Returns
        -------
        ID : str
            Formatted name
            
        """

        ID = str(ID)
        ID = ID.lower()
        
        # Add naming exceptions here
        variants = {'KIC': ['kic', 'kplr', 'KIC'],
                    'Gaia DR2': ['gaia dr2', 'gdr2', 'dr2', 'Gaia DR2'],
                    'Gaia DR1': ['gaia dr1', 'gdr1', 'dr1', 'Gaia DR1'], 
                    'EPIC': ['epic', 'ktwo', 'EPIC'],
                    'TIC': ['tic', 'tess', 'TIC']
                }
        
        fname = None
        for key in variants:   
            for x in variants[key]:
                if x in ID:
                    fname = ID.replace(x,'')
                    fname = re.sub(r"\s+", "", fname, flags=re.UNICODE)
                    fname = key+' '+str(int(fname))
                    return fname 
        return ID
 
    def _getTS(self, use_cached):
        """Get time series with lightkurve

        Parameters
        ----------
        exptime : int, optional
            The exposure time (cadence) of the data to get, by default None.

        Returns
        -------
        time : DeviceArray
            The timestamps of the time series.
        flux : DeviceArray
            The flux values of the time series.
        """
        # Waiting a short amount of time prevents multiple rapid requests from 
        # being rejected.
        time.sleep(np.random.uniform(1, 5))

        # TODO this should be set depending on numax.
        wlen = int(4e6/self.lk_kwargs['exptime'])-1
        if wlen % 2 == 0:
            wlen += 1
        
        try:
            LCcol = self.search_lightcurve(use_cached=use_cached, cache_expire=30)  
        except:
            LCcol = self.search_lightcurve(use_cached=use_cached, cache_expire=0)
              
        lc = LCcol.stitch().normalize().remove_nans().remove_outliers().flatten(window_length=wlen)

        self.time, self.flux = jnp.array(lc.time.value), jnp.array(lc.flux.value)
 
        return self.time, self.flux

    def load_fits(self, files):
        """ Read fitsfiles into a Lightkurve object
        
        Parameters
        ----------
        files : list
            List of pathnames to fits files
        mission : str
            Which mission to download the data from.
        
        Returns
        -------
        lc : lightkurve.lightcurve.KeplerLightCurve object
            Lightkurve light curve object containing the concatenated set of 
            quarters.
            
        """
        if self.lk_kwargs['mission'] in ['Kepler', 'K2']:
            lcs = [lk.lightcurvefile.KeplerLightCurveFile(file) for file in files]

            lcCol = lk.LightCurveCollection(lcs)
        
        elif self.lk_kwargs['mission'] == 'TESS':
            lcs = [lk.lightcurvefile.TessLightCurveFile(file) for file in files if os.path.basename(file).startswith('tess')]
        
            lcCol = lk.LightCurveCollection(lcs)
        
        return lcCol
            
    def search_and_dump(self, ID, search_cache):
        """ Get lightkurve search result online.
        
        Uses the lightkurve search_lightcurve to find the list of available data 
        for a target ID. 
        
        Stores the result in the ~/.lightkurve/cache/searchResult directory as a 
        dictionary with the search result object and a timestamp.
        
        Parameters
        ----------
        ID : str
            ID string of the target
        lkwargs : dict
            Dictionary to be passed to LightKurve
        search_cache : str
            Directory to store the search results in. 
            
        Returns
        -------
        resultDict : dict
            Dictionary with the search result object and timestamp.    
        """
        
        current_date = datetime.now().isoformat()

        store_date = current_date[:current_date.index('T')].replace('-','')

        search = lk.search_lightcurve(ID, 
                                      exptime=self.lk_kwargs['exptime'], 
                                      mission=self.lk_kwargs['mission'], 
                                      author=self.lk_kwargs['author'])
        
        resultDict = {'result': search,
                      'timestamp': store_date}
        
        fname = os.path.join(*[search_cache, f"{ID}_{self.lk_kwargs['exptime']}.lksearchresult"])
        
        pickle.dump(resultDict, open(fname, "wb"))
        
        return resultDict   

    def getMASTidentifier(self,):
        """ return KIC/TIC/EPIC for given ID.
        
        If input ID is not a KIC/TIC/EPIC identifier then the target is looked 
        up on MAST and the identifier is retried. If a mission is not specified 
        the set of observations with the most quarters/sectors etc. will be 
        used. 
        
        Parameters
        ----------
        ID : str
            Target ID
        lkwargs : dict
            Dictionary with arguments to be passed to lightkurve. In this case
            mission and exptime.
        
        Returns
        -------
        ID : str
            The KIC/TIC/EPIC ID of the target.    
        """
        
        if not any([x in self.ID for x in ['KIC', 'TIC', 'EPIC']]):
            
            search = lk.search_lightcurve(self.ID)

            match = (self.lk_kwargs['exptime'] in search.exptime.value) & \
                    (self.lk_kwargs['author'] in search.author) & \
                    any([self.lk_kwargs['mission'] in x for x in search.mission])

            if len(search) == 0:
                raise ValueError(f'No results for {self.ID} found on MAST')
            elif not match:
                print(search)
                print()
                raise ValueError(f'Unable to find anything for {self.ID} matching criteria in lk_kwargs.')
            #elif not (self.lk_kwargs['exptime'] in search.exptime.value):



            maxFreqName = max(set(list(search.table['target_name'])), key = list(search.table['target_name']).count)

            maxFreqObsCol = max(set(list(search.table['obs_collection'])), key = list(search.table['obs_collection']).count)

            if maxFreqObsCol == 'TESS':
                prefix = 'TIC'
            else:
                prefix = ''

            temp_id = prefix + maxFreqName

            ID = self._set_ID(temp_id).replace(' ', '')

            self.lk_kwargs['mission'] = maxFreqObsCol

        else:

            ID = self.ID.replace(' ', '')

        return ID

    def check_sr_cache(self, ID, use_cached, cache_expire):
        """ check search results cache
        
        Preferentially accesses cached search results, otherwise searches the 
        MAST archive.
        
        Parameters
        ----------
        use_cached : bool, optional
            Whether or not to use the cached time series. Default is True.
        download_dir : str, optional.
            Directory for fits file and search results caches. Default is 
            ~/.lightkurve/cache. 
        cache_expire : int, optional.
            Expiration time for the search cache results. Files older than this 
            will be. The default is 30 days.
            
        Returns
        -------
        search : lightkurve.search.SearchResult
            Search result from MAST.  
        """
        
        # Set default lightkurve cache directory if nothing else is given
        if self.downloadDir is None:
            downloadDir = os.path.join(*[os.path.expanduser('~'), '.lightkurve', 'cache'])
        else:
            downloadDir = self.downloadDir

        # Make the search cache dir if it doesn't exist
        cachepath = os.path.join(*[downloadDir, 'searchResults', self.lk_kwargs['mission']])
        if not os.path.isdir(cachepath):
            os.makedirs(cachepath)

        filepath = os.path.join(*[cachepath, f"{ID}_{self.lk_kwargs['exptime']}.lksearchresult"])
         
        if os.path.exists(filepath) and use_cached:  
            
            resultDict = pickle.load(open(filepath, "rb"))

            fdate = resultDict['timestamp'] 

            ddate = datetime.now() - datetime(int(fdate[:4]), int(fdate[4:6]), int(fdate[6:]))
            
            # If file is saved more than cache_expire days ago, a new search is performed
            if ddate.days > cache_expire:   
                print(f'Last search was performed more than {cache_expire} days ago, checking for new data.')
                resultDict = self.search_and_dump(ID, cachepath)
            else:
                print('Using cached search result.')    
        else:
            print('No cached search results, searching MAST')
            resultDict = self.search_and_dump(ID, cachepath)
            
        return resultDict['result']

    def check_fits_cache(self, search):
        """ Query cache directory or download fits files.
        
        Searches the Lightkurve cache directory set by download_dir for fits files
        matching the search query, and returns a list of path names of the fits
        files.
        
        If not cache either doesn't exist or doesn't contain all the files in the
        search, all the fits files will be downloaded again.
        
        Parameters
        ----------
        search : lightkurve.search.SearchResult
            Search result from MAST. 
        mission : str
            Which mission to download the data from.
        download_dir : str, optional.
            Top level of the Lightkurve cache directory. default is 
            ~/.lightkurve/cache
            
        Returns
        -------
        files_in_cache : list
            List of path names to the fits files in the cache directory
        """
            
        if self.downloadDir is None:
            download_dir = os.path.join(*[os.path.expanduser('~'), '.lightkurve', 'cache'])
        else:
            download_dir = self.downloadDir
            
        files_in_cache = []

        for i, row in enumerate(search.table):
            fname = os.path.join(*[download_dir, 'mastDownload', self.lk_kwargs['mission'], row['obs_id'], row['productFilename']])
            
            if os.path.exists(fname):
                files_in_cache.append(fname)
        
        if len(files_in_cache) != len(search):
            if len(files_in_cache) == 0:
                print('No files in cache, downloading.')
            
            elif len(files_in_cache) > 0:
                print('Search result did not match cached fits files, downloading.')  
                
            search.download_all(download_dir=download_dir)
            
            files_in_cache = [os.path.join(*[download_dir, 'mastDownload', self.lk_kwargs['mission'], row['obs_id'], row['productFilename']]) for row in search.table]
        
        else:
            print('Loading fits files from cache.')

        return files_in_cache

    def search_lightcurve(self, use_cached, cache_expire=30):
        """ Get time series using LightKurve
        
        Performs a search for available fits files on MAST and then downloads them
        if nessary.
        
        The search results are cached with an expiration of 30 days. If a search
        result is found, the fits file cache is searched for a matching file list
        which is then used.
        
        Parameters
        ----------
        ID : str
            ID string of the target
        download_dir : str
            Directory for fits file and search results caches. 
        lkwargs : dict
            Dictionary to be passed to LightKurve  
        
        Returns
        -------
        lcCol : Lightkurve.LightCurveCollection instance
            Contains a list of all the sectors/quarters of data either freshly 
            downloaded or from the cache.
        """
        

        
        ID = self.getMASTidentifier()
        
        search = self.check_sr_cache(ID, use_cached=use_cached, cache_expire=cache_expire)
        
        fitsFiles = self.check_fits_cache(search)

        lcCol = self.load_fits(fitsFiles)

        return lcCol


def _get_outpath(self, fname):
        """  Get basepath or make full file path name.
        
        Convenience function for either setting the base path name for the star,
        or if given fname as input, will append this to the basepath name to 
        create a full path to the file in question. 

        Parameters
        ----------
        fname : str, optional
            If not None, will append this to the pathname of the star. Use this
            to store files such as plots or tables.
        
        Returns
        -------
        path : str
            If fname is None, path is the path name of the star. Otherwise it is
            the full file path name for the file in question.
        """
    
        if fname is None:
            return self.path
        elif isinstance(fname, str):
            path = os.path.join(*[self.path, fname])
        else:
            raise ValueError(f'Unrecognized input {fname}.')
        
        if not os.path.isdir(self.path):
            raise IOError(f'You are trying to access {self.path} which is a directory that does not exist.')
        else:
            return path

def _set_outpath(ID, rootPath):
    """ Sets the path attribute for star

    If path is a string it is assumed to be a path name, if not the
    current working directory will be used.

    Attempts to create an output directory for all the results that PBjam
    produces. A directory is created when a star class instance is
    initialized, so a session might create multiple directories.

    Parameters
    ----------
    rootPath : str
        Directory to place the star subdirectory.

    """

    if rootPath is None:
        rootPath = os.getcwd()

    if not os.path.basename == ID:
        path = os.path.join(*[rootPath, f'{ID}'])
    else:
        path = rootPath

    # Check if self.path exists, if not try to create it
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except Exception as ex:
            message = "Could not create directory for Star {0} because an exception of type {1} occurred. Arguments:\n{2!r}".format(ID, type(ex).__name__, ex.args)
            print(message)
    
    return path

def get_priorpath():
    """ Get default prior path name
    
    Returns
    -------
    prior_file : str
        Default path to the prior in the package directory structure.
    """
    
    return os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])


def clean_lc(lc):
    """ Perform Lightkurve operations on object.

    Performes basic cleaning of a light curve, removing nans, outliers,
    median filtering etc.

    Parameters
    ----------
    lc : Lightkurve.LightCurve instance
        Lightkurve object to be cleaned

    Returns
    -------
    lc : Lightkurve.LightCurve instance
        The cleaned Lightkurve object
        
    """

    lc = lc.remove_nans().flatten(window_length=4001).remove_outliers()

    return lc

def squish(time, dt, gapSize=27):
    """ Remove gaps

    Adjusts timestamps to remove gaps of a given size. Large gaps influence
    the statistics we use for the detection quite strongly.

    Parameters
    ----------
    gapSize : float
        Size of the gaps to consider, in units of the timestamps. Gaps
        larger than this will be removed. Default is 27 days.

    Returns
    -------
    t : array
        Adjusted timestamps
    """

    tsquish = time.copy()

    for i in np.where(np.diff(tsquish) > gapSize)[0]:
        diff = tsquish[i] - tsquish[i+1]

        tsquish[i+1:] = tsquish[i+1:] + diff + dt

    return tsquish

def nyquist(cad):
    """ Nyquist freqeuncy in muHz

    Parameters
    ----------
    cad : float
        Observational cadence in seconds.

    Returns
    -------
    Nyquist
        Nyquist frequency in muHz.
    """

    return 1/(2*cad) 


def _querySimbad(ID):
    """ Query any ID at Simbad for Gaia DR2 source ID.
    
    Looks up the target ID on Simbad to check if it has a Gaia DR2 ID.
    
    The input ID can be any commonly used identifier, such as a Bayer 
    designation, HD number or KIC.
    
    Returns None if there is not a valid cross-match with GDR2 on Simbad.
    
    Notes
    -----
    TIC numbers are note currently listed on Simbad. Do a separate MAST quiry 
    for this.
    
    Parameters
    ----------
    ID : str
        Target identifier. If Simbad can resolve the name then it should work. 
        
    Returns
    -------
    gaiaID : str
        Gaia DR2 source ID. Returns None if no Gaia ID is found.   
    """
    
    print('Querying Simbad for Gaia ID')

    try:
        job = Simbad.query_objectids(ID)
    except:
        print(f'Unable to resolve {ID} with Simbad')
        return None
    
    for line in job['ID']:
        if 'Gaia DR2' in line:
            return line.replace('Gaia DR2 ', '')
    return None

def _queryTIC(ID, radius = 20):
    """ Query TIC for bp-rp value
    
    Queries the TIC at MAST to search for a target ID to return bp-rp value. The
    TIC is already cross-matched with the Gaia catalog, so it contains a bp-rp 
    value for many targets (not all though).
    
    For some reason it does a cone search, which may return more than one 
    target. In which case the target matching the ID is found in the returned
    list. 
    
    Returns None if the target does not have a GDR3 ID.
    
    Parameters
    ----------
    ID : str
        The TIC identifier to search for.
    radius : float, optional
        Radius in arcseconds to use for the sky cone search. Default is 20".
    
    Returns
    -------
    bp_rp : float
        Gaia bp-rp value from the TIC.   
    """
    
    print('Querying TIC for Gaia bp-rp values.')
    job = Catalogs.query_object(objectname=ID, catalog='TIC', objType='STAR', 
                                radius = radius*units.arcsec)

    if len(job) > 0:
        idx = job['ID'] == str(ID.replace('TIC','').replace(' ', ''))
        return float(job['gaiabp'][idx] - job['gaiarp'][idx]) #This should crash if len(result) > 1.
    else:
        return None

def _queryMAST(ID):
    """ Query any ID at MAST
    
    Sends a query for a target ID to MAST which returns an Astropy Skycoords 
    object with the target coordinates.
    
    ID can be any commonly used identifier such as a Bayer designation, HD, KIC,
    2MASS or other name.
    
    Parameters
    ----------
    ID : str
        Target identifier
    
    Returns
    -------
    job : astropy.Skycoords
        An Astropy Skycoords object with the target coordinates.
    
    """

    print(f'Querying MAST for the {ID} coordinates.')
    mastobs = AsqMastObsCl()

    try:            
        return mastobs.resolve_object(objectname=ID)
    except:
        return None

def _queryGaia(ID=None, coords=None, radius=2):
    """ Query Gaia archive for bp-rp
    
    Sends an ADQL query to the Gaia archive to look up a requested target ID or
    set of coordinates. 
        
    If the query is based on coordinates a cone search will be performed and the
    closest target is returned. Provided coordinates must be astropy.Skycoords.
    
    Parameters
    ----------
    ID : str
        Gaia source ID to search for.
    coord : astropy.Skycoords
        An Astropy Skycoords object with the target coordinates. Must only 
        contain one target.
    radius : float, optional
        Radius in arcseconds to use for the sky cone search. Default is 20".
    
    Returns
    -------
    bp_rp : float
        Gaia bp-rp value of the requested target from the Gaia archive.  
    """
    
    from astroquery.gaia import Gaia

    if ID is not None:
        print('Querying Gaia archive for bp-rp values by target ID.')
        adql_query = "select * from gaiadr2.gaia_source where source_id=%s" % (ID)
        try:
            job = Gaia.launch_job(adql_query).get_results()
        except:
            return None
        return float(job['bp_rp'][0])
    
    elif coords is not None:
        print('Querying Gaia archive for bp-rp values by target coordinates.')
        ra = coords.ra.value
        dec = coords.dec.value
        adql_query = f"SELECT DISTANCE(POINT('ICRS', {ra}, {dec}), POINT('ICRS', ra, dec)) AS dist, * FROM gaiaedr3.gaia_source WHERE 1=CONTAINS(  POINT('ICRS', {ra}, {dec}),  CIRCLE('ICRS', ra, dec,{radius})) ORDER BY dist ASC"
        try:
            job = Gaia.launch_job(adql_query).get_results()
        except:
            return None
        return float(job['bp_rp'][0])
    else:
        raise ValueError('No ID or coordinates provided when querying the Gaia archive.')

def _format_name(ID):
    """ Format input ID
    
    Users tend to be inconsistent in naming targets, which is an issue for 
    looking stuff up on, e.g., Simbad. This function formats the ID so that 
    Simbad doesn't throw a fit.
    
    If the name doesn't look like anything in the variant list it will only be 
    changed to a lower-case string.
    
    Parameters
    ----------
    ID : str
        Name to be formatted.
    
    Returns
    -------
    ID : str
        Formatted name
        
    """

    ID = str(ID)
    ID = ID.lower()
    
    # Add naming exceptions here
    variants = {'KIC': ['kic', 'kplr', 'KIC'],
                'Gaia EDR3': ['gaia edr3', 'gedr3', 'edr3', 'Gaia EDR3'],
                'Gaia DR2': ['gaia dr2', 'gdr2', 'dr2', 'Gaia DR2'],
                'Gaia DR1': ['gaia dr1', 'gdr1', 'dr1', 'Gaia DR1'], 
                'EPIC': ['epic', 'ktwo', 'EPIC'],
                'TIC': ['tic', 'tess', 'TIC']
               }
    
    fname = None
    for key in variants:   
        for x in variants[key]:
            if x in ID:
                fname = ID.replace(x,'')
                fname = re.sub(r"\s+", "", fname, flags=re.UNICODE)
                fname = key+' '+str(int(fname))
                return fname 
    return ID

def get_bp_rp(ID):
    """ Search online for bp_rp values based on ID.
       
    First a check is made to see if the target is a TIC number, in which case 
    the TIC will be queried, since this is already cross-matched with Gaia DR2. 
    
    If it is not a TIC number, Simbad is queried to identify a possible Gaia 
    source ID. 
    
    As a last resort MAST is queried to provide the target coordinates, after 
    which a Gaia query is launched to find the closest target. The default 
    search radius is 20" around the provided coordinates. 
    
    Parameters
    ----------
    ID : str
        Target identifier to search for.
    
    Returns
    -------
    bp_rp : float
        Gaia bp-rp value for the target. Is nan if no result is found or the
        queries failed. 
    """
    
    time.sleep(1) # Sleep timer to prevent temporary blacklisting by CDS servers.
    
    ID = _format_name(ID)
    
    if 'TIC' in ID:
        bp_rp = _queryTIC(ID)          

    else:
        try:
            gaiaID = _querySimbad(ID)
            bp_rp = _queryGaia(ID=gaiaID)
        except:
            try:
                coords = _queryMAST(ID)
                bp_rp = _queryGaia(coords=coords)
            except:
                print(f'Unable to retrieve a bp_rp value for {ID}.')
                bp_rp = np.nan
    return bp_rp