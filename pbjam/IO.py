"""

This module contains classes and functions for handling I/O related matters,
including downloading time series and computing power density spectra.

"""

from . import PACKAGEDIR
import jax.numpy as jnp
import numpy as np
from scipy.integrate import simpson
import os, time
import lightkurve as lk
from lightkurve.periodogram import Periodogram
from astropy.timeseries import LombScargle
from astropy import units
 
 
class psd():
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
    
    def __init__(self, ID, lk_kwargs={}, time=None, flux=None, flux_err=None, useWeighted=False, 
                 downloadDir=None, fit_mean=False, timeConversion=86400, badIdx=None, numax=None):
 
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        self.TS = timeSeries(ID, lk_kwargs, time, flux, flux_err, self.badIdx, self.downloadDir, numax=self.numax)
                          
        if useWeighted:
            # Init Astropy LS class with weights
            assert flux_err is not None, 'To compute the weighted spectrum, provide a flux_err argument.'

            self.ls = LombScargle(self.TS.time * self.timeConversion,
                                  self.TS.flux, center_data=True,
                                  fit_mean=self.fit_mean,
                                  dy=self.TS.flux_err,)
        else:
            # Init Astropy LS class without weights
            self.ls = LombScargle(self.TS.time * self.timeConversion,
                                    self.TS.flux, center_data=True,
                                    fit_mean=self.fit_mean)
   
        self.Nyquist = 1/(2 * self.timeConversion * self.TS.dt) # Hz

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
 
    

    def getTSWindowFunction(self, tmin=None, tmax=None, cadenceMargin=1.01):
        """
        Generates a time series window function.

        Parameters
        ----------
        tmin : float, optional
            Minimum time value for padding. If None, uses the minimum of self.time. Default is None.
        tmax : float, optional
            Maximum time value for padding. If None, uses the maximum of self.time. Default is None.
        cadenceMargin : float, optional
            Margin factor to identify gaps in the time series. Default is 1.01.

        Returns
        -------
        tuple
            A tuple containing the adjusted time array and the corresponding window function array.

        Notes
        -----
        - The method first initializes the time (`t`) and window function (`w`) arrays.
        - It then identifies gaps in the time series larger than `cadenceMargin * self.dt` and fills them with zeros in the window function.
        - The method ensures the length of the time series does not exceed a break counter of 100 to avoid infinite loops.
        - Padding is added at the start and end of the time series if `tmin` or `tmax` are specified and exceed the current bounds of `t`.
        """

        if tmin is None:
            tmin = min(self.TS.time)

        if tmax is None:
            tmax = max(self.TS.time)

        t = self.TS.time.copy() 

        w = np.ones_like(t)

        break_counter = 0

        epsilon = 0.0001 # this is a tiny scaling of dt to avoid numerical issues

        while any(np.diff(t) > cadenceMargin * self.TS.dt):

            idx = np.where(np.diff(t)>cadenceMargin*self.TS.dt)[0][0]

            t_gap_fill = np.arange(t[idx], t[idx+1]-epsilon*self.TS.dt, self.TS.dt)

            w_gap_fill = np.zeros(len(t_gap_fill))

            w_gap_fill[0] = 1

            t = np.concatenate((t[:idx], t_gap_fill, t[idx+1:]))

            w = np.concatenate((w[:idx], w_gap_fill, w[idx+1:]))

            break_counter +=1

            if break_counter == 100:
                break

        if (tmin is not None) and (tmin < t[0]):
            padLow = np.arange(tmin, t[0], self.TS.dt)

            t = np.append(padLow, t)
            
            w = np.append(np.zeros_like(padLow), w)

        if (tmax is not None) and (t[0] < tmax):
            padHi = np.arange(t[-1], tmax, self.TS.dt)
            
            t = np.append(t, padHi)
            
            w = np.append(w, np.zeros_like(padHi))

        return t, w

    

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
        df = 1/(self.timeConversion*(np.nanmax(self.TS.time) - np.nanmin(self.TS.time))) # Hz

        # Compute the window function
        freq, window = self.windowfunction(df, width=100*df, oversampling=5) # oversampling for integral accuracy

        # Integrate the windowfunction to get the corrected frequency resolution
        df = simpson(window, x=freq)

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

    def __init__(self, ID, lk_kwargs, time, flux, flux_err=None, badIdx=None, downloadDir=None, numax=None):
        
        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        
        if (self.time is None) and (self.flux is None):

            self.time, self.flux, self.flux_err = self._getTS()

            self.flux = (self.flux/np.nanmedian(self.flux) - 1) * 1e6

        self._getBadIndex(self.time, self.flux, self.flux_err, self.badIdx)

        self.goodIdx = np.invert(self.badIdx)

        self.time, self.flux = self.time[self.goodIdx], self.flux[self.goodIdx]

        if self.flux_err is not None:
            self.flux_err = self.flux_err[self.goodIdx]
        
        self.dt = self._getSampling()

        self.dT = self.time.max() - self.time.min()

        self.NT = len(self.time)

        self.dutyCycle = self._getDutyCycle()

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
 
        dutyCycle = len(self.time) / nomLen

        return dutyCycle
    
    def _getBadIndex(self, time, flux, flux_err, badIdx):
        """ Identify indices with nan/inf values

        Flags array indices where either the timestamps, flux values, or flux errors
        are nan or inf.
        """

        if badIdx is None:
            badIdx = np.zeros(len(time), dtype=bool)
        
        if flux_err is None:
            flux_err = np.zeros(len(time), dtype=bool)
        
        self.badIdx = np.invert(np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & np.invert(badIdx)) 
        
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
         
        dt = np.median(np.diff(self.time))

        return dt
    
    def _getTS(self, outlierRejection=5):
        """Get time series with lightkurve

        Parameters
        ----------
        outlierRejection : float, optional
            Number of sigma to use for sigma clipping in the time series cleaning.
            Used as input the the lightkurve remove_outliers method.

        Returns
        -------
        time : DeviceArray
            The timestamps of the time series.
        flux : DeviceArray
            The flux values of the time series.
        """
        
        # Force a short wait time to prevent multiple rapid requests from 
        # being rejected.
        time.sleep(np.random.uniform(1, 5))

        search = lk.search_lightcurve(self.ID, **self.lk_kwargs)
        
        LCcol = search.download_all(download_dir=self.downloadDir)

        lc = LCcol.stitch()
              
        lc = self.cleanLC(lc, outlierRejection)

        _time, _flux, _flux_err = jnp.array(lc.time.value), jnp.array(lc.flux.value), jnp.array(lc.flux_err.value)
 
        return _time, _flux, _flux_err
    
    def cleanLC(self, lc, outlierRejection):
        """ Perform Lightkurve operations on object.

        Performes basic cleaning of a light curve, removing nans, outliers,
        median filtering etc.

        Parameters
        ----------
        lc : Lightkurve.LightCurve instance
            Lightkurve object to be cleaned
        outlierRejection : float, optional
            Number of sigma to use for sigma clipping in the time series cleaning.
            Used as input the the lightkurve remove_outliers method.

        Returns
        -------
        lc : Lightkurve.LightCurve instance
            The cleaned Lightkurve object
        """

        if self.numax is None:
            wlen = int(4e6 / self.lk_kwargs['exptime'])
        else:
            wlen = int(1/(self.numax * 1e-3)*1e6 / self.lk_kwargs['exptime'])

        if wlen % 2 == 0:
            wlen += 1
            
        lc = lc.normalize().remove_nans().remove_outliers(outlierRejection).flatten(window_length=wlen) #remove_nans().flatten(window_length=4001).remove_outliers()

        return lc

def _getOutpath(self, fname):
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

def _setOutpath(name, rootPath):
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

    if not os.path.basename == name:
        path = os.path.join(*[rootPath, f'{name}'])
    else:
        path = rootPath

    # Check if self.path exists, if not try to create it
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except Exception as ex:
            message = "Could not create directory for Star {0} because an exception of type {1} occurred. Arguments:\n{2!r}".format(name, type(ex).__name__, ex.args)
            print(message)
    
    return path

def getPriorPath():
    """ Get default prior path name
    
    Returns
    -------
    prior_file : str
        Default path to the prior in the package directory structure.
    """
    
    return os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])


