import scipy.interpolate as interpolate
import numpy as np
from scipy.stats import chi2
import scipy.special as sc
import multiprocessing as mp
import scipy.stats as st
from pbjam import jar
from pbjam.jar import constants as c
from pbjam.jar import scalingRelations as sr



class detect():

    def __init__(self, f, s):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])
        pass

    def get_bkg(self, a=0.66, b=0.88, skips=100):
        """ Estimate the background

        Takes an average of the power at linearly spaced points along the
        log(frequency) axis, where the width of the averaging window increases
        as a power law.

        The mean power values are interpolated back onto the full linear
        frequency axis to estimate the background noise level at all
        frequencies.

        Returns
        -------
        b : array
            Array of psd values approximating the background.
        """

        freq_skips = np.exp(np.linspace(np.log(self.f[0]), np.log(self.f[-1]), skips))

        m = [np.median(self.s[np.abs(self.f-fi) < a*fi**b]) for fi in freq_skips]

        m = interpolate.interp1d(freq_skips, m, bounds_error=False)

        return m(self.f)/np.log(2)










class scalingRelations:
    """ Helper functions related to asteroseismic scaling relations """

    def pmode_env(self, f, numax, Amax, Teff):
        """ p-mode envelope as a function of frequency.

        The p-mode envelope is assumed to be a Guassian.

        Parameters
        ----------
        f : array
            Frequency bins in the spectrum, in muHz.
        numax : float
            Frequency to place the p-mode envelope at, in muHz.
        Amax : float
            Amplitude of the p-mode envelope at numax, in ppm.

        Returns
        -------
        envelope : array
            Predicted Guassian p-mode envelope, in ppm^2/muHz.
        """

        Henv = self.env_height(numax, Amax)

        stdenv = jar.scalingRelations.envWidth(numax, Teff) / 2 / np.sqrt(2*np.log(2))

        envelope = jar.gaussian(f, Henv, numax, stdenv)
                                
        return envelope

    # def env_beta(self, numax, Teff):
    #     """ Compute beta correction

    #     Computes the beta correction factor for Amax. This has the effect of
    #     reducing the amplitude for hotter solar-like stars that are close to
    #     the red edge of the delta-scuti instability strip, according to the
    #     observed reduction in the amplitude.

    #     This method was originally applied by Chaplin et al. 2011, who used a
    #     Delta_Teff = 1250K, this was later updated (private communcation) to
    #     Delta_Teff = 1550K.

    #     Parameters
    #     ----------
    #     numax : float
    #         Value of numax in muHz to compute the beta correction at.
    #     Teff0 : float, optional
    #         Solar effective temperature in K. Default is 5777K.
    #     TeffRed0 : float, optional
    #         Red edge temperature in K for a 1 solar luminosity star. Default is
    #         8907K.
    #     numax0: float, optional
    #         Solar numax. Default is 3050 muHz.
    #     Delta_Teff : float, optional
    #         The fall-off rate of the beta correction factor. Default is 1550K

    #     Returns
    #     -------
    #     beta : float
    #         The correction factor for Amax.
    #     """

    #     TeffRed = c.TeffRed0 * (numax/c.numax0)**0.11 * (Teff/c.Teff0)**-0.47

    #     beta = 1.0 - np.exp(-(TeffRed-Teff)/c.Delta_Teff)

    #     if isinstance(beta, (list, np.ndarray)):
    #         beta[beta<=0] = np.exp(-1250)

    #     elif beta <=0:
    #         beta = np.exp(-1250)

    #     return beta


    def env_Amax(self, numax):
        """ Compute Amax

        Computes the mode amplitude of a notional radial mode at nu_max, based
        on scaling relations.

        This includes the beta correction factor.

        Parameters
        ----------
        numax : float
            Value of numax in muHz to compute the beta correction at, in muHz.
        numax0: float, optional
            Solar numax. Default is 3050 muHz.
        Teff0 : float, optional
            Solar effective temperature. . Default is 5777K


        Returns
        -------
        Amax : float
            Amplitude in ppm of a radial order if it were exactly at nu_max.
        """

        if self.mission == 'TESS':
            V = 0.85
        else:
            V = 0.95

        beta = sr.env_beta(numax, self.Teff)

        Amax = V * beta * (numax/c.numax0)**-1 * (self.Teff/c.Teff0)**1.5

        return Amax # solar units

    def env_height(self, numax, Amax, alpha=0.791):
        """ Scaling relation for the envelope height

        Parameters
        ----------
        numax : float
            Frequency of maximum power of the p-mode envelope.
        Amax : float
            Envelope amplitude in ppm^2.
        numax0: float, optional
            Solar numax in muHz. Default is 3090 muHz.
        Teff0 : float, optional
            Solar effective temperature in K. Default is 5777 K.
        alpha : float
            Exponent of the dnu/numax scaling relation. Default is 0.804

        Returns
        -------
        Henv : float
            Height of the p-mode envelope
        """
        

        eta = np.sinc(0.5 * (numax/self.Nyquist))

        Henv = c.Henv0 * eta**2 * self.dilution**2 * (numax/jar.constants.numax0)**-alpha * Amax**2

        return Henv

    # def env_width(self, numax, Teff0=5777):
    #     """ Scaling relation for the envelope height

    #     Currently just a crude estimate. This can probably
    #     be improved.

    #     Full width at half maximum.

    #     Parameters
    #     ----------
    #     numax : float
    #         Frequency of maximum power of the p-mode envelope.
    #     Teff0 : float, optional
    #         Solar effective temperature in K. Default is 5777 K.

    #     Returns
    #     -------
    #     width : float
    #         Envelope width in muHz

    #     """
    #     if self.Teff <= 5600:
    #         width = 0.66*numax**0.88

    #     else:
    #         width = 0.66*numax**0.88*(1+(self.Teff-Teff0)*6e-4)

    #     return width

    # def dnuScale(self, nu, gamma=0.0, p=[0.79101684, -0.63285292]):
    #     """ Compute dnu from numax

    #     Computes an estimate of the large separation from a given value of numax,
    #     assuming the two parameters scale as a polynomial in log(dnu) and
    #     log(numax).

    #     The default is a linear function in log(dnu) and log(numax), estimated
    #     based on a polynomial fit performed on a set of main-sequence and sub-
    #     giant stars in the literature.

    #     The output may be scaled by a factor gamma, e.g., for setting credible
    #     intervals.

    #     Parameters
    #     ----------
    #     nu : float, array
    #         Value(s) at which to compute dnu, assuming nu corresponds to numax.
    #     gamma : float
    #         Scaling factor to apply.
    #     p : array-like, optional
    #         Polynomial coefficients to use for log(dnu), log(numax), starting with
    #         the coefficient of the Nth order term and ending at the bias term.

    #     Returns
    #     -------
    #     dnu : float, array
    #         Estimate of the large separation dnu
    #     """

    #     return 10**(np.polyval(p, np.log10(nu))+gamma)


class PEDetection(scalingRelations):
    """ Power excess detection

    The PEDetection class is used to estimate the probability that a frequency
    bin in the AARPS exibits power that is inconsistent with the background
    while being consistent with the power expected from the scaling relations.

    The variable list includes Parameters which are input parameters and
    Attributes, which are class attributes set when the class instance is
    either initalized or called.

    Parameters
    ----------
    freq : array
        Frequency array of the AARPS.
    power : array
        Power array of the AARPS.
    Teff : float
        Effective temperature of the star.
    Radius : float
        Radius of the star.
    Bin_width : float
        Value in muHz to bin the spectrum by.
    maskPoints : list
        List of tuples of the form (peak frequency, peak width) for each peak
        in the spectrum of a requested target.
    dilution : float
        Dilution parameter to scale the envelope height. This is <1 for targets
        with significant flux from other sources in the aperture. Default is 1.
    falseAlarm : float
        False alarm probability to use the the threshold prior.
    sigmaNumax : float
        Width of the log-normal function to use as a prior on numax.
    sigmaAmax : float
        Width of the normal distribution in log-Amax to use for the amplitude
        marginalization.
    Nint : int
        Number of bins to use to approximate the log-Amax normal distribution.
    mission : 'str'
        Set which mission the data is from. This is used to set the visibility
        of the modes. Currently undetermined for plato? Default is Kepler.

    Attributes
    ----------
    background :array
        Array of psd values approximating the background
    df :
        Frequency resolution of the unbinned spectrum.
    Nyquist :
        Nyquest frequency of the unbbined spectrum.
    Nbin :
        Number of bins to bin the spectrum by.
    fb : array
        Frequencies of the binned spectrum.
    pb : array
        Power of the binned spectrum.
    bb : array
        Background noise of the binned spectrum.
    dfb : float
        Frequency of the binned spectrum.
    Amax : array
        2D array of values for Amax, including the uncertainty at each test
        frequency.
    SNR : array
        Observed signal-to-noise ratio of the spectrum.
    SNRPred : array
        Predictd signal-to-noise ratio of the spectrum.
    dof : array
        The number of degrees of freedom at each test frequency.
    logProbabilities : dict
        Dictionary of arrays of log-probabilities pertaining to the calculation
        of the posterior on H1.
    TeffRed : float
        Red-edge temperature of the delta-scuti instability strip. Used to
        compute the beta correction factor.
    PH1 : array
        The posterior probability of H1
    PH1marg : array
        The posterior probability of H1 marginalized over the estimated Amax
        uncertainty.
    merit : array
        Value of the merit function at each test frequency. Used to assert a
        detection. Default is that this is identical to PH1marg
    numax_guess : float
        Very rough guess for where numax should be based on estimates of Teff
        and Radius. Using in computing the numax prior.
    numaxpdf : array
        Probability density function for numax.
    """


    def __init__(self, freq, power, Teff=None, Radius=None, Bin_width=1,
                 maskPoints=[], dilution=1, falseAlarm=0.01, sigmaNumax=0.5,
                 sigmaAmax=0.1, Nint=20, mission='Kepler'):

        # Spectrum
        self.f = freq
        self.p = power
        self.Nyquist = self.f.max()

        ## Background estimation
        self.background = self.get_bkg()

        # Masking artefact peaks
        self.p = self.applyMask(maskPoints)

        # Sort out any potential bins with nan values
        idxBad = np.isnan(self.p/self.background) | np.isinf(self.p/self.background)
        self.f = self.f[~idxBad]
        self.p = self.p[~idxBad]
        self.background = self.background[~idxBad]

        self.falseAlarm = falseAlarm

        self.sigmaNumax = sigmaNumax

        self.Teff = Teff

        self.Radius = Radius

        self.numax_guess = self.getNumax(self.Teff, self.Radius)

        self.sigmaAmax = sigmaAmax

        self.mission = mission

        self.dilution = dilution

        # Binning
        self.df = self.f[1]-self.f[0]
        self.Bin_width = max([Bin_width, self.df])
        self.Nbin = int(self.Bin_width / self.df)
        self.fb = self.bin(self.f)
        self.pb = self.bin(self.p)
        self.bb = self.bin(self.background)
        self.dfb = self.fb[1] - self.fb[0]

        # Marginalization integration points must be uneven
        self.Nint = Nint
        if self.Nint % 2 == 0:
            self.Nint += 1

        # Empties
        self.Amax = np.zeros((len(self.fb), self.Nint))
        self.SNR = np.zeros((len(self.fb), self.Nint))
        self.SNRPred = np.zeros((len(self.fb), self.Nint))
        self.dof = np.zeros(len(self.fb))
        self.logProbabilities = {key: np.zeros((len(self.fb), self.Nint)) for key in ['H0', 'H1', 'falseAlarm', 'numaxObs', 'wA']}


    def __call__(self):
        """ Call to compute detection merit function

        Computes the posterior probability of the H1 hypothesis as well as
        the probability density of numax.

        """

        self.numaxLogProbability()

        self.PH1, self.PH1marg = self.computePosterior(zeronan=True)

        self.merit = self.PH1marg

    def get_bkg(self, a=0.66, b=0.88, skips=100):
        """ Estimate the background

        Takes an average of the power at linearly spaced points along the
        log(frequency) axis, where the width of the averaging window increases
        as a power law.

        The mean power values are interpolated back onto the full linear
        frequency axis to estimate the background noise level at all
        frequencies.

        Returns
        -------
        b : array
            Array of psd values approximating the background.
        """

        freq_skips = np.exp(np.linspace(np.log(self.f[0]), np.log(self.f[-1]), skips))

        m = [np.median(self.p[np.abs(self.f-fi) < a*fi**b]) for fi in freq_skips]

        m = interpolate.interp1d(freq_skips, m, bounds_error=False)

        return m(self.f)/np.log(2)

    def applyMask(self, maskPoints):
        """ Apply mask to remove artefacts

        Masks out frequencies with known artefact frequencies and replaces it
        with random values drawn from an exponential distribution and scaled to
        the background estimate.

        This is only needed for testing on Kepler and TESS data.

        Parameters
        ----------
        maskePoints : list
            List of tuples, one for each peak to mask out. Each tuple consists
            a central frequency and the full-width of the range around that
            frequency to mask out.

        Returns
        -------
        pp : array
            Array of power with masked frequencies replaced with random values.
        """

        pp = self.p.copy()

        for i in range(len(maskPoints)):

            idx = abs(self.freq-maskPoints[i][0]) < maskPoints[i][1]/2

            pp[idx] = np.random.exponential(size=len(self.freq[idx]))*self.background[idx]

        return pp



    def bin(self, x):
        """ Bin x by a factor n

        If len(x) is not equal to an integer number of n, the remaining
        frequency bins are discarded. Half at low frequency and half at high
        frequency.

        Parameters
        ----------
        x : array
            Array of values to bin.

        Returns
        -------
        xbin : array
            The binned version of the input array
        """

        # Number of frequency bins per requested bin width
        n = int(self.Bin_width / self.df)

        # The input array isn't always an integer number of the binning factor
        # A bit of the input array is therefore trimmed a low and high end.
        trim = (len(x)//n)*n

        half_rest = (len(x)-trim)//2

        x = x[half_rest:half_rest+trim] # Trim the input array

        xbin = x.reshape((-1, n)).mean(axis = 1) # reshape and average

        return xbin

    def thresholdPrior(self, SNRPred, dof):
        """ Computes the false alarm probability prior

        Estimates where we can be reasonably sure that the predicted p-modes
        will be visible, given the observed background. This is then used as a
        prior.

        Parameters
        ----------
        SNRPred : array
            Array of predicted SNR values in the spectrum (Ptot_pred/Btot).
        dof : int
            Degress of freedom of the chi^2 distribution

        Returns
        -------
        logp_thresh : array
            The log of the threshold prior at each frequency.
        """

        # Find the value of x corresponding to a false alarm probability
        x = chi2.isf(self.falseAlarm, df=dof, scale=1./dof)

        # This is redundant given below, but for clarity
        SNRThresh = x-1

        # False alarm (threshold) prior
        logp_thresh = chi2.logsf((1+SNRThresh)/(1+SNRPred), df=dof,
                                 scale=1./dof)

        return logp_thresh

    def numaxLogProbability(self, Nsig=5):
        """ Computed numax log probability along frequency axis

        Computes the log-probabilities associated with estimating whether a
        p-mode envelope is present in the spectrum.

        These estimates are based on the scaling relations for the envelope
        height and width. Where the power in a predicted envelope is compared
        to the estimated noise level and the power predicted by the scaling
        relations.

        A range of envelope amplitudes are computed in order to marginalize over
        the uncertainty in the scaling relations. The uncertainty can be
        adjusted using the sigmaAmax argument when initializing the Detection
        class.

        A prior on numax from Gaia observations is applied and a prior based on
        the prediction having a low false alarm probability.

        Parameters
        ----------
        Nsig : int
            The amplitude marginalization is computed to +/-Nsig around the
            scaling relation. Nsig ~ 1 is probably insufficient as it might not
            capture the spread in true Amax values. Default is 5.

        """

        # Loop over all test frequencies
        for i, nu in enumerate(self.fb):

            # The numax prior and the H0 probabilities are independent of the
            # predicted envelope amplitude (no marginalization), so we compute
            # these first to get them out of the way. Note the scaling of the
            # width to 1.5*sigma
            width = 1.5 * jar.scalingRelations.envWidth(nu) / (2*np.sqrt(2*np.log(2)))

            # Set the range of the envelope in terms of frequency array indices.
            envRange = abs(self.fb-nu) < (width+self.dfb)

            # Total observed power in the range of the notial envelope
            Ptot = np.sum(self.pb[envRange]-self.bb[envRange])

            # Total background power in the range of the notial envelope
            Btot = np.sum(self.bb[envRange])

            # Degrees of freedom of the chi2 distribution
            dof = 2 * len(self.pb[envRange])*self.Nbin
            self.dof[i] = dof

            # Compute observed SNR
            self.SNR[i, :] = Ptot / Btot

            # Compute the probability of numax given photometric constraints.
            self.logProbabilities['numaxObs'][i, :] = self.obsNumaxPrior(nu)

            # Compute the H0 probability. Can maybe use chi2.logsf instead?
            self.logProbabilities['H0'][i, :] = chi2logpdf(1 + self.SNR[i,:], df=dof, scale=1./dof) # scalar -> array

            # We assume that the envelope amplitude at numax, Amax, generally
            # follows the scaling relation given by muAmax, but with some
            # scatter around it given by sigmaAmax, and that the scatter in
            # true values of Amax are symmetric around log(Amax) from the
            # scaling relations, and roughly Gaussian.
            muAmax = np.log10(self.env_Amax(nu))

            # Define the range of amplitudes to try out (integrate over). This
            # is centered on muAmax with a spread of 2*Nsig*sigmaAmax. Here Nsig
            # determines the number of sigma the amplitude range should span.
            self.Amax[i,:] = np.linspace(-Nsig * self.sigmaAmax, Nsig * self.sigmaAmax, self.Nint) + muAmax

            # The weights are assumed to be Gaussian distributed around muAmax.
            # The normalization can probably be computed outside the loop to
            # speed things a bit? Also This could potentially be turned into log-wA
            # to get rid of the exp call.
            self.logProbabilities['wA'][i, :] = np.exp(-(self.Amax[i,:] - muAmax)**2 / (2*self.sigmaAmax**2)) / np.sqrt(2*np.pi*self.sigmaAmax**2)

            # Compute the predicted p-mode envelope. Now, one for each value of
            # Amax that we want to integrate over. Note that pmode_env takes
            # linear values of Amax. The Amax array above is in log-amplitude.
            env = np.array([self.pmode_env(self.fb, nu, 10**a, self.Teff) for a in self.Amax[i,:]])

            # Compute the total power under the predicted envelope
            PtotPred = np.sum(env[:, envRange], axis = 1)

            # Compute the predicted SNR
            self.SNRPred[i, :] = PtotPred / Btot

            # Compute the H1 likelihood.
            self.logProbabilities['H1'][i, :] = chi2logpdf((1 + self.SNR[i, :]) / (1 + self.SNRPred[i, :]), df=dof, scale=1./dof)

            # Compute the false alarm (threshold) probability
            self.logProbabilities['falseAlarm'][i, :] = self.thresholdPrior(self.SNRPred[i, :], dof)


    def obsNumaxPrior(self, nu):
        """ Prior on numax based on non-seismic observations

        Evaluates the prior probability of observing a given value nu.

        Parameters
        ----------
        nu : float
            Frequency to evaluate the prior at
        numax_guess : float
            Prior distribution mean/center.
        sigma : float
            Prior distribution width

        Returns
        -------
        logp : float
            Logarithmic probability. Prior evaluated at nu.
        """

        logp = self.logNumaxPrior(nu, self.numax_guess, self.sigmaNumax)

        return logp


    def computePosterior(self, zeronan=False, Priors=[1,1]):
        """ Compute the posterior H1

        Assumes the posterior of H1 is given by
        P_H1 = pT*pN*pH1 / (pT*pN*pH1 + (1-pTpN)*pH0)

        When the probabilities are computed as log-probabilities, adding them up
        requires a bit of care to maintain numerical precision. The above
        equation may evaluate as 1 even when pH1 is very very small, so long as
        pH0 is smaller.

        Parameters
        ----------
        zeronan : bool
            Switch for setting all nan values in the resulting posterior array
            to 0.

        Returns
        -------
        PH1 : array
            The posterior probability (not log) of the H1 hypothesis for all the
            Amax guesses. The result is a 2D (M, N) array where M is the length
            of the power spectrum and N is the number of Amax values that are
            tested.
        PH1marg : array
            The posterior probability (not log) of the H1 hypothesis where
            the Amax uncertainty is marginalized. The result isa 1D array of
            length equal to the power spectrum.

        """

        #Adding in the 0.5 so that the priors tend to uninformative rather than
        # the opposite.
        if Priors == [1, 1]:
            logpT = self.logProbabilities['falseAlarm'] + np.log(0.5)
            logpN = self.logProbabilities['numaxObs']
        elif Priors == [1, 0]:
            logpN = self.logProbabilities['numaxObs'] + np.log(0.5)
            logpT = np.log(np.ones_like(self.logProbabilities['falseAlarm']))
        elif Priors == [0, 0]:
            logpN = np.log(np.ones_like(self.logProbabilities['numaxObs'])*0.5)
            logpT = np.log(np.ones_like(self.logProbabilities['falseAlarm']))
        elif Priors == [0, 1]:
            logpN = np.log(np.ones_like(self.logProbabilities['numaxObs'])*0.5)
            logpT = self.logProbabilities['falseAlarm']

        logpH0 = self.logProbabilities['H0']

        logpH1 = self.logProbabilities['H1']

        wA = self.logProbabilities['wA']

        X = -logpN - logpT + logpH0 - logpH1
        Y = logpH0 - logpH1

        PH1 = 1 / (1 + np.exp(X) - np.exp(Y))

        if zeronan:
            PH1 = np.nan_to_num(PH1, nan=0, posinf=0, neginf=0)

        # Marginalize over the probabilities weighted by wA
        PH1marg = np.trapz(wA*PH1, self.Amax, axis=1)

        X = self.logProbabilities['H1'] - self.logProbabilities['H0'] + self.logProbabilities['falseAlarm'] + self.logProbabilities['numaxObs']

        logLnumax= X[:, self.Nint//2]

        maxlogL = np.nanmax(logLnumax[~np.isinf(logLnumax)])

        logLprime = logLnumax-maxlogL

        numaxpdf = np.exp(logLprime)/np.nansum(np.exp(logLprime))/self.dfb

        self.numaxpdf = np.nan_to_num(numaxpdf, nan=0, posinf=0, neginf=0)

        return PH1, PH1marg

    def logNumaxPrior(self, nu, numaxGuess, sigmaNumax):
         """ Prior on numax

         The log-normal function in frequency for the prior on numax.

         Parameters
         ----------
         nu : float
             Frequency to evaluate the prior at, in muHz.
         numaxGuess : float
             Prior mean/center, in muHz.
         sigmaNumax : float
             Width of the log-normal function.

         Returns
         -------
         logp : float
             Logarithmic probability. Prior evaluated at nu.
         """

         p = np.exp(-(np.log(nu/numaxGuess))**2 / (2*sigmaNumax**2))

         return np.log(p)

    def getNumax(self, teff, R, alpha=0.804, teff0=5777, numax0=3050):
        """ Compute numax from scaling relations

        Computes an estimate of numax based on the 'massless' scaling relation.

        Parameters
        ----------
        teff : float
            Estimate of the target surface temperature in K.
        R : float
            Estimate of the target radius in solar radii.
        alpha : float, optional
            Exponent to use in the dnu \propto numax**n relation. Default is
            0.804.
        teff0 : float, optional
            Surface temperature of the Sun. Default is 5777K.
        numax0 : float, optional
            numax of the Sun. Default is 3050 muHz.

        Returns
        -------
        numax : float
            Scaling relation estimate of numax in muHz.
        """

        A = 0.5/(0.5 - alpha)

        B = -0.25/(0.5 - alpha)

        numax = numax0*R**A*(teff/teff0)**B

        return numax


def chi2logpdf(x, df, scale, normed=True):
    """ Compute log prob of chi2 dist.

    If normed=True this is equivalent to using the scipy.stats chi2 as
    chi2.logpdf(x, df=df, loc=0, scale=scale)

    If normed=False the normalization to unit area is discarded to speed
    up the computation. This is a fudge for using this with MCMC etc.

    Parameters
    ----------
    x : array
        Points at which to evaluate the log-pdf
    df : int
        Degrees of freedom of the chi2 distribution
    scale : float
        Scale factor for the pdf. Same as the scale parameter in the
        scipy.stats.chi2 implementation.

    Returns
    -------
    logpdf : array
        Log of the pdf of a chi^2 distribution with df degrees of freedom.

    """

    x /=scale

    if normed:
        return -(df/2)*np.log(2) - sc.gammaln(df/2) - x/2 - np.log(scale) + np.log(x)*(df/2-1)
    else:
        return np.log(x)*(df/2-1) - x/2
    



class RPDetection(scalingRelations):
    """ Computes the collapsed ACF of the filtered time series.

    Computes the ACF of the pass-band filtered time series by taking the inverse
    Fourier transform of the power spectrum. The S/N spectrum is used to reduce
    the effect of granulation noise on the resulting ACF.

    The ACF is computed with the pass-band filter placed at all the test
    frequencies. As the pass-band filter we use the Hanning function with a
    width equivalent to a notional p-mode envelope at the test frequency. The
    filter is proportional to cos^2 in then range of the envelope and zero
    otherwise. For each test frequency the pass-band filter is shifted and the
    ACF is computed.

    When the filter passes over the p-mode envelope, the ACF will show a peak
    at a time series lag proportional to 1/dnu. Otherwise the ACF will tend to
    a contant mean value.

    The average of the resulting ACF at each test frequency is computed to
    generate the collapsed ACF, to show the most likely location of numax.

    The variable list includes Parameters which are input parameters and
    Attributes, which are class attributes set when the class instance is
    either initalized or called.

    Parameters
    ----------
    freq : float, array
        Frequency axis of the S/N spectrum. The pass-band filter is iteratively
        applied at each frequency in this array.
    snr : float array
        S/N spectrum used to compute the ACF.
    Teff : float
        Effective temperature of the star in K. Used to estimate the envelope
        width.
    duty : float
        Duty cycle of the time series. This is used to scale the the mean
        background in the collapsed ACF such that the SNR of the collapsed ACF
        is ~1 for white noise.
    PE : object
        PEDetection class instance from MSAP3_01A. Contains the probabilities
        and meta data used to compute the merit function of the collapsed ACF.
    minNumax : float, optional
        Lower limit on numax to consider. Influences the range of tau to
        consider as well by way of the scaling relation between dnu and numax.
    maxNumax : float, optional
        Upper limit on numax to consider. Influences the range of tau to
        consider as well by way of the scaling relation between dnu and numax.
    timeLen : float, optional
        Maximum lag to consider in units of days. The spectrum will be binned
        to match this notional time series length.


    Attributes
    ----------
    tau : float, array
        ACF lags in seconds. Proportional to 1/dnu where dnu is in Hz
        (not muHz!)
    idxTau : bool
        Selection of lags, tau, to use. The numpy iFFT computes the double-sided
        inverse FFT, we only need the first half. Furthermore given a lower
        limit on numax, 100muHz for example, an upper limit on tau may be set
        through the dnu, numax relation.
    AvrgNumax : float, array
        1D average of the ACF collapsed along the dnu direction.
    scaleLims : float, array
        Limits on tau used at each test frequency to average the ACF map.
    Nh : array
        Number of frequency bins in the Hanning filter at each test frequency.
    Nheff : array
        Effective number of frequency bins in the Hanning filter at each
        test frequencym, accounting for a reduced width near the Nyquist
        frequency.
    merit : array
        Value of the merit function at each test frequency. Used to assert a
        detection.
    df : float
        Frequency resolution of the power spectrum
    dT : float
        Length of the time series
    Bin_width : float
        Width in frequency by which to bin the spectrum so that it approximates
        a time series of length timeLen
    tauMin : float
        Minimum tau value to consider, determined by numaxMax.
    tauMax : float
        Maximimum tau value to consider, determined by numaxMin.
    dnu : array
        Inverse of tau. Units are in muHz.
    PE : object
        PEdetection object from MSAP3_01
    cACFweights : array
        Weights to apply in the sum for the collapsed ACF along the numax
        axis, i.e., to get dnu.
    numaxPrior : array
        Prior on numax from external photometry and parallax etc.
    collapsedACF : dict
        Dictionary containing the resulting collapsed ACF along the numax and
        dnu axes.
    scaleLims : array
        Limits in dnu and numax to consider, such that the results are consistent
        with the scaling relation between dnu and numax.
    logLnumax : array
        log-likelihood for numax, determined by the collapsed ACF onto the numax
        axis.
    numaxpdf : array
        Probability density of numax on a linear scale, normalized to unit
        integral between 0 and the Nyquist frequencies. Note, this does not
        account for correlations between bins.
    detection : bool
        True if any point in the merit array exceeds a given threshold.
    """

    def __init__(self, freq, power, Teff, duty, NT, PE, minNumax=None, maxNumax=None,
                 timeLen=180):

        self.freq = freq
        self.power = power
        self.Teff = Teff
        self.duty = duty
        self.NT = NT
        self.Nyquist = self.freq.max()
        self.background = self.get_bkg()
        self.snr = self.power/self.background

        # Sort out any potential bins with nan values
        idxBad = np.isnan(self.snr) | np.isinf(self.snr)
        self.freq = self.freq[~idxBad]
        self.power = self.power[~idxBad]
        self.background = self.background[~idxBad]
        self.snr = self.snr[~idxBad]


        self.df = self.freq[1]-self.freq[0]
        self.dT = 1/(self.df*1e-6)/60/60/24

        self.Bin_width = 1/(timeLen*60*60*24)*1e6

        if self.Bin_width > self.df:
            self.freq = self.bin(self.freq)
            self.power = self.bin(self.power)
            self.snr = self.bin(self.snr)
            self.Bin_factor = (self.freq[1]-self.freq[0]) / self.df
        else:
            self.Bin_factor = 1


        # Set limits on numax, note this effectively puts a uniform prior on
        # the numax estimate.
        if minNumax is None:
            self.minNumax = self.freq[0]
        else:
            self.minNumax=minNumax

        if maxNumax is None:
            self.maxNumax = self.Nyquist
        else:
            self.maxNumax=maxNumax

         # lags to compute the ACF at
        self.tau = np.arange(len(self.snr))*(1/2/self.freq[-1])*1e6 + 1e-10 #add tiny offset to avoid problems at 0.


        # Above limits on numax correspond to limits in tau
        self.tauMin = 0
        self.tauMax = self.tau[len(self.tau)//2]

        self.idxTau = (self.tauMin < self.tau) & (self.tau < self.tauMax)
        self.tau = self.tau[self.idxTau]

        self.dnu = 1e6/self.tau[1:] # This is the dnu axis. The zero tau bin is discarded.


        # Use things from PE
        self.testFreqs = PE['IDP_128_POWER_EXCESS_METRICS']['Test frequency']
        self.numaxPrior = PE['IDP_128_POWER_EXCESS_METRICS']['numax prior']
        self.numaxPrior = self.numaxPrior[:, self.numaxPrior.shape[1]//2]

        PEmerit = PE['IDP_128_POWER_EXCESS_PROBABILITY']['MSAP3_01 merit']
        idxBad = np.isnan(PEmerit) | np.isinf(PEmerit)
        PEmerit[idxBad] = 0


        self.setcACFweights(PEmerit)


        # Empties
        self.collapsedACF = {'dnu': np.zeros_like(self.tau),
                             'numax': np.zeros_like(self.testFreqs)}
        self.Nheff = np.zeros(len(self.testFreqs), dtype=int)
        self.Ntaus = np.zeros(len(self.testFreqs), dtype=int)
        self.Nnu = np.zeros(len(self.tau), dtype=int)
        self.calibration = np.zeros(len(self.tau), dtype=float)

    def __call__(self, scaleInt=0.2):
        """ Call to detect pmodes

        Parameters
        ----------
        testFreqs : float, array
            The test frequencies to set the pass-band filter at and compute the
            collapsed ACF. MSAP3_01B doesn't know about the frequencies that are
            used in MSAP3_01A.
        threshold : float
            Threshold to use as a detection criterion.
        scaleInt : float
            The scaling of the dnu/numax relation use to define the interval
            used for computing the collapsed ACF along the dnu direction.  The
            default is 0.12
        """

        self.computeCollapsedACF(scaleInt)

        self.computeLikelihood()

    def setcACFweights(self, PEmerit):
        """ set cACF-dnu weights

        Sets the weights on the summation of the ACF when computing dnu.

        The are uniform in an interval determined by either the numax prior
        or the merit from the PE module.

        Parameters
        ----------
        PEmerit : np.array
            The merit function from the PE module.
        """

        if any(PEmerit > 0.5):
            cACFweights = PEmerit.copy()
        else:
            cACFweights = np.exp(self.numaxPrior)
        idxW = cACFweights >= 0.05
        cACFweights[idxW] = 1
        cACFweights[~idxW] = 0

        idxN = np.isnan(cACFweights) | np.isinf(cACFweights)
        cACFweights[idxN] = 0
        #cACFweights /= np.nansum(cACFweights)
        self.cACFweights = cACFweights

    def obsNumaxPrior(self, nu):
        """ Prior on numax based on non-seismic observations

        Evaluates the prior probability of observing a given value nu.

        Parameters
        ----------
        nu : float
            Frequency to evaluate the prior at
        numax_guess : float
            Prior distribution mean/center.
        sigma : float
            Prior distribution width

        Returns
        -------
        logp : float
            Logarithmic probability. Prior evaluated at nu.
        """

        logp = self.logNumaxPrior(nu, self.numax_guess, self.sigmaNumax)

        return logp




    def bin(self, inp):
        """ Bin x by a factor n

        If len(x) is not equal to an integer number of n, the remaining
        frequency bins are discarded. Half at low frequency and half at high
        frequency.

        Parameters
        ----------
        inp : array
            Array of values to bin.

        Returns
        -------
        xbin : array
            The binned version of the input array
        """

        x = inp.copy()

        # Number of frequency bins per requested bin width
        n = int(self.Bin_width / self.df)

        # The input array isn't always an integer number of the binning factor
        # A bit of the input array is therefore trimmed a low and high end.
        trim = (len(x)//n)*n

        half_rest = (len(x)-trim)//2

        x = x[half_rest:half_rest+trim] # Trim the input array

        xbin = x.reshape((-1, n)).mean(axis = 1) # reshape and average

        return xbin

    def _computeFilteredACF(self, W):
        """ Compute the ACF of the filtered S/N spectrum.

        Computes the autocorrelation of the S/N spectrum, with a pass-band
        filter W applied, via the inverse Fourier transform method.

        The result is normalized to unity at zero lag.

        Parameters
        ----------
        snr : array
            The S/N spectrum.
        W : array
            The pass-band filter to be applied to the S/N spectrum. Set to ones
            if no filtering should be used.

        Returns
        -------
        A : array
            The absolute squared autocorrelation, normalized to unity at zero
            lag.
        """

        C = np.fft.ifft(self.snr*W, n=len(self.snr))

        A = np.abs(C**2)/np.abs(C[0]**2)

        return A

    def _makeFilter(self, i, Nh):
        """ Create Hanning filter

        Creates an array of length equal to the frequency axis. The Hanning
        function acts as a pass-band filter proptional to cos^2, centered on the
        test frequency and with a width equal to the expected envelope width.
        Outside the range of the envelope the filter is 0.

        Parameters
        ----------
        i : int
            Index of the test frequency
        Nh : int
            Number of frequency bins inside the envelope

        Returns
        -------
        W : array
            The pass-band filter.
        Nheff : float
            Effective number of bins where W is != 0. This is reduced when the
            test frequency approaches the Nyquist frequency.
        """

        # Init a zero array for the filter. Can this be moved out to speed up?
        W = np.zeros_like(self.freq)

        i0 = max([0, i-int(Nh/2)])

        i1 = i+int(Nh/2)

        Nheff = len(W[i0:i1])

        # I feel like the filter width should be Nh, not Nheff, but that doesn't
        # seem to work so well at the high frequency end
        W[i0:i1] = np.hanning(Nheff)

        return W, Nheff



    def make1DACF(self, testFreq, Nh):
        """ Computes the 1D ACF of filtered TS

        Computes the 1D ACF with a pass-band filter placed at a frequency nu, to
        filter the time series. The filter has a total width derived based on
        the scaling relations for the envelope width.

        Parameters
        ----------
        testFreq : float
            Frequency at which to place the pass-band filter.
        Nh : int
            Number of frequency bins spanned by the filter.

        Returns
        -------
        A : float, array
            Residual of the time series ACF and the iFFT of the pass-band
            filter.
        """

        # Index in frequency array to place the filter
        i_nu = np.argmin(np.abs(self.freq-testFreq))

        # Create the Hanning filter at nu_i
        W, Nheff = self._makeFilter(i_nu, Nh)

        # Compute filtered ACF
        A = self._computeFilteredACF(W)

        return A[self.idxTau], max([Nheff, 1])

    def computeCollapsedACF(self, scaleInt):
        """ Compute collapsed ACF along dnu axis

        Computes the 1D ACF of the band-pass filtered time series at each test
        frequency. Takes the average along the dnu axis. Any repeating pattern
        will show up as an excess at the frequency where the pass-band filter
        was placed.

        Only the parts of the 1D ACF that are consistent with dnu for a given
        numax=test frequency are considered in the average. A seperate average
        is recorded for the range corresponding to 2*dnu, so that the harmonic
        may be analyzed as well.

        Parameters
        ----------
        scaleInt : float
            The scaling of the dnu/numax relation use to define the interval
            used for computing the collapsed ACF.
        """

        # Envelope widths for the filter
        widths = 6 * self.env_width(self.testFreqs) / (2*np.sqrt(2.*np.log(2.)))

        # Number of frequency bins in the envelope width
        self.Nh = (widths/np.median(np.diff(self.freq))).astype(int)

         # Compute interval to consider
        self.scaleLims = np.array([1./(self.dnuScale(self.testFreqs,  scaleInt))*1e6,
                                   1./(self.dnuScale(self.testFreqs, -scaleInt))*1e6])
        self.scaleLims[self.scaleLims >= self.tauMax] = self.tauMax


        # Set up multiprocessing workers and task queues
        nWorkers = 1 #mp.cpu_count()
        result = mp.Queue()
        task = mp.Queue()

        # The range of test freqs is divided into chuncks that each worker will
        # handle.
        L = len(self.testFreqs)//nWorkers

        # Init the workers
        workers = [mp.Process(target=self._workerTask, args=(task, result,
                                                             self.testFreqs,
                                                             self.collapsedACF,
                                                             self.Ntaus,
                                                             self.Nnu,
                                                             self.Nheff,
                                                             self.calibration)) for i in range(nWorkers)]

        # Start each worker sequentially
        for each in workers:
            each.start()

        # Give a unique frequency range to each worker
        for i in range(len(workers)):
            if i == len(workers)-1:
                task.put([i*L, len(self.testFreqs)])
            else:
                task.put([i*L, (i+1)*L])

        # Note sure what this bit does
        task.close()
        task.join_thread()

        # Get the results from the workers
        while nWorkers != 0:

            out = result.get()

            self.collapsedACF['numax'] += out[0]

            self.Nheff += out[1]

            self.Ntaus += out[2]

            self.collapsedACF['dnu'] += out[3]

            self.Nnu += out[4]

            self.calibration += out[5]

            nWorkers -=1


    def _workerTask(self, task, result, testFreqs, collapsedACF, Ntaus, Nnu, Nheff, calib):
        """ Task for each mp worker

        The task for each worker in the multiprocessing queue. For each test
        frequency in the range that the worker is given, the 1D ACF is computed
        and averaged in the range corresponding to the expected dnu.

        Parameters
        ----------
        task : multiprocessing.queue object
            The multiprocessing.queue object that provides the frequency range
            for the worker.
        result : multiprocessing.queue object
            The multiprocessing.queue object where the result should be stored.
        AvrgNumax : array
            Array of zeros. The results will be stored in the indices
            corresponding to the frequencies in testFreqs that the worker deals
            with.
        testFreqs : array
            The list of frequencies that the 1D ACF is computed at.
        Nheff : array
            Empty array to contain the effective number of frequency bins that
            the filter function spans.
        """

        # Get the start and end indices in testFreqs that the worker is
        # responsible for.
        st, sl = task.get()

        # Loop over each frequency
        for i in range(st, sl):

            # Make 1D ACF
            ACF, Nheff[i] = self.make1DACF(testFreqs[i], self.Nh[i])

            idxDnu = (self.scaleLims[0, i] < self.tau) & (self.tau < self.scaleLims[1, i])

            collapsedACF['numax'][i] = np.nanmean(ACF[idxDnu])

            Ntaus[i] = len(ACF[idxDnu])


            # Values outside the range of interest.
            idxS = np.isnan(ACF) | np.isinf(ACF) | (self.scaleLims[0, i] > self.tau) | (self.tau > self.scaleLims[1, i]) | (self.tau > self.tauMax) | (self.tau < self.tauMin) | (testFreqs[i] < self.minNumax) | (testFreqs[i] > self.maxNumax)
            ACF[idxS] = 0

            w = self.dnuCollapsedACFWeight(testFreqs[i], self.cACFweights[i], Mlim=1.4)

            w[idxS] = 0

            Nnu += np.invert(idxS).astype(int) #(w > 0).astype(int) #

            calib += ACF

            collapsedACF['dnu'] += ACF * w

        result.put([collapsedACF['numax'], Nheff, Ntaus, collapsedACF['dnu'], Nnu, calib])


    def dnuCollapsedACFWeight(self, testFreq, MSAP3_01_Weight, alpha=3.6,
                              dnu0=135.1, Teff0=5777, numax0=3050,
                              sigmaTeff=150, Mlim=2):
        """ Compute weights for collapsed ACF along test frequency to get dnu

        The weights applied to the ACF when collapsing along test frequency
        to get dnu are a combination of results from MSAP3_01 and a mass limit.

        The weight from MSAP3_01 can either be the prior or posterior
        probability density of numax.

        The mass limit acts to reduce the impact of mixed modes on the
        resulting probability density for dnu.

        Parameters
        ----------
        testFreq : np.array
            Array of test frequencies
        MSAP3_01_Weight : np.array
            Array of weights from MSAP3_01. This can either be the posterior
            probability density or the numax prior.
        alpha : float, optional
            Exponent on the mass in the dnu(M,Teff,numax) scaling relation.
            Normally this is 4, but can be changed to account for the variation
            of mass in numax. The default is 3.6.
        dnu0 : float, optional
            Solar dnu. The default is 135.1.
        Teff0 : float, optional
            Solar Teff. The default is 5777.
        numax0 : float, optional
            Solar numax. The default is 3050.
        sigmaTeff : float, optional
            Uncertainty in Teff. The default is 150.
        Mlim : float, optional
            Mass limit. The default is 2.

        Returns
        -------
        w : np.array
            Array of weights to apply when collapsing the ACF array.
        """

        dnuLim = Mlim**(-1/alpha) * (self.Teff/Teff0)**(3/(2*alpha)) * (testFreq/numax0)**(3/alpha) * dnu0

        dnu = 1/self.tau*1e6

        w = np.ones_like(self.tau)

        idx = dnu < dnuLim

        # # This is experimental, instead of a making a hard cut, the mass limit
        # # weight can be made smooth by accounting for uncertainty in Teff.
        # Teffprime = (dnu[idx]/dnu0)**(2*alpha/3) * Mlim**(2/3) * (testFreq/numax0)**(-2) * Teff0
        # w[idx] = np.exp(-(Teffprime - self.Teff)**2 / (2*sigmaTeff**2))
        # w *= MSAP3_01_Weight

        w = MSAP3_01_Weight * np.invert(idx).astype(int)

        return w

    def computeLikelihood(self):
        """ compute dnu and numax PDFs.

        Computes the PDF of numax first which is then used to compute the
        PDF for dnu. The PDF of numax is used to define the ranges where
        that are used as calibration points. The points are chosen such that
        they do not overlap with the expected dnu for the given numax pdf.

        """

        self.computenumaxPDF()

        numax_guess = getCurvePercentiles(self.testFreqs, self.numaxpdf, [0.5])

        self.computeDnuPDF(numax_guess)

    def computenumaxPDF(self):
        """ compute numax likelihoods and merit function from collapsed ACF

        Computes the merit function for a set test frequencies, which
        approximates the probability that an observed value of the collapsed ACF
        is inconsistent with noise.

        We assume that the noise is chi2 distributed with 2N degrees of freedom,
        where N is the number of bins in tau, that were used to compute the
        collapsed ACF onto the frequency axis.


        Parameters
        ----------
        bkg : array
            Estimate of the background level. Default is None.
        dof : array
            Degrees of freedom of the distribution of the collapsed ACF SNR at
            each test frequency.
        log : bool
            If log is True, returns the log probability. Not normalized to unit
            area. Otherwise returns the linear probability, normalized to unit
            area.
        a : float
            Scaling of the DOF of the chi^2 distribution used to compute the
            likelihood function of numax.
        """

        alpha, beta = self.getnumaxGammaPars()

        # This is the collapsed ACF SNR
        s = self.collapsedACF['numax']

        logLnumax = -st.gamma.logpdf(s, a=alpha, loc=0, scale=1/beta)

        self.logLnumax = logLnumax

        # Adding numax prior
        logLnumax += self.numaxPrior + np.log(0.5)

        idx = (self.testFreqs < self.minNumax) | (self.testFreqs > self.maxNumax)

        logLnumax[idx] = np.nan

        # Compute normalized merit function
        maxlogL = np.nanmax(logLnumax[~np.isinf(logLnumax)])

        logLprime = logLnumax-maxlogL

        self.merit = np.exp(logLprime - np.logaddexp(-maxlogL, logLprime))

        # Compute numax pdf
        numaxpdf = np.exp(logLprime)/np.nansum(np.exp(logLprime))/(self.testFreqs[1]-self.testFreqs[0])

        numaxpdf = np.nan_to_num(numaxpdf, nan=0, posinf=0, neginf=0)

        self.numaxpdf = numaxpdf

    def getnumaxGammaPars(self):
        """ Get alpha and beta for numax distribution.

        Based on the expected behavior of the mean and variance of the noise
        in the collapsed ACF we can compute the shape and scale parameters
        for the noise distribution which we expect to be Gamma distributed.

        Returns
        -------
        alpha : np.array
            Expected shape parameter for the Gamma distributed noise.
        beta : np.array
            Expected scale parameter for the Gamma distributed noise.
        """

        model_mu = 3/2 /self.Nheff / self.duty / self.Bin_factor

        alpha = (self.Nheff) * self.Ntaus/(self.NT+self.Ntaus) * self.Bin_factor

        beta = alpha/model_mu

        return alpha, beta


    def getDnuGammaPars(self, s, numax, bounds=np.array([[0.01, 2], [180, 250]]),
                        theta=np.array([1.81060231, 0.36093552, 1.54467413])):
        """ Get alpha and beta for dnu distribution.

        Based on the expected behavior of the mean and variance of the noise
        in the collapsed ACF we can compute the shape and scale parameters
        for the noise distribution which we expect to be Gamma distributed.

        Unlike for numax, the expected noise distribution requires some
        calibration.

        Parameters
        ----------
        numax : float
            Initial numax guess.
        bounds : 2d array
            bounds in frequency at which to estimate the scaling
        theta: array
            Parameters of model resulting from fit to simulations

        Returns
        -------
        alpha : np.array
            Expected shape parameter for the Gamma distributed noise.
        beta : np.array
            Expected scale parameter for the Gamma distributed noise.
        """

        a1, a2, b = theta

        mean = (b + (self.tau**a1 / self.Nnu)**a2)

        if 0.7 * self.dnuScale(numax) - bounds[0][1] < 0:
            bounds[0] += (1.3 * self.dnuScale(numax) - bounds[0][0])

        elif 1.3 * self.dnuScale(numax) - bounds[1][0] > 0:
            bounds[1] -= (bounds[1][1] - 0.7 * self.dnuScale(numax))

        self.calibpts = [(self.tau < 10**6 / x[0]) & (self.tau > 10**6 / x[1]) for x in bounds]

        percs = np.array([np.median(self.calibration[self.calibpts[i]]/self.Nnu[self.calibpts[i]]) for i in (0, 1)])

        obs   = np.array([np.median(mean[self.calibpts[i]]) for i in (0, 1)])

        idxGood = np.isfinite(obs) & np.isfinite(percs)

        ratio = np.nanmedian(obs[idxGood])/np.nanmedian(percs[idxGood])

        model_mu = mean / ratio

        model_var = (model_mu)**2

        beta = model_mu/model_var

        alpha = beta*model_mu

        return alpha, beta


    def computeDnuPDF(self, numax):
        """ Compute the probability density for dnu.

        Computes the probability density of dnu based on the expected
        distribution of noise from white noise simulations.

        Parameters
        ----------
        numax : np.float
            Guess for numax. Used to set calibration ranges.
        """

        s = self.collapsedACF['dnu']/self.Nnu

        alpha, beta = self.getDnuGammaPars(s, numax)

        logL = -gammalogsf(s, alpha, beta)

        self.logLdnu = logL

        # Normalizing
        maxlogL = np.nanmax(logL[~np.isinf(logL)])

        logLprime = logL - maxlogL

        #Normalize to unit integral in 1/tau.
        log_Delta_dnu = np.log(-np.diff(10**6/self.tau))

        dnupdf = np.exp(logLprime[1:] - np.log(np.nansum(np.exp(logLprime[1:] + log_Delta_dnu))))

        # Note self.dnupdf and self.nu are arrays ordered in increasing tau.
        self.dnupdf = dnupdf

def getCurvePercentiles(x, y, percentiles):
    """ Compute percentiles of value along a curve

    Computes the cumulative sum of y, normalized to unit maximum. The returned
    percentiles values are where the cumulative sum exceeds the requested
    percentiles.

    Parameters
    ----------
    x : array
        Support for y.
    y : array
        Array
    percentiles: array

    Returns
    -------
    percs : array
        Values of y at the requested percentiles.
    """

    cSumNorm = np.cumsum(y)/np.sum(y)

    percs = np.array([x[cSumNorm > p][0] for p in percentiles])

    return np.sort(percs)

def gammalogsf(x, alpha, beta):
    """ log-survival function of Gamma distribution

    Computes the survival function (1-CDF) of the Gamma distribution for a
    given set of shape and scale paramters alpha and beta.

    Parameters
    ----------
    x : np.array
        Array to evaluate the log-sf on.
    alpha : np.array
        Shape parameter of the Gamma distribution.
    beta : np.array
        Scale parameter of the Gamma distribution.

    Returns
    -------
    logsf : np.array
        The log of the survival function of the Gamma distribution.

    """

    def logfac(n):
        """ Compute approximate log(n!) """
        if n == 0:
            return 0
        else:
            return n*np.log(n) - n + (np.log(n*(1+4*n*(1+2*n))))/6 + np.log(np.pi)/2

    if np.isscalar(alpha):
        alpha = np.zeros(len(x)) + alpha

    if np.isscalar(beta):
        beta = np.zeros(len(x)) + beta

    a = np.floor(alpha - 1).astype(int)

    logsf = np.zeros_like(x)

    for r in range(1, a.max()):
        logsf[r<a] = np.logaddexp(logsf[r<a], r * np.log(x[r<a]) - logfac(r) + r * np.log(beta[r<a]))

    logsf -= beta*x

    return logsf

def chi2logpdf(x, df, scale, normed=True):
    """ Compute log prob of chi2 dist.

    If normed=True this is equivalent to using the scipy.stats chi2 as
    chi2.logpdf(x, df=df, loc=0, scale=scale)

    If normed=False the normalization to unit area is discarded to speed
    up the computation. This is a fudge for using this with MCMC etc.

    Parameters
    ----------
    x : array
        Points at which to evaluate the log-pdf
    df : int
        Degrees of freedom of the chi2 distribution
    scale : float
        Scale factor for the pdf. Same as the scale parameter in the
        scipy.stats.chi2 implementation.

    Returns
    -------
    logpdf : array
        Log of the pdf of a chi^2 distribution with df degrees of freedom.

    """

    x /=scale

    if normed:
        return -(df/2)*np.log(2) - sc.gammaln(df/2) - x/2 - np.log(scale) + np.log(x)*(df/2-1)
    else:
        return np.log(x)*(df/2-1) - x/2

def chi2logsf(x, df, scale):
    
    """ log-chi2 survival function

    The scipy.stats implimentation of the log of the chi^2 survival function
    doesn't work when the degrees of freedom and x are large. This is because it
    takes the log of the upper incomplete gamma function, which tends to 0 for
    large x. So it eventually hits the machine precision limit.

    The implimentation below rewrites the log of the integral using the upper
    incomplete gamma function recursion formula, and makes use of the
    scipy.special.logsumexp function and the Ramanujan approximation for
    log(n!). This allows the log of the survival function to be evaluated even
    for large x. Whether that makes sense is another issue.

    Notes
    -----
    This is only exactly equal to chi2.logsf when df is even. If df is odd you
    can use np.log(sc.chdtrc(df, x/scale)).

    This is slower than the scipy implemenation because of the recursion.

    Parameters
    ----------
    x : array
        Values to evaluate the function at
    df : int
        Degrees of freedom of the chi^2 distribution. Must be even.
    scale : int
        Scale parameter of x.

    Returns
    -------
    logsf : array
        The log of the chi^2 survival function evaluated at x.
    """

    def logfac(n):
        """ Compute approximate log(n!) """
        if n == 0:
            return 0
        else:
            return n*np.log(n) - n + (np.log(n*(1+4*n*(1+2*n))))/6 + np.log(np.pi)/2

    # Scale the x input
    x = x/(2*scale)

    if np.isscalar(df):
        df = np.zeros(len(x)) + df

    a = np.floor(0.5*df).astype(int)

    logsf = np.zeros_like(x)

    for r in range(1, a.max()):
        logsf[r<a] = np.logaddexp(logsf[r<a], r * np.log(x[r<a]) - logfac(r))

    logsf -= x

    return logsf




    

 