import numpy as np
import scipy.interpolate as interpolate
import dynesty, jax
from msap import utils
from dynesty import utils as dyfunc
import jax.numpy as jnp
from functools import partial

class bkgSampler():
    def __init__(self, f, s, numax, lower_limit=10.):
        """ Sample the background noise in the PSD

        Samples the model parameters of the form given in Kallinger et al. (2014).

        The spectrum is first binned and the background is estimated based on 
        the binned spectrum. The binning is frequency dependent, and the 
        likelihood is adjusted accordingly.

        We use the dynesty sampler to estimate the posterior distribution of the
        model parameters.

        Parameters
        ----------
        f : array
            Full set of frequency bins from 0 to Nyquist.
        s : array
            Full power density spectrum.
        numax : tuple
            Estimated numax and error for the target.
        lower_limit : float, optional
            Lower frequency limit to consider in the background modelinger, 
            by default 10.
        """

        self.f = jnp.array(f)

        self.s = jnp.array(s)
        
        self.numax = numax
        
        self.lower_limit = lower_limit
        
        self.bkgPts, self.bkgVals, _, self.bins = self.get_bkg(self.f, self.s)
    
        envWidth = utils.env_width(self.numax[0])

        # Frequency array mask including the cut at low frequency
        envMask = (abs(self.bkgPts - self.numax[0]) < envWidth) | (self.bkgPts < self.lower_limit)

        self.mask = ~envMask
    
        self.eta = utils.attenuation(self.bkgPts[self.mask], self.f[-1])

        self.priors, self.labels = self.setPriors(self.bkgVals[self.mask])

        self.ndim = len(self.labels)

    def get_bkg(self, freq, power, a=0.66, b=0.88, skips=100):
        """ Estimate the background

        Takes an average of the power at logarithmically spaced points along the
        frequency axis, where the width of the averaging window increases as a
        power law. The default is that this averaging window increases as the
        scaling of the p-mode envelope width as a function of numax.

        Finally the mean power values are interpolated back onto the full
        frequency axis.

        Notes
        -----
        This is exactly the same as the get_bkg functions in the other 
        sub-modules except that it outputs a few other objects in addition to 
        the background estimate.

        Parameters
        ----------
        freq : array
            Frequency array of the AARPS.
        power : array
            Power array of the AARPS.
        a : float
            Power law factor to use for the background estimation. Default is
            0.66.
        b : float
            Power law exponent to use for the background estimation. Default is
            0.88.
        skips : float
            In the background estimation, compute the median at an interval of
            skips. Default is 100.

        Returns
        -------
        freq_bins : array
            Frequencies of the interval centers
        power_bins : array
            Power in each inteval center
        bkg : array
            Array of psd values approximating the background
        s : array
            Binning factor for used for each interval
        """

        freq_bins = np.exp(np.linspace(np.log(freq[0]), np.log(freq[-1]), skips))

        s = [len(power[np.abs(freq-fi) < a*fi**b]) for fi in freq_bins]

        power_bins = np.array([np.median(power[np.abs(freq-fi) < a*fi**b]) for fi in freq_bins])/np.log(2)

        mInt = interpolate.interp1d(freq_bins, power_bins, bounds_error=False)

        bkg = mInt(freq)/np.log(2)

        return jnp.array(freq_bins), jnp.array(power_bins), jnp.array(bkg), jnp.array(s)

    def __call__(self):
        """ Get background model samples

        Samples the posterior distribution of the binned AARPS given the
        background model.

        The model consists of two Harvey-like profiles to capture the 
        granulation contributions to the background, a low-frequency Harvey 
        profile to account for the frequency dependent instrumental variability,
        and a constant white noise term to account for the shot noise.

        We fit the model to the binned spectrum, where the width of each bin is
        given by the width of the notional oscillation envelope at that 
        frequency. This provides a good estimate of the background which also 
        partially disregards the contribution from the oscillation envelope.

        We also mask out a range around numax to further reduce any contribution
        from the p-modes to the model posterior. The lowest frequency bins of 
        the spectrum are also masked out.
        
        Returns
        -------
        samples : np.array
            Array of shape (nsteps, nwalkers, ndim) of the samples drawn during the
            sampling process.
        args: list
            Additional arguments used during the sampling. This is only relevant
            for plotting a few things later.    
        """

        sampler = dynesty.NestedSampler(self.lnlikelihood, 
                                        self.ptform, 
                                        self.ndim, 
                                        nlive=100, 
                                        sample='rwalk', 
                                        logl_args=[self.bkgPts[self.mask], 
                                                   self.bkgVals[self.mask], 
                                                   self.bins[self.mask],
                                                   self.eta])
            
        sampler.run_nested(print_progress=True)

        result = sampler.results

        unweighted_samples, weights = result.samples, jnp.exp(result.logwt - result.logz[-1])

        samples = dyfunc.resample_equal(unweighted_samples, weights)

        return sampler, samples

    def setPriors(self, p,):
        """ Set prior distributions for model parameters

        The priors are added to a dictionary as class instances, where each 
        distribution class has a ppf method that Dynesty calls.

        Parameters
        ----------
        p : jax device array
            The PSD (binned or not), used to determine average levels in at 
            various points in the spectrum.
         
        Returns
        -------
        priors : dict
            A dictionary of distribution class instances with methods that can be
            called to evaluate the logpdf of the distribution.
        """

        priors = {}

        # This is the ordering of the inputs to the model
        labels = ['b1', 'b2', 'a', 'w', 'c1', 'c2', 'bI', 'aI']

        priors['b1'] = self.set_b1()

        priors['b2'] = self.set_b2()

        priors['a'] = self.set_a(p,)

        priors['w'] = self.set_w(p,)

        priors['c1'] = self.set_c1()

        priors['c2'] = self.set_c2()

        priors['bI'] = self.set_bI()

        priors['aI'] = self.set_aI(p,)

        return priors, labels

    def set_b1(self,):
        """ Set the b2 prior

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.

        Notes
        -----
        This prior is set based on the scaling relation by Kallinger et al. (2014)
  
        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf, ppf and logpdf methods.
        """

        mu = 0.317 * self.numax[0] ** 0.97

        err = 0.3

        vmin = (1-err)*mu

        vmax = (1+err)*mu

        prior = utils.beta(a=1.2, b=1.2, scale=vmax-vmin, loc=vmin)

        return prior

    def set_b2(self,):
        """ Set the b2 prior 

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.

        Notes
        -----
        This prior is set based on the scaling relation by Kallinger et al. (2014).
        The margin for error is larger than the reported value though.

        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf and logpdf methods.

        """

        mu = 0.948 * self.numax[0] ** 0.992

        err = 0.6 # relative uncertainty on mu

        vmin = (1-err)*mu

        vmax = (1+err)*mu

        prior = utils.beta(a=1.2, b=1.2, scale=vmax-vmin, loc=vmin)

        return prior

    def set_bI(self,):
        """ Set the bI prior

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.

        This term is meant to absorb any contribution from instrumental background
        variability. No formal relation exists for this, so the distribution is
        simply set to encompas approximately 0 - 30 muHz.
    
        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf, ppf and logpdf methods.
        """

        mu = 10

        err = 0.99 # relative uncertainty on mu

        vmin = (1-err)*mu

        vmax = (1+err)*mu

        prior = utils.beta(a=1.2, b=1.2, loc=vmin, scale=vmax-vmin)

        return prior

    def set_aI(self,p,):
        """ Set the aI prior

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.

        This term is meant to absorb any contribution from instrumental background
        variability. No formal relation exists for this, so the distribution is
        simply set to encompas approximately 1 order of magnitude around the PSD
        of the first bin in the binned spectrum.

        Parameters
        ----------
        p : array
            The binned AARPS.

        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf, ppf and logpdf methods.
        """

        A = np.max([1, np.median(p[:2]) - p[-1]])

        mu = np.log10(np.sqrt(A*10))

        sigma = 1

        prior = utils.normal(mu=mu, sigma=sigma)

        return prior

    def set_a(self,p):
        """ Set the a prior

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.

        This is the amplitude (divided by 2 pi) of the two  Harvey-like profiles
        used to model contribution of the instrinsic stellar variability to the
        spectrum background.

        We find that the scaling relations by Kallinger et al. (2014) aren't very
        good for predicting the Harvey amplitude, so we simply take the difference
        in the PSD divided by the characeteristic granulation frequency as an
        estimate of a**2.

        Parameters
        ----------
        p : array
            The binned AARPS.
    
        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf, ppf and logpdf methods.
    
        """

        A = np.max([1, p[0]- p[-1]])

        b1 = 0.317 * self.numax[0] ** 0.97

        mu = np.log10(np.sqrt(A*b1))

        sigma = 1

        prior = utils.normal(mu=mu, sigma=sigma)

        return prior

    def set_w(self,p,):
        """ Set the w prior

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.

        This is the white noise (or shot noise) contribution to the background
        model. The prior is set by the PSD at the highest frequency of the binned
        spectrum.
    
        Parameters
        ----------
        p : array
            The binned AARPS.
    
        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf, ppf and logpdf methods.

        """

        mu = np.log10(p[-1])

        sigma = 1

        prior = utils.normal(mu=mu, sigma=sigma)

        return prior

    def set_c1(self,):
        """ Setup the c1 prior

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.

        This is the exponent parameter of the first Harvey-like profile. Based on
        Kallinger et al. (2014) and Karoff et al. (2013) we set the range of this
        parameter to be 2-5.

        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf, ppf and logpdf methods.
    
        """

        mu = 3.5

        vmin = mu-1.5

        vmax = mu+1.5

        prior = utils.beta(a=1.2, b=1.2, loc=vmin, scale=vmax-vmin)

        return prior

    def set_c2(self,):
        """ Setup the c2 prior

        Set a class instance with methods that can be called to evaluate the
        ppf of the distribution.
    
        This is the exponent parameter of the second Harvey-like profile. Kallinger et
        al. (2014) found this parameter to be ~4 for a sample of RG stars. However,
        we find that a value >4 is usually favored at least for the initial
        background model. This is similar to the findings in Karoff et al. (2013).
        We set the range of this parameter to be 2-9.

        Returns
        -------
        prior : object
            Distribution class instance. Must have the pdf, ppf and logpdf methods.
        
        """

        mu = 5.5

        vmin = mu-3.5

        vmax = mu+3.5

        prior = utils.beta(a=1.2, b=1.2, loc=vmin, scale=vmax-vmin)
    
        return  prior

    @partial(jax.jit, static_argnums=(0,))
    def harvey(self,f, a, b, c):
        """ Harvey-profile

        Parameters
        ----------
        f : np.array
            Frequency axis of the PSD.
        a : float
            The amplitude (divided by 2 pi) of the Harvey-like profile.
        b : float
            The characeteristic frequency of the Harvey-like profile.
        c : float
            The exponent parameter of the Harvey-like profile.

        Returns
        -------
        H : np.array
            The Harvey-like profile given the relevant parameters.
        """

        H = a**2/b * 1/(1+(f/b)**c)

        return H

    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta, f, eta):
        """ Background model

        The model used to estimate the background noise in the PSD.

        The model consists of three Harvey-like profiles, the first being due to
        the granulation signal at ~100-400 muHz, and the second being situated
        underneath the p-mode envelope.

        The instrumental contribution is esimated by a constant white noise term
        and another low frequency Harvey-like profile, where the characeteristic
        frequency is ~10muHz.

        We don't fit a Gausian profile for the p-mode envelope as this is masked
        out.

        Parameters
        ----------
        theta : list
            List of model parameters.
        f : np.array
            Frequency axis of the PSD.
        eta : np.array
            The attenuation of the PSD at each frequency due to the discrete
            sampling.

        Returns
        -------
        model : np.array
            The background model.

        """
        b1 = theta[0]

        b2 = theta[1]

        a = 10**theta[2]

        w = 10**theta[3]

        c1 = theta[4]

        c2 = theta[5]

        bI = theta[6]

        aI = 10**theta[7]

        # The first (low-frequency) Harvey profile
        H1 = self.harvey(f, a, b1, c1)

        # The first (high-frequency) Harvey profile
        H2 = self.harvey(f, a, b2, c2)

        # The attenuation correction is only applied to the granulation terms.
        # Not the instrumental noise terms.
        model = eta**2*(H1+H2) + w + self.harvey(f, aI, bI, 2)

        return model

    @partial(jax.jit, static_argnums=(0,))
    def lnlikelihood(self, theta, f, p, s, eta):
        """ log-likelihood function

        Estimates the probability of the model given the data.

        This likelihood is for the more general case of a binned spectrum.

        The binning factor is a list of value here, since we are fitting the
        result of a spectrum binned at different values over the range of
        frequencies.

        Notes
        -----
        The number of bins can become quite large, which causes the factorial term
        to explode, so we use the Ramanujan approximation to log(n!).

        Parameters
        ----------
        theta : list
            List of model parameters.
        f : np.array
            Frequency axis of the PSD.
        p : np.array
            The PSD spectrum.
        s : np.array
            The number of bins used to bin the spectrum at each frequency.
        eta : np.array
            The attenuation of the PSD at each frequency due to the discrete
            sampling.

        Returns
        -------
        lnlike : float
            The log-likelihood of the model.
        """

        M = self.model(theta, f, eta)
 
        lnlike = jnp.sum((s-1)*(jnp.log(s) + jnp.log(p)) - utils.lnfactorial(s-1) - s*(jnp.log(M)+p/M))

        return lnlike

    @partial(jax.jit, static_argnums=(0,))
    def ptform(self, u):
        
        """the prior transform function for the nested sampling
        
        Evaluates the ppf for a list of values drawn from the unit cube.
        
        Parameters
        ----------
        u : list
            List of floats between 0 and 1 with length equivalent to ndim. 
            
        Returns
        -------
        x : list
            List of floats of the prior pdfs evaluated at each point in u.
        """

        x = jnp.array([self.priors[key].ppf(u[i]) for i, key in enumerate(self.labels)])

        return x