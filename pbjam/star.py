"""

The `star' class is the core of PBjam and refers to a single target that is to 
be peakbagged. Each `star' instance is assigned an ID and physical input 
parameters, as well as a time series or power spectrum. 

The different steps in the peakbagging process are then passed the `star' 
instance, updating it with the results of each step. The outputs of each step
are stored in a dedicated directory created with the star ID.

The `session' class wraps one or more star class instances and peakbags them all
sequentially. The recommended use of PBjam is the use the `session' class, and
only use the `star' class for more granular control of the peakbagging process.

"""

from .peakbag import peakbag
from .jar import references
import pandas as pd
from pbjam import IO
from pbjam.modeID import modeIDsampler

class star():
    """ Class for each star to be peakbagged

    Additional attributes are added for each step of the peakbagging process

    Note spectrum is flattened (background divided out.)

    Examples
    --------
    Peakbag using the star class. Note that the star class only takes Lightkurve
    periodograms, pg, as spectrum input. 

    >>> st = pbjam.star(ID='KIC4448777', pg=pg, numax=[220.0, 3.0], 
                           dnu=[16.97, 0.01], teff=[4750, 100],
                           bp_rp = [1.34, 0.01])
    >>> st(make_plots=True)

    Parameters
    ----------
    ID : string, int
        Target identifier. If custom timeseries/periodogram is provided, it
        must be resolvable by LightKurve (KIC, TIC, EPIC, HD, etc.).
    pg : lightkurve.periodogram.Periodogram object
        A lightkurve periodogram object containing frequencies in units of
        microhertz and power (in arbitrary units).
    numax : list
        List of the form [numax, numax_error]. 
    dnu : list
        List of the form [dnu, dnu_error]. 
    teff : list
        List of the form [teff, teff_error]. 
    bp_rp : list
        List of the form [bp_rp, bp_rp_error]. 
    path : str, optional
        The path at which to store output. If no path is set but make_plots is
        True, output will be saved in the current working directory. Default is
        the current working directory.
    prior_file : str, optional
        Path to the csv file containing the prior data. Default is
        pbjam/data/prior_data.csv

    Attributes
    ----------
    f : array
        Array of power spectrum frequencies
    s : array
        power spectrum

    """

    def __init__(self, ID, f, s, addObs, outpath=None, priorpath=None):

        self.__dict__.update((k, v) for k, v in locals().items() if k not in ['self'])

        #self.references = references()

        #self.references._addRef(['numpy', 'python', 'astropy'])

        self.outpath = IO._set_outpath(ID, self.outpath)

        if priorpath is None:
            self.priorpath = IO.get_priorpath()
                
    # def run_peakbag(self, model_type='simple', tune=1500, nthreads=1,
    #                 make_plots=False, store_chains=False):
    def run_peakbag(self, modeID_result, snr=None, **kwargs):
        """  Run all steps involving peakbag.

        Performs fit using simple Lorentzian profile pairs to subsections of 
        the power spectrum, based on results from asy_peakbag.

        Parameters
        ----------
        model_type : str
            Can be either 'simple' or 'model_gp' which sets the type of mode
            width model. Defaults is 'simple'.
        tune : int, optional
            Numer of tuning steps passed to pm.sample. Default is 1500.
        nthreads : int, optional.
            Number of processes to spin up in pymc3. Default is 1.
        make_plots : bool, optional.
            Whether or not to produce plots of the results. Default is False.
        store_chains : bool, optional
            Whether or not to store posterior samples on disk. Default is False.

        """

        if (snr is None) and hasattr(self, 'snr'):
            snr = self.snr

        print('Starting peakbagging')


        self.peakbag = peakbag(self.f, snr, modeIDres=modeID_result)

        self.peakbag_sampler, self.peakbag_samples = self.peakbag(nlive=300)

        S = self.peakbag.unpackSamples(self.peakbag_samples)

        return S
        # Init
        # peakbag(self)

        # # Call
        # self.peakbag(model_type=model_type, tune=tune, nthreads=nthreads)

        # # Store
        # self.peakbag.summary.to_csv(IO._get_outpath(self, f'peakbag_summary_{self.ID}.csv'),
        #                             index_label='name')
            
        # if make_plots:
        #     self.peakbag.plot_spectrum(path=self.path, ID=self.ID,
        #                                savefig=make_plots)
        #     self.peakbag.plot_echelle(path=self.path, ID=self.ID, 
        #                               savefig=make_plots)
        #     self.references._addRef('matplotlib')

        # if store_chains:
        #     peakbag_samps = pd.DataFrame(self.peakbag.samples, columns=self.peakbag.par_names)
        #     peakbag_samps.to_csv(IO._get_outpath(self, f'peakbag_chains_{self.ID}.csv'), index=False)
            
    def run_modeID(self, addPriors={}, N_p=7, N_pca=100, PCAdims=8, **kwargs):
        
        self.modeID = modeIDsampler(self.f, self.s, self.addObs, addPriors, N_p=N_p, Npca=N_pca, PCAdims=PCAdims, priorpath=self.priorpath)

        self.modeID_sampler, self.modeID_samples = self.modeID()

        #TODO make storage and plots

        return self.modeID_sampler, self.modeID_samples

    def __call__(self, norders=8, modeID_kwargs={}, peakbag_kwargs={}):
        """ Perform all the PBjam steps

        Starts by running KDE, followed by Asy_peakbag and then finally peakbag.
        
        Parameters
        ----------
        bw_fac : float, optional.
            Scaling factor for the KDE bandwidth. By default the bandwidth is
            automatically set, but may be scaled to adjust for sparsity of the 
            prior sample. Default is 1.
        norders : int, optional.
            Number of orders to include in the fits. Default is 8.
        model_type : str, optional.
            Can be either 'simple' or 'model_gp' which sets the type of mode 
            width model. Defaults is 'simple'. 
        tune : int, optional
            Numer of tuning steps passed to pm.sample. Default is 1500.
        nthreads : int, optional.
            Number of processes to spin up in pymc3. Default is 1.
        make_plots : bool, optional.
            Whether or not to produce plots of the results. Default is False.
        store_chains : bool, optional
            Whether or not to store posterior samples on disk. Default is False.
        asy_sampling : string
            Method to be used for sampling the posterior in asy_peakbag. Options
            are 'emcee' or 'cpnest. Default method is 'emcee' that will call 
            emcee, alternative is 'cpnest' to call nested sampling with CPnest.
        developer_mode : bool
            Run asy_peakbag in developer mode. Currently just retains the input 
            value of dnu and numax as priors, for the purposes of expanding
            the prior sample. Important: This is not good practice for getting 
            science results!    
        """
 
        self.run_modeID(N_p=norders, **modeID_kwargs)

        samples_u = self.modeID.unpackSamples(self.modeID_samples)

        modeID_result = self.modeID.parseSamples(samples_u)

        muBkg = self.modeID.meanBkg(self.f, samples_u)

        self.snr = self.s / muBkg
 
        peakbag_summary = self.run_peakbag(modeID_result, **peakbag_kwargs)

        return peakbag_summary, modeID_result