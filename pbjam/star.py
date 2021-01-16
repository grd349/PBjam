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

import os, warnings, re, time, numbers
from .asy_peakbag import asymptotic_fit
from .priors import kde
from .peakbag import peakbag
from .jar import get_priorpath, to_log10, references
from .plotting import plotting
import pandas as pd
import numpy as np
from astroquery.mast import ObservationsClass as AsqMastObsCl
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
import astropy.units as units

import logging
from .jar import log, file_logging

logger = logging.getLogger(__name__)  # For module-level logging
logger.debug('Initialized module logger.')


class star(plotting):
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
    def __init__(self, ID, pg, numax, dnu, teff=[None,None], bp_rp=[None,None], 
                 path=None, prior_file=None):
        self.ID = ID

        if numax[0] < 25:
            warnings.warn('The input numax is less than 25. The prior is not well defined here, so be careful with the result.')
        self.numax = numax
        self.dnu = dnu

        self.references = references()
        self.references._addRef(['numpy', 'python', 'lightkurve', 'astropy'])
        
        teff, bp_rp = self._checkTeffBpRp(teff, bp_rp)
        self.teff = teff
        self.bp_rp = bp_rp

        self.pg = pg.flatten()  # in case user supplies unormalized spectrum
        self.f = self.pg.frequency.value
        self.s = self.pg.power.value

        self._obs = {'dnu': self.dnu, 'numax': self.numax, 'teff': self.teff,
                     'bp_rp': self.bp_rp}
        self._log_obs = {x: to_log10(*self._obs[x]) for x in self._obs.keys() if x != 'bp_rp'}

        self._set_outpath(path)

        if prior_file is None:
            self.prior_file = get_priorpath()
        else:
            self.prior_file = prior_file

        logger.info(f"Initialized star with ID {self.ID}.")

    def _checkTeffBpRp(self, teff, bp_rp):
        """ Set the Teff and/or bp_rp values
        
        Checks the input Teff and Gbp-Grp values to see if any are missing.
        
        If Gbp-Grp is missing it will be looked up online either from the TIC or 
        the Gaia archive.
        
        Teff and Gbp-Grp provide a lot of the same information, so only one of
        them need to be provided to start with. If one is not provided, PBjam
        will assume a wide prior on it.
        
        Parameters
        ----------
        teff : list
            List of the form [teff, teff_error]. For multiple targets, use a list
            of lists.
        bp_rp : list
            List of the form [bp_rp, bp_rp_error]. For multiple targets, use a list
            of lists.
        
        Returns
        -------
        teff : list
            The checked teff value. List of the form [teff, teff_error]. 
        bp_rp : list
            The checked bp_rp value. List of the form [bp_rp, bp_rp_error]. 
        """
        
        if not isinstance(bp_rp[0], numbers.Real):
            bp_rp = [get_bp_rp(self.ID), 0.1]
          
        teff_good = isinstance(teff[0], numbers.Real)
        bprp_good = isinstance(bp_rp[0], numbers.Real)
        
        if not teff_good and not bprp_good:
            raise ValueError('Must provide either teff or bp_rp arguments when initializing the star class.')
        elif not teff_good:
            teff = [4889, 1500] # these are rough esimates from the prior
        elif not bprp_good:
            bp_rp = [1.2927, 0.5] # these are rough esimates from the prior
            
        self.references._addRef(['Evans2018'])
        
        return teff, bp_rp

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

    def _set_outpath(self, path):
        """ Sets the path attribute for star

        If path is a string it is assumed to be a path name, if not the
        current working directory will be used.

        Attempts to create an output directory for all the results that PBjam
        produces. A directory is created when a star class instance is
        initialized, so a session might create multiple directories.

        Parameters
        ----------
        path : str
            Directory to place the star subdirectory.

        """

        if isinstance(path, str):
            # If path is str, presume user wants to put stuff somewhere specific.
            self.path = os.path.join(*[path, f'{self.ID}'])
        else:
            # Otherwise just create a subdir in cwd.
            self.path = os.path.join(*[os.getcwd(), f'{self.ID}'])

        # Check if self.path exists, if not try to create it
        if not os.path.isdir(self.path):
            try:
                os.makedirs(self.path)
            except Exception as ex:
                message = "Could not create directory for Star {0} because an exception of type {1} occurred. Arguments:\n{2!r}".format(self.ID, type(ex).__name__, ex.args)
                print(message)



    def run_kde(self, bw_fac=1.0, make_plots=False, store_chains=False):
        """ Run all steps involving KDE.

        Starts by creating a KDE based on the prior data sample. Then samples
        this KDE for initial starting positions for asy_peakbag.

        Parameters
        ----------
        bw_fac : float
            Scaling factor for the KDE bandwidth. By default the bandwidth is
            automatically set, but may be scaled to adjust for sparsity of the
            prior sample.
        make_plots : bool, optional
            Whether or not to produce plots of the results. Default is False.
        store_chains : bool, optional
            Whether or not to store posterior samples on disk. Default is False.

        """

        print('Starting KDE estimation')
        
        # Init
        kde(self)

        # Call
        self.kde(dnu=self.dnu, numax=self.numax, teff=self.teff,
                 bp_rp=self.bp_rp, bw_fac=bw_fac)

        # Store          
        if make_plots:
            self.kde.plot_corner(path=self.path, ID=self.ID,
                                 savefig=make_plots)
            self.kde.plot_spectrum(pg=self.pg, path=self.path, ID=self.ID,
                                   savefig=make_plots)
            self.kde.plot_echelle(path=self.path, ID=self.ID,
                                  savefig=make_plots)
            
            self.references._addRef('matplotlib')

        if store_chains:
            kde_samps = pd.DataFrame(self.kde.samples, columns=self.kde.par_names)
            kde_samps.to_csv(self._get_outpath(f'kde_chains_{self.ID}.csv'), index=False)
            

    def run_asy_peakbag(self, norders, make_plots=False,
                        store_chains=False, method='emcee', 
                        developer_mode=False):
        """ Run all steps involving asy_peakbag.

        Performs a fit of the asymptotic relation to the spectrum (l=2,0 only),
        using initial guesses and prior for the fit parameters from KDE.

        Parameters
        ----------
        norders : int
            Number of orders to include in the fits.
        make_plots : bool, optional
            Whether or not to produce plots of the results. Default is False.
        store_chains : bool, optional
            Whether or not to store posterior samples on disk. Default is False.
        method : string
            Method to be used for sampling the posterior. Options are 'emcee' or
            'cpnest. Default method is 'emcee' that will call emcee, alternative
            is 'cpnest' to call nested sampling with CPnest.
        developer_mode : bool
            Run asy_peakbag in developer mode. Currently just retains the input 
            value of dnu and numax as priors, for the purposes of expanding
            the prior sample. Important: This is not good practice for getting 
            science results!
            
        """

        print('Starting asymptotic peakbagging')
        # Init
        asymptotic_fit(self, norders=norders)

        # Call
        self.asy_fit(method, developer_mode)
        self.references._addRef(method)

        # Store
        self.asy_fit.summary.to_csv(self._get_outpath(f'asymptotic_fit_summary_{self.ID}.csv'),
                                    index=True, index_label='name')
        self.asy_fit.modeID.to_csv(self._get_outpath(f'asymptotic_fit_modeID_{self.ID}.csv'),
                                   index=False)
        if make_plots:
            self.asy_fit.plot_spectrum(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.asy_fit.plot_corner(path=self.path, ID=self.ID,
                                     savefig=make_plots)
            self.asy_fit.plot_echelle(path=self.path, ID=self.ID,
                                      savefig=make_plots)
            self.references._addRef('matplotlib')

        if store_chains:
            asy_samps = pd.DataFrame(self.asy_fit.samples, columns=self.asy_fit.par_names)
            asy_samps.to_csv(self._get_outpath(f'asymptotic_fit_chains_{self.ID}.csv'), index=False)

    
    def run_peakbag(self, model_type='simple', tune=1500, nthreads=1,
                    make_plots=False, store_chains=False):
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

        print('Starting peakbagging')
        # Init
        peakbag(self, self.asy_fit)

        # Call
        self.peakbag(model_type=model_type, tune=tune, nthreads=nthreads)

        # Store
        self.peakbag.summary.to_csv(self._get_outpath(f'peakbag_summary_{self.ID}.csv'),
                                    index_label='name')
            
        if make_plots:
            self.peakbag.plot_spectrum(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.peakbag.plot_echelle(path=self.path, ID=self.ID, 
                                      savefig=make_plots)
            self.references._addRef('matplotlib')

        if store_chains:
            peakbag_samps = pd.DataFrame(self.peakbag.samples, columns=self.peakbag.par_names)
            peakbag_samps.to_csv(self._get_outpath(f'peakbag_chains_{self.ID}.csv'), index=False)
            
    # def add_file_handler(self):
    #     logger = logging.getLogger('pbjam')  # <--- logs everything under pbjam
    #     fpath = os.path.join(self.path, 'star.log')
    #     self.handler = logging.FileHandler(fpath)
    #     self.handler.setFormatter(HANDLER_FMT)
    #     logger.addHandler(self.handler)
    
    # def remove_file_handler(self)
    #     logger = logging.getLogger('pbjam')
    #     logger.handlers.remove(self.handler)

    def __call__(self, bw_fac=1.0, norders=8, model_type='simple', tune=1500,
                 nthreads=1, make_plots=True, store_chains=False, 
                 asy_sampling='emcee', developer_mode=False):
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
        # self.add_file_handler()
        with file_logging(self.path):
            self.run_kde(bw_fac=bw_fac, make_plots=make_plots, store_chains=store_chains)

            self.run_asy_peakbag(norders=norders, make_plots=make_plots,
                                store_chains=store_chains, method=asy_sampling,
                                developer_mode=developer_mode)

            self.run_peakbag(model_type=model_type, tune=tune, nthreads=nthreads,
                            make_plots=make_plots, store_chains=store_chains)

            self.references._addRef('pandas')

         # self.remove_file_handler()

@log(logger)
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
    
    logger.debug('Querying Simbad for Gaia ID.')

    try:
        job = Simbad.query_objectids(ID)
    except:
        logger.debug(f'Unable to resolve {ID} with Simbad.')
        return None
    
    for line in job['ID']:
        if 'Gaia DR2' in line:
            return line.replace('Gaia DR2 ', '')
    return None

@log(logger)
def _queryTIC(ID, radius = 20):
    """ Query TIC for bp-rp value
    
    Queries the TIC at MAST to search for a target ID to return bp-rp value. The
    TIC is already cross-matched with the Gaia catalog, so it contains a bp-rp 
    value for many targets (not all though).
    
    For some reason it does a cone search, which may return more than one 
    target. In which case the target matching the ID is found in the returned
    list. 
    
    Returns None if the target does not have a GDR2 ID.
    
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
    
    logger.debug('Querying TIC for Gaia bp-rp values.')
    job = Catalogs.query_object(objectname=ID, catalog='TIC', objType='STAR', 
                                radius = radius*units.arcsec)

    if len(job) > 0:
        idx = job['ID'] == str(ID.replace('TIC','').replace(' ', ''))
        return float(job['gaiabp'][idx] - job['gaiarp'][idx]) #This should crash if len(result) > 1.
    else:
        return None

@log(logger)
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

    logger.debug(f'Querying MAST for the {ID} coordinates.')
    mastobs = AsqMastObsCl()
    try:            
        return mastobs.resolve_object(objectname = ID)
    except:
        return None

@log(logger)
def _queryGaia(ID=None,coords=None, radius = 20):
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
    
    logger.debug('Querying Gaia archive for bp-rp values.')
    
    from astroquery.gaia import Gaia

    if ID is not None:
        adql_query = "select * from gaiadr2.gaia_source where source_id=%s" % (ID)
        try:
            job = Gaia.launch_job(adql_query).get_results()
        except:
            logger.debug(f'Unable to query Gaia archive using ID={ID}.')
            return None
        return float(job['bp_rp'][0])
    
    elif coords is not None:
        ra = coords.to_value()
        dec = coords.to_value()
        adql_query = f"SELECT DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS dist, * FROM gaiadr2.gaia_source WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) ORDER BY dist ASC"

        try:
            job = Gaia.launch_job(adql_query).get_results()
        except:
            logger.debug('Unable to query Gaia archive using coords={coords}.')
            return None
        return float(job['bp_rp'][0])
    else:
        raise ValueError('No ID or coordinates provided when querying the Gaia archive.')

@log(logger)
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

@log(logger)
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
            except Exception as exc:
                # Note that logger.exception gives the full Traceback or just set exc_info
                logger.debug(f'Exception: {exc}.', exc_info=1)
                logger.warning(f'Unable to retrieve a bp_rp value for {ID}.')
                bp_rp = np.nan

    return bp_rp
