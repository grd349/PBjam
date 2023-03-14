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

import re, time
from .peakbag import peakbag
from .jar import to_log10, references
from .plotting import plotting
import pandas as pd
import numpy as np
from astroquery.mast import ObservationsClass as AsqMastObsCl
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
import astropy.units as units
from pbjam import IO
from pbjam.modeID import modeIDsampler


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

    def __init__(self, ID, pg, obs, outpath=None, priorpath=None):

        self.ID = ID

        self.obs = obs
 
        self.references = references()

        self.references._addRef(['numpy', 'python', 'lightkurve', 'astropy'])
               
        self.pg = pg  
        
        self.f = self.pg.frequency.value
        
        self.s = self.pg.power.value

        self.log_obs = {x: to_log10(*self.obs[x]) for x in self.obs.keys() if x  not in ['bp_rp', 'eps']}

        IO._set_outpath(self, outpath)

        if priorpath is None:
            self.priorpath = IO.get_priorpath()
        else:
            self.priorpath = priorpath
                
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
        peakbag(self)

        # Call
        self.peakbag(model_type=model_type, tune=tune, nthreads=nthreads)

        # Store
        self.peakbag.summary.to_csv(IO._get_outpath(self, f'peakbag_summary_{self.ID}.csv'),
                                    index_label='name')
            
        if make_plots:
            self.peakbag.plot_spectrum(path=self.path, ID=self.ID,
                                       savefig=make_plots)
            self.peakbag.plot_echelle(path=self.path, ID=self.ID, 
                                      savefig=make_plots)
            self.references._addRef('matplotlib')

        if store_chains:
            peakbag_samps = pd.DataFrame(self.peakbag.samples, columns=self.peakbag.par_names)
            peakbag_samps.to_csv(IO._get_outpath(self, f'peakbag_chains_{self.ID}.csv'), index=False)
            
    def run_modeID(self, norders, Npca):
        M = modeIDsampler(self.f, self.s, self.obs, norders, priorpath=self.priorpath, Npca=Npca, freq_limits=[10,2750])

        sampler, samples = M()

        return samples

    def __call__(self, norders=8, model_type='simple', tune=1500,
                 nthreads=1, make_plots=True, store_chains=False, 
                 developer_mode=False):
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
 

        self.run_modeID(norders)

        self.run_peakbag(model_type=model_type, tune=tune, nthreads=nthreads,
                         make_plots=make_plots, store_chains=store_chains)

        self.references._addRef('pandas')

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