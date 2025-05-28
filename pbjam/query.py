## Convenience functions for looking up bp_rp (and teff),
## rescued from pbjam1

import time
from astroquery.mast import ObservationsClass as AsqMastObsCl
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

def _querySimbad(ID):
    """ Query simbad for Gaia DR2 source ID.
    
    Looks up the target ID on Simbad to check if it has a Gaia DR2 ID.
    
    The input ID can be any commonly used identifier, such as a Bayer 
    designation, HD number or KIC.
    
    Notes
    -----
    TIC numbers are note currently listed on Simbad. Do a separate MAST quiry 
    for this.
    
    Parameters
    ----------
    ID : str
        Target identifier.
        
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
    
    for line in job['id']: # as of astroquery >= 0.4.8, this is lowercase
        if 'Gaia DR2' in line:
            return line.replace('Gaia DR2 ', '')
    return None

def _queryTIC(ID, radius = 20):
    """ Find bp_rp in TIC
    
    Queries the TIC at MAST to search for a target ID to return bp-rp value. The
    TIC is already cross-matched with the Gaia catalog, so it contains a bp-rp 
    value for many targets (not all though).
    
    For some reason it does a cone search, which may return more than one 
    target. In which case the target matching the ID is found in the returned
    list. 
    
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
    
    print('Querying TIC for Gaia values.')
    job = Catalogs.query_object(objectname=ID, catalog='TIC', 
                                radius = radius*units.arcsec)

    if len(job) > 0:
        idx = job['ID'] == str(ID.replace('TIC','').replace(' ', ''))
        return {
            'bp_rp': float(job['gaiabp'][idx] - job['gaiarp'][idx]), #This should crash if len(result) > 1.
            'teff': float(job['Teff'][idx])
        }
    else:
        return None

def _queryMAST(ID):
    """ Query ID at MAST
    
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
        return mastobs.resolve_object(objectname = ID)
    except:
        return None

def _queryGaia(ID=None,coords=None, radius = 20):
    """ Query Gaia archive
    
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
    
    if ID is not None:
        adql_query = "select * from gaiadr2.gaia_source where source_id=%s" % (ID)
        try:
            job = Gaia.launch_job(adql_query).get_results()
        except:
            return None
    
    elif coords is not None:
        ra = coords.to_value()
        dec = coords.to_value()
        adql_query = f"SELECT DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS dist, * FROM gaiadr2.gaia_source WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius})) ORDER BY dist ASC"

        try:
            job = Gaia.launch_job(adql_query).get_results()
        except:
            return None
    else:
        raise ValueError('No ID or coordinates provided when querying the Gaia archive.')

    return {
        'bp_rp': job['bp_rp'][0],
        'teff': job['teff_val'][0]
    }

def _format_name(name):
    """ Format input ID
    
    Users tend to be inconsistent in naming targets, which is an issue for 
    looking stuff up on, e.g., Simbad. 
    
    This function formats the name so that Simbad doesn't throw a fit.
    
    If the name doesn't look like anything in the variant list it will only be 
    changed to a lower-case string.
    
    Parameters
    ----------
    name : str
        Name to be formatted.
    
    Returns
    -------
    name : str
        Formatted name
        
    """
        
    name = str(name)
    name = name.lower()
    
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
            if x in name:
                fname = name.replace(x,'')
                fname = re.sub(r"\s+", "", fname, flags=re.UNICODE)
                fname = key+' '+fname
                return fname
            
    return name
           
        
def get_spec(ID):
    """ Search online for bp_rp and Teff values based on ID.
       
    First a check is made to see if the target is a TIC number, in which case 
    the TIC will be queried, since this is already cross-matched with Gaia DR2. 
    
    If it is not a TIC number, Simbad is queries to identify a possible Gaia 
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
    props : dict
        Gaia bp-rp and Teff value for the target. Is nan if no result is found or the
        queries failed.
    
    """
    
    time.sleep(1)

    ID = _format_name(ID)
    
    if 'TIC' in ID:
        bp_rp = _queryTIC(ID)          

    else:
        try:
            gaiaID = _querySimbad(ID)
            res = _queryGaia(ID=gaiaID)
        except:
            try:
                coords = _queryMAST(ID)
                res = _queryGaia(coords=coords)
            except:
                print(f'Unable to retrieve a bp_rp and Teff value for {ID}.')
                res = None

    return res