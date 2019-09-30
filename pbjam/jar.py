from . import PACKAGEDIR
import os

def get_priorpath():
    return os.path.join(*[PACKAGEDIR, 'data', 'prior_data.csv'])
