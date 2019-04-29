""" Peakbagging individual peaks

This module fits a model of lorentzians to the p-modes identified by the
asymptotic peakbag module. It can treat as many modes as necessary, given they
have an ID.
"""

import numpy as np
import pandas as pd
import os
import emcee
import matplotlib.pyplot as plt
import warnings

from . import PACKAGEDIR

def create_asterostan():
    pbstan = '''
    functions{
        real lorentzian(real loc, int l, int m, real f, real eps, real H, real w, real nus){
            return (eps * H) ./ (1 + (4/w^2) * (f - loc + m*nus)^2);
        }
    }
    data{
        int N;            // Number of data points
        int M;            // Number of modes
        vector[N] f;      // Frequency
        vector[N] p;      // Power
        real asy_locs[M]; // Mode locations (this will have to change for multiple n modes)
        int asy_ids[M];   // The ID's of the modes
    }
    parameters{
        real logAmp[M];         // Mode amplitude in log space
        real logGamma[M];       // Mode linewidth in log space
        real<lower=0> locs[M];  // True mode locations
        real<lower=0> vsini;    // Line of sight rotational frequency
        real<lower=0> nus;      // Rotational frequency splitting
    }
    transformed parameters{
        real sini;       // Sin of angle of inclination (rad)
        real i;          // Angle of inclination (rad)
        real H[M];       // Mode height
        real w[M];       // Mode linewidth
        matrix[4,4] eps; // Matrix of legendre polynomials
        eps = rep_matrix(i, 4, 4);

        sini = vsini / nus;       // Transform sin of angle of inclination from line of sight rotation frequency
        i = asin(sini);
        for (m in 1:M){
            w[m] = 10^logGamma[m];             // Transform mode linewidth from log space
            H[m] = 10^logAmp[m] / pi() / w[m]; // Transform mode amplitude to mode height
        }

        // Now I'll calculate all the legendre polynomials for this i
        eps[0+1,0+1] = 1.;
        eps[1+1,0+1] = cos(i)^2;
        eps[1+1,1+1] = 0.5 * sin(i)^2;
        eps[2+1,0+1] = 0.25 * (3. * cos(i)^2 - 1.)^2;
        eps[2+1,1+1] = (3./8.)*sin(2*i)^2;
        eps[2+1,2+1] = (3./8.) * sin(i)^4;
        eps[3+1,0+1] = (1./64.)*(5.*cos(3.*i) + 3.*cos(i))^2;
        eps[3+1,1+1] = (3./64.)*(5.*cos(2.*i) + 3.)^2 * sin(i)^2;
        eps[3+1,2+1] = (15./8.)*cos(i)^2 * sin(i)^4;
        eps[3+1,3+1] = (5./16.)*sin(i)^6;

    }
    model{
        vector[N] modes;
        int l;

        modes = rep_vector(1., N);
        for (mode in 1:M){        // Iterate over all modes passed in
            l = asy_ids[mode];    // Identify the Mode ID
            for (m in -l:l){      // Iterate over all m in a given l
                for (n in 1:N){
                    modes[n] += lorentzian(locs[l+1], l, m, f[n], eps[l+1,abs(m)+1], H[l+1], w[l+1], nus);
                }
            }
        }

        // Model drawn from a gamma distribution scaled to the model (Anderson+1990)
        p ~ gamma(1, 1../modes);

        //priors on the parameters
        logAmp ~ normal(1.5, 1);
        logGamma ~ normal(0, 0.01);
        locs ~ normal(asy_locs, 1);
        sini ~ uniform(0., 1.);
        nus ~ normal(0.411, 0.1);
    }
    '''
    model_path = 'pbstan.pkl'
    sm = pystan.StanModel(model_code = pbstan, model_name='pbstan')
    pkl_file =  open(model_path, 'wb')
    pickle.dump(sm, pkl_file)
    pkl_file.close()


class peakbag():
    def __init__(self, snr, locs, siglocs, modeids,
                    iters=5000, nchains=4):
        """Module to fit a model of lorentzians to the p-modes identified by the
        asymptotic peakbag module.

        Parameters
        ----------
        snr : lightkurve.periodogram.SNRPeriodogram
            The power spectrum normalized to a background noise
            level of 1.
        locs : array
            The locations in microhertz of any identified modes.
        siglocs : array
            The uncertainty in microhertz of any identified modes.
        modeids : array (int)
            The mode IDs of the identified modes.
        iters : float
            Number of HMC iterations to perform
        nchains : float
            Number of HMC chains to run

        Returns
        -------
        i : float
            The angle of inclination of the star in radians
        sigi : float
            The uncertainty on the angle of inclination in radians
        split : float
            The rotational splitting of the acoustic modes in microhertz
        sigsplit : float
            The uncertainty on the rotational splitting of the modes
        modes : pandas.DataFrame
            A dataframe containing the mode locations, mode ids, mode widths,
            and mode amplitudes (with appropriate uncertainties) for all modes
            passed in initially.
        """
        self.f = snr.frequency.value
        self.p = snr.power.value
        self.locs = locs
        self.siglocs = siglocs
        self.modeids = modeids

        self.iters = iters
        self.nchains = nchains

    def get_stanmodel(self):
        """If a stan model is saved locally, read that in. If not, create it
        """
        model_path = 'data/pbstan.pkl'
        if os.path.isfile(model_path):
            sm = pickle.load(open(model_path, 'rb'))
        else:
            warnings.warn('No stan model found, creating stan model.')
            create_pbstan(overwrite=True)
        return sm

    def run_stan(self):
        """Runs the PBJam peakbag model on the given input data.
        """
        data = {'N':len(self.f),
                'M': len(self.locs),
                'f':self.f,
                'p':self.p,
                'asy_locs':self.locs,
                'asy_ids':self.modeids}

        init = {'logAmp' :   np.ones(len(modelocs))*1.5,
                'logGamma' : np.zeros(len(modelocs)),
                'locs' : modelocs}

        fit = sm.sampling(data = self.dat,
                    iter= self.iters, chains=4, seed=1895,
                    init = [init for n in range(self.nchains)])

        return fit

    def get_output(self, fit):
        """Returns the output data given the completed fit
        """
        i = np.median(fit['i'])
        nus = np.median(fit['nus'])

        df = pd.DataFrame(columns={'modeid','loc','sig_loc','H','sig_H',
                                    'w','sig_w'})
        df['modeid'] = self.modeids
        df['loc'] = np.median(fit['locs'],axis=0)
        df['sig_loc'] = np.std(fit['locs'],axis=0)
        df['H'] = np.median(fit['H'],axis=0)
        df['sig_H'] = np.std(fit['H'],axis=0)
        df['w'] = np.median(fit['w'],axis=0)
        df['sig_w'] = np.std(fit['w'],axis=0)

        return i, nus, df

    def __call__(self):
        fit = self.run_stan()
        return get_output(fit)
