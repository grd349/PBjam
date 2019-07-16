# PBjam

PBjam is toolbox for modeling the oscillation spectra of solar-like stars. This involves two main parts: identifying a set of modes of interest, and accurately modeling those modes to measure their frequencies. 

Currently, the mode identification is based on fitting the asymptotic relation to the l=2,0 pairs, relying on the cumulative sum of prior knowledge gained from NASA's Kepler mission to inform the fitting process. 

Modeling the modes, or 'peakbagging' is done using the HMC sampler from pymc3, which fits the modes with much fewer priors in place, which allows us to more accurately model the spectrum of frequencies.


## Getting Started

### Prerequisites

- numpy
- scipy v1.2.1
- pandas
- emcee
- lightkurve
- astropy
- pymc3
- statsmodels v0.9.0
- corner

### Installing

Clone this repo from GitHub or pip install

### Quickstart

See working examples in the [Examples Notebook](https://github.com/grd349/PBjam/blob/master/Example.ipynb))

## Contributing

PBjam is open source and we welcome contributions, large or small. If you spot any bugs, have ideas for optimizing the code, want new nifty features, feel free to submit issues on the GitHub repo. Pull requests are also welcome, these will be reviewed by the main authors of the code before merging. 

## Authors

### Main Contributors
- Guy Davies
- Oliver Hall
- Martin Nielsen

### Chaos Engineers
- Warrick Ball

## License

## Acknowledgments

PBjam relies heavily on many open source software packages to operate. 
