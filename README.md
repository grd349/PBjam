# PBjam

PBjam is toolbox for peakbagging solar-like oscillators. This involves identifying a set of modes and then modeling the surrounding spectrum to accurately measure the mode frequencies. Currently, the mode identification is based on fitting the asymptotic relation to the l=2,0 pairs, relying on the cumulative sum of prior knowledge gained from NASA's Kepler to inform the fitting process. PBjam is meant to be modular, allowing for different approaches to this to be added. 

This provides precise initial estimates and mode IDs for further detailed peakbagging. 

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

PBjam is open source and we welcome contributions, large or small, by anyone with ideas for optimizing the current code, adding new nifty features, or just simply tidying things up. 

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
