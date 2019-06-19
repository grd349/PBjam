# PBjam - remove this before release
A repo for our peak baggin code and tips on jam

The rds directory for data storage (spectra or otherwise) is /rds/projects/n/nielsemb-plato-peakbagging/pbjam. Read/write permission should now have been granted to Guy, Oli and Ted. It should have 2TB free disk space, and if we need more Martin can request it.

To mount the directory on your local machine do: 
$ sudo mount -t cifs -o vers=3.0 -o domain=ADF -o username=USERNAME -o uid=USERNAME -o gid=USERNAME //its-rds.bham.ac.uk/rdsprojects/n/nielsemb-plato-peakbagging/pbjam /my/local/directory

# PBjam

PBjam is toolbox for peakbagging solar-like oscillators. This involves identifying a set of modes and then modeling the surrounding spectrum to accurately measure the mode frequencies. Currently, the mode identification is based on fitting the asymptotic relation to the l=2,0 pairs, relying on the cumulative sum of prior knowledge gained from NASA's Kepler to inform the fitting process. PBjam is meant to be modular, allowing for different approaches to this to be added. 

This provides precise initial estimates and mode IDs for further detailed peakbagging. 

## Getting Started

### Prerequisites

- numpy
- pandas
- emcee
- lightkurve
- astropy
- statsmodels v0.9.0
- scipy v1.2.1

### Installing

Clone repo from GitHub or pip install

### Quickstart

See working examples in the [Examples Notebook](https://github.com/grd349/PBjam/blob/master/Example.ipynb))

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Citing

Please cite us

## Authors

### Main Contributors
Guy Davies
Oliver Hall
Martin Nielsen

### Chaos Engineers
Warrick Ball

## License

## Acknowledgments

PBjam relies heavily on many open source software packages to operate 

* Hat tip to anyone whose code was used
* Inspiration
* etc
