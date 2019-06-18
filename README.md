# PBjam - remove this before release
A repo for our peak baggin code and tips on jam

The rds directory for data storage (spectra or otherwise) is /rds/projects/n/nielsemb-plato-peakbagging/pbjam. Read/write permission should now have been granted to Guy, Oli and Ted. It should have 2TB free disk space, and if we need more Martin can request it.

To mount the directory on your local machine do: 
$ sudo mount -t cifs -o vers=3.0 -o domain=ADF -o username=USERNAME -o uid=USERNAME -o gid=USERNAME //its-rds.bham.ac.uk/rdsprojects/n/nielsemb-plato-peakbagging/pbjam /my/local/directory

# PBjam

PBjam is toolbox for peakbagging solar-like oscillators. This process involves identifying the modes of interest and then modeling the surounding to accurately measure their frequencies. Currently, the mode identification is based on fitting the asymptotic relation to the l=20 pairs, relying on the cumulative sum of prior knowledge gained from Kepler to inform the fitting process. PBjam is meant to be modular, allowing for different approaches to this. 

This provides precise initial estimates and mode IDs for further detailed peakbagging. 

## Getting Started

Clone repo from GitHub or pip install

### Prerequisites

- numpy
- pandas
- emcee
- lightkurve
- astropy
- statsmodels v0.9.0
- scipy v1.2.1

### Installing

Start Jupyter Notebook (see Examply.ipynb)

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

The Usual Suspects

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
