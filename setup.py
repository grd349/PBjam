from setuptools import setup

setup(name='pbjam',
      packages=['pbjam'],
      url='https://pbjam.readthedocs.io/en/latest/',
      install_requires=['numpy', 'pandas', 'emcee', 'statsmodels>=0.10.0',
                        'lightkurve>=1.0.1', 'astropy', 'scipy>=1.3.0',
                        'psutil', 'corner', 'pymc3', 'matplotlib>=1.5.3'],
      include_package_data=True,
      )
