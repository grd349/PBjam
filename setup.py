from setuptools import setup

setup(name='pbjam',
      packages=['pbjam'],
      #url='TODO',
      install_requires=['numpy', 'pandas', 'emcee', 'statsmodels==0.9.0',
                        'lightkurve', 'astropy', 'scipy==1.2.1',
                        'psutil', 'corner', 'pymc3', 'matplotlib>=1.5.3'],
      include_package_data=True,
      )








