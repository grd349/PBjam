from setuptools import setup

setup(name='pbjam',
      packages=['pbjam'],
      #url='TODO',
      install_requires=['numpy', 'pandas', 'emcee', 'statsmodels',
                        'lightkurve', 'astropy', 'scipy'],
      include_package_data=True,
      )
