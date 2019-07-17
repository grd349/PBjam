from setuptools import setup

setup(name='pbjam',
      packages=['pbjam'],
      #url='TODO',
      install_requires=['numpy', 'pandas', 'emcee', 'statsmodels==0.9.0',
                        'lightkurve==1.0.1', 'astropy', 'scipy==1.2.1',
                        'psutil', 'corner'],
      include_package_data=True,
      )
