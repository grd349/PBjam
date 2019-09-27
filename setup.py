import setuptools

exec(open('pbjam/version.py').read())

setuptools.setup(
    name="pbjam",
    version=__version__,
    author="Martin Nielsen, Guy Davies, Oliver Hall",
    author_email="m.b.nielsen.1@bham.ac.uk",
    description="A package for peakbagging solar-like oscillators",
    long_description=open("README.rst").read(),
    url="https://pbjam.readthedocs.io/",
    packages=['pbjam'],
    install_requires=['numpy', 'pandas', 'emcee==3.0rc2', 'statsmodels>=0.10.0',
                      'lightkurve>=1.0.1', 'astropy', 'scipy>=1.3.0',
                      'psutil', 'corner', 'pymc3', 'matplotlib>=1.5.3'],
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
)
