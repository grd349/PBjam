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
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha"
    ],
)
