"""Pip installation script."""

from setuptools import find_packages, setup

setup(
    name='crystex',
    version="0.1",
    author='Maria S. Yankova, Adam J. Plowman',
    packages=find_packages(),
    install_requires=[
        'h5py',
        'plotly',
        'spglib',
        'numpy',
        'matplotlib',
        'mendeleev',
        'dropbox',
        'PyYAML',
        'scipy',
        'palettable',
        'scikit-image',
        'pandas',
        'vecmaths'
    ]
)
