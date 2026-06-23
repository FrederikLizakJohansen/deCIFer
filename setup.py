from setuptools import setup, find_packages

setup(
    name='decifer',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'pyyaml',
        'tqdm',
        'omegaconf',
        'h5py',
        'pymatgen',
        'periodictable',
        'scikit-learn',
    ],
)
