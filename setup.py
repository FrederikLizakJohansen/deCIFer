from setuptools import setup, find_packages

setup(
    name='decifer',
    version='1.0.0',
    python_requires='>=3.12,<3.14',
    packages=find_packages(),
    install_requires=[
        'braggcalculator>=0.1.0',
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
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
)
