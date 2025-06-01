from setuptools import setup, find_packages
from pathlib import Path

# Read version
about = {}
here  = Path(__file__).parent
with open(here / 'geoidlab' / '__version__.py', 'r') as f:
    exec(f.read(), about)

# Read license
with open(here / 'LICENSE', 'r') as f:
    license_text = f.read()

# Default requirements if requirements.txt is not available
requirements = [
    'numpy>=1.20.0',
    'scipy>=1.7.0',
    'requests>=2.25.0',
    'tqdm>=4.60.0',
    'beautifulsoup4>=4.9.0',
    'xarray>=0.19.0',
    'netCDF4>=1.5.7',
    'pandas>=1.3.0',
    'numba>=0.53.0',
    'numba-progress>=0.0.3',
    'openpyxl>=3.0.7',
    'rasterio>=1.2.0',
    'rioxarray>=0.5.0',
    'bottleneck>=1.3.2',
    'scikit-learn>=0.24.0',
    'joblib>=1.3.0',
    'matplotlib>=3.4.0',
    'pyproj>=3.0.0'
]

# Try to read requirements from requirements.txt if it exists
try:
    with open(here / 'requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    pass  # Use default requirements

setup(
    name='geoidlab',
    version=about['__version__'],
    packages=find_packages(include=['geoidlab', 'geoidlab.*']),
    license=license_text,
    author='Caleb Kelly',
    author_email='geo.calebkelly@gmail.com',
    description='A Modular and Automated Python Package for Geoid Computation',
    long_description=(here / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/cikelly/geoidlab',
    project_urls={
        'Documentation': 'https://geoidlab.readthedocs.io',
        'Source': 'https://github.com/cikelly/geoidlab',
        'Issues': 'https://github.com/cikelly/geoidlab/issues',
    },
    install_requires=requirements,
    python_requires='>=3.8,<3.13',
    entry_points={
        'console_scripts': [
            'geoidlab=geoidlab.cli.main:main',
        ],
    },
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Geodesy',
        'Topic :: Scientific/Engineering :: Geoscience',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.900',
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'twine>=3.4',
            'build>=0.7'
        ]
    }
)
