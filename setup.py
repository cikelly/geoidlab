from setuptools import setup, find_packages

setup(
    name='easy-pygeoid',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'compute_geoid=compute_geoid:main',
        ],
    },
)
