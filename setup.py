from setuptools import setup, find_packages

setup(
    name='easy-pygeoid',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'easy-pygeoid=cli.easy_pygeoid:main',
        ],
    },
)
