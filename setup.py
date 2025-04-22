from setuptools import setup, find_packages

setup(
    name='easy-pygeoid',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'easy-pygeoid=cli.easy_pygeoid:main',
            'easy-pygeoid-download=cli.tools.download_data:main',
            'easy-pygeoid-gravity=cli.tools.gravity_reduction:main',
            'easy-pygeoid-reference=cli.tools.reference_quantities:main',
            'easy-pygeoid-terrain=cli.tools.terrain_quantities:main',
        ],
    },
)
