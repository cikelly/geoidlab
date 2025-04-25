from setuptools import setup, find_packages

setup(
    name='geoidlab',
    version='1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'geoidlab=cli.geoidlab:main',
            'geoidlab-download=cli.tools.download_data:main',
            'geoidlab-gravity=cli.tools.gravity_reduction:main',
            'geoidlab-reference=cli.tools.reference_quantities:main',
            'geoidlab-terrain=cli.tools.terrain_quantities:main',
        ],
    },
)
