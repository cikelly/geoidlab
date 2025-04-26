from setuptools import setup, find_packages
from pathlib import Path

about = {}
here  = Path(__file__).parent
with open(here / 'geoidlab' / '__version__.py', 'r') as f:
    exec(f.read(), about)

with open(here / 'LICENSE', 'r') as f:
    lincense_text = f.read()



setup(
    name='geoidlab',
    version=about['__version__'],
    packages=find_packages(),
    license=lincense_text,
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
