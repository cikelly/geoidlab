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
            'geoidlab=geoidlab.cli.geoidApp:app',
            'geoidApp=geoidlab.cli.geoidApp:app',
            'ggmApp=geoidlab.cli.ggmApp:app',
            'terrainApp=geoidlab.cli.terrainApp:app'
        ],
    },
)
