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



setup(
    name='geoidlab',
    version=about['__version__'],
    packages=find_packages(include=['geoidlab', 'geoidlab.*']),
    license=license_text,
    author='Caleb Kelly',
    author_email='geo.calebkelly@gmail.com',
    description='Geoid computation workflow CLI tools',
    long_description=(here / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    entry_points={
        'console_scripts': [
            'geoidlab=geoidlab.cli.geoidApp:app',
            'geoidApp=geoidlab.cli.geoidApp:app',
            'ggmApp=geoidlab.cli.commands.reference:main',
            'terrainApp=geoidlab.cli.terrainApp:app'
        ],
    },
    python_requires='>=3.8',
    include_package_data=True
)
