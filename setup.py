from setuptools import setup, find_packages
from ditat_ml import __version__


with open('README.md', 'r') as f:
	long_description = f.read()


setup(
	author='ditat.io',
	description='Ditat Machine Learning Implementation',
    long_description=long_description,
	name='ditat_ml',
	version=__version__,
	packages=find_packages(include=['ditat_ml', 'ditat_ml.*']),
	python_requires='>=3.6',
	install_requires=[
		'pandas',
		'sklearn',
		'matplotlib',
		'fuzzywuzzy[speedup]'
	]

)
