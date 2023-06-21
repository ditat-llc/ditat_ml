from setuptools import setup, find_packages

__version__ = '0.7' 


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
		'scikit-learn',
		'matplotlib',
		'fuzzywuzzy[speedup]'
	]

)
