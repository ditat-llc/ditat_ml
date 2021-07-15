from setuptools import setup, find_packages


setup(
	author='ditat.io',
	description='Ditat Machine Learning Implementation - New pipeline kfold funcionality',
	name='ditat_ml',
	version='0.2.0',
	packages=find_packages(include=['ditat_ml', 'ditat_ml.*']),
	python_requires='>=3.6',
	install_requires=[
		'pandas',
		'sklearn',
		'matplotlib',
		'fuzzywuzzy[speedup]'
	]

)
