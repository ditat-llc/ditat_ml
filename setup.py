from setuptools import setup, find_packages


setup(
	author='ditat.io',
	description='Ditat Machine Learning Implementation',
	name='ditat_ml',
	version='0.1.2',
	packages=find_packages(include=['ditat_ml', 'ditat_ml.*']),
	python_requires='>=3.6',
	install_requires=[
		'pandas',
		'sklearn',
		'matplotlib'
	]

)
