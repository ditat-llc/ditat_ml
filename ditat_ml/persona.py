import re

import pandas as pd
import numpy as np
from fuzzywuzzy import process, fuzz

from .utility_functions import (
	dummies_with_options_and_limit,
	dummies_with_limit,
	time_it
	)


class Persona:
	MAPPING = {'Founder': {'seniority': 10, 'function': 'Firm Leader'}, 'CEO': {'seniority': 10, 'function': 'Firm Leader'}, 'Chief Executive Officer': {'seniority': 10, 'function': 'Firm Leader'}, 'Co-Founder': {'seniority': 10, 'function': 'Firm Leader'}, 'Co Founder': {'seniority': 10, 'function': 'Firm Leader'}, 'President': {'seniority': 10, 'function': 'Firm Leader'}, 'Executive Director': {'seniority': 10, 'function': 'Firm Leader'}, 'Chairman': {'seniority': 10, 'function': 'Firm Leader'}, 'Senior Managing Director': {'seniority': 10, 'function': 'Firm Leader'}, 'Senior Partner': {'seniority': 10, 'function': 'Firm Leader'}, 'Managing Principal': {'seniority': 10, 'function': 'Firm Leader'}, 'Managing Partner': {'seniority': 10, 'function': 'Firm Leader'}, 'Managing Director': {'seniority': 10, 'function': 'Firm Leader'}, 'Chief Operating Officer': {'seniority': 9, 'function': 'Operations'}, 'COO': {'seniority': 9, 'function': 'Operations'}, 'Operations': {'seniority': 5, 'function': 'Operations'}, 'Ops': {'seniority': 5, 'function': 'Operations'}, 'Chief Financial Officer': {'seniority': 9, 'function': 'Finance'}, 'CFO': {'seniority': 9, 'function': 'Finance'}, 'Controller': {'seniority': 7, 'function': 'Finance'}, 'Treasurer': {'seniority': 7, 'function': 'Finance'}, 'Fund Controller': {'seniority': 7, 'function': 'Finance'}, 'Vice President-Finance': {'seniority': 5, 'function': 'Finance'}, 'Vice President & Treasurer': {'seniority': 5, 'function': 'Finance'}, 'Director-Finance': {'seniority': 5, 'function': 'Finance'}, 'Director of Finance': {'seniority': 5, 'function': 'Finance'}, 'Finance Director': {'seniority': 5, 'function': 'Finance'}, 'Vice President-Finance & Administration': {'seniority': 5, 'function': 'Finance'}, 'Vice President-Finance & Treasurer': {'seniority': 5, 'function': 'Finance'}, 'VP Finance': {'seniority': 5, 'function': 'Finance'}, 'Senior Vice President-Finance': {'seniority': 5, 'function': 'Finance'}, 'Vice President & Controller': {'seniority': 5, 'function': 'Finance'}, 'Financial Controller': {'seniority': 5, 'function': 'Finance'}, 'Comptroller': {'seniority': 5, 'function': 'Finance'}, 'Vice President of Finance': {'seniority': 5, 'function': 'Finance'}, 'VP of Finance': {'seniority': 5, 'function': 'Finance'}, 'Director-Treasury': {'seniority': 5, 'function': 'Finance'}, 'Director-Finance & Administration': {'seniority': 5, 'function': 'Finance'}, 'Director-Treasury Services': {'seniority': 5, 'function': 'Finance'}, 'Vice President, Finance': {'seniority': 5, 'function': 'Finance'}, 'Vice President-Business & Finance': {'seniority': 5, 'function': 'Finance'}, 'Director-Treasury Operations': {'seniority': 5, 'function': 'Finance'}, 'Financial Director': {'seniority': 5, 'function': 'Finance'}, 'Finance Manager': {'seniority': 5, 'function': 'Finance'}, 'Accounting Manager': {'seniority': 5, 'function': 'Finance'}, 'Treasury Manager': {'seniority': 5, 'function': 'Finance'}, 'Manager-Treasury Operations': {'seniority': 5, 'function': 'Finance'}, 'Financial Manager': {'seniority': 5, 'function': 'Finance'}, 'Director-Accounting': {'seniority': 5, 'function': 'Finance'}, 'Assistant Treasurer': {'seniority': 2, 'function': 'Finance'}, 'Senior Accountant': {'seniority': 2, 'function': 'Finance'}, 'Financial Analyst': {'seniority': 2, 'function': 'Finance'}, 'Fund Accountant': {'seniority': 2, 'function': 'Finance'}, 'Fund Administrator': {'seniority': 2, 'function': 'Finance'}, 'Senior Financial Analyst': {'seniority': 2, 'function': 'Finance'}, 'Accountant': {'seniority': 2, 'function': 'Finance'}, 'Cash Manager': {'seniority': 2, 'function': 'Finance'}, 'Staff Accountant': {'seniority': 2, 'function': 'Finance'}, 'Finance Associate': {'seniority': 3, 'function': 'Finance'}, 'Finance Assistant': {'seniority': 2, 'function': 'Finance'}, 'CPA': {'seniority': 2, 'function': 'Finance'}, 'Associate Treasurer': {'seniority': 3, 'function': 'Finance'}, 'Investment Accountant': {'seniority': 2, 'function': 'Finance'}, 'Partner, Investor Relations': {'seniority': 7, 'function': 'Investor Relations'}, 'Head of Investor Relations': {'seniority': 7, 'function': 'Investor Relations'}, 'Head of IR': {'seniority': 7, 'function': 'Investor Relations'}, 'Managing Director, Investor Relations': {'seniority': 8, 'function': 'Investor Relations'}, 'Managing Director, Client Management': {'seniority': 7, 'function': 'Investor Relations'}, 'Senior Vice President-Investor Relations': {'seniority': 7, 'function': 'Investor Relations'}, 'Director of Marketing': {'seniority': 7, 'function': 'Investor Relations'}, 'Marketing Director': {'seniority': 7, 'function': 'Investor Relations'}, 'Director of Communications': {'seniority': 7, 'function': 'Investor Relations'}, 'Chief Marketing Officer': {'seniority': 8, 'function': 'Investor Relations'}, 'CMO': {'seniority': 8, 'function': 'Investor Relations'}, 'Investor Relations Associate': {'seniority': 4, 'function': 'Investor Relations'}, 'Associate, Investor Relations': {'seniority': 4, 'function': 'Investor Relations'}, 'Investor Relations Analyst': {'seniority': 3, 'function': 'Investor Relations'}, 'Investor Relations Coordinator': {'seniority': 3, 'function': 'Investor Relations'}, 'Marketing Associate': {'seniority': 4, 'function': 'Investor Relations'}, 'Marketing Coordinator': {'seniority': 3, 'function': 'Investor Relations'}, 'Marketing Assistant': {'seniority': 2, 'function': 'Investor Relations'}, 'Marketing': {'seniority': 5, 'function': 'Investor Relations'}, 'Partner': {'seniority': 9, 'function': 'Deal Team'}, 'Vice President': {'seniority': 6, 'function': 'Deal Team'}, 'Senior Vice President': {'seniority': 6, 'function': 'Firm Leader'}, 'Director': {'seniority': 7, 'function': 'Deal Team'}, 'Principal': {'seniority': 7, 'function': 'Deal Team'}, 'Associate': {'seniority': 4, 'function': 'Deal Team'}, 'Co': {'seniority': 8, 'function': 'Firm Leader'}, 'Finance': {'seniority': 3, 'function': 'Finance'}, 'General Partner': {'seniority': 8, 'function': 'Firm Leader'}, 'Investor Relations': {'seniority': 3, 'function': 'Investor Relations'}, 'Operating Partner': {'seniority': 8, 'function': 'Operations'}, 'VP': {'seniority': 6, 'function': 'Deal Team'}, 'Executive Vice President': {'seniority': 7, 'function': 'Firm Leader'}, 'Senior Associate': {'seniority': 5, 'function': 'Deal Team'}, 'Venture Partner': {'seniority': 8, 'function': 'Deal Team'}, 'Human Resources': {'seniority': 3, 'function': 'Deal Team'}, 'Analyst': {'seniority': 3, 'function': 'Deal Team'}, 'Manager': {'seniority': 8, 'function': 'Deal Team'}, 'Owner': {'seniority': 10, 'function': 'Firm Leader'}, 'Founding Partner': {'seniority': 10, 'function': 'Firm Leader'}, 'Investments': {'seniority': 4, 'function': 'Deal Team'}, 'Co,Founder': {'seniority': 10, 'function': 'Firm Leader'}, 'SVP': {'seniority': 8, 'function': 'Deal Team'}, 'Business Development': {'seniority': 4, 'function': 'Deal Team'}, 'Private Equity': {'seniority': 4, 'function': 'Deal Team'}, 'Senior Advisor': {'seniority': 7, 'function': 'Deal Team'}, 'Board Member': {'seniority': 9, 'function': 'Firm Leader'}, 'EA': {'seniority': 3, 'function': 'Operations'}, 'Fundraising': {'seniority': 4, 'function': 'Investor Relations'}, 'Consultant': {'seniority': 4, 'function': 'Deal Team'}, 'Senior Analyst': {'seniority': 4, 'function': 'Deal Team'}, 'Head of Business Development': {'seniority': 8, 'function': 'Deal Team'}, 'Secretary': {'seniority': 3, 'function': 'Operations'}, 'Growth Equity Investor': {'seniority': 5, 'function': 'Deal Team'}, 'Development': {'seniority': 4, 'function': 'Deal Team'}, 'Information Technology': {'seniority': 4, 'function': 'Technology'}, 'Vc': {'seniority': 5, 'function': 'Deal Team'}, 'Investment Professional': {'seniority': 4, 'function': 'Deal Team'}, 'Assistant Vice President': {'seniority': 5, 'function': 'Deal Team'}, 'Human Capital': {'seniority': 4, 'function': 'Deal Team'}, 'Chief Financial': {'seniority': 8, 'function': 'Finance'}, 'EVP': {'seniority': 8, 'function': 'Firm Leader'}, 'Trustee': {'seniority': 7, 'function': 'Deal Team'}, 'Media': {'seniority': 4, 'function': 'Deal Team'}, 'Telecommunications': {'seniority': 3, 'function': 'Technology'}, 'Investment Team': {'seniority': 4, 'function': 'Deal Team'}, 'Senior Principal': {'seniority': 8, 'function': 'Firm Leader'}, 'Client Services': {'seniority': 3, 'function': 'Investor Relations'}, 'Client Service Manager': {'seniority': 5, 'function': 'Investor Relations'}, 'Team Member': {'seniority': 3, 'function': 'Deal Team'}, 'Client Service': {'seniority': 3, 'function': 'Investor Relations'}, 'Senior Managing Partner': {'seniority': 8, 'function': 'Deal Team'}, 'Venture Capital': {'seniority': 7, 'function': 'Deal Team'}, 'Investment Committee': {'seniority': 6, 'function': 'Deal Team'}, 'Business Analyst': {'seniority': 3, 'function': 'Deal Team'}, }
	
	PERSONA_MAPPING = {
		'Finance': ['finance', 'financial', 'equity', 'portfolio', 'treasury', 'accounting', 'market', 'markets', 'tax', 'investor'],
		'Investor Relations': ['investor relations', 'ir', 'marketing', 'relationship', 'relationships', 'communications', 'communication', 'sales', 'account'],
		'Operations': ['operations', 'operating', 'administration', 'compliance', 'office', 'administrative'],
		'Firm Leader': ['executive', 'general', 'strategy'],
		'Technology': ['technology', 'it', 'software'],
		'Deal Team': ['associate','investment','investments'],
		'Business Development': ['business development']
	}

	SENIORITY_MAPPING = {
		'head': 9,
		'partner': 9,
		'manager': 8,
		'director': 8,
		'officer': 6,
		'counsel': 6,
		'associate': 4,
		'analyst': 3,
		'assistant': 2,
		'intern': 1,
		'administrator': 4,
		'vp': 5,
		'vice president': 5,
		'executive': 7,
		'chairman': 10,
		'management': 8
	}

	DEFAULT_SENIORITY = 3
	DEFAULT_PERSONA = 'Deal Team'

	REPLACEMENT_MAPPING_SYMBOLS = {
		'&': ',',
		'/': ',',
		'-': ',',
		'\+': ','
	}
	REPLACEMENT_MAPPING_WORDS  = {
		'and': ',',
	}
	
	def __init__(
		self,
		mapping: dict=None,
		persona_mapping: dict=None,
		seniority_mapping: dict=None,
		combine_mappings: bool=False,
		default_persona: str=None,
		default_seniority: int=None
		):
		'''
		'''
		self.combine_mappings = combine_mappings
		self.mapping = mapping
		self.seniority_mapping = seniority_mapping
		self.persona_mapping = persona_mapping

		self.default_seniority = type(self).DEFAULT_SENIORITY if default_seniority is None else default_seniority
		self.default_persona = type(self).DEFAULT_PERSONA if default_persona is None else default_persona

		self.verbose = False

	@property
	def mapping(self):
		return self._mapping

	@mapping.setter
	def mapping(self, value):
		if value is None:
			self._mapping = type(self).MAPPING.copy()
		elif not isinstance(value, dict):
			raise ValueError('mapping must be a dictionary.')
		elif self.combine_mappings:
			self._mapping = type(self).MAPPING.copy()
			for key, value in value.items():
				self._mapping[key] = value
		else:
			self._mapping = value

	@property
	def seniority_mapping(self):
		return self._seniority_mapping

	@seniority_mapping.setter
	def seniority_mapping(self, value):
		if value is None:
			self._seniority_mapping = type(self).SENIORITY_MAPPING.copy()
		elif not isinstance(value, dict):
			raise ValueError('seniority_mapping must be a dictionary.')
		elif self.combine_mappings:
			self._seniority_mapping = type(self).SENIORITY_MAPPING.copy()
			for key, value in value.items():
				self._seniority_mapping[key] = value
		else:
			self._seniority_mapping = value

		self._seniority_list = list(self._seniority_mapping.keys())

	@property
	def persona_mapping(self):
		return self._persona_mapping

	@persona_mapping.setter
	def persona_mapping(self, value):
		if value is None:
			self._persona_mapping = type(self).PERSONA_MAPPING.copy()
		elif not isinstance(value, dict):
			raise ValueError('persona_mapping must be a dictionary.')
		elif self.combine_mappings:
			self._persona_mapping = type(self).PERSONA_MAPPING.copy()
			for key, value in value.items():
				self._persona_mapping[key] = value
		else:
			self._persona_mapping = value

		self._persona_list = list(self._persona_mapping.keys())

	@property
	def seniority_list(self):
		return self._seniority_list

	@property
	def persona_list(self):
		return self._persona_list

	def printif(self, value):
		if self.verbose:
			print(value)

	def _internal_process(
		self,
		value: str,
		fuzzywuzzy_usage=True
		):
		if value in ['', None]:
			return None

		# Lower casing titles.
		value = value.lower()
		mapping = {i.lower(): j for i, j in self.mapping.items()}

		if value in mapping:
			# Return if exists in MAPPING.
			return mapping[value]

		last_word = value.split()[-1]

		###########################
		# I. CXO excluding CEO that's part of the mapping.
		if re.search(r'^c[efiotc]o$', last_word):
			temp_map = {
				'f': 'Finance',
				'o': 'Operations',
				't': 'Technology',
				'i': 'Finance',
				'm': 'Investor Relations',
				'e': 'Firm Leader',
				'c': self.default_persona
			}
			temp_function = temp_map.get(last_word[1])
			# Arbitrary if CEO -> 10 else 8
			seniority = 10 if temp_function == 'e' else 8
			if seniority and temp_function:
				return {'seniority': seniority, 'function': temp_function}

		elif 'chief' in value and 'officer' in value:
			# Chief {Something} Officer
			second = value.split('chief')[1].split('office')[0].strip()
			temp_function = None
			for persona, synonym in self.persona_mapping.items():
				if second in synonym:
					temp_function = persona
					break
			seniority = 10 if temp_function == 'executive' else 8
			temp_function = temp_function or self.default_persona
			if seniority and temp_function:
				return {'seniority': seniority, 'function': temp_function}

		###########################
		# II. "Function seniority". 2-word combination like "Finance Manager" (Function Seniority)
		elif len(value.split()) == 2:
			temp_function = None
			first, second = value.split()
			for persona, synonym in self.persona_mapping.items():
				if first in synonym:
					temp_function = persona
					break
			temp_seniority = self.seniority_mapping.get(second)
			if temp_function and temp_seniority:
				return {'seniority': temp_seniority, 'function': temp_function}

		###########################
		# III. Same logic as II, but instead of Finance manager, it is Manager of finance.
		elif len(value.split()) == 3 and value.split()[1] == 'of':
			temp_function = None
			first, second, third = value.split()
			for persona, synonym in self.persona_mapping.items():
				if third in synonym:
					temp_function = persona
					break
			temp_seniority = self.seniority_mapping.get(first)
			if temp_function and temp_seniority:
				return {'seniority': temp_seniority, 'function': temp_function}

		###########################
		# IV. For single words. It assumes that the single references a seniority like (Manager),
		# it this does not happen we assume the single word references a Function with 'default_' seniority.
		elif len(value.split()) == 1:
			temp_function = None
			temp_seniority = None
			if value in self.seniority_mapping:
				temp_seniority = self.seniority_mapping[value]
				temp_function = self.default_persona
			else:
				for persona, synonym in self.persona_mapping.items():
					if value in synonym:
						temp_function = persona
						break		
				temp_seniority = self.default_seniority	
			if temp_function and temp_seniority:
				return {'seniority': temp_seniority, 'function': temp_function}

		###########################
		# V. Other cases.
		temp_function = None
		temp_seniority = None

		for persona, synonym in self.persona_mapping.items():
			for v in value.split():
				if v in synonym:
					temp_function = persona
					break

		if temp_function:
			for s in self.seniority_mapping:
				if s in value.split():
					temp_seniority = self.seniority_mapping[s]
		
		if temp_function and temp_seniority:
			return {'seniority': temp_seniority, 'function': temp_function}

		###########################
		# FuzzyWuzzy strategy
		# [fuzz.WRatio, fuzz.QRatio, fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.partial_token_set_ratio, fuzz.partial_token_sort_ratio, fuzz.UWRatio, fuzz.UQRatio]
		if fuzzywuzzy_usage:
			th = 70
			scorer = fuzz.WRatio

			s_name, s_value = process.extractOne(value, self.seniority_list, scorer=scorer)
			p_name, p_value = process.extractOne(value, self.persona_list, scorer=scorer)

			if s_value >= th and p_value >= th:
				result = {'seniority': self.seniority_mapping[s_name], 'function': p_name}

		return None

	# @time_it()
	def process(
		self,
		value,
		fuzzywuzzy_usage=True,
		return_tuples=False
		):
		'''
		Process job title into
			{'senioryty': int[1, 10], 'function': str[from PERSONA_SYNONYMS]}
		
		Args:
			value (str): String to be processed.
			fuzzywuzzy_uage (bool, default=True): Use this as a last option
				where the hard-coded  options haven't returned a match.
			return_tuples (bool, default=False): Sometimes the value provided is "comma"
				separated array of values. By default we return the one with the highest
				seniority. You can also return the list of matches

		Returns:
			(dict): ex: "{'senioryty': 10, 'function': 'Firm Leader'}"
				You can also return if return_tuples is True, a list of results.
		'''
		self.printif('--------------------')
		self.printif(value)
		if not value or not isinstance(value, str) or len(value) < 2:
			return None

		# Check if value is present in self.mapping
		# This logic also exists inside self_internal_process()
		if value in self.mapping:
			return self.mapping[value]

		# Some exceptions
		value  = value.replace('Co-', 'Co ')

		for k, v in type(self).REPLACEMENT_MAPPING_SYMBOLS.items():
			value = value.replace(k, v)

		split_values = value.split()
		values = []

		for split_value in split_values:
			new_value = type(self).REPLACEMENT_MAPPING_WORDS.get(split_value) or split_value
			values.append(new_value)

		values = ' '.join(values)
		values = values.split(',')
		values = [x.strip() for x in values if x]

		self.printif(values)
		results = []
		for v in values:
			r = self._internal_process(value=v, fuzzywuzzy_usage=fuzzywuzzy_usage)
			results.append(r)

		results = [x for x in results if x]
		self.printif(results)

		if not results:
			result = None

		elif return_tuples and len(results) >= 1:
			result = results

		elif len(results) > 1:
			result = max(results, key=lambda x: x.get('seniority'))

		else:
			result = results[0]

		self.printif(result)
		return result


	def process_dataframe(self, dataframe, column, drop=True):
		'''
		Convenient method to self.process() a column in a dataframe
		'''
		dataframe = dataframe.copy()
		dataframe['process'] = dataframe[column].apply(lambda x: self.process(x, return_tuples=False))
		dataframe['seniority'] = dataframe['process'].apply(lambda x: x['seniority'] if x else x)
		dataframe['function'] = dataframe['process'].apply(lambda x: x['function'] if x else x)
		dataframe.drop('process', axis=1, inplace=True)
		if drop:
			dataframe.drop(column, axis=1, inplace=True)
		return dataframe









	# @classmethod
	# def process_dataframe(cls, dataframe, column, drop=False):
	# 	'''
	# 	'''
	# 	dataframe_original = dataframe.copy()
	# 	dataframe = dataframe[[column]]

	# 	dataframe = dataframe_original.replace({column: cls.REPLACEMENT_MAPPING}, regex=True)

	# 	dataframe = dummies_with_options_and_limit(
	# 		dataframe=dataframe,
	# 		col=column,
	# 		verbose=False,
	# 		prefix=False,
	# 		return_unique_list=False,
	# 		limit=None,
	# 		return_value_counts=False
	# 	)

	# 	results = {col: cls.process(col) for col in dataframe.columns}

	# 	dataframe['seniority'] = 0
		
	# 	func_cols = [f"function_{persona}" for persona in cls.PERSONA_SYNONYMS]
	# 	for f in func_cols:
	# 		dataframe[f] = 0

	# 	for col in dataframe:
	# 		if results.get(col) and col != 'seniority' and col not in func_cols:
	# 			# Assignment for seniority and keeps the highest value if more than one function
	# 			dataframe['seniority'] = np.where(
	# 				(dataframe[col] == 1) & (results[col]['seniority'] > dataframe['seniority']),
	# 				results[col]['seniority'],
	# 				dataframe['seniority']
	# 			)
	# 			# Assignment for function_{col} according to persona synonyms
	# 			dataframe.loc[dataframe[col] == 1 , f"function_{results[col]['function']}"] = 1

	# 	dataframe = dataframe[['seniority'] + func_cols]

	# 	dataframe_original = dataframe_original.join(dataframe)

	# 	if drop:
	# 		dataframe_original.drop(column, axis=1, inplace=True)

	# 	return dataframe_original


