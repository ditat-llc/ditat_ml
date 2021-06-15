import re

import pandas as pd
import numpy as np
from fuzzywuzzy import process

from .utility_functions import (
	dummies_with_options_and_limit,
	dummies_with_limit,
	time_it
	)


class Persona:
	MAPPING = {'Founder': {'seniority': 10, 'function': 'Firm Leader'}, 'CEO': {'seniority': 10, 'function': 'Firm Leader'}, 'Chief Executive Officer': {'seniority': 10, 'function': 'Firm Leader'}, 'Co-Founder': {'seniority': 10, 'function': 'Firm Leader'}, 'President': {'seniority': 10, 'function': 'Firm Leader'}, 'Executive Director': {'seniority': 10, 'function': 'Firm Leader'}, 'Chairman': {'seniority': 10, 'function': 'Firm Leader'}, 'Senior Managing Director': {'seniority': 10, 'function': 'Firm Leader'}, 'Senior Partner': {'seniority': 10, 'function': 'Firm Leader'}, 'Managing Principal': {'seniority': 10, 'function': 'Firm Leader'}, 'Managing Partner': {'seniority': 10, 'function': 'Firm Leader'}, 'Managing Director': {'seniority': 10, 'function': 'Firm Leader'}, 'Chief Operating Officer': {'seniority': 9, 'function': 'Operations'}, 'COO': {'seniority': 9, 'function': 'Operations'}, 'Operations': {'seniority': 5, 'function': 'Operations'}, 'Ops': {'seniority': 5, 'function': 'Operations'}, 'Chief Financial Officer': {'seniority': 9, 'function': 'Finance'}, 'CFO': {'seniority': 9, 'function': 'Finance'}, 'Controller': {'seniority': 7, 'function': 'Finance'}, 'Treasurer': {'seniority': 7, 'function': 'Finance'}, 'Fund Controller': {'seniority': 7, 'function': 'Finance'}, 'Vice President-Finance': {'seniority': 5, 'function': 'Finance'}, 'Vice President & Treasurer': {'seniority': 5, 'function': 'Finance'}, 'Director-Finance': {'seniority': 5, 'function': 'Finance'}, 'Director of Finance': {'seniority': 5, 'function': 'Finance'}, 'Finance Director': {'seniority': 5, 'function': 'Finance'}, 'Vice President-Finance & Administration': {'seniority': 5, 'function': 'Finance'}, 'Vice President-Finance & Treasurer': {'seniority': 5, 'function': 'Finance'}, 'VP Finance': {'seniority': 5, 'function': 'Finance'}, 'Senior Vice President-Finance': {'seniority': 5, 'function': 'Finance'}, 'Vice President & Controller': {'seniority': 5, 'function': 'Finance'}, 'Financial Controller': {'seniority': 5, 'function': 'Finance'}, 'Comptroller': {'seniority': 5, 'function': 'Finance'}, 'Vice President of Finance': {'seniority': 5, 'function': 'Finance'}, 'VP of Finance': {'seniority': 5, 'function': 'Finance'}, 'Director-Treasury': {'seniority': 5, 'function': 'Finance'}, 'Director-Finance & Administration': {'seniority': 5, 'function': 'Finance'}, 'Director-Treasury Services': {'seniority': 5, 'function': 'Finance'}, 'Vice President, Finance': {'seniority': 5, 'function': 'Finance'}, 'Vice President-Business & Finance': {'seniority': 5, 'function': 'Finance'}, 'Director-Treasury Operations': {'seniority': 5, 'function': 'Finance'}, 'Financial Director': {'seniority': 5, 'function': 'Finance'}, 'Finance Manager': {'seniority': 5, 'function': 'Finance'}, 'Accounting Manager': {'seniority': 5, 'function': 'Finance'}, 'Treasury Manager': {'seniority': 5, 'function': 'Finance'}, 'Manager-Treasury Operations': {'seniority': 5, 'function': 'Finance'}, 'Financial Manager': {'seniority': 5, 'function': 'Finance'}, 'Director-Accounting': {'seniority': 5, 'function': 'Finance'}, 'Assistant Treasurer': {'seniority': 2, 'function': 'Finance'}, 'Senior Accountant': {'seniority': 2, 'function': 'Finance'}, 'Financial Analyst': {'seniority': 2, 'function': 'Finance'}, 'Fund Accountant': {'seniority': 2, 'function': 'Finance'}, 'Fund Administrator': {'seniority': 2, 'function': 'Finance'}, 'Senior Financial Analyst': {'seniority': 2, 'function': 'Finance'}, 'Accountant': {'seniority': 2, 'function': 'Finance'}, 'Cash Manager': {'seniority': 2, 'function': 'Finance'}, 'Staff Accountant': {'seniority': 2, 'function': 'Finance'}, 'Finance Associate': {'seniority': 3, 'function': 'Finance'}, 'Finance Assistant': {'seniority': 2, 'function': 'Finance'}, 'CPA': {'seniority': 2, 'function': 'Finance'}, 'Associate Treasurer': {'seniority': 3, 'function': 'Finance'}, 'Investment Accountant': {'seniority': 2, 'function': 'Finance'}, 'Partner, Investor Relations': {'seniority': 7, 'function': 'Investor Relations'}, 'Head of Investor Relations': {'seniority': 7, 'function': 'Investor Relations'}, 'Head of IR': {'seniority': 7, 'function': 'Investor Relations'}, 'Managing Director, Investor Relations': {'seniority': 8, 'function': 'Investor Relations'}, 'Managing Director, Client Management': {'seniority': 7, 'function': 'Investor Relations'}, 'Senior Vice President-Investor Relations': {'seniority': 7, 'function': 'Investor Relations'}, 'Director of Marketing': {'seniority': 7, 'function': 'Investor Relations'}, 'Marketing Director': {'seniority': 7, 'function': 'Investor Relations'}, 'Director of Communications': {'seniority': 7, 'function': 'Investor Relations'}, 'Chief Marketing Officer': {'seniority': 8, 'function': 'Investor Relations'}, 'CMO': {'seniority': 8, 'function': 'Investor Relations'}, 'Investor Relations Associate': {'seniority': 4, 'function': 'Investor Relations'}, 'Associate, Investor Relations': {'seniority': 4, 'function': 'Investor Relations'}, 'Investor Relations Analyst': {'seniority': 3, 'function': 'Investor Relations'}, 'Investor Relations Coordinator': {'seniority': 3, 'function': 'Investor Relations'}, 'Marketing Associate': {'seniority': 4, 'function': 'Investor Relations'}, 'Marketing Coordinator': {'seniority': 3, 'function': 'Investor Relations'}, 'Marketing Assistant': {'seniority': 2, 'function': 'Investor Relations'}, 'Marketing': {'seniority': 5, 'function': 'Investor Relations'}, 'Partner': {'seniority': 9, 'function': 'Deal Team'}, 'Vice President': {'seniority': 6, 'function': 'Deal Team'}, 'Senior Vice President': {'seniority': 6, 'function': 'Firm Leader'}, 'Director': {'seniority': 7, 'function': 'Deal Team'}, 'Principal': {'seniority': 7, 'function': 'Deal Team'}, 'Associate': {'seniority': 4, 'function': 'Deal Team'}, 'Co': {'seniority': 8, 'function': 'Firm Leader'}, 'Finance': {'seniority': 3, 'function': 'Finance'}, 'General Partner': {'seniority': 8, 'function': 'Firm Leader'}, 'Investor Relations': {'seniority': 3, 'function': 'Investor Relations'}, 'Operating Partner': {'seniority': 8, 'function': 'Operations'}, 'VP': {'seniority': 6, 'function': 'Deal Team'}, 'Executive Vice President': {'seniority': 7, 'function': 'Firm Leader'}, 'Senior Associate': {'seniority': 5, 'function': 'Deal Team'}, 'Venture Partner': {'seniority': 8, 'function': 'Deal Team'}, 'Human Resources': {'seniority': 3, 'function': 'Deal Team'}, 'Analyst': {'seniority': 3, 'function': 'Deal Team'}, 'Manager': {'seniority': 8, 'function': 'Deal Team'}, 'Owner': {'seniority': 10, 'function': 'Firm Leader'}, 'Founding Partner': {'seniority': 10, 'function': 'Firm Leader'}, 'Investments': {'seniority': 4, 'function': 'Deal Team'}, 'Co,Founder': {'seniority': 10, 'function': 'Firm Leader'}, 'SVP': {'seniority': 8, 'function': 'Deal Team'}, 'Business Development': {'seniority': 4, 'function': 'Deal Team'}, 'Private Equity': {'seniority': 4, 'function': 'Deal Team'}, 'Senior Advisor': {'seniority': 7, 'function': 'Deal Team'}, 'Board Member': {'seniority': 9, 'function': 'Firm Leader'}, 'EA': {'seniority': 3, 'function': 'Operations'}, 'Fundraising': {'seniority': 4, 'function': 'Investor Relations'}, 'Consultant': {'seniority': 4, 'function': 'Deal Team'}, 'Senior Analyst': {'seniority': 4, 'function': 'Deal Team'}, 'Head of Business Development': {'seniority': 8, 'function': 'Deal Team'}, 'Secretary': {'seniority': 3, 'function': 'Operations'}, 'Growth Equity Investor': {'seniority': 5, 'function': 'Deal Team'}, 'Development': {'seniority': 4, 'function': 'Deal Team'}, 'Information Technology': {'seniority': 4, 'function': 'Technology'}, 'Vc': {'seniority': 5, 'function': 'Deal Team'}, 'Investment Professional': {'seniority': 4, 'function': 'Deal Team'}, 'Assistant Vice President': {'seniority': 5, 'function': 'Deal Team'}, 'Human Capital': {'seniority': 4, 'function': 'Deal Team'}, 'Chief Financial': {'seniority': 8, 'function': 'Finance'}, 'EVP': {'seniority': 8, 'function': 'Firm Leader'}, 'Trustee': {'seniority': 7, 'function': 'Deal Team'}, 'Media': {'seniority': 4, 'function': 'Deal Team'}, 'Telecommunications': {'seniority': 3, 'function': 'Technology'}, 'Investment Team': {'seniority': 4, 'function': 'Deal Team'}, 'Senior Principal': {'seniority': 8, 'function': 'Firm Leader'}, 'Client Services': {'seniority': 3, 'function': 'Investor Relations'}, 'Client Service Manager': {'seniority': 5, 'function': 'Investor Relations'}, 'Team Member': {'seniority': 3, 'function': 'Deal Team'}, 'Client Service': {'seniority': 3, 'function': 'Investor Relations'}, 'Senior Managing Partner': {'seniority': 8, 'function': 'Deal Team'}, 'Venture Capital': {'seniority': 7, 'function': 'Deal Team'}, 'Investment Committee': {'seniority': 6, 'function': 'Deal Team'}, 'Business Analyst': {'seniority': 3, 'function': 'Deal Team'}, }
	
	PERSONA_SYNONYMS = {
		'Finance': ['finance', 'financial', 'equity', 'portfolio', 'treasury', 'accounting', 'market', 'markets', 'tax', 'investor'],
		'Investor Relations': ['investor relations', 'ir', 'marketing', 'relationship', 'relationships', 'communications', 'communication', 'sales', 'account'],
		'Operations': ['operations', 'operating', 'administration', 'compliance', 'office', 'administrative'],
		'Firm Leader': ['executive', 'general', 'strategy'],
		'Technology': ['technology', 'it', 'software'],
		'Deal Team': ['associate','investment','investments'],
		'Business Development': ['business development'],
		# 'Deal Team': ['research', 'legal']
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
		'default_': 3,
		'administrator': 4,
		'vp': 5,
		'vice president': 5,
		'executive': 7,
		'chairman': 10,
		'management': 8
	}

	REPLACEMENT_MAPPING = {
		'&': ',',
		'and': ',',
		'/': ',',
		'-': ',',
		'\+': ','
	}

	SENIORITY_LIST = list(SENIORITY_MAPPING.keys())
	PERSONA_LIST = list(PERSONA_SYNONYMS.keys())
	
	def __init__(self):
		pass

	@classmethod
	# @time_it()
	def process(cls, value):
		'''
		Process job title into
			{'senioryty': int[1, 10], 'function': str[from PERSONA_SYNONYMS]}
		
		Args:
			value (str): String to be processed.

		Returns:
			(dict): ex: "{'senioryty': 10, 'function': 'Firm Leader'"
		'''
		if not value or not isinstance(value, str):
			return None

		# Lower casing titles.
		value = value.lower()
		mapping = {i.lower(): j for i, j in cls.MAPPING.items()}

		if value in mapping:
			# Return if exists in MAPPING.
			return mapping[value]

		last_word = value.split()[-1]

		# I. CXO excluding CEO that's part of the mapping.
		if re.search(r'^c[efiotc]o$', last_word):
			temp_map = {
				'f': 'Finance',
				'o': 'Operations',
				't': 'Technology',
				'i': 'Finance',
				'm': 'Investor Relations',
				'e': 'Firm Leader',
				'c': 'Deal Team'
			}
			temp_function = last_word[1]
			# Arbitrary if CEO -> 10 else 8
			seniority = 10 if temp_function == 'e' else 8
			return {'seniority': seniority, 'function': temp_map[temp_function]}

		elif 'chief' in value and 'officer' in value:
			# Chief {Something} Officer
			second = value.split('chief')[1].split('office')[0].strip()
			temp_function = None
			for persona, synonym in cls.PERSONA_SYNONYMS.items():
				if second in synonym:
					temp_function = persona
					break
			seniority = 10 if temp_function == 'executive' else 8
			return {'seniority': seniority, 'function': temp_function}

		# II. "Function seniority". 2-word combination like "Finance Manager" (Function Seniority)
		elif len(value.split()) == 2:
			temp_function = None
			first, second = value.split()
			for persona, synonym in cls.PERSONA_SYNONYMS.items():
				if first in synonym:
					temp_function = persona
					break
			temp_seniority = cls.SENIORITY_MAPPING.get(second)
			if temp_function and temp_seniority:
				return {'seniority': temp_seniority, 'function': temp_function}

		# III. Same logic as II, but instead of Finance manager, it is Manager of finance.
		elif len(value.split()) == 3 and value.split()[1] == 'of':
			temp_function = None
			first, second, third = value.split()
			for persona, synonym in cls.PERSONA_SYNONYMS.items():
				if third in synonym:
					temp_function = persona
					break
			temp_seniority = cls.SENIORITY_MAPPING.get(first)
			if temp_function and temp_seniority:
				return {'seniority': temp_seniority, 'function': temp_function}

		# IV. For single words. It assumes that the single references a seniority like (Manager),
		# it this does not happen we assume the single word references a Function with 'default_' seniority.
		elif len(value.split()) == 1:
			temp_function = None
			temp_seniority = None
			if value in cls.SENIORITY_MAPPING:
				temp_seniority = cls.SENIORITY_MAPPING[value]
				temp_function = 'Deal Team'
			else:
				for persona, synonym in cls.PERSONA_SYNONYMS.items():
					if value in synonym:
						temp_function = persona
						break
				temp_seniority = cls.SENIORITY_MAPPING['default_']			
			if temp_function and temp_seniority:
				return {'seniority': temp_seniority, 'function': temp_function}

		# Other cases.
		temp_function = None
		temp_seniority = None

		for persona, synonym in cls.PERSONA_SYNONYMS.items():
			for v in value.split():
				if v in synonym:
					temp_function = persona
					break

		if temp_function:
			for s in cls.SENIORITY_MAPPING:
				if s in value.split():
					temp_seniority = cls.SENIORITY_MAPPING[s]
		
		if temp_function and temp_seniority:
			return {'seniority': temp_seniority, 'function': temp_function}


		# # PolyFuzzy Cases		
		# cls.polyfuzz_model.match([value], cls.SENIORITY_LIST)
		# s_df = cls.polyfuzz_model.get_matches()

		# cls.polyfuzz_model.match([value], cls.PERSONA_LIST)
		# p_df = cls.polyfuzz_model.get_matches()

		# df = pd.merge(s_df, p_df, on='From', how='left')
		# df.rename(columns={
		# 		'To_x': 'seniority',
		# 		'To_y': 'function',
		# 		'Similarity_x': 's_score',
		# 		'Similarity_y': 'f_score'
		# 	}, inplace=True)
		
		# th = 0.3
		# if df.s_score.iloc[0] >= th and df.f_score.iloc[0] >= th:
		# 	resp =  {
		# 		'seniority': cls.SENIORITY_MAPPING[df.seniority.iloc[0]] ,
		# 		'function': df.function.iloc[0]
		# 	}
		# 	print(value, resp, df.seniority.iloc[0])
		# 	return resp

		# WuzzyFuzzy
		th = 40
		seniority, s_value = process.extractOne(value, cls.SENIORITY_LIST)
		persona, p_value = process.extractOne(value, cls.PERSONA_LIST)

		if s_value >= th and p_value >= th:
			return {'seniority': cls.SENIORITY_MAPPING[seniority], 'function': persona}

		return None

	@classmethod
	def process_dataframe(cls, dataframe, column, drop=False):
		'''
		'''
		dataframe_original = dataframe.copy()
		dataframe = dataframe[[column]]

		dataframe = dataframe_original.replace({column: cls.REPLACEMENT_MAPPING}, regex=True)

		dataframe = dummies_with_options_and_limit(
			dataframe=dataframe,
			col=column,
			verbose=False,
			prefix=False,
			return_unique_list=False,
			limit=None,
			return_value_counts=False
		)

		results = {col: cls.process(col) for col in dataframe.columns}

		dataframe['seniority'] = 0
		
		func_cols = [f"function_{persona}" for persona in cls.PERSONA_SYNONYMS]
		for f in func_cols:
			dataframe[f] = 0

		for col in dataframe:
			if results.get(col) and col != 'seniority' and col not in func_cols:
				# Assignment for seniority and keeps the highest value if more than one function
				dataframe['seniority'] = np.where(
					(dataframe[col] == 1) & (results[col]['seniority'] > dataframe['seniority']),
					results[col]['seniority'],
					dataframe['seniority']
				)
				# Assignment for function_{col} according to persona synonyms
				dataframe.loc[dataframe[col] == 1 , f"function_{results[col]['function']}"] = 1

		dataframe = dataframe[['seniority'] + func_cols]

		dataframe_original = dataframe_original.join(dataframe)

		if drop:
			dataframe_original.drop(column, axis=1, inplace=True)

		return dataframe_original


