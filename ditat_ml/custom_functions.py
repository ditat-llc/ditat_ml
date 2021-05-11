'''
Custom functions


Notes:
	This could be transformed into a single class and module.
'''
import re

import pandas as pd
import numpy as np

from . utility_functions import dummies_with_options_and_limit, dummies_with_limit, time_it


job_title_mapping = {
	'Founder': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'CEO': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Chief Executive Officer': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Co-Founder': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'President': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Executive Director': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Chairman': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Senior Managing Director': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Senior Partner': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Managing Principal': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Managing Partner': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Managing Director': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Chief Operating Officer': {
		'seniority': 9,
		'function': 'Operations'
	},
	'COO': {
		'seniority': 9,
		'function': 'Operations'
	},
	'Operations': {
		'seniority': 5,
		'function': 'Operations'
	},
	'Ops': {
		'seniority': 5,
		'function': 'Operations'
	},
	'Chief Financial Officer': {
		'seniority': 9,
		'function': 'Finance'
	},
	'CFO': {
		'seniority': 9,
		'function': 'Finance'
	},
	'Controller': {
		'seniority': 7,
		'function': 'Finance'
	},
	'Treasurer': {
		'seniority': 7,
		'function': 'Finance'
	},
	'Fund Controller': {
		'seniority': 7,
		'function': 'Finance'
	},
	'Vice President-Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Vice President & Treasurer': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Director-Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Director of Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Finance Director': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Vice President-Finance & Administration': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Vice President-Finance & Treasurer': {
		'seniority': 5,
		'function': 'Finance'
	},
	'VP Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Senior Vice President-Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Vice President & Controller': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Financial Controller': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Comptroller': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Vice President of Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'VP of Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Director-Treasury': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Director-Finance & Administration': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Director-Treasury Services': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Vice President, Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Vice President-Business & Finance': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Director-Treasury Operations': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Financial Director': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Finance Manager': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Accounting Manager': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Treasury Manager': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Manager-Treasury Operations': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Financial Manager': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Director-Accounting': {
		'seniority': 5,
		'function': 'Finance'
	},
	'Assistant Treasurer': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Senior Accountant': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Financial Analyst': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Fund Accountant': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Fund Administrator': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Senior Financial Analyst': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Accountant': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Cash Manager': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Staff Accountant': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Finance Associate': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Finance Assistant': {
		'seniority': 2,
		'function': 'Finance'
	},
	'CPA': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Associate Treasurer': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Investment Accountant': {
		'seniority': 2,
		'function': 'Finance'
	},
	'Partner, Investor Relations': {
		'seniority': 7,
		'function': 'Investor Relations'
	},
	'Head of Investor Relations': {
		'seniority': 7,
		'function': 'Investor Relations'
	},
	'Head of IR': {
		'seniority': 8,
		'function': 'Investor Relations'
	},
	'Managing Director, Investor Relations': {
		'seniority': 8,
		'function': 'Investor Relations'
	},
	'Managing Director, Client Management': {
		'seniority': 7,
		'function': 'Investor Relations'
	},
	'Senior Vice President-Investor Relations': {
		'seniority': 7,
		'function': 'Investor Relations'
	},
	'Director of Marketing': {
		'seniority': 7,
		'function': 'Investor Relations'
	},
	'Marketing Director': {
		'seniority': 7,
		'function': 'Investor Relations'
	},
	'Director of Communications': {
		'seniority': 7,
		'function': 'Investor Relations'
	},
	'Chief Marketing Officer': {
		'seniority': 8,
		'function': 'Investor Relations'
	},
	'CMO': {
		'seniority': 8,
		'function': 'Investor Relations'
	},
	'Investor Relations Associate': {
		'seniority': 4,
		'function': 'Investor Relations'
	},
	'Associate, Investor Relations': {
		'seniority': 4,
		'function': 'Investor Relations'
	},
	'Investor Relations Analyst': {
		'seniority': 3,
		'function': 'Investor Relations'
	},
	'Investor Relations Coordinator': {
		'seniority': 3,
		'function': 'Investor Relations'
	},
	'Marketing Associate': {
		'seniority': 4,
		'function': 'Investor Relations'
	},
	'Marketing Coordinator': {
		'seniority': 3,
		'function': 'Investor Relations'
	},
	'Marketing Assistant': {
		'seniority': 2,
		'function': 'Investor Relations'
	},
	'Marketing': {
		'seniority': 5,
		'function': 'Investor Relations'
	},

	'Partner': {
		'seniority': 9,
		'function': 'Firm Leader'
	},
	'Vice President': {
		'seniority': 5,
		'function': 'Firm Leader'
	},
	'Senior Vice President': {
		'seniority': 6,
		'function': 'Firm Leader'
	},
	'Director': {
		'seniority': 8,
		'function': 'Firm Leader'
	},
	'Principal': {
		'seniority': 8,
		'function': 'Firm Leader'
	},
	'Associate': {
		'seniority': 4,
		'function': 'Other'
	},
	'Co': {
		'seniority': 8,
		'function': 'Firm Leader'
	},
	'Finance': {
		'seniority': 3,
		'function': 'Finance'
	},
	'General Partner': {
		'seniority': 8,
		'function': 'Firm Leader'
	},
	'Investor Relations': {
		'seniority': 3,
		'function': 'Investor Relations'
	},
	'Operating Partner': {
		'seniority': 8,
		'function': 'Firm Leader'
	},
	'VP': {
		'seniority': 5,
		'function': 'Firm Leader'
	},
	'Executive Vice President': {
		'seniority': 6,
		'function': 'Firm Leader'
	},
	'Senior Associate': {
		'seniority': 5,
		'function': 'Other'
	},
	'Venture Partner': {
		'seniority': 8,
		'function': 'Firm Leader'
	},
	'Human Resources': {
		'seniority': 3,
		'function': 'Other'
	},
	'Analyst': {
		'seniority': 3,
		'function': 'Other'
	},
	'Manager': {
		'seniority': 8,
		'function': 'Firm Leader'
	},
	'Owner': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Founding Partner': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
	'Investments': {
		'seniority': 4,
		'function': 'Finance'
	},
	'Co,Founder': {
		'seniority': 10,
		'function': 'Firm Leader'
	},
}

persona_synonyms = {
	'Finance': ['finance', 'financial', 'investment', 'investments', 'equity', 'portfolio', 'treasury', 'accounting', 'market', 'markets', 'tax', 'investor'],
	'Investor Relations': ['investor relations', 'ir', 'marketing', 'relationship', 'relationships', 'communications', 'communication', 'sales', 'account'],
	'Operations': ['operations', 'operating', 'administration', 'compliance', 'office', 'administrative'],
	'Firm Leader': ['executive', 'general', 'associate', 'strategy'],
	'Technology': ['technology', 'it', 'software'],
	'Deal Team': [],
	'Other': ['research', 'legal']
}

seniority_mapping = {
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


def job_title_processor(value):
	'''
	Job title processor. It gives meaning to a value.

	Evaluates 1) Area/Function/Persona 2) Seniority
	
	Some references:
		seniority https://docs.microsoft.com/en-us/linkedin/shared/references/reference-tables/seniority-codes
		job function https://docs.microsoft.com/en-us/linkedin/shared/references/reference-tables/job-function-codes

		'unpaid': 1,
		'training': 2,
		'entry': 3,
		'senior': 4,
		'manager': 5,
		'director': 6,
		'vice president': 7,
		'cxo': 8,
		'partner': 9,
		'owner': 10

	'''
	if not value or not isinstance(value, str):
		return None

	# Lower casing titles.
	value = value.lower()
	mapping = {i.lower(): j for i, j in job_title_mapping.items()}

	if value in mapping:
		# If value exists in the predefined mapping (job_title_mapping)

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
			'c': 'Other'
		}
		temp_function = last_word[1]
		# Arbitrary if CEO -> 10 else 8
		seniority = 10 if temp_function == 'e' else 8
		return {'seniority': seniority, 'function': temp_map[temp_function]}

	elif 'chief' in value and 'officer' in value:
		# Chief {Something} Officer
		second = value.split('chief')[1].split('office')[0].strip()
		temp_function = None
		for persona, synonym in persona_synonyms.items():
			if second in synonym:
				temp_function = persona
				break
		seniority = 10 if temp_function == 'executive' else 8
		return {'seniority': seniority, 'function': temp_function}

	# II. "Function seniority". 2-word combination like "Finance Manager" (Function Seniority)
	elif len(value.split()) == 2:
		temp_function = None
		first, second = value.split()
		for persona, synonym in persona_synonyms.items():
			if first in synonym:
				temp_function = persona
				break
		temp_seniority = seniority_mapping.get(second)
		if temp_function and temp_seniority:
			return {'seniority': temp_seniority, 'function': temp_function}

	# III. Same logic as II, but instead of Finance manager, it is Manager of finance.
	elif len(value.split()) == 3 and value.split()[1] == 'of':
		temp_function = None
		first, second, third = value.split()
		for persona, synonym in persona_synonyms.items():
			if third in synonym:
				temp_function = persona
				break
		temp_seniority = seniority_mapping.get(first)
		if temp_function and temp_seniority:
			return {'seniority': temp_seniority, 'function': temp_function}

	# IV. For single words. It assumes that the single references a seniority like (Manager),
	# it this does not happen we assume the single word references a Function with 'default_' seniority.
	elif len(value.split()) == 1:
		temp_function = None
		temp_seniority = None
		if value in seniority_mapping:
			temp_seniority = seniority_mapping[value]
			temp_function = 'Other'
		else:
			for persona, synonym in persona_synonyms.items():
				if value in synonym:
					temp_function = persona
					break
			temp_seniority = seniority_mapping['default_']			
		if temp_function and temp_seniority:
			return {'seniority': temp_seniority, 'function': temp_function}

	# Other cases.
	temp_function = None
	temp_seniority = None

	for persona, synonym in persona_synonyms.items():
		for syn in synonym:
			if syn in value:
				temp_function = persona
				break

	if temp_function:
		for s in seniority_mapping:
			if s in value:
				temp_seniority = seniority_mapping[s]
	
	if temp_function and temp_seniority:
		return {'seniority': temp_seniority, 'function': temp_function}

	return None

# @time_it()
def job_title_processor_df(dataframe, column, drop=False):
	'''
	Dataframe functionality for job_title_processor()
	'''
	dataframe_original = dataframe.copy()
	dataframe = dataframe[[column]]

	replacement_mapping = {
		'&': ',',
		'and': ',',
		'/': ',',
		'-': ',',
		'\+': ','
	}

	dataframe = dataframe_original.replace({column: replacement_mapping}, regex=True)

	dataframe = dummies_with_options_and_limit(
		dataframe=dataframe,
		col=column,
		verbose=False,
		prefix=False,
		return_unique_list=False,
		limit=None,
		return_value_counts=False
	)

	results = {col: job_title_processor(col) for col in dataframe.columns}

	dataframe['seniority'] = 0
	
	func_cols = [f"function_{persona}" for persona in persona_synonyms]
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




