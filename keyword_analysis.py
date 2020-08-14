from collections import Counter

import numpy as np
from tqdm.autonotebook import tqdm

from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
from sklearn.feature_extraction.text import CountVectorizer


def pointwise_mutual_information(contingency_matrix):
	expected_freq_matrix = expected_freq(contingency_matrix)
	return {'pmi': np.log2(contingency_matrix[0][0]/expected_freq_matrix[0][0])}

def bootstrap_test(freq_c1, freq_c2, n_samples):
	"""
	Python implementation of the bootstrap test proposed by Lijffijt et al. (2016). 
	See: Lijffijt et al. 2016, Significance Testing of Word Frequencies in Corpora
	"""
	h = lambda v1, v2: 1 if v1 > v2 else (0.5 if v1 == v2 else 0)
	n = min([len(freq_c1), len(freq_c2)])
	p1 = 0
	for i in range(n_samples):
		p1 += h(np.mean(np.random.choice(freq_c1, size=n, replace=True)), 
				np.mean(np.random.choice(freq_c2, size=n, replace=True))) 
	p1 = p1/n_samples
	p2 = (1+ 2 * n_samples * min(p1,1 - p1))/(1 + n_samples)
	return {'bootstrap_p': p2}

def chi_squared(contingency_matrix, signed=True):
	chisq, p, _, _ = chi2_contingency(contingency_matrix)
	if signed:
		if contingency_matrix[0][0] / contingency_matrix[0][1] < contingency_matrix[1][0] / contingency_matrix[1][1]: 
			chisq = -chisq
	return {'chi': chisq, 'chi_p': p}

def loglikelihood(contingency_matrix, signed=True):
	llr, p, _, _ = chi2_contingency(contingency_matrix, lambda_='log-likelihood')
	if signed:
		if contingency_matrix[0][0] / contingency_matrix[0][1] < contingency_matrix[1][0] / contingency_matrix[1][1]: 
			llr = -llr
	return {'llr': llr, 'llr_p': p}

def get_dt_matrix(target_corpus, reference_corpus):
	cv = CountVectorizer(analyzer=lambda x: (word for word in x))
	cv.fit(np.concatenate((target_corpus, reference_corpus)))
	dt_target = cv.transform(target_corpus).toarray()
	dt_target_norm = dt_target / np.sum(dt_target, axis=1).reshape(-1, 1)
	dt_reference = cv.transform(reference_corpus).toarray()
	dt_reference_norm = dt_reference / np.sum(dt_reference, axis=1).reshape(-1, 1)
	return dt_target_norm, dt_reference_norm, cv.get_feature_names()

def significant_words(target_corpus, reference_corpus, tests=('chi', 'pmi'), bootstrap_samples=9999, 
						signed=True, normalize_counts=False):
	func_dict = {'pmi': pointwise_mutual_information, 
				'chi': chi_squared, 
				'bootstrap': bootstrap_test,
				'llr': loglikelihood}
	word_statistics = []
	if 'bootstrap' in tests:
		dt_target_norm, dt_reference_norm, feature_names = get_dt_matrix(target_corpus, reference_corpus)
	target_corpus_conc = np.concatenate(target_corpus)
	reference_corpus_conc = np.concatenate(reference_corpus)
	length_c1 = target_corpus_conc.shape[0]
	length_c2 = reference_corpus_conc.shape[0]
	count_dict_c1 = Counter(target_corpus_conc)
	count_dict_c2 = Counter(reference_corpus_conc)
	for word_c1 in tqdm(count_dict_c1.keys()):
		count_c1 = count_dict_c1[word_c1] 
		count_c2 = count_dict_c2[word_c1]
		if normalize_counts:
			assert type(normalize_counts) == int, 'If not false, normalize_counts should be an integer'
			freq_c1 = (count_c1 / length_c1) * normalize_counts
			freq_c2 = (count_c2 / length_c2) * normalize_counts
		else:
			freq_c1, freq_c2 = count_c1, count_c2
		result_dict = {'freq_c1': freq_c1, 'freq_c2': freq_c2, 'word': word_c1}
		contingency_matrix = np.array([[count_c1, length_c1 - count_c1], [count_c2, length_c2 - count_c2]])
		for test in tests:
			if test == 'pmi':
				results = func_dict[test](contingency_matrix)
			elif test == 'bootstrap':
				results = func_dict[test](dt_target_norm[:,feature_names.index(word_c1)], 
										dt_reference_norm[:,feature_names.index(word_c1)],
										n_samples=bootstrap_samples)
			else:
				results = func_dict[test](contingency_matrix, signed=signed)
			result_dict.update(results)
		word_statistics.append(result_dict)
	return word_statistics
