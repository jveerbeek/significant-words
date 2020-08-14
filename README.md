# Significant words

A Python script to compare the word frequencies of two corpora, AntConc style. Tests implemented: chi-squared, log-likelihood ratio, pointwise mutual information, and [the bootstrap test](https://users.ics.aalto.fi/lijffijt/bootstraptest/)

## Working example

```python
from keyword_analysis import significant_words

import numpy as np 
import pandas as pd

from nltk.tokenize import RegexpTokenize
from sklearn.datasets import fetch_20newsgroups

# Load dataset and tokenize it (could be any tokenizer)
tokenizer = RegexpTokenizer('\w+')
newsgroups_train = fetch_20newsgroups(remove=('headers', 'footers'))
tokenized_texts = np.array([[word.lower() for word in tokenizer.tokenize(text)] for text in newsgroups_train.data])
target_corpus = tokenized_texts[newsgroups_train.target == newsgroups_train.target_names.index('talk.politics.guns')]
reference_corpus = tokenized_texts[newsgroups_train.target != newsgroups_train.target_names.index('talk.politics.guns')]

results = significant_words(target_corpus, 
                         reference_corpus, 
                        tests=('pmi', 'chi', 'llr', 'bootstrap'),
                         bootstrap_samples=1000
                        )

df = pd.DataFrame(results)
print(df.sort_values('chi', ascending=False).head(10))

```
# significant-words
