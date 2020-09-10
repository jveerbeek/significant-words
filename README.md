# Significant words

A Python script to compare the word frequencies of two corpora, AntConc style. Tests implemented: chi-squared, log-likelihood ratio, pointwise mutual information, and [the bootstrap test](https://users.ics.aalto.fi/lijffijt/bootstraptest/).


### Dependencies
* numpy
* scikit-learn
* scipy
* tqdm

### Working example

```python
from keyword_analysis import significant_words

import numpy as np 
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups

# Load dataset and tokenize it (could be any tokenizer)

tokenizer = RegexpTokenizer('\w+')
newsgroups_train = fetch_20newsgroups(remove=('headers', 'footers'))
tokenized_texts = np.array([[word.lower() for word in tokenizer.tokenize(text)] 
							for text in newsgroups_train.data])

class_of_interest = 'talk.politics.guns'
index_class = newsgroups_train.target_names.index(class_of_interest)
target_corpus = tokenized_texts[newsgroups_train.target == index_class]
reference_corpus = tokenized_texts[newsgroups_train.target != index_class]

results = significant_words(target_corpus, 
                         reference_corpus, 
                        tests=('pmi', 'chi', 'llr'),
                         bootstrap_samples=1000
                        )

df = pd.DataFrame(results)
print(df.sort_values('chi', ascending=False).head(10))

#       freq_c1  freq_c2       word   pmi       chi  chi_p      llr  llr_p
# 429       820      165        gun  3.81  10512.14    0.0  3757.37    0.0
# 218       419       77       guns  3.83   5455.10    0.0  1942.65    0.0
# 476       294       11   firearms  4.02   4444.88    0.0  1560.55    0.0
# 43        307      144    weapons  3.52   3101.10    0.0  1182.49    0.0
# 3042      192       45    militia  3.77   2374.27    0.0   854.85    0.0
# 908       143        0    firearm  4.07   2244.80    0.0   797.85    0.0
# 857       142        3    handgun  4.04   2177.02    0.0   766.24    0.0
# 464       168       44     weapon  3.74   2023.32    0.0   733.18    0.0
# 2080      106        3       rkba  4.03   1608.15    0.0   565.02    0.0
# 257       231      243       fire  3.03   1543.96    0.0   674.40    0.0
# 655       133       42       batf  3.68   1522.92    0.0   559.17    0.0
# 623       194      177        fbi  3.14   1416.42    0.0   600.52    0.0
# 169       100       11   handguns  3.92   1390.03    0.0   489.18    0.0
# 885       207      245      crime  2.94   1276.26    0.0   572.64    0.0
# 1527      149      117  amendment  3.24   1183.26    0.0   487.58    0.0
# 920        92       18        nra  3.81   1173.13    0.0   419.12    0.0
# 1247      116       55   compound  3.51   1159.97    0.0   443.26    0.0
# 595        83       12       roby  3.88   1111.45    0.0   393.25    0.0
# 147       115       61       semi  3.46   1099.43    0.0   426.12    0.0
# 543        84       15        atf  3.83   1087.81    0.0   387.33    0.0

```

