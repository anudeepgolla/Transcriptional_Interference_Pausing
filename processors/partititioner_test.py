import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

kmers = 10
print(kmers)

df = pd.read_csv('../data/seq/sequence_ds_kmers_{}_no_spike_selective_v1.csv'.format(kmers))
print(df.columns)

target = np.array(df['pause_status'])


print(df['pause_context_kmers'][0])

corpus = np.array(df['pause_context_kmers'])
print(corpus.shape)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(len(vectorizer.get_feature_names()))

X_data = X.toarray()

print(X_data.shape)
print(np.sum(X_data[0]))

np.save('../data/seq/special/X_data_kmers_{}_vectorized_no_spike_selective.npy'.format(kmers), X_data)
np.save('../data/seq/special/y_data_kmers_{}_vectorized_no_spike_selective.npy'.format(kmers), target)



# #np.save('../data/extra/X_data_kmers_{}_vectorized_no_spike_selective.npy'.format(kmers), X_data)
# np.save('../data/extra/y_data_kmers_{}_vectorized_no_spike_selective.npy'.format(kmers), target)


