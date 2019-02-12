import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer



# df = pd.read_csv('ata/processed_genomic_data_set_v2.csv')
#
# cols = list(df.columns)
#
# print(df.head(5))
# print(df.dtypes)
# print(cols)
# print(len(cols))
# print(df.nunique()


# df = pd.read_csv('../data/seq/sequence_ds_onehot_no_spike_selective_v1.csv')
#
# print(df.columns)
# print(df.shape)
#
# print(df.head(5))
# print(df.tail(5))


# df.to_csv('../data/seq/sequence_ds_kmers_6_given_spike_loose_v2.csv')


# X_data = np.array([])
#
# for i in range(len)

# target = np.array(df['pause_status'])
# print(target.shape)
# rem_cols = ['Unnamed: 0', 'position', 'pause_status']
# for c in df.columns:
#     if c[:9] == 'pause_seq':
#         rem_cols.append(c)
# df = df.drop(rem_cols, axis=1)
# print(df.shape)
#
# X_data = np.array(df.values.tolist())
# print(X_data.shape)
# print(X_data[0].shape)
# X_data = X_data.reshape((7790, 218, 4))
# print(X_data.shape)
# print(X_data[0])
#
# np.save('../data/seq/special/X_data_onehot_no_spike_selective.npy', X_data)
# np.save('../data/seq/special/y_data_onehot_no_spike_selective.npy', target)





# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_given_spike_selective_v1.csv')
# print(df.columns)
#
# df = pd.read_csv('../data/seq/sequence_ds_onehot_given_spike_selective_v1.csv')
# print(df.columns)

# df = pd.read_csv('../data/seq/sequence_ds_ordinal_given_spike_selective_v1.csv')
# print(df.columns)
#
# target = np.array(df['pause_status'])
# rem_cols = ['Unnamed: 0', 'position', 'pause_status']
# for c in df.columns:
#     if c[:9] == 'pause_seq':
#         rem_cols.append(c)
# df = df.drop(rem_cols, axis=1)
# print(df.shape)
# print(df.columns)
# X_data = np.array(df.values.tolist())
# print(X_data.shape)
# print(X_data[-1])
# print(type(X_data[-1][0]))
#
# map = {
#     'A': 0.25,
#     'C': 0.50,
#     'G': 0.75,
#     'T': 1.00,
#     '0.25': 0.25,
#     '0.5': 0.50,
#     '0.75': 0.75,
#     '1.0': 1.00,
# }
#
# for i in range(X_data.shape[0]):
#     for j in range(X_data.shape[1]):
#         X_data[i][j] = map[X_data[i][j]]
#     if i % 50 == 0:
#         print('Step {:4}/{:4} = {:2}%'.format(i, 7790, (i*100/7790)))
#
# print(X_data[-1])
# print(type(X_data[-1][0]))
#
# np.save('../data/seq/special/X_data_ordinal_no_spike_selective.npy', X_data)
# np.save('../data/seq/special/y_data_ordinal_no_spike_selective.npy', target)
#




# vocab = {}


# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_no_spike_selective_v1.csv')
# print(df.columns)
#
# target = np.array(df['pause_status'])
#
#
# print(df['pause_context_kmers'][0])
#
# corpus = np.array(df['pause_context_kmers'])
# print(corpus.shape)
#
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(len(vectorizer.get_feature_names()))
#
# X_data = X.toarray()
#
# print(X_data.shape)
# print(np.sum(X_data[0]))
#
# np.save('../data/seq/special/X_data_kmers_6_vectorized_no_spike_selective.npy', X_data)
# np.save('../data/seq/special/y_data_kmers_6_vectorized_no_spike_selective.npy', target)


# X = np.load('../data/seq/special/X_data_kmers_4_vectorized_no_spike_selective.npy')
# y = np.load('../data/seq/special/y_data_kmers_4_vectorized_no_spike_selective.npy')
# print(X.shape)
# print(y.shape)

X = np.load('../data/seq/special/X_data_kmers_6_vectorized_no_spike_selective.npy')
print(X.shape)
print(X[0])
