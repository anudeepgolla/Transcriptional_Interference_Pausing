import pandas as pd
import numpy as np


# df = pd.read_csv('data/sequence_ds_v1.csv')
# print(df.shape)
#
# # Sequence Indexing
# ps_df = list(df['pause_seq'].str)
#
# for i in range(len(ps_df)):
#
#     df['pause_seq_{}'.format(i)] = ps_df[i]
#
# pc_df = list(df['pause_context'].str)
#
# for i in range(len(pc_df)):
#     df['pause_context_{}'.format(i)] = pc_df[i]
#
# print(df.shape)
#
# df.to_csv('data/sequence_ds_indexed_v2.csv')




# df = pd.read_csv('data/sequence_ds_indexed_v2.csv')
# print(df.shape)
#
# # Ordinal Encoding
# ordinal_cols = []
# for c in df.columns:
#     if c[:10] == 'pause_seq_' or c[:14] == 'pause_context_':
#         ordinal_cols.append(c)
#
# ordinal_mapper = {'A': 0.25, 'C': 0.50, 'G': 0.75, 'T': 1.00}
#
# for i in range(df.shape[0]):
#     for c in ordinal_cols:
#         df.loc[i, c] = ordinal_mapper[df.loc[i, c]]
#
#     if (i+1) % 100 == 0:
#         print('Step {:5}/{:5} = {:2}%'.format(i+1, 68893, (i+1)*100//68893))
#
# print(df.head(5))
#
# df.to_csv('data/sequence_ds_ordinal_v3.csv')
# print('Saved!')





df = pd.read_csv('../data/seq/sequence_ds_indexed_v2.csv')
print(df.shape)

# One-Hot Encoding
onehot_cols = []
for c in df.columns:
    if c[:10] == 'pause_seq_' or c[:14] == 'pause_context_':
        onehot_cols.append(c)

# for c in df.columns:
#     if c not in onehot_cols:
#         print(c)

dummies = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols)
# df = pd.concat([df, dummies], axis=1)
# print(df.shape)

print(dummies.shape)
print(dummies.columns)
# for c in dummies.columns[:]:
#     print(c)
#
# dummies.to_csv('data/sequence_ds_onehot_v1.csv')
# print('Saved!')



# df = pd.read_csv('../data/seq/sequence_ds_v1.csv')
# print(df.shape)
#
# # KMers Encoding
# def getKmers(sequence, size):
#     return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
#
# kmer_size = 12
# df['pause_seq_kmers'] = np.nan
# df['pause_context_kmers'] = np.nan
# for i in range(df.shape[0]):
#     df.loc[i, 'pause_seq_kmers'] = ' '.join(getKmers(df.loc[i, 'pause_seq'], size=kmer_size))
#     df.loc[i, 'pause_context_kmers'] = ' '.join(getKmers(df.loc[i, 'pause_context'], size=kmer_size))
#
#     if (i + 1) % 100 == 0:
#         print('Step {:5}/{:5} = {:2}%'.format(i+1, 68893, (i+1)*100//68893))
#
# print(df.shape)
# print(df.head(10))
#
# df.to_csv('../data/seq/sequence_ds_kmers_12_v1.csv')
