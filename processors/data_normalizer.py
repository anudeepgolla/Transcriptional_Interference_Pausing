import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

"""

data_normalizer.py : transform data with data of all types into a dataset that is only numbers for training

"""

# df = pd.read_csv('data/genomic_data_set_v2.csv')
#
# # df = df[:][:500]
#
# # print(df.nunique())
# # print(df.columns, len(df.columns))
# # df.columns = ['Unnamed: 0', 'position.drop', 'pause_status', 'position', 'ref_base',
# #        'pause_seq', 'pause_G', 'pause_C', 'pause_context', 'context_G',
# #        'context_C', 'gene', 'gene_start', 'gene_end', 'start_dist_abs',
# #        'start_dist_rel', 'end_dist_abs', 'end_dist_rel', 'trans_base']
# # df = df.drop(['position.drop', 'Unnamed: 0'], axis = 1)
# # print(df.columns)
# # print(df['position'].head(5))
# # df.to_csv('data/genomic_data_set_v2.csv')
# # print(df['gene_start'].head(5))
# # print(df.dtypes)
#
#
#
# min_max_cols = ['position', 'gene_start', 'gene_end']
#
# norm_cols = ['pause_G', 'pause_C', 'context_G', 'context_C', 'start_dist_abs',
#              'start_dist_rel', 'end_dist_abs', 'end_dist_rel']
#
# cat_cols = ['pause_status', 'ref_base', 'trans_base']
#
# seq_cols = ['pause_seq', 'pause_context']
#
# spec_cols = ['gene', 'pause_status']
#
# na_cols = ['gene_start', 'gene_end', 'start_dist_abs', 'start_dist_rel', 'end_dist_abs', 'end_dist_rel']
#
# drop_cols = ['pause_status', 'ref_base', 'trans_base', 'pause_seq', 'pause_context', 'Unnamed: 0']
#
#
# # indicator scale
# for col in spec_cols:
#     if col == 'gene':
#         for i in range(len(df[col])):
#             if pd.isna(df[col][i]):
#                 df.set_value(i, col, 0)
#             else:
#                 df.set_value(i, col, 1)
#     elif col == 'pause_status':
#         for i in range(len(df[col])):
#             if df[col][i] == 'HC True':
#                 df.set_value(i, col, 'hc_true')
#             elif df[col][i] == 'LC True':
#                 df.set_value(i, col, 'lc_true')
#             elif df[col][i] == 'FALSE':
#                 df.set_value(i, col, 'false')
#             elif df[col][i] == False:
#                 df.set_value(i, col, 'false')
#
#
# # print('SEQUENCE B4', df['pause_seq'][0])
# for col in seq_cols:
#     seq_len = len(df[col][0])
#     for j in range(seq_len):
#         # print(col, j)
#         df['{}_{}'.format(col, j)] = np.nan
#     # print(df.columns)
#     for i in range(len(df[col])):
#         seq = list(str(df[col][i]))
#         # print(seq)
#         for j in range(len(seq)):
#             # print(i, '{}_{}'.format(col, j), str(seq[j]))
#             # print(df['{}_{}'.format(col, j)][i])
#             # df.set_value(i, '{}_{}'.format(col, j), str(seq[j]))
#             # print(i, '{}_{}'.format(col, j), seq[j])
#             df.loc[i, '{}_{}'.format(col, j)] = str(seq[j])
#
#         if i % 100 == 0:
#             print("{} Modifcation Step: {}/{}".format(col, i, len(df[col])))
#
# print(df['pause_seq'][0])
# print('SEQUENCE After', df['pause_seq_0'][0], df['pause_seq_1'][0], df['pause_seq_2'][0], df['pause_seq_15'][0])
#
#
#
# # min_max scale
# for col in min_max_cols:
#     df[col] = (df[col]-df[col].min(skipna=True))/(df[col].max(skipna=True)-df[col].min(skipna=True))
#
# # gaussian scale
# for col in norm_cols:
#     df[col] = (df[col] - df[col].mean(skipna=True)) / df[col].std(skipna=True)
#
# new_cat_cols = []
# for c in df.columns:
#     if c != 'pause_seq' and c!= 'pause_context':
#         if c[:9] == 'pause_seq':
#             new_cat_cols.append(c)
#         if c[:13] == 'pause_context':
#             new_cat_cols.append(c)
# cat_cols.extend(new_cat_cols)
#
#
# # one-hot encode
# for col in cat_cols:
#     dummies = pd.get_dummies(df[col], prefix = col)
#     df = pd.concat([df, dummies], axis=1, sort=False)
#
# # fill nan
# for col in na_cols:
#     df[col].fillna(0)
#
# # drop
# drop_cols.extend(new_cat_cols)
# df = df.drop(drop_cols, axis=1)
#
#
# print(df.head(5))
# print(df.columns)
# print(df.dtypes)
#
#
#
#
# df.to_csv('data/processed_genomic_data_set_v2.csv')

# load data
df = pd.read_csv('../data/util/genomic_data_set_v2.csv')
print(df.head(10))

# remove unneccesary cols
drop_cols = ['pause_seq', 'pause_context', 'gene_start', 'gene_end', 'start_dist_abs', 'end_dist_abs']

# params that are pure numerical
num_cols = ['position', 'pause_G', 'pause_C', 'context_G', 'context_C', 'gene_len', 'start_dist_rel', 'end_dist_rel']
# params that are pure numerical and follow gaussian distribution
norm_cols = ['pause_G', 'pause_C', 'context_G', 'context_C', 'gene_len']
# params that are pure numerical and follow linear distribution
lin_cols = ['position', 'start_dist_rel', 'end_dist_rel']
# params that are categorical
cat_cols = ['ref_base', 'trans_base']

# for c in df.columns:
#     print(c, df[c].isna().sum(), df[c].nunique())

# print(df['pause_status'].unique())

# choose all data that are not LC True pause sites
df = df.loc[df['pause_status'] != 'LC True']
df.reset_index(drop=True, inplace=True)

# create new column for gene-len based on gene_start and gene_end
df['gene_len'] = df['gene_end'] - df['gene_start']
# iterate through all samples in the dataset
for i in range(df.shape[0]):
    # based on pause state, change string into either 0 or 1
    if df.loc[i, 'pause_status'] in {'HC True'}:
        df.loc[i, 'pause_status'] = 1
    else:
        df.loc[i, 'pause_status'] = 0

    if df.loc[i, 'gene']:
        df.loc[i, 'gene'] = 1

    if i %1000==0:
        print('STEP {}/{}'.format(i, df.shape[0]))


# make all params in cat_cols categorically encoded (one-hot encoding)
df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

# adjust all linear parameters so they fall into [0,1] rangee
for col in lin_cols:
    mean, max, min  = df[col].mean(), df[col].max(), df[col].min()
    # linearize
    df[col] = df[col].apply(lambda x: (x - mean)/(max - min))

# adjust gaussian params by normalizing the column with (z-scores)
for col in norm_cols:
    mean, std = df[col].mean(), df[col].std()
    # normalize
    df[col] = df[col].apply(lambda x: (x - mean) / (std))

# remove unneccesary columns
df = df.drop(drop_cols, axis=1)
# fill all nan or empty cells with 0
df = df.fillna(0)

for c in df.columns:
    print(c)
    print(df[c].head(5))


# # plt.hist(x='context_G')
# hist = df.hist(column='start_dist_rel', by='pause_status')
# plt.show()


# save data to csv
df.to_csv('../data/param/param_ds_prcs_selectivelabel_v1.csv')

