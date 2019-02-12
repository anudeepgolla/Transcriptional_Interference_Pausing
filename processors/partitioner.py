import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# df = pd.read_csv('../data/param/param_ds_prcs_looselabel_v1.csv')
# print(df.shape)
# print(df['pause_status'].value_counts())
#
#
# df = pd.read_csv('../data/param/param_ds_prcs_strictlabel_v1.csv')
# print(df.shape)
# print(df['pause_status'].value_counts())
#
#
# df = pd.read_csv('../data/seq/sequence_ds_indexed_v1.csv')
# print(df.shape)
# print(df['pause_status'].value_counts())
# print(df['energy_spike'].value_counts())
#
# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_v1.csv')
# print(df.shape)
# print(df['pause_status'].value_counts())
# print(df['energy_spike'].value_counts())
#
# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df['pause_status'].value_counts())
# print(df['energy_spike'].value_counts())
#
# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v1.csv')
# print(df.shape)
# print(df['pause_status'].value_counts())
# print(df['energy_spike'].value_counts())
#
# df = pd.read_csv('../data/seq/sequence_ds_v1.csv')
# print(df.shape)
# print(df['pause_status'].value_counts())
# print(df['energy_spike'].value_counts())





# param_cols = ['pause_status', 'position', 'pause_G',
#        'pause_C', 'context_G', 'context_C', 'gene', 'start_dist_rel',
#        'end_dist_rel', 'gene_len', 'ref_base_A', 'ref_base_C', 'ref_base_G',
#        'ref_base_T', 'trans_base_A', 'trans_base_C', 'trans_base_G',
#        'trans_base_T']
#
# df = pd.read_csv('../data/param/param_ds_prcs_selectivelabel_v1.csv')
# print(df.shape)
# df = df[param_cols]
# print(df.shape)
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
#
# print(df_false.shape)
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# print(df_false.shape)
#
#
# df_fin = pd.concat([df_true]*5, ignore_index=True)
# df_fin = pd.concat([df_fin, df_false], ignore_index=True)
# print(df_fin.shape)
#
# df_fin.to_csv('../data/param/param_ds_prcs_selectivelabel_sample_adj_v1.csv')





# param_cols = ['pause_status', 'position', 'pause_G',
#        'pause_C', 'context_G', 'context_C', 'gene', 'start_dist_rel',
#        'end_dist_rel', 'gene_len', 'ref_base_A', 'ref_base_C', 'ref_base_G',
#        'ref_base_T', 'trans_base_A', 'trans_base_C', 'trans_base_G',
#        'trans_base_T']
#
# df = pd.read_csv('../data/param/param_ds_prcs_strictlabel_v1.csv')
#
# print(df.shape)
# df = df[param_cols]
# print(df.shape)
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
#
# print(df_false.shape)
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# print(df_false.shape)
#
#
# df_fin = pd.concat([df_true]*5, ignore_index=True)
# df_fin = pd.concat([df_fin, df_false], ignore_index=True)
# print(df_fin.shape)
#
# df_fin.to_csv('../data/param/param_ds_strictlabel_sample_adj_v1.csv')




# param_cols = ['pause_status', 'position', 'pause_G',
#        'pause_C', 'context_G', 'context_C', 'gene', 'start_dist_rel',
#        'end_dist_rel', 'gene_len', 'ref_base_A', 'ref_base_C', 'ref_base_G',
#        'ref_base_T', 'trans_base_A', 'trans_base_C', 'trans_base_G',
#        'trans_base_T']
#
# df = pd.read_csv('../data/param/param_ds_prcs_looselabel_v1.csv')
#
# print(df.shape)
# df = df[param_cols]
# print(df.shape)
#
# df.to_csv('../data/param/param_ds_looselabel_sample_adj_v1.csv')








# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v2.csv')
# print(df.shape)
# print(df.columns)
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'position',
#        'pause_status', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# df.to_csv('../data/seq/sequence_ds_ordinal_spike_v1.csv')

#
# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df.columns)
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'position',
#        'pause_status', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# df.to_csv('../data/seq/sequence_ds_onehot_spike_v1.csv')







#
# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_v1.csv')
# print(df.shape)
# print(df.columns)
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'position', 'pause_status', 'pause_seq',
#        'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# df.to_csv('../data/seq/sequence_ds_kmers_spike_v1.csv')











# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v2.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     df.loc[i, 'pause_status'] = 1 if df.loc[i, 'pause_status'] in {'HC True', 'LC True'} else 0
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
#
# df.to_csv('../data/seq/sequence_ds_ordinal_given_spike_loose_v1.csv')





# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v2.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df.reset_index(drop=True, inplace=True)
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     elif df.loc[i, 'pause_status'] in {'LC True'}:
#         df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
#
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_ordinal_given_spike_selective_v1.csv')
#
#
#
#
#
#
# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v2.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     # elif df.loc[i, 'pause_status'] in {'LC True'}:
#     #     df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
#
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_ordinal_given_spike_strict_v1.csv')










# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     df.loc[i, 'pause_status'] = 1 if df.loc[i, 'pause_status'] in {'HC True', 'LC True'} else 0
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
#
# df.to_csv('../data/seq/sequence_ds_onehot_given_spike_loose_v1.csv')
#
#
#
#
#
# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     elif df.loc[i, 'pause_status'] in {'LC True'}:
#         df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
#
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_onehot_given_spike_selective_v1.csv')
#
#
#
#
#
#
# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     # elif df.loc[i, 'pause_status'] in {'LC True'}:
#     #     df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
#
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_onehot_given_spike_strict_v1.csv')








# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_v1.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     df.loc[i, 'pause_status'] = 1 if df.loc[i, 'pause_status'] in {'HC True', 'LC True'} else 0
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
#
# df.to_csv('../data/seq/sequence_ds_kmers_6_given_spike_loose_v1.csv')





# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_v1.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     elif df.loc[i, 'pause_status'] in {'LC True'}:
#         df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
#
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_kmers_6_given_spike_selective_v1.csv')






# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_v1.csv')
# print(df.shape)
# print(df.columns)
# df = df.loc[df['energy_spike'] == 1]
# df = df.drop(['energy_spike'], axis=1)
# print(df.shape)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     # elif df.loc[i, 'pause_status'] in {'LC True'}:
#     #     df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# print(df_false.shape)
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
#
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'pause_seq', 'pause_context'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_kmers_6_given_spike_strict_v1.csv')












# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v2.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     df.loc[i, 'pause_status'] = 1 if df.loc[i, 'pause_status'] in {'HC True', 'LC True'} else 0
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
#
# print(df_true.shape)
# print(df_false.shape)
#
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 15000)]
#
# df = pd.concat([df_true, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
#
# df.to_csv('../data/seq/sequence_ds_ordinal_no_spike_loose_v1.csv')





# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v2.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     elif df.loc[i, 'pause_status'] in {'LC True'}:
#         df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
#
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_ordinal_no_spike_selective_v1.csv')
#
#
#
#
#
#
# df = pd.read_csv('../data/seq/sequence_ds_ordinal_v2.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     # elif df.loc[i, 'pause_status'] in {'LC True'}:
#     #     df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_ordinal_no_spike_strict_v1.csv')












# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     df.loc[i, 'pause_status'] = 1 if df.loc[i, 'pause_status'] in {'HC True', 'LC True'} else 0
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
#
# print(df_true.shape)
# print(df_false.shape)
#
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 15000)]
#
# df = pd.concat([df_true, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
#
# df.to_csv('../data/seq/sequence_ds_onehot_no_spike_loose_v1.csv')
#
#
#
#
#
# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     elif df.loc[i, 'pause_status'] in {'LC True'}:
#         df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
#
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_onehot_no_spike_selective_v1.csv')






# df = pd.read_csv('../data/seq/sequence_ds_onehot_v1.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     # elif df.loc[i, 'pause_status'] in {'LC True'}:
#     #     df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_onehot_no_spike_strict_v1.csv')












# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_v1.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     df.loc[i, 'pause_status'] = 1 if df.loc[i, 'pause_status'] in {'HC True', 'LC True'} else 0
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false.reset_index(drop=True, inplace=True)
#
# print(df_true.shape)
# print(df_false.shape)
#
# df_false = df_false.iloc[np.random.random_integers(0, df_false.shape[0], 15000)]
#
# df = pd.concat([df_true, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
#
# df.to_csv('../data/seq/sequence_ds_kmers_6_no_spike_loose_v1.csv')
#
#
#
#
#

# kmers = 8
# df = pd.read_csv('../data/seq/sequence_ds_kmers_{}_v1.csv'.format(kmers))
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     elif df.loc[i, 'pause_status'] in {'LC True'}:
#         df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df.reset_index(drop=True, inplace=True)
#
#
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_kmers_{}_no_spike_selective_v1.csv'.format(kmers))
#
# print('CSV Created!')
#
# df = pd.read_csv('../data/seq/sequence_ds_kmers_{}_no_spike_selective_v1.csv'.format(kmers))
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
# np.save('../data/seq/special/X_data_kmers_{}_vectorized_no_spike_selective.npy'.format(kmers), X_data)
# np.save('../data/seq/special/y_data_kmers_{}_vectorized_no_spike_selective.npy'.format(kmers), target)

#
#
#
#
#
# df = pd.read_csv('../data/seq/sequence_ds_kmers_6_v1.csv')
# print(df.shape)
# print(df.columns)
#
# for i in range(df.shape[0]):
#     if df.loc[i, 'pause_status'] in {'HC True'}:
#         df.loc[i, 'pause_status'] = 1
#     # elif df.loc[i, 'pause_status'] in {'LC True'}:
#     #     df.loc[i, 'pause_status'] = -1
#     else:
#         df.loc[i, 'pause_status'] = 0
#
#
# df_true = df.loc[df['pause_status'] == 1]
# df_false = df.loc[df['pause_status'] == 0]
# df_false = df_false.iloc[np.random.random_integers(0, len(df_false), 4000)]
# df = pd.concat([df_true]*5, ignore_index=True)
# df = pd.concat([df, df_false], ignore_index=True)
# df.reset_index(drop=True, inplace=True)
#
# print(df['pause_status'].value_counts())
# drops = {'Unnamed: 0', 'Unnamed: 0.1', 'pause_seq', 'pause_context', 'energy_spike'}
# keep = []
# for c in df.columns:
#     if c not in drops:
#         keep.append(c)
#
# df = df[keep]
# print(df.shape)
# print(df.columns)
#
# df.to_csv('../data/seq/sequence_ds_kmers_6_no_spike_strict_v1.csv')




