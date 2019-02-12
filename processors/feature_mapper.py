import pandas as pd
import numpy as np

# feats = ['gene_len', 'ref_base_T', 'ref_base_G', 'trans_base_T', 'ref_base_A', 'trans_base_C', 'trans_base_A', 'ref_base_C', 'pause_C', 'context_C', 'position', 'start_dist_rel', 'end_dist_rel', 'trans_base_G', 'context_G', 'pause_G']
# scores = [55.18983154545005, 18.126120478746113, 12.53161782895101, 11.166928380028734, 5.83868268666578, 3.440434021525543, 3.2416484983165716, 2.229669696622637, 1.0916392266607504, 1.0916392157692882, 0.708069854206974, 0.23128571159504283, 0.2312857115950422, 0.08461163996144426, 0.027062574473532337, 0.027062570903871742, 0.0]

# feats = np.load('../data/extra/kmers_4_hp_X_data_column_names.npy')
# scores = []
#
# df = pd.read_csv('../data/extra/kmers_4_from_log.csv')
# print(df.shape)
#
# for i in range(df.shape[0]):
#     scores.append(df.loc[i, 'a'])
#     scores.append(df.loc[i, 'b'])
#     scores.append(df.loc[i, 'c'])
#     scores.append(df.loc[i, 'd'])

# feats = ['a', 'b', 'c', 'd']
# scores = [1, 4, 2, 8]

feat_scores = zip(feats, scores[:len(feats)])
print(feat_scores)
feat_scores = sorted(feat_scores, key=lambda x: x[1], reverse=True)
print(feat_scores)

df_cols = ['features', 'scores']
df = pd.DataFrame(columns=df_cols)

ind = 0
for s in feat_scores:
    df.loc[ind] = [s[0], s[1]]
    ind += 1

print(df.head(10))

df.to_csv('../data/extra/seq_kmers_4_features.csv')



