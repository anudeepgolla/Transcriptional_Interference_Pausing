import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import Entrez
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Entrez.email = 'anudeepgolla3@gmail.com'
handle = "../sequence.gb"
print(handle)
record = SeqIO.read(handle,"genbank")
print(record.id)

# this script is ideally customizable to any desired DNA sequence
# seq = input('Enter sequence: ')


#interest = input('pause, promoter, or rbs?: ') # this script can distinguish between promoter or pause site features - or maybe RBS
complement=record.seq.complement() # find the reverse complement (antisense strand) of the sense strand
mutgen = record.seq.tomutable() # convert imported sense strand into a mutable object (i.e a string of letters)
mutcomp=complement.tomutable() # convert complement into mutable object



df = pd.read_csv("../data/util/genomic_data_set_v2.csv")


# df_true = df[df['pause_status'] == 'HC True']
# # df = df.ix[(df['pause_status'] != 'HC True') & (df['pause_status'] != 'LC True')]
# # df_true = df.ix[(df['pause_status'] == 'HC True') | (df['pause_status'] == 'LC True')]
# positions_true = np.array(df_true['position'])
#
#
# cols = ['A', 'C', 'G', 'T']
# inds = np.arange(16)
# print(inds)
#
# hMap_true = pd.DataFrame(index=inds, columns=cols)
# hMap_true = hMap_true.fillna(0)
# print(hMap_true)
#
# ct = 0
# for pos in positions_true:
#     for i in range(16):
#         hMap_true[str(mutgen[pos - 16 + i])][i] += 1
#     ct +=1
#     # print(ct, pos/len(positions))
#
# for c in hMap_true.columns:
#     hMap_true[c] = hMap_true[c].div(len(positions_true))
#
# print(hMap_true)




# df = df[df['pause_status'] == 'LC True']
# df = df.ix[(df['pause_status'] != 'HC True') & (df['pause_status'] != 'LC True')]
df_false = df.ix[(df['pause_status'] != 'HC True') & (df['pause_status'] != 'LC True')]
positions_false = np.array(df_false['position'])


cols = ['A', 'C', 'G', 'T']
inds = np.arange(16)
print(inds)

hMap_false = pd.DataFrame(index=inds, columns=cols)
hMap_false = hMap_false.fillna(0)
print(hMap_false)

ct = 0
for pos in positions_false:
    for i in range(16):
        hMap_false[str(mutgen[pos - 16 + i])][i] += 1
    ct +=1
    # print(ct, pos/len(positions))

for c in hMap_false.columns:
    hMap_false[c] = hMap_false[c].div(len(positions_false))

print(hMap_false)


# diffMap = hMap_true - hMap_false
# print(diffMap)
#
#
#
# ax = sns.heatmap(diffMap, linewidth=0.5, vmin=-0.02, vmax=0.05)
# plt.savefig('../figures/heatmap_initial_diff.png')
# plt.show()



model_preds = np.load('../data/extra/model_predictions.npy')

cols = ['A', 'C', 'G', 'T']
inds = np.arange(16)
print(inds)

modelMap = pd.DataFrame(index=inds, columns=cols)
modelMap = modelMap.fillna(0)
print(modelMap)

for pos in model_preds:
    for i in range(16):
        modelMap[str(mutgen[pos - 16 + i])][i] += 1

for c in modelMap.columns:
    modelMap[c] = modelMap[c].div(len(model_preds))

print(modelMap)


diffMap = modelMap - hMap_false

ax = sns.heatmap(diffMap, linewidth=0.5, vmin=-0.02, vmax=0.05)
plt.savefig('../figures/heatmap_model_pred_diff.png')
plt.show()

