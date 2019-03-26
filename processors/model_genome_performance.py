import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio import Entrez
import pickle


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


model = pickle.load(open('../models/model_files/final/seq_onehot_final_v1_max_depth.sav', 'rb'))


# X = np.load('../data/seq/special/X_data_onehot_flat_no_spike_selective.npy')
#
# print(X[0])
# print(len(X[0]))


def get_onehot_flat(pos):
    start = pos - 110
    end = pos + 108
    seg = mutgen[start: end]
    flat = ''
    onehot_patch = {'A': '1000', 'C': '0100', 'G': '0010', 'T': '0001'}
    for base in seg:
        flat += onehot_patch[base]
    flat_vector = []
    for c in flat:
        flat_vector.append(int(c))
    return flat_vector


# model_position_preds = []
#
# # print(get_onehot_flat(200))
# for i in range(110, len(mutgen)-110):
#     if (model.predict(np.array([get_onehot_flat(i)]))[0] == 1):
#         model_position_preds.append(i)
#     if i % 10000 == 0:
#         print('MODEL Step {}/{} = {}%'.format(i, len(mutgen), i*100//len(mutgen)))
#
# print('TOTAL NUMBER OF MODEL PREDS: ', len(model_position_preds))
# model_position_preds = np.array(model_position_preds)
# np.save('../data/extra/model_predictions.npy', model_position_preds)

model_position_preds = np.load('../data/extra/model_predictions.npy')
print(len(model_position_preds))


df = pd.read_csv('../data/util/genomic_data_set_v2.csv')

df_hc = df[df['pause_status'] == 'HC True']
hc_pos = np.array(df_hc['position'])
print('NUMBER OF HC POS: ', len(hc_pos))

df_lc = df[df['pause_status'] == 'LC True']
lc_pos = np.array(df_lc['position'])
print('NUMBER OF LC POS: ', len(lc_pos))



def find_pos_seg(pos, search_arr):
    for elem in search_arr:
        if elem <= pos + 16 and elem >= pos - 16:
            return True
    return False

hc_match_ct = 0
for p in range(len(hc_pos)):
    if find_pos_seg(hc_pos[p], model_position_preds):
        hc_match_ct += 1
    if p % 100 == 0:
        print('HC Step {}/{} = {}%'.format(p, len(hc_pos), p*100//len(hc_pos)))

lc_match_ct = 0
for p in range(len(lc_pos)):
    if find_pos_seg(lc_pos[p], model_position_preds):
        lc_match_ct += 1
    if p % 100 == 0:
        print('LC Step {}/{} = {}%'.format(p, len(lc_pos), p*100//len(lc_pos)))

print('TOTAL NUMBER OF MODEL PREDS: ', len(model_position_preds))
print('NUMBER OF HC POS: ', len(hc_pos))
print('NUMBER OF LC POS: ', len(lc_pos))
print('Total Match Count HC: ', hc_match_ct)
print('Total Match Count LC: ', lc_match_ct)



"""
RESULTS:
TOTAL NUMBER OF MODEL PREDS:  25499
NUMBER OF HC POS:  758
NUMBER OF LC POS:  13744
Total Match Count HC:  758
Total Match Count LC:  2909
"""
