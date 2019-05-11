import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio import Entrez
import pickle

"""

model_genome_performance.py : performs thorough testing of a model (pretrained) on the entire ecoli genome
by running every base pair in the genome into the model and checking which pause sites were predicted

"""


# credentials for data access
Entrez.email = 'anudeepgolla3@gmail.com'
handle = "../sequence.gb"
print(handle)
# specifc data request in sequence.gb
record = SeqIO.read(handle,"genbank")
print(record.id)

# this script is ideally customizable to any desired DNA sequence
# seq = input('Enter sequence: ')
#interest = input('pause, promoter, or rbs?: ') # this script can distinguish between promoter or pause site features - or maybe RBS
complement=record.seq.complement() # find the reverse complement (antisense strand) of the sense strand
mutgen = record.seq.tomutable() # convert imported sense strand into a mutable object (i.e a string of letters)
mutcomp=complement.tomutable() # convert complement into mutable object


# get locally stored model file using pickle, model is now the trained algorithm
model = pickle.load(open('../models/model_files/final/seq_onehot_final_v1_max_depth.sav', 'rb'))

# get context sequence at pos index of genome in onehot encoded vector
# that is flattened. Ex (length 4): [A, C, G, T] ==> [1, 0, 0, 0,   0, 1, 0, 0,   0, 0, 1, 0,   0, 0, 0, 1]
def get_onehot_flat(pos):
    start = pos - 110
    end = pos + 108
    # get sub list from start to end
    seg = mutgen[start: end]
    flat = ''
    onehot_patch = {'A': '1000', 'C': '0100', 'G': '0010', 'T': '0001'}
    for base in seg:
        flat += onehot_patch[base]
    flat_vector = []
    for c in flat:
        flat_vector.append(int(c))
    return flat_vector


model_position_preds = []

# from base pair at index 110 to base pair at len(mutgen)-110, run model and get prediction
for i in range(110, len(mutgen)-110):
    # if model predicts 1 = it is a pause site, add index of base pair to model_position_preds
    if (model.predict(np.array([get_onehot_flat(i)]))[0] == 1):
        model_position_preds.append(i)
    # print the progress every 10,000 bps
    if i % 10000 == 0:
        print('MODEL Step {}/{} = {}%'.format(i, len(mutgen), i*100//len(mutgen)))

print('TOTAL NUMBER OF MODEL PREDS: ', len(model_position_preds))
# turn it into a numpy array and save the array using np.save(filepath, array)
model_position_preds = np.array(model_position_preds)
np.save('../data/extra/model_predictions.npy', model_position_preds)

# load the saved numpy array
model_position_preds = np.load('../data/extra/model_predictions.npy')
print(len(model_position_preds))

# load genomic_data_set_v2 into df (dataframe object of pandas)
df = pd.read_csv('../data/util/genomic_data_set_v2.csv')

# df_hc is a dataframe of only HC True pause sites
df_hc = df[df['pause_status'] == 'HC True']
# of this data, get a array of all the HC True positions (indexes of bps)
hc_pos = np.array(df_hc['position'])
print('NUMBER OF HC POS: ', len(hc_pos))

# df_lc is a dataframe of only LC True pause sites
df_lc = df[df['pause_status'] == 'LC True']
# of this data, get a array of all the LC True positions (indexes of bps)
lc_pos = np.array(df_lc['position'])
print('NUMBER OF LC POS: ', len(lc_pos))


# method will return true if pos is +/- 16 bps away from any index in search_arr
def find_pos_seg(pos, search_arr):
    for elem in search_arr:
        if elem <= pos + 16 and elem >= pos - 16:
            return True
    return False


hc_match_ct = 0
# go through all HC True pause sites and count it if it is near one of the model predictions
for p in range(len(hc_pos)):
    if find_pos_seg(hc_pos[p], model_position_preds):
        hc_match_ct += 1
    # print progress every 100
    if p % 100 == 0:
        print('HC Step {}/{} = {}%'.format(p, len(hc_pos), p*100//len(hc_pos)))

lc_match_ct = 0
# go through all LC True pause sites and count it if it is near one of the model predictions
for p in range(len(lc_pos)):
    if find_pos_seg(lc_pos[p], model_position_preds):
        lc_match_ct += 1
    # print progress every 100
    if p % 100 == 0:
        print('LC Step {}/{} = {}%'.format(p, len(lc_pos), p*100//len(lc_pos)))

# output all statistics
print('TOTAL NUMBER OF MODEL PREDS: ', len(model_position_preds))
print('NUMBER OF HC POS: ', len(hc_pos))
print('NUMBER OF LC POS: ', len(lc_pos))
print('Total Match Count HC: ', hc_match_ct)
print('Total Match Count LC: ', lc_match_ct)


# real results of running this file
"""
RESULTS:
TOTAL NUMBER OF MODEL PREDS:  25499
NUMBER OF HC POS:  758
NUMBER OF LC POS:  13744
Total Match Count HC:  758
Total Match Count LC:  2909
"""
