import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import Entrez
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""

heatmap.py : looks at pause site data and model predictions, and produces heatmaps for bps prediction for both

"""


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


# load data (original genomic data from genome parsing for params)
df = pd.read_csv("../data/util/genomic_data_set_v2.csv")


def data_true_map():
    # choose true pause sites and convert into numpy array
    df_true = df[df['pause_status'] == 'HC True']
    # df = df.ix[(df['pause_status'] != 'HC True') & (df['pause_status'] != 'LC True')]
    # df_true = df.ix[(df['pause_status'] == 'HC True') | (df['pause_status'] == 'LC True')]
    positions_true = np.array(df_true['position'])

    # create table with cols as the 4 columns with 16 sample rows filled 
    # with 0 representing pause sequence
    cols = ['A', 'C', 'G', 'T']
    inds = np.arange(16)
    hMap_true = pd.DataFrame(index=inds, columns=cols)
    hMap_true = hMap_true.fillna(0)
    print(hMap_true)

    ct = 0
    # iterate through all true pause site positions
    for pos in positions_true:
        # based on pause sequence at the positions, increment the respective A, C, G, T column for that row
        for i in range(16):
            hMap_true[str(mutgen[pos - 16 + i])][i] += 1
        ct +=1
        # print(ct, pos/len(positions))

    # divide each cell value by number of pause sites to get percentage
    for c in hMap_true.columns:
        hMap_true[c] = hMap_true[c].div(len(positions_true))

    print(hMap_true)



def data_false_map():
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


# draws a heatmap given a dataframe 
def drawMap(map):
    diffMap = map
    ax = sns.heatmap(diffMap, linewidth=0.5, vmin=-0.02, vmax=0.05)
    # save
    plt.savefig('../figures/heatmap_initial_diff.png')
    plt.show()



def model_map():
    # get save model predictions data
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


    ax = sns.heatmap(modelMap, linewidth=0.5, vmin=-0.02, vmax=0.05)
    plt.savefig('../figures/heatmap_model_pred.png')
    plt.show()

