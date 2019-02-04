import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio import Entrez
import pandas as pd
import numpy as np

Entrez.email = 'anudeepgolla3@gmail.com'
handle = "sequence.gb"
print(handle)
record = SeqIO.read(handle,"genbank")
print(record.id)

# this script is ideally customizable to any desired DNA sequence
# seq = input('Enter sequence: ')


#interest = input('pause, promoter, or rbs?: ') # this script can distinguish between promoter or pause site features - or maybe RBS
complement=record.seq.complement() # find the reverse complement (antisense strand) of the sense strand
mutgen = record.seq.tomutable() # convert imported sense strand into a mutable object (i.e a string of letters)
mutcomp=complement.tomutable() # convert complement into mutable object

mutgen = mutcomp
print(len(mutcomp))

df_param = pd.read_csv('data/genomic_data_set_v2.csv')

df_param = df_param.loc[df_param['pause_status'] != 'HC True']
print(df_param.head(10))
print(df_param.shape)
positions = set(df_param['position'])
print(len(positions))


ct = 0
motif = 'GNNNNNNTGCG'
for i in range(len(mutgen) - len(motif) + 1):
    if mutgen[i:i+len(motif)] == 'G{}TGCG'.format(mutgen[i+1:i+7]):
        #print(i, mutgen[i:i+len(motif)])
        ct += 1
print(ct)
