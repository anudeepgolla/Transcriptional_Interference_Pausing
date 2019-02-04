# import E Coli K12 MG1655 genome data using Entrez database
# U00096 is the ID for MG1655
# lines taken from BioPython tutorial Chp 9
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



# gene_params = ['gene', 'gene_start', 'gene_end']
# df = pd.DataFrame(columns=gene_params)
#
# for feature in record.features:
#     if feature.type == 'CDS':
#         df = df.append({'gene': str(feature.qualifiers['gene'][0]),
#                         'gene_start': int(feature.location.nofuzzy_start),
#                         'gene_end': int(feature.location.nofuzzy_end)}, ignore_index=True)
#
# print(df.shape)
# print(df.head(10))
#
# df.to_csv('data/gene_data.csv')


# print(len(mutgen))
# pos_i = 1
# df_g = pd.DataFrame(index=np.arange(4641652), columns=['position', 'gene'])
# df_g['position'] = np.arange(1, 4641653)
# df_g['gene'] = np.nan
# print('Gene Map Initialized!')
# df_gene = pd.read_csv('data/gene_data.csv')
# print('Gene Data Downloaded!')
#
# for g_i in range(df_gene.shape[0]):
#     gene, st, end = df_gene.loc[g_i, 'gene'], df_gene.loc[g_i, 'gene_start'], df_gene.loc[g_i, 'gene_end']
#     print(gene, st, end)
#     for p_i in range(st, end+1):
#         df_g.loc[p_i - 1, 'gene'] = gene
#         print(p_i)
#     if g_i + 1 % 10 == 0:
#         print('Step {:7}/{:7} --- {:2}% complete'.format(g_i + 1, df_gene.shape[0], ((g_i + 1)*100)/df_gene.shape[0]))
#
# print(df_g.head(300))
# print('Loading Dataset...')
# df_g.to_csv('data/gene_map.csv')
# print('Data Saved!')


df = pd.read_csv('data/genomic_data_set_v2.csv')
df = df[['position', 'pause_status', 'pause_seq', 'pause_context']]
df['energy_spike'] = 1
print(df.head(5))
print(df.shape)

# print(len(df['pause_seq'][0]), len(df['pause_context'][0]))
for i in range(df.shape[0]):
    if df.loc[i, 'pause_status'] not in {'HC True', 'LC True'}:
        df.loc[i, 'pause_status'] = 'false'

print(df['pause_status'][-10:])

randinds = np.random.randint(110, 4641500, 35000)
print(randinds[:10])
n = len(randinds)
print(n)

ct = 0
for ri in randinds:
    df = df.append({'position': ri,
                            'pause_status': 'false',
                            'pause_seq': mutgen[ri-16: ri],
                            'pause_context': mutgen[ri-109: ri+109],
                            'energy_spike': 0},  ignore_index=True)
    ct += 1

    if ct % 100 == 0:
        print('Step {:5}/{:5} = {:2}%'.format(ct, n, (ct*100)//n))


print(df.shape)

df.to_csv(('data/sequence_ds_v1.csv'))
print('DataSaved!')
