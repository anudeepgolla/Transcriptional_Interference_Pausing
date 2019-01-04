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




"""
Nolans Parse
"""

#
# # n=len(seq) # could be useful for automating the mapping of each base in the submitted sequence
# n=16 # for now just assume sequence is 16 bp
# last=0 # can ignore
# i=0 # counter
# k=0 # counter
# for m in range(n,len(mutgen)):
#     if (mutgen[m-n]=='G' and mutgen[m-n+1]=='G' and mutgen[m-n+2]==('C' or 'G') and mutgen[m-n+3]!='T'
#         and mutgen[m-n+5]=='A' and mutgen[m-n+6]==('A' or 'G') and mutgen[m-n+7]=='T' and mutgen[m-n+8]=='T'
#         and mutgen[m - n + 9] == 'G' and mutgen[m-n+10]==('C' or 'T') and mutgen[m-n+11]=='G'
#         and mutgen[m - n + 12] == 'G' and mutgen[m-n+13]=='C' and mutgen[m-n+14]=='C' and mutgen[m-n+15]=='G'):
#             print('STRICT SENSE pause starting at: ', m-n)
#             k=k+1 # look for the pause seq presented in Larson...Landick (Science, 2014)
#     if (mutgen[m - n] == 'G' and mutgen[m - n + 1] == 'G'
#     and mutgen[m - n + 8] == 'T' and mutgen[m - n + 9] == 'G' and mutgen[m - n + 10] == ('C' or 'T')
#     and mutgen[m - n + 11] == 'G' and mutgen[m - n + 12] == 'G' and mutgen[m - n + 13] == 'C'):
#         print('probable SENSE pause starting at: ', m - n)
#         i = i + 1  # count the seq since it doesn't overlap
#         last = m
#     predicted = len(mutgen) % 4^8 * 2 # how many possible pause sites if each bp independent
#
#
# j=0
# last =0
# l=0
# for m in range(n-1,len(mutcomp)):
#         if (mutcomp[m-n]=='G' and mutcomp[m-n+1]=='G' and mutcomp[m-n+2]==('C' or 'G') and mutcomp[m-n+3]!='T'
#         and mutcomp[m-n+5]=='A' and mutcomp[m-n+6]==('A' or 'G') and mutcomp[m-n+7]=='T' and mutcomp[m-n+8]=='T'
#                 and mutcomp[m - n + 9] == 'G' and mutcomp[m-n+10]==('C' or 'T') and mutcomp[m-n+11]=='G'
#           and mutcomp[m - n + 12] == 'G' and mutcomp[m-n+13]=='C' and mutcomp[m-n+14]=='C' and mutcomp[m-n+15]=='G'):
#             print('STRICT ANTISENSE pause starting at: ', m-n)
#             l=l+1
# # do the same for the complement strand
#         if (mutcomp[m - n] == 'G' and mutcomp[m - n + 1] == 'G'  and mutcomp[
#             m - n + 8] == 'T' and mutcomp[m - n + 9] == 'G' and mutcomp[m - n + 10] == ('C' or 'T') and mutcomp[m - n + 11] == 'G' and mutcomp[m - n + 12] == 'G' and mutcomp[m - n + 13] == 'C'):
#             print('probable ANTISENSE pause starting at: ', m-n)
#             if (m > 16 and m - last > 16):
#                 j = j + 1 # count the seq since it doesn't overlap
#                 last = m
#
# print(i, " probable pause sites in sense strand")
# print(j, " probable pause sites in antisense strand")
# print(k, " strict sense sites in sense strand")
# print(l, " strict antisense sites in antisense strand")
#
# print(predicted , " total possibilities in sense strand given random DNA sequences")


col_params = ['pause_status', 'position', 'ref_base', 'pause_seq', 'pause _G', 'pause_C',
              'pause_context', 'context_G', 'context_C', 'gene', 'gene_start', 'gene_end',
              'start_dist_abs', 'start_dist_rel', 'end_dist_abs', 'end_dist_rel',
              'trans_base']  # pause_seq is last 17bp, trans_base is pos+1, pause_context is 109 before and after

# print(len(col_params))

df = pd.DataFrame(columns=col_params)

pos_arr = np.array([])
datas = ['true_pie_hi_data.csv', 'true_pie_low_data.csv', 'false_pie_data.csv']
for ds in datas:
    df_ = pd.read_csv('data/' + ds)
    arr_ = np.array(df_['Position'])
    arr_ = np.reshape(arr_, (-1, arr_.shape[0]))
    pos_arr = np.append(pos_arr, arr_)
    # print(len(arr_[0]))
    # print(len(arr_), arr_[-1])

# print("pos arr", len(pos_arr))
print(len(pos_arr))

print(len(mutgen))




def data_parse_loader(genome, positions, dataf):
    print('GENOME: ', len(genome))
    # print(np.isnan(positions[758]))

    for pos_i in range(len(positions)):
        if not np.isnan(positions[pos_i]) and (positions[pos_i] + 109) <= 4641652:

            # if type(positions[pos_i]) != int and type(positions[pos_i]) != np.float64:
            #     print(pos_i, type(positions[pos_i]))

            # print(positions[pos_i], pos_i)
            pos = int(positions[pos_i])
            pos_res = []

            if pos_i<759:
                pos_res.append('HC True')
            elif pos_i<14505:
                pos_res.append('LC True')
            else:
                pos_res.append('False')

            pos_res.append(pos)
            # print(pos)
            pos_res.append(genome[pos-1])
            pos_res.append(genome[pos-17:pos-1])

            pauseg = 0
            pausec = 0

            for i in range(17):
                if genome[pos-17+i] == 'G':
                    pauseg += 1
                if genome[pos-17+i] == 'C':
                    pausec += 1

            pos_res.append(pauseg/17)
            pos_res.append(pausec/17)
            pos_res.append(genome[pos - 110:pos + 108])

            contextg = 0
            contextc = 0

            for i in range(219):
                if genome[pos-110+i] == 'G':
                    contextg += 1
                if genome[pos-110+i] == 'C':
                    contextc += 1

            pos_res.append(pauseg / 219)
            pos_res.append(pausec / 219)

            for feature in record.features:  # loop each position through whole genome
                # In this particular case I'm interested in focusing on cds, but
                # in others, I may be interested in other feature types?
                if feature.type == "CDS":
                    if feature.location.nofuzzy_start <= pos and feature.location.nofuzzy_end >= pos:
                        pos_res.append(feature.qualifiers['gene'][0])
                        pos_res.append(feature.location.nofuzzy_start)
                        pos_res.append(feature.location.nofuzzy_end)
                        pos_res.append(pos - feature.location.nofuzzy_start)
                        pos_res.append((pos - feature.location.nofuzzy_start)/(feature.location.nofuzzy_end - feature.location.nofuzzy_start))
                        pos_res.append(feature.location.nofuzzy_end - pos)
                        pos_res.append((feature.location.nofuzzy_end - pos) / (feature.location.nofuzzy_end - feature.location.nofuzzy_start))
                        break

            if (len(pos_res)==9):
                pos_res.append(np.nan)
                pos_res.append(np.nan)
                pos_res.append(np.nan)
                pos_res.append(np.nan)
                pos_res.append(np.nan)
                pos_res.append(np.nan)
                pos_res.append(np.nan)



            pos_res.append(genome[pos+1])

            # print(pos_res[16])

            dataf.loc[pos_i] = pos_res
            if pos_i % 100 == 0:
                print("Step {}/{}".format(pos_i, 33938))



data_parse_loader(mutgen, pos_arr, df)
df.to_csv('genomic_data_set.csv')
print(df.shape)
print(df.head(10))




# 4645179
# 4641652
