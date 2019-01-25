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

# for feature in record.features:  # loop each position through whole genome
#     # In this particular case I'm interested in focusing on cds, but
#     # in others, I may be interested in other feature types?
#     if feature.type == "CDS":
#         if feature.location.nofuzzy_start <= pos and feature.location.nofuzzy_end >= pos:
#             pos_res.append(feature.qualifiers['gene'][0])
#             pos_res.append(feature.location.nofuzzy_start)
#             pos_res.append(feature.location.nofuzzy_end)
#             pos_res.append(pos - feature.location.nofuzzy_start)
#             pos_res.append((pos - feature.location.nofuzzy_start) / (
#                     feature.location.nofuzzy_end - feature.location.nofuzzy_start))
#             pos_res.append(feature.location.nofuzzy_end - pos)
#             pos_res.append(
#                 (feature.location.nofuzzy_end - pos) / (feature.location.nofuzzy_end - feature.location.nofuzzy_start))
#             break

print(type(record))
print(record)
