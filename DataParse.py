# import E Coli K12 MG1655 genome data using Entrez database
# U00096 is the ID for MG1655
# lines taken from BioPython tutorial Chp 9
from Bio import SeqIO
from Bio import Entrez
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
# n=len(seq) # could be useful for automating the mapping of each base in the submitted sequence
n=16 # for now just assume sequence is 16 bp
last=0 # can ignore
i=0 # counter
k=0 # counter
for m in range(n,len(mutgen)):
    if (mutgen[m-n]=='G' and mutgen[m-n+1]=='G' and mutgen[m-n+2]==('C' or 'G') and mutgen[m-n+3]!='T'
        and mutgen[m-n+5]=='A' and mutgen[m-n+6]==('A' or 'G') and mutgen[m-n+7]=='T' and mutgen[m-n+8]=='T'
        and mutgen[m - n + 9] == 'G' and mutgen[m-n+10]==('C' or 'T') and mutgen[m-n+11]=='G'
        and mutgen[m - n + 12] == 'G' and mutgen[m-n+13]=='C' and mutgen[m-n+14]=='C' and mutgen[m-n+15]=='G'):
            print('STRICT SENSE pause starting at: ', m-n)
            k=k+1 # look for the pause seq presented in Larson...Landick (Science, 2014)
    if (mutgen[m - n] == 'G' and mutgen[m - n + 1] == 'G'
    and mutgen[m - n + 8] == 'T' and mutgen[m - n + 9] == 'G' and mutgen[m - n + 10] == ('C' or 'T')
    and mutgen[m - n + 11] == 'G' and mutgen[m - n + 12] == 'G' and mutgen[m - n + 13] == 'C'):
        print('probable SENSE pause starting at: ', m - n)
        i = i + 1  # count the seq since it doesn't overlap
        last = m
    predicted = len(mutgen) % 4^8 * 2 # how many possible pause sites if each bp independent


j=0
last =0
l=0
for m in range(n-1,len(mutcomp)):
        if (mutcomp[m-n]=='G' and mutcomp[m-n+1]=='G' and mutcomp[m-n+2]==('C' or 'G') and mutcomp[m-n+3]!='T'
        and mutcomp[m-n+5]=='A' and mutcomp[m-n+6]==('A' or 'G') and mutcomp[m-n+7]=='T' and mutcomp[m-n+8]=='T'
                and mutcomp[m - n + 9] == 'G' and mutcomp[m-n+10]==('C' or 'T') and mutcomp[m-n+11]=='G'
          and mutcomp[m - n + 12] == 'G' and mutcomp[m-n+13]=='C' and mutcomp[m-n+14]=='C' and mutcomp[m-n+15]=='G'):
            print('STRICT ANTISENSE pause starting at: ', m-n)
            l=l+1
# do the same for the complement strand
        if (mutcomp[m - n] == 'G' and mutcomp[m - n + 1] == 'G'  and mutcomp[
            m - n + 8] == 'T' and mutcomp[m - n + 9] == 'G' and mutcomp[m - n + 10] == ('C' or 'T') and mutcomp[m - n + 11] == 'G' and mutcomp[m - n + 12] == 'G' and mutcomp[m - n + 13] == 'C'):
            print('probable ANTISENSE pause starting at: ', m-n)
            if (m > 16 and m - last > 16):
                j = j + 1 # count the seq since it doesn't overlap
                last = m

print(i, " probable pause sites in sense strand")
print(j, " probable pause sites in antisense strand")
print(k, " strict sense sites in sense strand")
print(l, " strict antisense sites in antisense strand")

print(predicted , " total possibilities in sense strand given random DNA sequences")