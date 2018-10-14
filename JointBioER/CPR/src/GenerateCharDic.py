# -*- coding: utf-8 -*-

import codecs as cs
#读入abstract
fp = cs.open('../../ChemProt_Corpus/chemprot_training/chemprot_training_abstracts.tsv','r','utf-8')
text = fp.read().split('\n')[:-1]
fp.close()

chardic = {}
index = 0

for t in text:
    content = ''.join(t.split('\t')[1:])
    for char in content:
        if char not in chardic:
            chardic[char] = index
            index +=1

fp = cs.open('../../token2vec/char_dic_CPR.txt','w','utf-8')
for each in chardic:
    fp.write(each+'\t'+str(chardic[each])+'\n')
fp.close()