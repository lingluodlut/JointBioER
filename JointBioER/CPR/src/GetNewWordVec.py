# -*- coding: utf-8 -*-


import codecs as cs

fp = cs.open('../../token2vec/chemdner_pubmed_biov5_drug.token4_d200','r','utf-8')
lines = fp.read().split('\n')[1:-1]
fp.close()

alltoken= {}
mode = ['train','vaild','test']
for m in mode:
    fpc = cs.open('../data/CPR_%s_Y.txt'%m,'r','utf-8')
    sentences = fpc.read().split('\n\n')[:-1]
    fpc.close()
    for s in sentences:
        tokens = s.split('\n')
        for t in tokens:
            alltoken[t.split('\t')[0].lower()] = 1

fp = cs.open('../../token2vec/chemdner_pubmed_biov5_drug.token4_d200_CPR','w','utf-8')
fp.write('28000\t100\n')
for line in lines:
    word = line.split(' ')[0]
    if word in alltoken:
        fp.write(line+'\n')
fp.close()