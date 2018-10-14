# -*- coding: utf-8 -*-


import codecs as cs
from utils import LoadGoldEntity,LoadGoldRelation,loadtokens

testtokens = loadtokens(u'../data/CPR_test_Y.txt')
gold_e = LoadGoldEntity('../data/goldEntityAnswer_test.txt')#标注的实体位置
pre_e = LoadGoldEntity('../data/predictE_test.txt')
gold_r = LoadGoldRelation('../data/goldRelationAnswer_test.txt')#标注的关系的实体位置及类别
pre_r = LoadGoldRelation('../data/predictR_test.txt')

fp = cs.open('../data/CompareAns.txt','w','utf-8')
for i in range(len(testtokens)):
    fp.write(' '.join(testtokens[i])+'\n')
    text = ''.join(testtokens[i])
    
    fp.write(u'【GOLD_E】\n')
    for e in gold_e[i]:
        fp.write(text[e[0]:e[1]+1]+'\t'+e[2]+'\n')
    
    fp.write(u'【PRE_E】\n')
    for e in pre_e[i]:
        fp.write(text[e[0]:e[1]+1]+'\t'+e[2]+'\n')
    
    fp.write(u'【GOLD_R】\n')
    for r in gold_r[i]:
        fp.write(text[r[0]:r[1]+1]+'\t'+text[r[2]:r[3]+1]+'\t'+r[4]+'\n')
    
    fp.write(u'【PRE_R】\n')
    for r in pre_r[i]:
        fp.write(text[r[0]:r[1]+1]+'\t'+text[r[2]:r[3]+1]+'\t'+r[4]+'\n')
    fp.write('\n')
    
    