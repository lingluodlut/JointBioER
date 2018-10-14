# -*- coding: utf-8 -*-

import pickle
import codecs as cs
f = open('../data/predict/en2t_crf+bils250+ls100.pkl','rb')
eindex2tindex = pickle.load(f)
f.close()
from utils import loadtokens
testtokens = loadtokens(u'../data/DDI_test_BIOES.txt')
fp = cs.open('../data/testsample.txt','w','utf-8')
for i in range(len(testtokens)):
    enum = len(eindex2tindex[i])
    if enum>1:
        for j in range(0,enum-1):
            for k in range(j+1,enum):
                fp.write(str(eindex2tindex[i][j][0])+' '+str(eindex2tindex[i][j][-1])+'|')
                fp.write(str(eindex2tindex[i][k][0])+' '+str(eindex2tindex[i][k][-1])+'|')
                fp.write(testtokens[i][0])
                for t in testtokens[i][1:]:
                    fp.write(' '+t)
                fp.write('\n')
fp.close()
                    