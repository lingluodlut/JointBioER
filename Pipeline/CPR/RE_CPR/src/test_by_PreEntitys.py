# -*- coding: utf-8 -*-

import numpy as np
import codecs as cs

from keras.models import Model
from keras.layers import Dense,Bidirectional,Dropout,Embedding,Input
from keras.layers import LSTM,concatenate
from keras.optimizers import RMSprop
from utils import computeFr
from utils import save_model,load_model,ypre2label
from constants import *
from RepresentLayer import RepresentationLayer
#import Eval
#import Evalmicro
#import FileUtil
from constants import i2l_dic

max_len = 150
word_vec_dim = 100
position_vec_dim = 30
epoch_size = 80
bils = 600
#ls =100

reptest = RepresentationLayer(wordvec_file = '../../token2vec/chemdner_pubmed_biov5_drug.token4_d100_CPR',
                          frequency=20000,len_s = max_len,pattern = 'test',use_pre_e = True)
ytest, xtest, pos1test, pos2test = reptest.represent_instances()

epostest,indexmap_test,golde, goldr = reptest.EntityPos,reptest.indexmap,reptest.gold_e,reptest.gold_r
print (len(goldr))


    
word = Input(shape=(max_len,), dtype='int32', name='word')
distance_e1 = Input(shape=(max_len,), dtype='int32', name='distance_e1')
distance_e2 = Input(shape=(max_len,), dtype='int32', name='distance_e2')
word_emb = Embedding(reptest.vec_table.shape[0], reptest.vec_table.shape[1],weights = [reptest.vec_table], input_length=max_len)
position_emb = Embedding(max_len * 2 + 1, position_vec_dim, input_length=max_len)
word_vec = word_emb(word)
distance1_vec = position_emb(distance_e1)
distance2_vec = position_emb(distance_e2)
concatenated  = concatenate([word_vec, distance1_vec, distance2_vec],axis = 2)
#concatenated = Dropout(0.4)(concatenated)

bilstm = Bidirectional(LSTM(bils, return_sequences=(False),dropout=0.4,recurrent_dropout=0.4,implementation=2), input_shape=(max_len ,word_vec_dim+2* position_vec_dim))(concatenated)
#dense = Dense(ls, activation='tanh')(bilstm)
#dense = Dropout(0.4)(dense)
#    cnn = Convolution1D(nb_filter=100, filter_length=3, activation='tanh')(concat_vec)
#    cnn = Convolution1D(nb_filter=100, filter_length=3, activation='tanh')(cnn)
#    flattened = Flatten()(cnn)
#    dense = Dense(100, activation='tanh')(flattened)
#dense = Dense(50, activation='tanh')(dense)
predict = Dense(len(reptest.l2i_dic), activation='softmax')(bilstm)
model = Model(input=[word, distance_e1, distance_e2], output=predict)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=opt)
model.summary()

      
print ('load the model')
load_model('../data/model/bils%s.h5'%(str(bils)),model)
answer_array_d = model.predict([xtest, pos1test, pos2test], batch_size=512)
pre_label = ypre2label(answer_array_d)

pre_r = [[] for i in range(len(goldr))]
for i in range(len(pre_label)):
    if pre_label[i] != 0:
        label = reptest.i2l_dic[pre_label[i]]#类别
        sen_index = indexmap_test[i]#句子索引
        ans = epostest[i]+[label]
        pre_r[sen_index].append(ans)

p,r,f,f_label = computeFr(goldr,pre_r)
print 'p=%.3f r=%.3f f=%.3f'%(p,r,f)
print f_label[1:]
#fp = cs.open('../data/pre_relation_pipline.txt','w','utf-8')
#for each in pre_r:
#    for r in each:
#        for w in r:
#            fp.write(str(w)+'\t')
#        fp.write('\t')
#    fp.write('\n')
#fp.close()