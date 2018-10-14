# -*- coding: utf-8 -*-


import numpy as np
import codecs as cs

from keras.models import Model
from keras.layers import Dense,Bidirectional,Dropout,Embedding,Input
from keras.layers import LSTM,concatenate
from keras.optimizers import RMSprop#, SGD, Adam, Adadelta, Adagrad
from utils import computeFr
from utils import save_model,load_model,ypre2label
from constants import *
from RepresentLayer import RepresentationLayer
import Eval
import Evalmicro
#import FileUtil
from constants import i2l_dic

max_len = 170
word_vec_dim = 200
position_vec_dim = 20
epoch_size = 100
bils = 360
ls =100
#读取训练集和测试集
reptrain = RepresentationLayer(wordvec_file = '../../token2vec/wordemb_d200.emb',
                          frequency=60000,len_s = max_len,pattern = 'train')
ytrain, xtrain, pos1train, pos2train = reptrain.represent_instances()

reptest = RepresentationLayer(wordvec_file = '../../token2vec/wordemb_d200.emb',
                          frequency=60000,len_s = max_len,pattern = 'test')
ytest, xtest, pos1test, pos2test = reptest.represent_instances()
epostest,indexmap_test,golde, goldr = reptest.EntityPos,reptest.indexmap,reptest.gold_e,reptest.gold_r
#
#ytrain_list, xtrain_list, pos1train_list, pos2train_list = ytrain.tolist(), xtrain.tolist(), pos1train.tolist(), pos2train.tolist()
#ydev,xdev,pos1dev,pos2dev = [],[],[],[]
#ytrain, xtrain, pos1train, pos2train = [],[],[],[]
#for i in range(len(ytrain_list)):
#    if i > 25012:
#        ydev.append(ytrain_list[i])
#        xdev.append(xtrain_list[i])
#        pos1dev.append(pos1train_list[i])
#        pos2dev.append(pos2train_list[i])
#    else:
#        ytrain.append(ytrain_list[i])
#        xtrain.append(xtrain_list[i])
#        pos1train.append(pos1train_list[i])
#        pos2train.append(pos2train_list[i])
#ydev,xdev,pos1dev,pos2dev = np.array(ydev),np.array(xdev),np.array(pos1dev),np.array(pos2dev)
#ytrain, xtrain, pos1train, pos2train = np.array(ytrain),np.array(xtrain),np.array(pos1train),np.array(pos2train)

#搭建keras模型
#dis_vec_table = np.random.uniform(-1,1,(max_len * 2 + 1, position_vec_dim))
#for i in range(position_vec_dim):
#    dis_vec_table[0][i] = 0.
    
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

bilstm = Bidirectional(LSTM(bils, return_sequences=(False),dropout=0.5,recurrent_dropout=0.5,implementation=2), input_shape=(max_len ,word_vec_dim+2* position_vec_dim))(concatenated)
dense = Dropout(0.4)(bilstm)
dense = Dense(200, activation='tanh')(dense)
dense = Dropout(0.4)(dense)
#    cnn = Convolution1D(nb_filter=100, filter_length=3, activation='tanh')(concat_vec)
#    cnn = Convolution1D(nb_filter=100, filter_length=3, activation='tanh')(cnn)
#    flattened = Flatten()(cnn)
#    dense = Dense(100, activation='tanh')(flattened)
#dense = Dense(50, activation='tanh')(dense)
predict = Dense(len(reptest.l2i_dic), activation='softmax')(dense)
model = Model(input=[word, distance_e1, distance_e2], output=predict)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=opt)
model.summary()

 
best_f = 0
for epoch in xrange(epoch_size):
    print 'running the epoch:', (epoch + 1)
    model.fit([xtrain, pos1train, pos2train],ytrain, batch_size=512, epochs=1)
    answer_array_d = model.predict([xtest, pos1test, pos2test], batch_size=256)
    pre_label = ypre2label(answer_array_d)
    pre_r = [[] for i in range(len(goldr))]
#    
    for i in range(len(pre_label)):
        if pre_label[i] != 0:
            label = i2l_dic[pre_label[i]]#类别
            sen_index = indexmap_test[i]#句子索引
            ans = epostest[i]+[label]
            pre_r[sen_index].append(ans)
    print pre_r[10:15]
    p,r,f,f_label = computeFr(goldr,pre_r)
    if f > best_f:
        print 'New Best F-score'
        best_f = f
        save_model('../data/model/bils%s.h5'%(str(bils)),model)
    else:
        print 'Lower than before'
      

print 'load the model'
load_model('../data/model/bils%s.h5'%(str(bils)),model)
print 'predicting ...'
answer_array_d = model.predict([xtest, pos1test, pos2test], batch_size=256)
pre_label = ypre2label(answer_array_d)

pre_r = [[] for i in range(len(goldr))]
for i in range(len(pre_label)):
    if pre_label[i] != 0:
        label = i2l_dic[pre_label[i]]#类别
        sen_index = indexmap_test[i]#句子索引
        ans = epostest[i]+[label]
        pre_r[sen_index].append(ans)

p,r,f,f_label = computeFr(goldr,pre_r)
print p,r,f
#print f_label[1:]
#p, r, f = Evalmicro.eval_mulclass(ytest, answer_array_d)
#print f