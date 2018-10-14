# -*- coding: utf-8 -*-


import numpy as np
import codecs as cs

from keras.models import Model
from keras.layers import Dense,Bidirectional,Embedding,Dropout,Input
from keras.layers import TimeDistributed,Conv1D,GlobalMaxPooling1D,concatenate,LSTM
from keras.utils import np_utils
from keras.optimizers import RMSprop#, SGD, Adam, Adadelta, Adagrad
from ChainCRF import ChainCRF
from utils import ypre2label,label2answer,computeFe
from GenerateXY import GetXY
from utils import save_model,load_model,LoadGoldEntity
from utils import loadtokens,SaveGoldRelation,SaveGoldEntity,GetModel
from constants import num_class,len_sentence,len_word
from constants import bils,ls,wv
import pickle
#根据超参数来选择词向量模型
mask = 0
wordvec = u'../../token2vec/chemdner_pubmed_biov5_drug.token4_d100_CPR'
#wordvec = u'../../token2vec/medline_chemdner_pubmed_biov5_drug.token4_d50'
wordvecmodel = GetModel(wordvec,mask=mask)#得到所有token的词向量，按索引排列，python-list
#导入训练集和测试集对应的tokens，二维python-list,用于根据预测标签来得到实体位置
testtokens = loadtokens(u'../data/CPR_test_Y.txt')
traintokens = loadtokens(u'../data/CPR_train_Y.txt')
vaildtokens = loadtokens(u'../data/CPR_vaild_Y.txt')
#导入标注的实体和关系 二维python-list
gold_e = LoadGoldEntity('../data/goldEntityAnswer_test.txt')#标注的实体位置
gold_e_train = LoadGoldEntity('../data/goldEntityAnswer_train.txt')
gold_e_vaild = LoadGoldEntity('../data/goldEntityAnswer_vaild.txt')
#导入用于训练的X,Y，并将Y转化为one-hot形式
xtrain,y_train,xctrain= GetXY(u'../data/CPR_train_Y.txt',mask)
xtest,y_test,xctest= GetXY(u'../data/CPR_test_Y.txt',mask)
xvaild,y_vaild,xcvaild= GetXY(u'../data/CPR_vaild_Y.txt',mask)
num_train = len(y_train)
num_test = len(y_test)
num_vaild = len(y_vaild)
ytest = np_utils.to_categorical(y_test, num_class)#将数值型标签转换为多分类数组
ytest = np.reshape(ytest, (num_test,len_sentence,num_class))#重新reshape
ytrain = np_utils.to_categorical(y_train, num_class)
ytrain = np.reshape(ytrain, (num_train,len_sentence, num_class))
yvaild = np_utils.to_categorical(y_vaild, num_class)
yvaild = np.reshape(yvaild, (num_vaild,len_sentence, num_class))


word_input = Input(shape=(len_sentence,), dtype='int32', name='word_input')
words = Embedding(wordvecmodel.shape[0],wordvecmodel.shape[1],weights =[wordvecmodel],trainable=True,mask_zero= mask, input_length=len_sentence)(word_input)
#words = Dropout(0.4)(words)

chars_input = Input(shape=(len_sentence,len_word), dtype='int32', name='char_input')
chars = TimeDistributed(Embedding(148, output_dim=20, trainable=True, mask_zero=mask), name='char_emd')(chars_input)
chars = TimeDistributed(Conv1D(20, 3, padding='same'), name="char_cnn")(chars)
chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)
#chars = Dropout(0.4)(chars)

x =  concatenate([words,chars],axis=2)
bilstm = Bidirectional(LSTM(300, return_sequences=(True),dropout=0.4,recurrent_dropout=0.4),name = 'bilstm')(x)
bilstm = Bidirectional(LSTM(300, return_sequences=(True),dropout=0.4,recurrent_dropout=0.4),name = 'bilstm')(bilstm)
#lstm = LSTM(ls, return_sequences=(True),dropout=0.4,recurrent_dropout=0.4)(bilstm)
dense = TimeDistributed(Dense(200, activation='tanh',name = 'dense_cl'))(bilstm)
lstm = Dropout(0.4)(dense)
dense = TimeDistributed(Dense(num_class, activation=None,name = 'dense_cl'))(lstm)
crf = ChainCRF()
output1 = crf(dense)
model = Model(inputs = [word_input,chars_input],outputs = output1)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss=crf.loss,optimizer = opt,metrics=['accuracy'])
model.summary()

#为每组实验记录F值得迭代
maxf_e = 0.0
fpath = '../data/fscore/crf+bils-bils%s-ls%s-wv%s.txt'%(str(bils),str(ls),str(wv))
mpath = '../data/model/crf+bils-bils%s-ls%s-wv%s.h5'%(str(bils),str(ls),str(wv))
fpout = cs.open(fpath,'w','utf-8')
#
for i in range(40):
    print (u'第%d轮次：'%(i+1))
    model.fit([xtrain,xctrain], ytrain, batch_size=128, epochs=1)#epochs练过程中数据将被“轮”多少次
    #使用开发集的f来决定是否保存模型，实体F最大的模型和关系F最大的模型分别保存
    y_predict = model.predict([xvaild,xcvaild], batch_size=64)
    pre_label = ypre2label(y_predict)
    pre_e,temp1,temp2 = label2answer(pre_label,vaildtokens)
    pe,re,fe,fe_label= computeFe(gold_e_vaild,pre_e)
    fpout.write(u'第%s次开发集F值分别为：%f\n'%(str(i+1),fe))
    if fe > maxf_e:
        maxf_e = fe
        save_model(mpath, model)
    #每轮迭代都额外考察在测试集上的数据，但测试集不参与模型的选择
    y_predict = model.predict([xtest,xctest], batch_size=128)
    pre_label = ypre2label(y_predict)
    pre_e,temp1,temp2= label2answer(pre_label,testtokens)
    pe,re,fe,fe_label= computeFe(gold_e,pre_e)
    fpout.write(u'第%s次测试集F值分别为：%f\n'%(str(i+1),fe))
fpout.close()

#下载关系的F值最高的模型，并用其对测试集进行预测
load_model(mpath,model)
y_predict = model.predict([xtest,xctest], batch_size=32)
pre_label = ypre2label(y_predict)
pre_e,e2t_c,e2t_g= label2answer(pre_label,testtokens)
pe,re,fe,fe_label= computeFe(gold_e,pre_e)
#存储预测结果
SaveGoldEntity('../data/predictE_test.txt',pre_e)
pickle.dump(e2t_c,open('../data/e2t_c.pkl','wb'))
pickle.dump(e2t_g,open('../data/e2t_g.pkl','wb'))
#输出预测结果的F值
print ('\nThe final performance on test corpus is :')
print('entity:p ,r ,f %f %f %f' % (pe,re,fe))
print('chem:%f geneY:%f geneN:%f' % (fe_label[0],fe_label[1],fe_label[2]))
print ('tis is : crf+bils-bils%s-ls%s'%(str(bils),str(ls)))