# -*- coding: utf-8 -*-

import numpy as np
import codecs as cs
from keras.models import Sequential,Model
from keras.layers import Dense,Bidirectional,Embedding,Dropout,Input,Activation
from keras.layers import TimeDistributed,Conv1D,GlobalMaxPooling1D,concatenate,LSTM
from keras.utils import np_utils
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad
from ChainCRF import ChainCRF
from GenarateXY import GetXY
from utils import save_model,load_model,LoadGoldEntity,loadtokens,SaveGoldEntity,GetModel
from utils import ypre2label,label2answer,computeFe
from constants import num_class,bils,ls,label_mode,len_sentence,len_word
import pickle
#统计信息
maxf_e = 0.0
mask = 0
#导入所有句子、标注的实体位置、标注的关系、训练集和测试集
wordvecmodel = GetModel(u'../../token2vec/medline_chemdner_pubmed_biov5_drug.token4_d50',mask)
testtokens = loadtokens(u'../data/DDI_test_%s.txt'%(label_mode))
traintokens_all = loadtokens(u'../data/DDI_train_%s.txt'%(label_mode))
gold_e = LoadGoldEntity('../data/goldEntityAnswer.txt')#标注的实体位置
gold_e_train_all = LoadGoldEntity('../data/goldEntityAnswer_train.txt')
xtrain_all,y_train_all,xctrain_all= GetXY(u'../data/DDI_train_%s.txt'%(label_mode),mask)
xtest,y_test,xctest= GetXY(u'../data/DDI_test_%s.txt'%(label_mode),mask)
numtest = len(y_test)
numtrain = len(y_train_all)
ytest = np_utils.to_categorical(y_test, num_class)#将数值型标签转换为多分类数组
ytest = np.reshape(ytest, (numtest,len_sentence,num_class))#重新reshape
ytrain_all = np_utils.to_categorical(y_train_all, num_class)
ytrain_all = np.reshape(ytrain_all, (numtrain ,len_sentence, num_class))

xvaild = []
xcvaild = []
yvaild = []
xtrain = []
xctrain = []
ytrain = []
xtrain_all = xtrain_all.tolist()
xctrain_all = xctrain_all.tolist()
ytrain_all = ytrain_all.tolist()
vaildtokens = []
traintokens = []
gold_e_train = []
gold_e_vaild = []
for i in range(len(xtrain_all)):
    if i > 6055:
        xvaild.append(xtrain_all[i])
        xcvaild.append(xctrain_all[i])
        yvaild.append(ytrain_all[i])
        vaildtokens.append(traintokens_all[i])
        gold_e_vaild.append(gold_e_train_all[i])
    else:
        xtrain.append(xtrain_all[i])
        xctrain.append(xctrain_all[i])
        ytrain.append(ytrain_all[i])
        traintokens.append(traintokens_all[i])
        gold_e_train.append(gold_e_train_all[i])
xtrain = np.array(xtrain)
xctrain = np.array(xctrain)
ytrain = np.array(ytrain)
xvaild = np.array(xvaild)
xcvaild = np.array(xcvaild)
yvaild = np.array(yvaild)
#模型搭建
word_input = Input(shape=(len_sentence,), dtype='int32', name='word_input')
chars_input = Input(shape=(len_sentence,len_word), dtype='int32', name='char_input')
chars = TimeDistributed(Embedding(84, output_dim=30, trainable=True, mask_zero=mask), name='char_emd')(chars_input)
chars = TimeDistributed(Conv1D(50, 3, padding='same'), name="char_cnn")(chars)
chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)
chars = Dropout(0.4)(chars)
words = Embedding(wordvecmodel.shape[0],wordvecmodel.shape[1],weights =[wordvecmodel],mask_zero= mask, input_length=len_sentence)(word_input)
words = Dropout(0.4)(words)
x =  concatenate([words,chars],axis=2)
bilstm = Bidirectional(LSTM(300, return_sequences=(True),dropout=0.4,recurrent_dropout=0.4), input_shape=(len_sentence, 50))(x)
bilstm = Bidirectional(LSTM(300, return_sequences=(True),dropout=0.4,recurrent_dropout=0.4), input_shape=(len_sentence, 50))(bilstm)
#lstm = LSTM(ls,return_sequences=(True))(bilstm)
dense = TimeDistributed(Dense(200, activation='tanh',name = 'dense_cl'))(bilstm)
lstm = Dropout(0.4)(dense)
dense = TimeDistributed(Dense(num_class, activation=None))(lstm)
crf = ChainCRF()
output1 = crf(dense)
model = Model(inputs = [word_input,chars_input],outputs = output1)
#opt = Adagrad(lr=0.01, epsilon=1e-06)
#opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss=crf.loss,optimizer = opt,metrics=['accuracy'])
model.summary()

#为每组实验记录F值得迭代
fpath = u'../data/fscore/Fscore-crf-bils%s+ls100.txt'%(str(bils))
mpath = u'../data/model/crf+bils%s+ls100.h5'%(str(bils))
prepath1 = u'../data/predict/pre_e_crf+bils%s+ls100.txt'%(str(bils))
prepath2 = u'../data/predict/en2t_crf+bils%s+ls100.pkl'%(str(bils))

fpout = cs.open(fpath,'w','utf-8')
for i in range(120):
    model.fit([xtrain,xctrain], ytrain, batch_size=128, epochs=1)
    #使用开发集的f来决定是否保存模型，实体F最大的模型和关系F最大的模型分别保存
    y_predict = model.predict([xvaild,xcvaild], batch_size=128)
    pre_label = ypre2label(y_predict)
    pre_e,e_token= label2answer(pre_label,vaildtokens)
    print (u'第%s次训练开发集的结果为'%(str(i)))
    pe,re,fe = computeFe(gold_e_vaild,pre_e)
    fpout.write(u'第%s次开发集F值分别为：%f\n'%(str(i+1),fe))
    if fe > maxf_e:
        maxf_e = fe
        save_model(mpath, model)
    #每轮迭代都额外考察在测试集上的数据，但测试集不参与模型的选择
    y_predict = model.predict([xtest,xctest], batch_size=128)
    pre_label= ypre2label(y_predict)
    pre_e, e_token= label2answer(pre_label,testtokens)
    print (u'第%s次训练测试集的结果为'%(str(i)))
    pe,re,fe = computeFe(gold_e,pre_e)
    fpout.write(u'第%s次测试集F值分别为：%f\n\n'%(str(i+1),fe))
fpout.close()

#下载实体的F值最高的模型，并用其对测试集进行预测，保存预测出的实体及关系
load_model(mpath,model)
y_predict = model.predict([xtest,xctest], batch_size=128)
pre_label = ypre2label(y_predict)
#得到预测的实体位置及实体到token的映射
pre_e,e_token = label2answer(pre_label,testtokens)
SaveGoldEntity(prepath1,pre_e)
f1 = open(prepath2, 'wb') 
pickle.dump(e_token, f1)
f1.close()
#计算实体的F值
pe,re,fe = computeFe(gold_e,pre_e)
print('测试实体的准确率 召回率 F值是%f %f %f' % (pe,re,fe))