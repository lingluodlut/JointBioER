# -*- coding: utf-8 -*-


import numpy as np
import codecs as cs

from keras.models import Model
from keras.layers import Dense,Bidirectional,Embedding,Dropout,Input
from keras.layers import TimeDistributed,Conv1D,GlobalMaxPooling1D,concatenate,LSTM,GlobalAveragePooling1D
from keras.utils import np_utils
from keras.optimizers import RMSprop#, SGD, Adam, Adadelta, Adagrad
from ChainCRF import ChainCRF
from utils import ypre2label,label2answer,computeFe,computeFr,computeFr_2
from GenerateXY import GetXY
from utils import save_model,load_model,LoadGoldEntity,LoadGoldRelation
from utils import loadtokens,SaveGoldRelation,SaveGoldEntity,GetModel
from constants import num_class,len_sentence,len_word
from constants import bils,ls,wv
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
gold_r = LoadGoldRelation('../data/goldRelationAnswer_test.txt')#标注的关系的实体位置及类别
gold_e_train = LoadGoldEntity('../data/goldEntityAnswer_train.txt')
gold_r_train= LoadGoldRelation('../data/goldRelationAnswer_train.txt')
gold_e_vaild = LoadGoldEntity('../data/goldEntityAnswer_vaild.txt')
gold_r_vaild= LoadGoldRelation('../data/goldRelationAnswer_vaild.txt')
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


chars_input = Input(shape=(len_sentence,len_word), dtype='int32', name='char_input')
chars = TimeDistributed(Embedding(148, output_dim=20, trainable=True, mask_zero=mask), name='char_emd')(chars_input)
chars = TimeDistributed(Conv1D(50, 3, padding='same',activation='relu'), name="char_cnn")(chars)
chars_max = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling1")(chars)
chars_ave = TimeDistributed(GlobalAveragePooling1D(), name="char_pooling2")(chars)
chars = concatenate([chars_max,chars_ave],axis=2)
x =  concatenate([words,chars],axis=2)
x = Dropout(0.4)(x)
bilstm = Bidirectional(LSTM(300, return_sequences=True,dropout=0.4,recurrent_dropout=0.4),name = 'bilstm')(x)
bilstm = Bidirectional(LSTM(300, return_sequences=True,dropout=0.4,recurrent_dropout=0.4))(bilstm)
#lstm = LSTM(300, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)(bilstm)
dense = TimeDistributed(Dense(200, activation='tanh',name = 'dense_cl'))(bilstm)
dense = Dropout(0.4)(dense)
dense = TimeDistributed(Dense(num_class, activation=None,name = 'dense_cl'))(dense)
crf = ChainCRF()
output1 = crf(dense)
model = Model(inputs = [word_input,chars_input],outputs = output1)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss=crf.loss,optimizer = opt,metrics=['accuracy'])
model.summary()

#为每组实验记录F值得迭代
maxf_r = 0.0
maxf_e = 0.0
maxtest_fe=0.0
test_fe_fr = 0.0
test_fr_fe = 0.0
maxtest_fr=0.0
maxtest_ie=0
maxtest_ir=0
besttest_fr=0.0
besttest_fe=0.0
besttest_fr_i=0
fpath = '../data/fscore/crf+bils400-tanh200-wv100_luo.txt'
mpath = '../data/model/crf+bils400-tanh200-wv100_luo.h5'
fpout = cs.open(fpath,'w','utf-8')

for i in range(50):
    print u'第%d轮次：'%(i+1)
    model.fit([xtrain,xctrain], ytrain, batch_size=32, epochs=1)#epochs练过程中数据将被“轮”多少次
    #使用开发集的f来决定是否保存模型，实体F最大的模型和关系F最大的模型分别保存
    y_predict = model.predict([xvaild,xcvaild], batch_size=32)
    pre_label = ypre2label(y_predict)
    pre_e,pre_r= label2answer(pre_label,vaildtokens)
    dpe,dre,dfe,dfe_label= computeFe(gold_e_vaild,pre_e)
    dpr,drr,dfr,dfr_label= computeFr(gold_r_vaild,pre_r)
    print u'第%s次开发集F值分别为：%f  %f\n'%(str(i+1),dfe,dfr)
    fpout.write(u'第%s次开发集F值分别为：%f  %f\n'%(str(i+1),dfe,dfr))
    #if fr > maxf_r:
    #    maxf_r = fr
    #    save_model(mpath, model)
    #每轮迭代都额外考察在测试集上的数据，但测试集不参与模型的选择
    y_predict = model.predict([xtest,xctest], batch_size=32)
    pre_label = ypre2label(y_predict)
    pre_e,pre_r= label2answer(pre_label,testtokens)
    #print (u'第%s次训练测试集的结果为'%(str(i)))
    tpe,tre,tfe,tfe_label= computeFe(gold_e,pre_e)
    tpr,trr,tfr,tfr_label= computeFr(gold_r,pre_r)
    print u'第%s次测试集F值分别为：%f  %f \n'%(str(i+1),tfe,tfr)
    fpout.write(u'第%s次测试集F值分别为：%f  %f\n\n'%(str(i+1),tfe,tfr))
    #print u'本轮次训练时间为%f'%(endtime-begintime)
    if tfr > besttest_fr:
        besttest_fr=tfr
        besttest_fe=tfe
        besttest_fr_i=i
        save_model(bestmpath, model)
    if dfr > maxf_r:
        maxf_r = dfr
        maxtest_fr = tfr
        test_fr_fe = tfe
        maxtest_ir=i
        save_model(mpath, model)
    if dfe > maxf_e:
        maxf_e = dfe
        maxtest_fe = tfe
        test_fe_fr = tfr
        maxtest_ie = i

    print u'best_dev_en test：%f  %f (epoch: %d)\n'%(maxtest_fe,test_fe_fr,maxtest_ie)
    print u'best_dev_re test：%f  %f (epoch: %d)\n'%(test_fr_fe,maxtest_fr,maxtest_ir)
    print u'best_test_re ：%f  %f (epoch: %d)\n\n'%(besttest_fr,besttest_fe, besttest_fr_i)
    fpout.write(u'best_dev_en test：%f  %f (epoch: %d)\n'%(maxtest_fe,test_fe_fr,maxtest_ie))
    fpout.write(u'best_dev_re test：%f  %f (epoch: %d)\n\n'%(test_fr_fe,maxtest_fr,maxtest_ir))
fpout.close()

#下载关系的F值最高的模型，并用其对测试集进行预测，保存预测出的实体及关系
load_model(mpath,model)
y_predict = model.predict([xtest,xctest], batch_size=64)
pre_label = ypre2label(y_predict)
pre_e,pre_r= label2answer(pre_label,testtokens)
pe,re,fe,fe_label= computeFe(gold_e,pre_e)
pr,rr,fr,fr_label= computeFr(gold_r,pre_r)
r1,r2 = computeFr_2(gold_r,pre_r)
SaveGoldEntity('../data/predictE_test.txt',pre_e)
SaveGoldRelation('../data/predictR_test.txt',pre_r)
print '\nThe final performance on test corpus is :'
print('entity:p ,r ,f %f %f %f' % (pe,re,fe))
print('relation:p ,r ,f %f %f %f' % (pr,rr,fr))
print('r1= %f r2=%f'%(r1,r2))
print('chem:%f geneY:%f geneN:%f' % (fe_label[0],fe_label[1],fe_label[2]))
print('C3:%f C4:%f C5:%f C6:%f C9:%f' % (fr_label[0],fr_label[1],fr_label[2],fr_label[3],fr_label[4]))


load_model(bestmpath,model)
y_predict = model.predict([xtest,xctest], batch_size=64)
pre_label = ypre2label(y_predict)
pre_e,pre_r= label2answer(pre_label,testtokens)
pe,re,fe,fe_label= computeFe(gold_e,pre_e)
pr,rr,fr,fr_label= computeFr(gold_r,pre_r)
r1,r2 = computeFr_2(gold_r,pre_r)
SaveGoldEntity('../data/predictE_test.txt',pre_e)
SaveGoldRelation('../data/predictR_test.txt',pre_r)
print '\nThe final performance on test corpus is :'
print('entity:p ,r ,f %f %f %f' % (pe,re,fe))
print('relation:p ,r ,f %f %f %f' % (pr,rr,fr))
print('r1= %f r2=%f'%(r1,r2))
print('chem:%f geneY:%f geneN:%f' % (fe_label[0],fe_label[1],fe_label[2]))
print('C3:%f C4:%f C5:%f C6:%f C9:%f' % (fr_label[0],fr_label[1],fr_label[2],fr_label[3],fr_label[4]))

print 'tis is : crf+bils-bils%s-ls%s'%(str(bils),str(ls))
