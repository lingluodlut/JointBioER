# -*- coding: utf-8 -*-

import numpy as np
import codecs as cs

from keras.models import Model
from keras.layers import Dense,Bidirectional,Embedding,Dropout,Input
from keras.layers import TimeDistributed,Conv1D,GlobalMaxPooling1D,GlobalAveragePooling1D,concatenate,LSTM
from keras.optimizers import RMSprop, SGD, Adam, Adadelta, Adagrad
from ChainCRF import ChainCRF
from utils import ypre2label,label2answer,computeFe,computeFr,computeFr_2
from GenarateXY import GetXY
from utils import save_model,load_model,LoadGoldEntity,LoadGoldRelation
from utils import loadtokens,SaveGoldRelation,SaveGoldEntity,GetModel
from constants import num_class,wv,len_sentence,len_word

#是否加附加特征
if_lemma = 0
if_pos = 0
if_chunk = 0
if_ner = 0
dimdic = {'chardim':15, 'posdim':20, 'chunkdim':10, 'nerdim':10, 'convdim':25}
mask = 0

wordvecmodel = GetModel(u'../../token2vec/chemdner_pubmed_biov5_drug.token4_d100',mask=mask)#得到所有token的词向量，按索引排列，python-list
#导入训练集和测试集对应的tokens，二维python-list,用于根据预测标签来得到实体位置
testtokens = loadtokens(u'../data/DDI_test_BIOES.genia_fea')
traintokens_all = loadtokens(u'../data/DDI_train_BIOES.genia_fea')
traintokens = traintokens_all[0:6066]
vaildtokens = traintokens_all[6066:]
#导入标注的实体和关系 二维python-list
gold_e = LoadGoldEntity('../data/goldEntityAnswer.txt')#标注的实体位置
gold_r = LoadGoldRelation('../data/goldRelationAnswer.txt')#标注的关系的实体位置及类别
gold_e_train_all = LoadGoldEntity('../data/goldEntityAnswer_train.txt')
gold_r_train_all = LoadGoldRelation('../data/goldRelationAnswer_train.txt')
gold_e_train = gold_e_train_all[:6066]
gold_e_vaild = gold_e_train_all[6066:]
gold_r_train = gold_r_train_all[:6066]
gold_r_vaild = gold_r_train_all[6066:]
#导入用于训练的X,Y，并将Y转化为one-hot形式
Data_train_all= GetXY(u'../data/DDI_train_BIOES.genia_fea',mask,if_lemma,if_pos,if_chunk,if_ner)
Data_test= GetXY(u'../data/DDI_test_BIOES.genia_fea',mask,if_lemma,if_pos,if_chunk,if_ner)

def SplitVaild(Data_train_all):
    Data_train = {}
    Data_vaild = {}
    for name in Data_train_all:
        array = Data_train_all[name].tolist()
        train_array = np.array(array[:6066])
        vaild_array = np.array(array[6066:])
        Data_train[name] = train_array
        Data_vaild[name] = vaild_array
    return Data_train,Data_vaild
Data_train,Data_vaild = SplitVaild(Data_train_all)
print u'验证集划分完成'

xtrain,xctrain,ytrain = Data_train['word'],Data_train['char'],Data_train['label']
xvaild,xcvaild,yvaild = Data_vaild['word'],Data_vaild['char'],Data_vaild['label']
xtest,xctest,ytest = Data_test['word'],Data_test['char'],Data_test['label']
input_train = [xtrain,xctrain]
input_vaild = [xvaild,xcvaild]
input_test = [xtest,xctest]
if if_lemma:
    input_train.append(Data_train['lemma'])
    input_vaild.append(Data_vaild['lemma'])
    input_test.append(Data_test['lemma'])
if if_pos:
    input_train.append(Data_train['pos'])
    input_vaild.append(Data_vaild['pos'])
    input_test.append(Data_test['pos'])
if if_chunk:
    input_train.append(Data_train['chunk'])
    input_vaild.append(Data_vaild['chunk'])
    input_test.append(Data_test['chunk'])
if if_ner:
    input_train.append(Data_train['ner'])
    input_vaild.append(Data_vaild['ner'])
    input_test.append(Data_test['ner'])

all_input = []#输入层的列表
all_data = []#merge层需要的层
input_len = 0#输入特征维度的和

word_input = Input(shape=(len_sentence,), dtype='int32', name='word_input')
wordembedding=Embedding(wordvecmodel.shape[0],wordvecmodel.shape[1],weights =[wordvecmodel],trainable=True,mask_zero= mask, input_length=len_sentence)
words = wordembedding(word_input)
#words = Dropout(0.4)(words)
all_input.append(word_input)
all_data.append(words)
input_len += wordvecmodel.shape[1]

chars_input = Input(shape=(len_sentence,len_word), dtype='int32', name='char_input')
chars = TimeDistributed(Embedding(84, output_dim=dimdic['chardim'], trainable=True, mask_zero=mask), name='char_emd')(chars_input)
#chars = Dropout(0.4)(chars)
chars = TimeDistributed(Conv1D(dimdic['convdim'], 3, padding='same',activation='relu'), name="char_cnn1")(chars)
#chars = TimeDistributed(Conv1D(dimdic['convdim'], 3, padding='same',activation='relu'), name="char_cnn2")(chars)
chars_max = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling_max")(chars)
chars_ave = TimeDistributed(GlobalAveragePooling1D(), name="char_pooling_ave")(chars)
chars =  concatenate([chars_max,chars_ave],axis=2)
#chars = Dropout(0.4)(chars)
input_len += dimdic['convdim']*2

all_input.append(chars_input)
all_data.append(chars)

if if_lemma:
    lemma_input = Input(shape=(len_sentence,), dtype='int32', name='lemma_input')
    lemmas = wordembedding(lemma_input)
    #lemmas = Dropout(0.4)(lemmas)
    all_input.append(lemma_input)
    all_data.append(lemmas)
    input_len += wordvecmodel.shape[1]
if if_pos:
    pos_input = Input(shape=(len_sentence,), dtype='int32', name='pos_input')
    poss = Embedding(46, output_dim=dimdic['posdim'], trainable=True, mask_zero=mask)(pos_input)
    poss = Dropout(0.4)(poss)
    all_input.append(pos_input)
    all_data.append(poss)
    input_len += dimdic['posdim']
if if_chunk:
    chunk_input = Input(shape=(len_sentence,), dtype='int32', name='chunk_input')
    chunks = Embedding(19, output_dim=dimdic['chunkdim'], trainable=True, mask_zero=mask)(chunk_input)
    chunks = Dropout(0.4)(chunks)
    all_input.append(chunk_input)
    all_data.append(chunks)
    input_len+= dimdic['chunkdim']
if if_ner:
    ner_input = Input(shape=(len_sentence,), dtype='int32', name='ner_input')
    ners = Embedding(12, output_dim=dimdic['nerdim'], trainable=True, mask_zero=mask)(ner_input)
    ners = Dropout(0.4)(ners)
    all_input.append(ner_input)
    all_data.append(ners)
    input_len += dimdic['nerdim']

bils = input_len*3 
ls = input_len*2 

x =  concatenate(all_data,axis=2)
x = Dropout(0.4)(x)
bilstm = Bidirectional(LSTM(300, return_sequences=True,dropout=0.4,recurrent_dropout=0.4),name = 'bilstm')(x)
bilstm = Bidirectional(LSTM(300, return_sequences=True,dropout=0.4,recurrent_dropout=0.4))(bilstm)
dense = TimeDistributed(Dense(200, activation='tanh',name = 'dense_cl'))(bilstm)
#lstm = LSTM(ls, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)(bilstm)
#dense = lstm
dense = Dropout(0.4)(dense)
dense = TimeDistributed(Dense(num_class, activation=None,name = 'dense_cl'))(dense)
crf = ChainCRF()
output1 = crf(dense)
model = Model(inputs = all_input ,outputs = output1)
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#opt = Adam()
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
fpath = '../data/fscore/crf+bilstm_luo.txt'
mpath = '../data/model/crf+bilstm_luo.h5'
fpout = cs.open(fpath,'w','utf-8')

#load_model(mpath,model)

for i in range(100):
    print u'第%d轮次：'%(i+1)
    model.fit(input_train, ytrain, batch_size=32, epochs=1)#epochs练过程中数据将被“轮”多少次
    #使用开发集的f来决定是否保存模型，实体F最大的模型和关系F最大的模型分别保存
    y_predict = model.predict(input_vaild, batch_size=32)
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
    y_predict = model.predict(input_test, batch_size=32)
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
y_predict = model.predict(input_test, batch_size=64)
pre_label = ypre2label(y_predict)
pre_e,pre_r= label2answer(pre_label,testtokens)
pe,re,fe,fe_label= computeFe(gold_e,pre_e)
pr,rr,fr,fr_label= computeFr(gold_r,pre_r)
r1,r2 = computeFr_2(gold_r,pre_r)
#SaveGoldRelation('../data/prer_cl.txt',pre_r)
print '\nThe final performance on test corpus is :'
print('entity:p ,r ,f %f %f %f' % (pe,re,fe))
print('relation:p ,r ,f %f %f %f' % (pr,rr,fr))
print('r1= %f r2=%f'%(r1,r2))
print('drug:%f group:%f brand:%f drug_n:%f' % (fe_label[0],fe_label[1],fe_label[2],fe_label[3]))
print('ME:%f AD:%f EF:%f IN:%f' % (fr_label[0],fr_label[1],fr_label[2],fr_label[3]))

print 'tis is : crf+bils%s-ls%s-lemma%s-pos%s-chunk%s-ner%s'%(bils,ls,if_lemma,if_pos,if_chunk,if_ner)
