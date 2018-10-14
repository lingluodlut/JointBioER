# -*- coding: utf-8 -*-

import theano
import numpy as np
import codecs as cs
from keras.models import Sequential
from keras.layers import Dense,Bidirectional,Dropout,Embedding
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import RMSprop#, SGD, Adam, Adadelta, Adagrad
from utils import save_model,load_model,LoadGoldEntity,LoadGoldRelation
from utils import loadtokens,SaveGoldRelation,SaveGoldEntity,GetModel
from constants import i2l_dic,rl2i,el2i,B_label,S_label
from constants import num_class,bils,ls,if_repeat,label_mode,len_sentence

#导入训练集和测试集对应的tokens，二维python-list,用于根据预测标签来得到实体位置
testtokens = loadtokens(u'../data/DDI_test_%s_Y.txt'%(label_mode))
#导入标注的实体和关系 二维python-list
gold_e = LoadGoldEntity('../data/goldEntityAnswer.txt')#标注的实体位置
gold_r = LoadGoldRelation('../data/goldRelationAnswer.txt')#标注的关系的实体位置及类别
#导入用于训练的X,Y，并将Y转化为one-hot形式
y_test = np.load("../data/testY-%s.npy"%(str(len_sentence)))
def label2answer(predict_label,tokens):
    '''
    根据每句话中每个token的标签来得到实体位置
    输入： Python-list  2D，值为每个token的类别序号
    输出： Python-list  3d，第三维为每个实体的起始位置列表[begin，end]（含头含尾）
          Python-list  3d，第三维为每个关系的组成及类别[begin，end，begin，end，label]（含头含尾）
    '''
    pre_entity = []#列表 存放预测出的实体位置及类别
    pre_elabel = []#列表 存在每个实体的首个token的标签
    for i in range(len(tokens)):
        num_token = len(tokens[i])#每个句子的token数
        if num_token > len_sentence:#句子长度最大为50
            num_token = len_sentence
        sen_entity = []#每句话中的实体位置列表
        sen_elabel = []#每个实体的首个token的标签列表，用于寻找配对关系
        begin = -1
        end = -1
        j = 0#token的序号
        while(j < num_token):
            begin = end + 1#确定当前token的起止位置 含头含尾
            end = end + len(tokens[i][j])
            if predict_label[i][j] in S_label:#如果类别为S 则直接将该token的位置加入列表
                tlabel = i2l_dic[predict_label[i][j]]#取出这个token的标签
                sen_elabel.append(tlabel)
                elabel = tlabel.split('-')[1]#切分出实体的类别
                sen_entity.append([begin,end,elabel])
                j += 1
            elif predict_label[i][j] in B_label:#如果类别为B 则...
                for endj in range(j,num_token):
                    if predict_label[i][endj] == 0:
                        break
                if endj == num_token-1 and predict_label[i][endj]!=0:
                    endj = num_token
                for tempj in range(j+1,endj):
                    end = end + len(tokens[i][tempj])
                tlabel = i2l_dic[predict_label[i][j]]#取出token的标签
                sen_elabel.append(tlabel)
                elabel = tlabel.split('-')[1]#切分出实体的类别
                sen_entity.append([begin,end,elabel])
                j = endj
            else:
                j+=1
        pre_entity.append(sen_entity)
        pre_elabel.append(sen_elabel)
        
    pre_relation = []
    for i in range(len(pre_elabel)):#每个句子
        if len(pre_elabel[i])<2:#如果实体数小于2 肯定无关系
            pre_relation.append([])
            continue
        sen_relation = []
        #past_token = []
        for j in range(len(pre_elabel[i])):#每个token
            infors = pre_elabel[i][j].split('-')
            if len(infors)>2:#如果是包含关系的实体
                rlabel = infors[2]#取出当前词的关系类型
                newlabel = ''#可以与之配对的标签
                if infors[3] == '1':
                    newlabel = rlabel + '-2'
                else:
                    newlabel = rlabel + '-1'
                for k in range(j+1,len(pre_elabel[i])):
                    if len(pre_elabel[i][k].split('-')) == 4:
                        if pre_elabel[i][k][-4:] == newlabel: #and k not in past_token :
                           sen_relation.append([pre_entity[i][j][0],pre_entity[i][j][1],pre_entity[i][k][0],pre_entity[i][k][1],rlabel])    
                       #past_token.append(k)
        pre_relation.append(sen_relation)
    return pre_entity,pre_relation
        
def computeFe(gold_entity,pre_entity):
    '''
    根据标注的实体位置和预测的实体位置，计算prf,完全匹配
    输入： Python-list  3D，值为每个实体的起始位置列表[begin，end]
    输出： float
    '''    
    truenum = 0
    prenum = 0
    goldnum = 0
    for i in range(len(gold_entity)):
        goldnum += len(gold_entity[i])
        prenum  += len(pre_entity[i])
        if len(gold_entity[i]) == 0 or len(pre_entity[i]) == 0:        
            continue
        else:
            for pre in pre_entity[i]:
                for gold in gold_entity[i]:
                    if pre[0] == gold[0] and pre[1] == gold[1] and pre[2] == gold[2]:
                        truenum +=1
                        break
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall)) 
    except:
        precise = recall = f = 0
    print('本轮实体的F值是 %f' %(f))
    labeltrue = [0,0,0,0]#四种类别分别的预测正确数量
    labelpre = [0,0,0,0]
    labelgold = [0,0,0,0]
    for i in range(len(pre_entity)):#每句话
        for pre in pre_entity[i]:
            label = pre[2]#得到该预测实体的类别
            labelpre[el2i[label]] += 1#该类别的预测关系数+1
            for gold in gold_entity[i]:
                 if cmp(pre, gold)==0:   
                    labeltrue[el2i[label]] +=1#该类别的预测正确关系数+1
                    break
    for i in range(len(gold_entity)):#每句话
        for gold in gold_entity[i]:
            label = gold[2]#得到该预测关系的类别
            labelgold[el2i[label]] += 1#该类别的预测关系数+1
    f_label = []
    for i in range(4):
        try:
            labelp = float(labeltrue[i]) / float(labelpre[i])
            labelr = float(labeltrue[i]) / float(labelgold[i])
            labelf = float(2 * labelp * labelr /( labelp + labelr))
            f_label.append(labelf)
        except:
            f_label.append(0.0)
    return precise,recall,f
def computeFr(gold_relation,pre_relation):
    '''
    根据标注的关系和预测的关系，计算prf,完全匹配
    输入： Python-list  3D，第三维为每个关系的组成及类别[begin，end，begin，end，label]（含头含尾）
    输出： float
    '''
    truenum = 0
    prenum = 0
    goldnum = 0
    for i in range(len(gold_relation)):
        goldnum += len(gold_relation[i])
        prenum  += len(pre_relation[i])
        if len(gold_relation[i]) == 0 or len(pre_relation[i]) == 0:        
            continue
        else:
            for pre in pre_relation[i]:
                for gold in gold_relation[i]:
                    #if pre[0] == gold[0] and pre[1] == gold[1] and pre[2] == gold[2] and pre[3] == gold[3] and pre[4] == gold[4]:
                     if cmp(pre, gold)==0:   
                        truenum +=1
                        break
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall)) 
    except:
        precise = recall = f = 0
    print('本轮关系的F值是%f' %(f))
    labeltrue = [0,0,0,0]#四种类别分别的预测正确数量
    labelpre = [0,0,0,0]
    labelgold = [0,0,0,0]
    for i in range(len(pre_relation)):#每句话
        for pre in pre_relation[i]:
            label = pre[4]#得到该预测关系的类别
            labelpre[rl2i[label]] += 1#该类别的预测关系数+1
            for gold in gold_relation[i]:
                 if cmp(pre, gold)==0:   
                    labeltrue[rl2i[label]] +=1#该类别的预测正确关系数+1
                    break
    for i in range(len(gold_relation)):#每句话
        for gold in gold_relation[i]:
            label = gold[4]#得到该预测关系的类别
            labelgold[rl2i[label]] += 1#该类别的预测关系数+1
    f_label = []
    for i in range(4):
        try:
            labelp = float(labeltrue[i]) / float(labelpre[i])
            labelr = float(labeltrue[i]) / float(labelgold[i])
            labelf = float(2 * labelp * labelr /( labelp + labelr))
            f_label.append(labelf)
        except:
            f_label.append(0.0)
    return precise,recall,f,f_label
predict_e,predict_r =label2answer(y_test,testtokens)
pe,re,fe = computeFe(gold_e,predict_e)
pr,rr,fr,frlabel = computeFr(gold_r,predict_r)
print u'由goldy得到的实体的PRF为%f %f %f'%(pe,re,fe)
print u'由goldy得到的关系的PRF为%f %f %f'%(pr,rr,fr)
