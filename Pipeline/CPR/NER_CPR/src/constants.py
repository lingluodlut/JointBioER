# -*- coding: utf-8 -*-

import codecs as cs
#一些超参数
bils = 500#神经网络双向LSTM神经元数
ls = 200#神经网络单向LSTm神经元数
wv = 100#词向量维度
label_mode = 'BIOES'#标注模式
if_repeat = 'Y'#是否允许实体参与多个关系
len_sentence = 150#句子的最大长度，即每个样本的token序列长度
len_word = 10#每个token的最大长度
el2i = {'chem':0,'geneY':1,'geneN':2}#实体类别到索引的映射
#生成标签和索引的映射
l2i_dic = {}
index = 0
fp = cs.open('../data/CPR_train_Y.txt','r','utf-8')
text = fp.read().split('\n\n')[:-1]
fp.close()

for s in text:
    tokens = s.split('\n')
    for t in tokens:
        label = t.split('\t')[1]
        if label not in l2i_dic:
            l2i_dic[label] = index
            index+=1
i2l_dic = {}
for label in l2i_dic:
    index = l2i_dic[label]
    i2l_dic[index] = label

num_class = len(l2i_dic)
#
B_label = {}
I_label = {}
E_label = {}
S_label = {}
O_label = {}

for label in l2i_dic:
    if label[0] == 'B':
        B_label[l2i_dic[label]] = ''
    elif label[0] == 'I':
        I_label[l2i_dic[label]] = ''
    elif label[0] == 'E':
        E_label[l2i_dic[label]] = ''
    elif label[0] == 'S':
        S_label[l2i_dic[label]] = ''
    else:
        O_label[l2i_dic[label]] = ''


