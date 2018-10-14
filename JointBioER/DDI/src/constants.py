# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:56:05 2017

@author: cmy
"""
#一些超参数
#bils = 250#神经网络双向LSTM神经元数
#ls = 100#神经网络单向LSTm神经元数
wv = 100#词向量维度
label_mode = 'BIOES'#标注模式
if_repeat = 'Y'#是否允许实体参与多个关系
len_sentence = 117#句子的最大长度，即每个样本的token序列长度，最长为117
len_word = 25
rl2i = {'ME':0,'AD':1 ,'EF':2,'IN':3}#关系类别到索引的映射
el2i = {'drug':0,'group':1,'brand':2,'drug_n':3}#实体类别到索引的映射

label = ['B','I','E','S']
entit = ['drug','group','brand','drug_n']
relat = ['EF','AD','ME','IN','MU']
dirct = ['1','2','M']
label2index = {}
index2label = {}
B_label = []
I_label = []
E_label = []
S_label = []
O_label = []
nowindex = 0
label2index['O'] = nowindex
nowindex += 1
for l in label:
    for e in entit:
        label2index[l+'-'+e] = nowindex
        if l == 'B':
            B_label.append(nowindex)
        elif l == 'I':
            I_label.append(nowindex)
        elif l == 'E':
            E_label.append(nowindex)
        elif l == 'S':
            S_label.append(nowindex)
        nowindex += 1
        
for l in label:
    for e in entit:
        for r in relat:
            for d in dirct:
                label2index[l+'-'+e+'-'+r+'-'+d] = nowindex
                if l == 'B':
                    B_label.append(nowindex)
                elif l == 'I':
                    I_label.append(nowindex)
                elif l == 'E':
                    E_label.append(nowindex)
                elif l == 'S':
                    S_label.append(nowindex)
                nowindex +=1
for each in label2index:
    index2label[label2index[each]] = each
    
i2l_dic = index2label
l2i_dic = label2index
num_class = len(l2i_dic)#多分类类别

relationmap = {'ME':['ME','MU'],'AD':['AD','MU'],'EF':['EF','MU'],'IN':['IN','MU'],'MU':['ME','AD','EF','IN','MU']}
dirmap = {'1':['2','M'],'2':['1','M'],'M':['1','2','M']}

    


