# -*- coding: utf-8 -*-

import codecs as cs
def loadtokens(path):#从BIOES模式的语料中下载每句话的tokens，二维list,schme为train/test
    '''
    下载测试集所有的句子，存入一个python-list
    '''
    f = cs.open(path,'r','utf-8')
    text = f.read()
    f.close()
    tokens = []#整个训练集的token，二维list
    sentences = text.split(u'\n\n')[:-1]
    for i in range(len(sentences)):#读取预料中的每句话、每个词及标签
        sentence_token = []
        sentence = sentences[i]
        sentence = sentence.split('\n')
        for k in range(len(sentence)):
            word1 = sentence[k].split('\t')[0]
            sentence_token.append(word1)
        tokens.append(sentence_token)
    return tokens

def LoadGoldRelation(path):#导入关系,三维python-list
    fp = cs.open(path,'r','utf-8')
    goldrelation = []
    text = fp.read().split('\n')[0:-1]
    for sen in text:
        if len(sen)==0:
            goldrelation.append([])
            continue
        locs = sen.split('\t')
        senrelation = []
        for i in range(0,len(locs),5):
            senrelation.append([int(locs[i]),int(locs[i+1]),int(locs[i+2]),int(locs[i+3]),locs[i+4]])
        goldrelation.append(senrelation)
    return goldrelation

def LoadGoldEntity(path):#导入实体位置 三维python-list
    fp = cs.open(path,'r','utf-8')
    goldentity = []
    text = fp.read().split('\n')[0:-1]
    for sen in text:
        if len(sen)==0:
            goldentity.append([])
            continue
        locs = sen.split('\t')
        senentity = []
        for i in range(0,len(locs),3):
            senentity.append([int(locs[i]),int(locs[i+1]),locs[i+2]])
        goldentity.append(senentity)
    return goldentity

testtokens = loadtokens(u'../data/DDI_test_BIOES_Y.txt')

goldr = LoadGoldRelation(u'../data/goldRelationAnswer.txt')
prer_pipline = LoadGoldRelation(u'../data/relation_pipline.txt')
prer_cl = LoadGoldRelation(u'../data/relation_joint.txt')

golde = LoadGoldEntity(u'../data/goldEntityAnswer.txt')
pree_pipline = LoadGoldEntity(u'../data/entity_pipline.txt')
pree_cl = LoadGoldEntity(u'../data/entity_joint.txt')

for i in range(len(testtokens)):
    nowtext = ' '.join(testtokens[i])
    if nowtext == 'Warfarin users who initiated citalopram , fluoxetine , paroxetine , amitriptyline , or mirtazapine had an increased risk of hospitalization for gastrointestinal bleeding .':
        print pree_pipline[i]
        print pree_cl[i]
        
        print prer_pipline[i]
        print prer_cl[i]
#    if len(goldr[i])==1:
#        if cmp(goldr[i],prer_cl[i])== 0 and cmp(goldr[i],prer_pipline[i])!= 0:
#            print ' '.join(testtokens[i])
#            print goldr[i]
#            
#            print prer_pipline[i]
#            print prer_cl[i]
#            
#            print pree_pipline[i]
#            print pree_cl[i]
            
    
