# -*- coding: utf-8 -*-


import codecs as cs
import xml.etree.ElementTree as ET
import os
import nltk
from utils import SaveGoldEntity,sample_token4
from constants import label_mode
tvt = 'train'#train or test
if tvt == 'train':
    dir1 = '../../DDI/DDICorpus/Train/DrugBank'
    dir2 = '../../DDI/DDICorpus/Train/MedLine'
    outdir = '../data/DDI_%s_%s.txt'%(tvt,label_mode)
    goldedir = '../data/goldEntityAnswer_train.txt'
else:
    dir1 = '../../DDI/DDICorpus/Test/Test for DDI Extraction task/DrugBank'
    dir2 = '../../DDI/DDICorpus/Test/Test for DDI Extraction task/MedLine'
    outdir = '../data/DDI_%s_%s.txt'%(tvt,label_mode)
    goldedir = '../data/goldEntityAnswer.txt'
#生成所有文件的路径
files = []#训练集所有文件的路径
for each in os.listdir(dir1):
    files.append(os.path.join(dir1,each))
for each in os.listdir(dir2):
    files.append(os.path.join(dir2,each))
#得到一个句子索引to文件名的映射，便于查看特殊情况
senindex2file = []   
#开始从语料中下载文本 实体 及关系
SentencesList = []
def LoadArticle(filepath,num_e,fuhe_e):
    root = ET.parse(filepath)
    sentences = root.findall('sentence')
    senslist = []#整个文章 列表
    for sentence in sentences:
        sendic = {}#每一句话 字典
        text = sentence.attrib['text']
        sendic['text'] = text    
        
        entitys = sentence.findall('entity')#句子中所有实体节点
        entityslist = []
        for entity in entitys:#对于句子中的每一个实体
            num_e += 1
            try:#非复合实体
                index = entity.attrib['charOffset']
                left = int(index.split('-')[0])#左边界
                right = int(index.split('-')[1])#右边界
                label = entity.attrib['type']#实体类型
                entityslist.append([left,right,label])
            except:#对于复合实体 将第一个实体截断
                fuhe_e += 1
                index = entity.attrib['charOffset'].split(';')[0]
                left = int(index.split('-')[0])#左边界
                right = int(index.split('-')[1])#右边界
                label = entity.attrib['type']#实体类型
                entityslist.append([left,right,label])
        sendic['entity'] = entityslist
        senslist.append(sendic)
    return senslist,num_e,fuhe_e
num_e = fuhe_e = 0#实体总数 关系总数 复合实体数
for i in range(len(files)):
    eachfile = files[i]
    senslist,num_e,fuhe_e = LoadArticle(eachfile,num_e,fuhe_e)
    SentencesList.extend(senslist)
    for i in range(len(senslist)):
        senindex2file.append(eachfile)
    
def ChangeIndex(senslist):#将实体位置调整为忽略空格的位置
    for sentence in senslist:#每个句子：字典
        text = sentence['text']#取出文本
        while(' ' in text):#当文本中有空格时
            for i in range(len(text)):#i即为当前第一个空格空格出现的位置
                if text[i] == ' ':
                    break
            for entity in sentence['entity']:#句子中的每个实体 [left,right,label]
                if entity[0] > i:#如果该实体在空格后面，则将它的左右边界同时减一
                    entity[0] -= 1
                    entity[1] -= 1
                if entity[0] < i and entity[1] > i:#如果该实体包含空格，则将它的右边界减一
                    entity[1] -= 1
            text = text[0:i] + text[i+1:]

def GenarateBIO(senslist, schema,num_toolong):#将转换成BIO标注的二维list
    article = []#文章级别的列表
    for i in range(len(senslist)):
        text = senslist[i]['text']#取出文本
        text = sample_token4(text)
        text = nltk.tokenize.word_tokenize(text)
        entitys = senslist[i]['entity']
        word_label = []#句子级别的列表
        left = -1
        right = -1
        for token in text:
            left = right + 1#当前token的左右边界（含头含尾）
            right = right + len(token)
            ifBI = 0
            if schema == 'BIO':
                for entity in entitys:#一个实体的列表[left,right,label]
                    if left == entity[0]:
                        word_label.append([token,'B'])
                        ifBI = 1
                        break
                    elif left > entity[0] and right <= entity[1]:
                        word_label.append([token,'I'])
                        ifBI = 1
                        break
                    elif left == entity[0] and right > entity[1]:
                        num_toolong += 1
                if ifBI == 0:
                    word_label.append([token,'O'])
            if schema == 'BIOES':
                for entity in entitys:#一个实体的列表[left,right,label]
                    if left == entity[0] and right == entity[1] :
                        word_label.append([token,'S'])
                        ifBI = 1
                        break
                    elif left == entity[0] and right < entity[1]:
                        word_label.append([token,'B'])
                        ifBI = 1
                        break
                    elif left > entity[0] and right < entity[1]:
                        word_label.append([token,'I'])
                        ifBI = 1
                        break
                    elif left > entity[0] and right == entity[1]:
                        word_label.append([token,'E'])
                        ifBI = 1
                        break
                    elif left == entity[0] and right > entity[1]:
#                        print token
#                        print left,right
#                        print entitys
#                        print senslist[i]['text']
#                        print senindex2file[i] +'\n'
                        num_toolong += 1
                if ifBI == 0:
                    word_label.append([token,'O'])
        article.append(word_label)
        
    return article,num_toolong

def GetGoldAnwer(SentencesList):#保存标注的ytest的实体位置和关系位置/类别
    gold_entity = []#所有实体的位置
    for sentence in SentencesList:#每句话
        text = sentence['text']
        text = sample_token4(text)
        text = nltk.tokenize.word_tokenize(text)
        if len(text)==0 or len(text)==1:#如果这句话是空白或只有句号则跳过
            continue
        entity_s = []#一句话中的实体
        for entity in sentence['entity']:#将这句话中实体们的位置保存
            entity_s.append([entity[0],entity[1]])
        gold_entity.append(entity_s)
    return gold_entity

ChangeIndex(SentencesList)#将实体的位置改成无空格情况的位置
gold_entity = GetGoldAnwer(SentencesList)#保存标注的实体和关系信息
SaveGoldEntity(goldedir,gold_entity)
num_toolong = 0#用于统计切分token后 token长度大于实体长度的个数
tokenandlabel,num_toolong= GenarateBIO(SentencesList,label_mode,num_toolong)#生成token：label的BIOES标签
fp = cs.open(outdir,'w','utf-8')
num_pass = 0
for sentence in tokenandlabel:
    if len(sentence) == 0 or len(sentence) == 1:#空句子被忽略
        num_pass += 1
        continue
    for token in sentence:#
        fp.write(token[0]+'\t'+token[1]+'\n')
    fp.write('\n')
