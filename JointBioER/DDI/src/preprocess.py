# -*- coding: utf-8 -*-


import codecs as cs
import xml.etree.ElementTree as ET
import os
import nltk
from utils import SaveGoldEntity,SaveGoldRelation,sample_token4
from constants import label_mode,if_repeat
pattern = 'test'#train or test
RelationAbbr = {'effect':'EF','advise':'AD','mechanism':'ME','int':'IN'}#四种关系类型到缩写的映射
if pattern == 'train':
    dir1 = '../../DDI/DDICorpus/Train/DrugBank'
    dir2 = '../../DDI/DDICorpus/Train/MedLine'
    outdir = '../data/DDI_%s_%s_%s.txt'%(pattern,label_mode,if_repeat)
    goldedir = '../data/goldEntityAnswer_train.txt'
    goldrdir = '../data/goldRelationAnswer_train.txt'
else:
    dir1 = '../../DDI/DDICorpus/Test/Test for DDI Extraction task/DrugBank'
    dir2 = '../../DDI/DDICorpus/Test/Test for DDI Extraction task/MedLine'
    outdir = '../data/DDI_%s_%s_%s.txt'%(pattern,label_mode,if_repeat)
    goldedir = '../data/goldEntityAnswer.txt'
    goldrdir = '../data/goldRelationAnswer.txt'
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
def LoadArticle(filepath,num_e,num_r,fuhe_e):
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
        
        pairs = sentence.findall('pair')#所有关系节点
        pairslist = []
        for pair in pairs:
            if pair.attrib['ddi'] == 'true':
                num_r += 1
                en1 = int(pair.attrib['e1'].split('.')[-1][1:])#实体1在句子中的索引
                en2 = int(pair.attrib['e2'].split('.')[-1][1:])#实体2在句子中的索引
                label = pair.attrib['type']
                pairslist.append([en1,en2,label])
        sendic['pair'] = pairslist
        senslist.append(sendic)
    return senslist,num_e,num_r,fuhe_e

num_e = num_r = fuhe_e = 0#实体总数 关系总数 复合实体数
for i in range(len(files)):
    eachfile = files[i]
    senslist,num_e,num_r,fuhe_e = LoadArticle(eachfile,num_e,num_r,fuhe_e)
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
                        word_label.append([token,'B-'+entity[2]])
                        ifBI = 1
                        break
                    elif left > entity[0] and right <= entity[1]:
                        word_label.append([token,'I-'+entity[2]])
                        ifBI = 1
                        break
                    elif left == entity[0] and right > entity[1]:
                        num_toolong += 1
                if ifBI == 0:
                    word_label.append([token,'O'])
            if schema == 'BIOES':
                for entity in entitys:#一个实体的列表[left,right,label]
                    if left == entity[0] and right == entity[1] :
                        word_label.append([token,'S-'+entity[2]])
                        ifBI = 1
                        break
                    elif left == entity[0] and right < entity[1]:
                        word_label.append([token,'B-'+entity[2]])
                        ifBI = 1
                        break
                    elif left > entity[0] and right < entity[1]:
                        word_label.append([token,'I-'+entity[2]])
                        ifBI = 1
                        break
                    elif left > entity[0] and right == entity[1]:
                        word_label.append([token,'E-'+entity[2]])
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


def AddRelation(senslist,token2BIOES):#实体可能有重复的关系
    for i in range(len(senslist)):#对于每句话
        pairs = senslist[i]['pair']#取出关系
        entitys = senslist[i]['entity']
        if len(pairs) == 0:
            continue
        for pair in pairs:#对于每个关系
            e1_index = pair[0]#实体1在句子中的索引
            e2_index = pair[1]#实体2在句子中的索引
            label = pair[2]#关系
            e1_dir = entitys[e1_index][:2]#实体1的左右边界（忽略空格）
            e2_dir = entitys[e2_index][:2]#实体2的左右边界（忽略空格）
            labelAbbr = RelationAbbr[label]#该关系的缩写
            left = -1
            right = -1
            
            for token in token2BIOES[i]:#遍历句子中的每个token
                left = right + 1#当前token的左右边界（含头含尾）
                right = right + len(token[0])
                if len(token) == 2:#如果之前没包含类别的标签
                    if left >= e1_dir[0] and right <= e1_dir[1]:
                        token.append('%s-%s-1'%(token[1],labelAbbr))
                    if left >= e2_dir[0] and right <= e2_dir[1]:
                        token.append('%s-%s-2'%(token[1],labelAbbr))
                elif len(token) > 2:
                    before_token_label = token[-1].split('-')[2]
                    before_token_dir = token[-1].split('-')[3]
                    if left >= e1_dir[0] and right <= e1_dir[1]:
                        if labelAbbr != before_token_label:
                            now_token_label = 'MU'
                        else:
                            now_token_label = labelAbbr
                        if before_token_dir != '1':
                            now_token_dir = 'M'
                        else:
                            now_token_dir = '1'
                        token.append('%s-%s-%s'%(token[1],now_token_label,now_token_dir))
                    
                    if left >= e2_dir[0] and right <= e2_dir[1]:   
                        if labelAbbr != before_token_label:
                            now_token_label = 'MU'
                        else:
                            now_token_label = labelAbbr
                        if before_token_dir != '2':
                            now_token_dir = 'M'
                        else:
                            now_token_dir = '2'
                        token.append('%s-%s-%s'%(token[1],now_token_label,now_token_dir))
    return token2BIOES

def AddRelation2(senslist,token2BIOES):#实体无重复的关系
    for i in range(len(senslist)):#对于每句话
        pairs = senslist[i]['pair']#取出关系
        entitys = senslist[i]['entity']
        if len(pairs) == 0:#如果这句话没有关系，则跳过
            continue
        relatedEn = []#这句话中存在关系的实体的索引
        for pair in pairs:#对于每个关系
            e1_index = pair[0]#实体1在句子中的索引
            e2_index = pair[1]#实体2在句子中的索引
            if e1_index in relatedEn or e2_index in relatedEn:#判断这两个实体是否已存在关系
                continue
            else:#若不存在，则将他们标记为已存在关系，进行进一步操作
                relatedEn.append(e1_index)
                relatedEn.append(e2_index)
            label = pair[2]#关系
            e1_dir = entitys[e1_index][:2]#实体1的左右边界（忽略空格）
            e2_dir = entitys[e2_index][:2]#实体2的左右边界（忽略空格）
            labelAbbr = RelationAbbr[label]#该关系的缩写
            left = -1
            right = -1
            for token in token2BIOES[i]:#遍历句子中的每个token
                left = right + 1#当前token的左右边界（含头含尾）
                right = right + len(token[0])
                if left >= e1_dir[0] and right <= e1_dir[1]:
                    token.append('%s-%s-1'%(token[1],labelAbbr))
                if left >= e2_dir[0] and right <= e2_dir[1]:
                    token.append('%s-%s-2'%(token[1],labelAbbr))
#        if len(token2BIOES[i])==0 or len(token2BIOES[i])==1:
#            print files[i]
    return token2BIOES


def GetGoldAnwer(SentencesList):#保存标注的ytest的实体位置和关系位置/类别
    gold_entity = []#所有实体的位置
    gold_relation = []#所有关系的实体位置及类别
    for sentence in SentencesList:#每句话
        text = sentence['text']
        text = sample_token4(text)
        text = nltk.tokenize.word_tokenize(text)
        if len(text)==0 or len(text)==1:#如果这句话是空白或只有句号则跳过
            continue
        entity_s = []#一句话中的实体
        relation_s = []#一句话中的关系
        for entity in sentence['entity']:#将这句话中实体们的位置保存
            entity_s.append(entity)
        for pair in sentence['pair']:
            e1_index = pair[0]#实体1在句子中的索引
            e2_index = pair[1]#实体2在句子中的索引
            label = pair[2]#关系
            e1_dir = sentence['entity'][e1_index][:2]#实体1的左右边界（忽略空格）
            e2_dir = sentence['entity'][e2_index][:2]#实体2的左右边界（忽略空格）
            labelAbbr = RelationAbbr[label]#该关系的缩写
            relation_s.append([e1_dir[0],e1_dir[1],e2_dir[0],e2_dir[1],labelAbbr])
        gold_entity.append(entity_s)
        gold_relation.append(relation_s)
    print len(gold_entity)
    return gold_entity,gold_relation

ChangeIndex(SentencesList)#将实体的位置改成无空格情况的位置
gold_entity,gold_relation = GetGoldAnwer(SentencesList)#保存标注的实体和关系信息
SaveGoldEntity(goldedir,gold_entity)
SaveGoldRelation(goldrdir,gold_relation)
num_toolong = 0#用于统计切分token后 token长度大于实体长度的个数
tokenandlabel,num_toolong= GenarateBIO(SentencesList,label_mode,num_toolong)#生成token：label的BIOES标签
fp = cs.open(outdir,'w','utf-8')
num_pass = 0
if if_repeat == 'N':#每个token不可以存在多重关系
    tokenandlabel = AddRelation2(SentencesList,tokenandlabel)
    for sentence in tokenandlabel:
        if len(sentence) == 0 or len(sentence) == 1:#空句子被忽略
            num_pass += 1
            continue
        for token in sentence:#
            if len(token)==2:
                fp.write(token[0]+'\t'+token[1]+'\n')
            else:
                fp.write(token[0]+'\t'+token[2]+'\n')
        fp.write('\n')
else:#每个token可以存在多重关系,它的标签会被最后一个关系覆盖
    tokenandlabel = AddRelation(SentencesList,tokenandlabel)
    for sentence in tokenandlabel:
        if len(sentence) == 0 or len(sentence) == 1:#空句子被忽略
            
            continue
        num_pass += 1
        #print num_pass
        for token in sentence:
            fp.write(token[0])
            if len(token) == 2:
                fp.write('\t'+token[1])
            else:
                fp.write('\t'+token[-1])
            fp.write('\n')
        fp.write('\n')
fp.close()

#查看由于标签产生的关系错误的位置
from GenarateXY import GetXY
from utils import label2answer,loadtokens,computeFr
testtokens = loadtokens(u'../data/DDI_test_%s_Y.txt'%(label_mode))
xtest,y_test,xctest =  GetXY(u'../data/DDI_test_%s_Y.txt'%(label_mode),mask=True)
predict_e,predict_r =label2answer(y_test,testtokens)

for i in range(len(gold_relation)):
    if cmp(gold_relation[i], predict_r[i])!= 0:
        print senindex2file[i]
        print ' '.join(testtokens[i])
        print gold_relation[i]
        print predict_r[i]
        print '\n'
pr,rr,fr,frlabel = computeFr(gold_relation,predict_r)
print u'由goldy得到的关系的PRF为%f %f %f'%(pr,rr,fr)

##计算重叠关系的数量
#num = 0
#for relas in gold_relation:
#    for i in range(len(relas)):
#        for j in range(len(relas)):
#            if j!= i:
#                if relas[i][0] == relas[j][0] and relas[i][1] == relas[j][1] or relas[i][0] == relas[j][2] and relas[i][1] == relas[j][3] or relas[i][2] == relas[j][0] and relas[i][3] == relas[j][1] or relas[i][2] == relas[j][2] and relas[i][3] == relas[j][3]:
#                    num+=1
#                    break
#print num