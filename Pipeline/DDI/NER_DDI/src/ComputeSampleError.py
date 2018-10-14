# -*- coding: utf-8 -*-

import codecs as cs
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
        for i in range(0,len(locs),2):
            senentity.append([int(locs[i]),int(locs[i+1])])
        goldentity.append(senentity)
    return goldentity

def LoadGoldEntity2(path):#导入实体位置 三维python-list
    fp = cs.open(path,'r','utf-8')
    goldentity = []
    text = fp.read().split('\r\n')[0:-1]
    for sen in text:
        if len(sen)==0:
            goldentity.append([])
            continue
        locs = sen.split('\t')
        senentity = []
        for i in range(0,len(locs),2):
            senentity.append([int(locs[i]),int(locs[i+1])])
        goldentity.append(senentity)
    return goldentity
def LoadGoldRelation(path):#导入关系,三维python-list
    fp = cs.open(path,'r','utf-8')
    goldrelation = []
    text = fp.read().split('\r\n')[0:-1]
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
def GenerateS(entity):
    samples = []
    for each in entity:
        s = []
        if len(each)>=2:
            for i in range(0,len(each)-1):
                for j in range(i+1,len(each)):
                    s.append([each[i][0],each[i][1],each[j][0],each[j][1]])
        samples.append(s)
    return samples

def computeFe(gold_entity,pre_entity):
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
                    if cmp(pre,gold)==0:
                        truenum +=1
                        break
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall)) 
    except:
        precise = recall = f = 0
    print('预测出的、实际的、相同的实例数分别是%d %d %d' %(prenum,goldnum,truenum))
    #print('本轮实体的准确率是%f %f %f' %(precise,recall,f))
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
                     if cmp(pre, gold[:-1])==0:   
                        truenum +=1
                        break
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall)) 
    except:
        precise = recall = f = 0
    print('预测出实例数是%d' %(prenum))
    print('实际的正例是%d' %(goldnum))
    print('预测实例中是真正正例的有%d' %(truenum))
    #print('本轮实体的准确率是%f %f %f' %(precise,recall,f))
    return precise,recall,f
golde= LoadGoldEntity('../data/goldEntityAnswer.txt')
pree = LoadGoldEntity2('../data/predict/pre_e_crf+bils250+ls100.txt')
goldsample = GenerateS(golde)
presample = GenerateS(pree)
p,r,f = computeFe(goldsample,presample)

goldr = LoadGoldRelation('../data/goldRelationAnswer.txt')
pr,rr,fr = computeFr(goldr,presample)