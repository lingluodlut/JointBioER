# -*- coding: utf-8 -*-

import codecs as cs
import numpy as np
import h5py
from constants import l2i_dic
def GetFiles(path):
    fp = cs.open(path,'r','utf-8')
    text = fp.read()
    fp.close()
    text = text.split('\n')[:-1]
    return text
def ypre2label(ypredict):
    '''
    将预测出的ypredict(50维list)取最大值索引，得到每个token的类别序号（0-36）
    输入： 神经网络预测出的结果，二维的np.array
    输出： Python-list  二维，值为每个token的类别序号
    
    '''
    predict_label = []
    y_predict = ypredict.tolist()
    for i in range(len(y_predict)):
        maxindex = y_predict[i].index(max(y_predict[i]))
        predict_label.append(maxindex)
    return predict_label
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
    labeltrue = [0,0,0,0,0]#四种类别分别的预测正确数量
    labelpre = [0,0,0,0,0]
    labelgold = [0,0,0,0,0]
    for i in range(len(pre_relation)):#每句话
        for pre in pre_relation[i]:
            label = pre[4]#得到该预测关系的类别
            labelpre[l2i_dic[label]] += 1#该类别的预测关系数+1
            for gold in gold_relation[i]:
                 if cmp(pre, gold)==0:   
                    labeltrue[l2i_dic[label]] +=1#该类别的预测正确关系数+1
                    break
    for i in range(len(gold_relation)):#每句话
        for gold in gold_relation[i]:
            label = gold[4]#得到该预测关系的类别
            labelgold[l2i_dic[label]] += 1#该类别的预测关系数+1
    f_label = []
    for i in range(5):
        try:
            labelp = float(labeltrue[i]) / float(labelpre[i])
            labelr = float(labeltrue[i]) / float(labelgold[i])
            labelf = float(2 * labelp * labelr /( labelp + labelr))
            f_label.append(labelf)
        except:
            f_label.append(0.0)
    return precise,recall,f,f_label

def save_model(address,model):#保存keras神经网络模型
    f = h5py.File(address,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        f.create_dataset('weight' + str(i),data = weight[i])
    f.close()
def load_model(address, model):#下载keras神经网络模型
    f = h5py.File(address, 'r')
    weight = []
    for i in range(len(f.keys())):
        weight.append(f['weight' + str(i)][:])
    model.set_weights(weight)
    
def SaveGoldEntity(path,goldentity):#保存实体位置 三维python-list
    fp = cs.open(path,'w','utf-8')
    for sen in goldentity:
        num = len(sen)
        if num == 0:
            fp.write('\n')
            continue
        for i in range(num):
            entity = sen[i]
            if i == (num -1):
                fp.write(str(entity[0])+'\t'+str(entity[1])+'\t'+entity[2]+'\n')
            else:
                fp.write(str(entity[0])+'\t'+str(entity[1])+'\t'+entity[2]+'\t')
    fp.close()
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

def SaveGoldRelation(path,goldrelation):#保存关系,三维python-list
    fp = cs.open(path,'w','utf-8')
    for sen in goldrelation:
        num = len(sen)
        if num == 0:
            fp.write('\n')
            continue
        for i in range(num):
            r = sen[i]
            if i == (num -1):
                fp.write(str(r[0])+'\t'+str(r[1])+'\t'+str(r[2])+'\t'+str(r[3])+'\t'+r[4]+'\n')
            else:
                fp.write(str(r[0])+'\t'+str(r[1])+'\t'+str(r[2])+'\t'+str(r[3])+'\t'+r[4]+'\t')
    fp.close()
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

def sample_token4(oristr):#对待匹配的实体进行字符串预处理
    '''
    对待匹配实体字符串进行预处理，用空格来替换一些字符
    @param:
        word:          python-string     待匹配实体
    @return:
        result:        python-string     处理后结果
    '''
    punctuation1=[u'.u',u'!',u'?',u'"']
    punctuation2=[u'•',u'●',u'-',u'—',u':',u';',u'%',u'+',u'=',u'~',u'#',u'$',u'&',u'*',u'/',u'@',\
                  u'"',u'?',u'!',u'[',u']',u'{',u'}',u'(',u')',u'<',u'>',\
                  u'→',u'↓',u'←',u'↔',u'↑',u'≃',u'⁺',u'···',u'®',u'≧',u'≈',u'⋅⋅⋅',u'·',u'…',u'...',u'‰',u'€',u'≥',u'∼',\
                  u'Δ',u'≤',u'δ',u'™',u'•',u'∗',u'⩾',u'Σ',u'Φ',u'∑',u'ƒ',u'≡',u'═',u'φ',u'Ψ',u'˙',u'Π',u'≫']
    punctuation3=[u',']#周围有空格才切
    punctuation4=[u"'"]#'s的情况，s'的情况，还有周围不是空格的情况

    line=oristr
    line=line.strip()
    for pun in punctuation2:
        line=line.replace(pun,u" "+pun+u" ")
#         print(line)
    if line!='':    
        if line[-1] in punctuation1:
            line=line[:-1]+u" "+line[-1]+u" "
    
    new_line=u""
    i=0
    
    while(len(line)!=0):
        if(i==len(line)-1): 
            new_line=new_line+line[0:i+1]
            break
        elif line[i] in punctuation3:
            if line[i-1]==u' ' or line[i+1]==u' ':
                new_line=new_line+line[:i]+u" "+line[i]+u" "
                line=line[i+1:]
                i=0
            else:
                i=i+1
        else:
            i=i+1
            
    line=new_line+u" "
    new_line=u""
    i=0
    while(len(line)!=0): 
        if(i==len(line)-1): 
            new_line=new_line+line[0:i+1]
            break
        elif line[i] in punctuation4:
            if line[i-1]==u' ' or line[i+1]==u' ':
                new_line=new_line+line[:i]+u" "+line[i]+u" "
                line=line[i+1:]
                i=0
            elif line[i+1]==u's' and line[i+2]==u' ':
                new_line=new_line+line[:i]+u" "+line[i]+u"s "
                line=line[i+2:]
                i=0
            else:
                i=i+1
        else:
            i=i+1
            
    words=new_line.split()
    new_line=u""
    for word in words:
        new_line+=word+u" "
            
    return new_line