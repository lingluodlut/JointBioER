# -*- coding: utf-8 -*-

import codecs as cs
import h5py
import numpy as np
from constants import B_label,S_label,i2l_dic
from constants import len_sentence,wv
def ypre2label(ypredict):
    '''
    将预测出的ypredict(50维list)取最大值索引，得到每个token的类别序号（0-36）
    输入： 神经网络预测出的结果，三维的np.array
    输出： Python-list  二维，值为每个token的类别序号
    
    '''
    predict_label = []
    y_predict = ypredict.tolist()
    for i in range(len(y_predict)):
        sentence_label = []
        for j in range(len(y_predict[i])):
            maxindex = y_predict[i][j].index(max(y_predict[i][j]))
            sentence_label.append(maxindex)
        predict_label.append(sentence_label)
    return predict_label
def label2answer(predict_label,tokens):
    '''
    根据每句话中每个token的标签来得到实体位置
    输入： Python-list  2D，值为每个token的类别序号
    输出： Python-list  3d，第三维为每个实体的起始位置列表[begin，end]（含头含尾）
          Python-list  3d，第三维为每个实体的对应的token序号[7，8，9]（含头含尾）
    '''
    pre_entity = []
    pre_entoken = []
    for i in range(len(tokens)):
        num_token = len(tokens[i])#每个句子的token数
        if num_token > len_sentence:#句子长度最大为50
            num_token = len_sentence
        sen_entity = []#每句话中的实体位置列表
        sen_etoken = []#每句话中的实体对应的token序号
        begin = -1
        end = -1
        j = 0#token的序号
        while(j < num_token):
            begin = end + 1#确定当前token的起止位置 含头含尾
            end = end + len(tokens[i][j])
            if predict_label[i][j] in S_label:#如果类别为S 则直接将该token的位置加入列表
                sen_entity.append([begin,end])
                sen_etoken.append([j])
                j = j + 1
            #并且的情况用elif!~!!!!!
            elif predict_label[i][j] in B_label:#如果类别为B 则...
                for endj in range(j,num_token):
                    if predict_label[i][endj] == 0:
                        break
                if endj == num_token-1 and predict_label[i][endj]!=0:
                    endj = num_token
                for tempj in range(j+1,endj):
                    end = end + len(tokens[i][tempj])
                sen_entity.append([begin,end])
                sen_etoken.append([k for k in range(j,endj)])
                j = endj
            else:
                j+=1
        pre_entity.append(sen_entity)
        pre_entoken.append(sen_etoken)
    return pre_entity,pre_entoken
        
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
                    if pre[0] == gold[0] and pre[1] == gold[1]:
                        truenum +=1
                        break
    try:
        precise = float(truenum) / float(prenum)
        recall = float(truenum) / float(goldnum)
        f = float(2 * precise * recall /( precise + recall)) 
    except:
        precise = recall = f = 0
    print('本轮实体的准确率是%f %f %f' %(precise,recall,f))
    return precise,recall,f
def GetModel(filepath,mask):#单词 to 向量的映射，dic形式
    model = []
    fp = cs.open(filepath,'r','utf-8')
    content = fp.readlines()[1:]
    fp.close()
    if mask:
        word = [0 for i in range(wv)]#增加一行对应index :0
        model.append(word)
    for each in content:
        word = []
        each = each.split(' ')
        for i in range(1,wv+1):
            word.append(float(each[i]))
        model.append(word)
    return np.array(model)
def GetToken2Index(filepath,mask):#从词向量模型中得到token to index的映射关系
    token2index = {}
    fp = cs.open(filepath,'r','utf-8')
    content = fp.readlines()[1:]
    fp.close()
    for i in range(len(content)):
        each = content[i].split(' ')
        if mask:
            token2index[each[0]] = i + 1 #####
        else:
            token2index[each[0]] = i
    return token2index
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
                fp.write(str(entity[0])+'\t'+str(entity[1])+'\n')
            else:
                fp.write(str(entity[0])+'\t'+str(entity[1])+'\t')
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

def GetCharMap(path):
    fp = cs.open(path,'r','utf-8')
    text = fp.read().split('\n')
    fp.close()
    dic = {}
    for c in text:
        char = c.split('\t')[0]
        index = c.split('\t')[1]
        dic[char] = int(index)
    return dic

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