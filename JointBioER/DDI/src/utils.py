# -*- coding: utf-8 -*-

import codecs as cs
import h5py
from constants import wv
import numpy as np
from constants import i2l_dic,rl2i,el2i,B_label,S_label,len_sentence,relationmap,dirmap
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
          Python-list  3d，第三维为每个关系的组成及类别[begin，end，begin，end，label]（含头含尾）
    '''
    pre_entity = []#列表 存放预测出的实体位置及类别
    pre_elabel = []#列表 存在每个实体的首个token的标签
    for i in range(len(tokens)):
        num_token = len(tokens[i])#每个句子的token数
        if num_token > len_sentence:
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
        exist_epair = []
        #past_token = []
        for j in range(len(pre_elabel[i])):#每个实体的首token
            infors = pre_elabel[i][j].split('-')
            if len(infors)>2:#如果是包含关系的token
                rlabel = infors[2]#取出当前token关系类型
                rdir = infors[3]#当前token的关系位置
                newlabel = []#可以与之配对的标签
                matchr = relationmap[rlabel]
                matchd = dirmap[rdir]
                for r in matchr:
                    for d in matchd:
                        newlabel.append(r+'-'+d)
                if rdir == '1' or rdir == 'M':
                    for k in range(j+1,len(pre_elabel[i])):
                        if len(pre_elabel[i][k].split('-')) == 4:
                            if pre_elabel[i][k][-4:] in newlabel and [j,k] not in exist_epair:
                                if rlabel == 'MU':
                                    if pre_elabel[i][k][-4:-2] !='MU':
                                        rlabel = pre_elabel[i][k][-4:-2]
                                    else:
                                        rlabel = 'EF'
                                sen_relation.append([pre_entity[i][j][0],pre_entity[i][j][1],pre_entity[i][k][0],pre_entity[i][k][1],rlabel])
                                exist_epair.append([j,k])
                                break
                if rdir == '2' or rdir == 'M':
                    for k in range(j-1,-1,-1):
                        if len(pre_elabel[i][k].split('-')) == 4:
                            if pre_elabel[i][k][-4:] in newlabel and [k,j] not in exist_epair:
                                if rlabel == 'MU':
                                    if pre_elabel[i][k][-4:-2] !='MU':
                                        rlabel = pre_elabel[i][k][-4:-2]
                                    else:
                                        rlabel = 'EF'
                                sen_relation.append([pre_entity[i][k][0],pre_entity[i][k][1],pre_entity[i][j][0],pre_entity[i][j][1],rlabel])
                                exist_epair.append([k,j])
                                break
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
    return precise,recall,f,f_label
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

def computeFr_2(gold_relation,pre_relation):
    '''
    先根据goldr判断哪些关系是重叠的，分别为重叠关系和非重叠关系计算prf
    输入： Python-list  3D，第三维为每个关系的组成及类别[begin，end，begin，end，label]（含头含尾）
    输出： float
    '''
    #判断重叠关系
    Relationlabel = []
    for relas in gold_relation:
        rl = []
        for i in range(len(relas)):
            if_repeat = 0
            for j in range(len(relas)):
                if j!= i:
                    if relas[i][0] == relas[j][0] and relas[i][1] == relas[j][1] or relas[i][0] == relas[j][2] and relas[i][1] == relas[j][3] or relas[i][2] == relas[j][0] and relas[i][3] == relas[j][1] or relas[i][2] == relas[j][2] and relas[i][3] == relas[j][3]:
                        if_repeat = 1
                        rl.append(1)
                        break
            if if_repeat == 0:
                rl.append(0)
        Relationlabel.append(rl)
    #计算prf
    truenum1 = 0#重叠的
    goldnum1 = 0
    truenum2 = 0#不重叠的
    goldnum2 = 0
    for i in range(len(gold_relation)):
        try:
            rnum = sum(Relationlabel[i])
        except:
            rnum = 0
        goldnum1 += rnum
        goldnum2 += len(gold_relation[i])-rnum
        if len(gold_relation[i]) == 0 or len(pre_relation[i]) == 0:        
            continue
        else:
            for j in range(len(gold_relation[i])):
                gold = gold_relation[i][j]
                for pre in pre_relation[i]:
                    #if pre[0] == gold[0] and pre[1] == gold[1] and pre[2] == gold[2] and pre[3] == gold[3] and pre[4] == gold[4]:
                    if cmp(pre, gold)==0 and Relationlabel[i][j]==1:   
                        truenum1 += 1
                        break
                    elif cmp(pre, gold)==0 and Relationlabel[i][j]==0:   
                        truenum2 += 1
                        break
    try:
        r1 = float(truenum1)/float(goldnum1)
        r2 = float(truenum2)/float(goldnum2)
    except:
        r1 = r2 = 0.0

    return r1,r2

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
        for i in range(0,len(locs),3):
            senentity.append([int(locs[i]),int(locs[i+1]),locs[i+2]])
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