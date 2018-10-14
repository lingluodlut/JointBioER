# -*- coding: utf-8 -*-

import codecs as cs
import xml.etree.ElementTree as ET
import os
import nltk
import numpy as np
from utils import sample_token4,LoadGoldEntity,GetFiles
from keras.utils import np_utils
import pickle
SPARSE = 'Sparse_word'
PADDING= 'padding'
E1_B = 'entity1begin'
E1_E = 'entity1end'
E2_B = 'entity2begin'
E2_E = 'entity2end'
class RepresentationLayer(object):
    
    
    def __init__(self, wordvec_file=None, frequency=60000,
                 len_s=170, pattern ='test',use_pre_e = False):
        self.len_sentence = len_s
        self.frequency = frequency
        self.l2i_dic = {'effect':1,'advise':2,'mechanism':3,'int':4,'none':0}
        self.i2l_dic = {1:'effect',2:'advise',3:'mechanism',4:'int',0:'none'}
        self.vec_table, self.word_2_index, self.index_2_word, self.vec_size = self.load_wordvecs(wordvec_file)
        self.distance_2_index = self.load_dis_index_table()
        self.pattern = pattern
        #以下是训练集和测试集不同的
        self.SentencesList = self.LoadCorpus(self.pattern)
        self.gold_e,self.gold_r = self.GetGoldAnwer()
        self.tokens,self.eindex2tindex,self.rindex2eindex = self.GetMap()
        if use_pre_e:
            #self.eindex2tindex = []
            f = open('../data/pre_e2token.pkl','rb')
            self.eindex2tindex = pickle.load(f)
            f.close()
            self.gold_e = LoadGoldEntity('../data/pre_e.txt')
        self.samples,self.EntityPos,self.indexmap = self.GenerateSample()
        
    def load_dis_index_table(self):
        distance_2_index = {}
        index = 1
        for i in range(-self.len_sentence, self.len_sentence):
            distance_2_index[i] = index
            index += 1
        return distance_2_index
    
    def LoadCorpus(self,pattern):
        if pattern == 'train':
            files = GetFiles('../data/trainfiles.txt')
        else:
            files = GetFiles('../data/testfiles.txt')
        #得到一个句子索引to文件名的映射，便于查看特殊情况 
        #开始从语料中下载文本 实体 及关系
        SentencesList = []
        for i in range(len(files)):
            eachfile = files[i]
            senslist= self.LoadArticle(eachfile)
            SentencesList.extend(senslist)
        for sentence in SentencesList:#每个句子：字典
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
        return SentencesList
    def LoadArticle(self,filepath):
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
                try:#非复合实体
                    index = entity.attrib['charOffset']
                    left = int(index.split('-')[0])#左边界
                    right = int(index.split('-')[1])#右边界
                    label = entity.attrib['type']#实体类型
                    entityslist.append([left,right,label])
                except:#对于复合实体 将第一个实体截断
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
                    en1 = int(pair.attrib['e1'].split('.')[-1][1:])#实体1在句子中的索引
                    en2 = int(pair.attrib['e2'].split('.')[-1][1:])#实体2在句子中的索引
                    label = pair.attrib['type']
                    pairslist.append([en1,en2,label])
            sendic['pair'] = pairslist
            senslist.append(sendic)
        return senslist
    def GetGoldAnwer(self):#保存标注的ytest的实体位置和关系位置/类别
        gold_entity = []#所有实体的位置
        gold_relation = []#所有关系的实体位置及类别
        for sentence in self.SentencesList:#每句话
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
                relation_s.append([e1_dir[0],e1_dir[1],e2_dir[0],e2_dir[1],label])
            gold_entity.append(entity_s)
            gold_relation.append(relation_s)
        return gold_entity,gold_relation
    def GetMap(self):
        alltokens = []#所有句子的tokens 二维列表
        eindex2tindex = []#所有句子的实体index to tokenindex的映射字典
        rindex2eindex = []#所有句子的关系index to 实体index的映射字典
        for i in range(len(self.SentencesList)):
            #提取tokens
            text = self.SentencesList[i]['text']#取出文本
            text = sample_token4(text)
            text = nltk.tokenize.word_tokenize(text)
            if len(text)==0 or len(text)==1:#如果这句话是空白或只有句号则跳过
                continue
            tokens = []
            for token in text:
                tokens.append(token.lower())
            alltokens.append(tokens)
            #提取entity2token的映射
            entitys = self.SentencesList[i]['entity']
            e2t = {}#句子级别的实体到token的映射
            for j in range(len(entitys)):#每个实体至少对应一个token！！！不然无法生成位置索引
                e2t[j] = []
                entity = entitys[j]
                left = -1
                right = -1
                for k in range(len(text)):
                    token = text[k]
                    left = right + 1#当前token的左右边界（含头含尾）
                    right = right + len(token)
                    if left == entity[0] or left > entity[0] and right <= entity[1] or left < entity[0] and right > entity[0]:
                        e2t[j].append(k)
                if len(e2t[j]) == 0:
                    print text
                    print entitys[j]
            eindex2tindex.append(e2t)
            #提取关系到实体的映射
            relations = self.SentencesList[i]['pair']
            rindex2eindex.append(relations)
        return alltokens,eindex2tindex,rindex2eindex
    def GenerateSample(self):
        #生成样本的tokens及实体位置，同时保存每个样本的实体位置及样本与句子的对应
        Samples = []
        sampleindex2sentence = []#每个样本到句子索引的映射
        ENTITYPOS = []#每个样本的实体信息
        index = 0#当前句子的索引
        for map in self.eindex2tindex:
            num_e = len(map)
            for i in range(num_e-1):#第一个实体的索引
                for j in range(i+1,num_e):#第二个实体的索引
                    #生成POS1 POS2
                    sample = {}
                    sample['tokens'] = self.tokens[index]
                    sample['e1b'] = map[i][0]
                    sample['e1e'] = map[i][-1]#实体1的首个token的索引
                    sample['e2b'] = map[j][0]
                    sample['e2e'] = map[j][-1]#实体1的首个token的索引

                    if_rela = 0
                    for rela in self.rindex2eindex[index]:
                        if i == rela[0] and j==rela[1]:
                            sample['label'] = self.l2i_dic[rela[2]]
                            if_rela = 1
                            break
                    if if_rela == 0:
                        sample['label'] = 0
                    #生成最后用于结合结果的entitypos
                    entitypos = []
                    entitypos.extend(self.gold_e[index][i][0:2])
                    entitypos.extend(self.gold_e[index][j][0:2])
                    ENTITYPOS.append(entitypos)
                    #将该样本对应的句子序号append进映射中
                    sampleindex2sentence.append(index)
                    Samples.append(sample)
            index += 1
        return Samples,ENTITYPOS,sampleindex2sentence
    
    def represent_instances(self):
        label_list = []
        word_index_list = []
        distance_e1_index_list = []
        distance_e2_index_list = []
        for i in range(len(self.samples)):
            sample = self.samples[i]
            label, word_indexs, distance_e1_indexs, distance_e2_indexs = self.represent_instance(sample)
            #there is 2 bug , e1 is superposition of e2
            if len(distance_e1_indexs) >170:
                distance_e1_indexs = distance_e1_indexs[0:170]
                distance_e2_indexs = distance_e2_indexs[0:170]
            label_list.append([label])
            word_index_list.append(word_indexs)
            distance_e1_index_list.append(distance_e1_indexs)
            distance_e2_index_list.append(distance_e2_indexs)

        label_array = np.array(label_list)
        label_array = np_utils.to_categorical(label_array, len(self.l2i_dic))#将数值型标签转换为多分类数组
        label_array = np.reshape(label_array, (len(word_index_list),len(self.l2i_dic)))
        #label_array = label_array.reshape((len(label_array)/self.y_dim, self.y_dim))

        word_array = np.array(word_index_list)
        #word_array = word_array.reshape((word_array.shape[0]/self.max_sent_len, self.max_sent_len))

        dis_e1_array = np.array(distance_e1_index_list)
        #dis_e1_array = dis_e1_array.reshape((dis_e1_array.shape[0]/self.max_sent_len, self.max_sent_len))

        dis_e2_array = np.array(distance_e2_index_list)
        #dis_e2_array = dis_e2_array.reshape((dis_e2_array.shape[0]/self.max_sent_len, self.max_sent_len))

        return label_array, word_array, dis_e1_array, dis_e2_array
    
    def represent_instance(self,sample):

        tokens = sample['tokens']
        e1_b,e1_e,e2_b,e2_e = sample['e1b'],sample['e1e'],sample['e2b'],sample['e2e']
        label = sample['label']
        # the max length sentence won't contain the
        # two entities
        left_part = tokens[:e1_b]
        e1 = tokens[e1_b:e1_e+1]
        middle_part = tokens[e1_e+1:e2_b]
        e2 = tokens[e2_b:e2_e+1]
        right_part = tokens[e2_e+1:] + [PADDING for i in range(self.len_sentence - len(tokens))]

        distance_e1, distance_e2 = self.generate_distance_features(left_part, e1, middle_part, e2, right_part)
        distance_e1_index_list = self.replace_distances_with_indexs(distance_e1)
        distance_e2_index_list =  self.replace_distances_with_indexs(distance_e2)
 
        tokens.extend([PADDING for i in range(self.len_sentence - len(tokens))])
        word_index_list = self.replace_words_with_indexs(tokens)


        return label, word_index_list, distance_e1_index_list, distance_e2_index_list
    
    def replace_words_with_indexs(self, words):
        
        word_indexs = []
        for word in words:
            #如果是有的词 或 pad
            if self.word_2_index.has_key(word.lower()):
                word_indexs.append(self.word_2_index[word.lower()])
            #如果是没有的词
            else:
                word_indexs.append(self.word_2_index[SPARSE])
        
        return word_indexs

    '''
        replace distance list with corresponding indexs

    '''

    def replace_distances_with_indexs(self, distances):

        distance_indexs = []
        for distance in distances:
            if distance == 0:
                distance_indexs.append(0)
                continue
            if self.distance_2_index.has_key(distance):
                distance_indexs.append(self.distance_2_index[distance])
            else:
                print 'Impossible! This program will stop!'
                # sys.exit(0)

        return distance_indexs
    def generate_distance_features(self, left_part, e1, middle_part, e2, right_part):
        distance_e1 = []
        distance_e2 = []
        len_left = len(left_part)
        len_middle = len(middle_part)
        len_right = len(right_part)

        ### left part
        for i in range(len_left):
            distance_e1.append(i - len_left)
            distance_e2.append(i - len_left - 1 - len_middle)
        ### entry1 part
        for e in e1:
            # for position feature about entry1
            distance_e1.append(-self.len_sentence)
            # for position feature about entry2
            distance_e2.append(-len_middle)
        ### middle part
        for i in range(len_middle):
            # for position feature about entry1
            distance_e1.append(i + 1)
            # simplify of -(len_middle - i)
            # for position feature about entry2
            distance_e2.append(i - len_middle)
        ### entry2 part
        for e in e2:
            # for position feature about entry1
            distance_e1.append(len_middle)
            # for position feature about entry2
            distance_e2.append(-self.len_sentence)
        ### right part
        for i in range(len_right):
            if right_part[i] == PADDING:
                distance_e1.append(0)
                distance_e2.append(0)
            else:
                # for position feature about entry1
                # where the first 1 stand for the len of entry2
                distance_e1.append(len_middle + 1 + i + 1)
                # for position feature about entry2
                distance_e2.append(i + 1)
        return distance_e1, distance_e2
    
    def load_wordvecs(self, wordvec_file):
        
        file = open(wordvec_file)
        first_line = file.readline()
        word_count = int(first_line.split()[0])
        dimension = int(first_line.split()[1])
        vec_table = np.zeros((word_count, dimension))
        word_2_index = {PADDING:0}
        index_2_word = {0:PADDING}
        padding_vector = np.zeros(dimension)
        for col in xrange(dimension):
            vec_table[0][col] = padding_vector[col]

        row = 1
        for line in file:
            if row < self.frequency:
                line_split = line[:-1].split()
                word_2_index[line_split[0]] = row
                index_2_word[row] = line_split[0]
                for col in xrange(dimension):
                    vec_table[row][col] = float(line_split[col + 1])
                row += 1
            else:
                break
        #忽略掉频率小于frequecy的词 并将它们用sparse代替，sparse的词向量为余下词的词向量取平均
        word_2_index[SPARSE] = row
        index_2_word[row] = SPARSE
        sparse_vectors = np.zeros(dimension)
        for line in file:
            line_split = line[:-1].split()[1:]
            for i in xrange(dimension):
                sparse_vectors[i] += float(line_split[i])

        sparse_vectors /= (word_count - self.frequency)

        for col in xrange(dimension):
            vec_table[row][col] = sparse_vectors[col]


        
        file.close()

        return vec_table, word_2_index, index_2_word, dimension
    
#rep = RepresentationLayer(wordvec_file = '../../token2vec/medline_chemdner_pubmed_biov5_drug.token4_d50', pattern ='test',frequency=60000)
#label, word_index_list, distance_e1_index_list, distance_e2_index_list= rep.represent_instances()
#EntityPos,indexmap= rep.EntityPos,rep.indexmap