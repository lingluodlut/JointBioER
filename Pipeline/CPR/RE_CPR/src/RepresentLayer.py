# -*- coding: utf-8 -*-

import codecs as cs
import nltk
import numpy as np
from utils import sample_token4,LoadGoldEntity
from keras.utils import np_utils
import pickle
SPARSE = 'Sparse_word'
PADDING= 'padding'
E1_B = 'entity1begin'
E1_E = 'entity1end'
E2_B = 'entity2begin'
E2_E = 'entity2end'
#一些用到的字典
small =  {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'}
eAbbr = {'CHEMICAL':'chem','GENE-Y':'geneY','GENE-N':'geneN'}
RelationAbbr = {'CPR:3':'C3','CPR:4':'C4','CPR:5':'C5','CPR:6':'C6','CPR:9':'C9'}

class RepresentationLayer(object):
    
    
    def __init__(self, wordvec_file=None, frequency=20000,
                 len_s=150, pattern ='test',use_pre_e = False):
        self.len_sentence = len_s
        self.frequency = frequency
        self.l2i_dic = {'none':0,'C3':1,'C4':2,'C5':3,'C6':4,'C9':5}
        self.i2l_dic = {0:'none',1:'C3',2:'C4',3:'C5',4:'C6',5:'C9'}
        self.vec_table, self.word_2_index, self.index_2_word, self.vec_size = self.load_wordvecs(wordvec_file)
        self.distance_2_index = self.load_dis_index_table()
        self.pattern = pattern
        #以下是训练集和测试集不同的
        self.SentencesList = self.LoadCorpus(self.pattern)
        self.gold_e,self.gold_r = self.GetGoldAnwer(self.SentencesList)
        self.tokens,self.eindex2tindex_C,self.eindex2tindex_G,self.rindex2eindex = self.GetMap()
        if use_pre_e:
            self.eindex2tindex_C = pickle.load(open('../data/e2t_c.pkl','rb'))
            self.eindex2tindex_G = pickle.load(open('../data/e2t_g.pkl','rb'))
            self.gold_e = LoadGoldEntity('../data/predictE_test.txt')
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
            abdir = '../../ChemProt_Corpus/chemprot_train_new/chemprot_training_abstracts.tsv'
            edir = '../../ChemProt_Corpus/chemprot_train_new/chemprot_training_entities.tsv'
            rdir = '../../ChemProt_Corpus/chemprot_train_new/chemprot_training_relations.tsv'
        elif pattern == 'vaild':
            abdir = '../../ChemProt_Corpus/chemprot_development_new/chemprot_development_abstracts.tsv'
            edir = '../../ChemProt_Corpus/chemprot_development_new/chemprot_development_entities.tsv'
            rdir = '../../ChemProt_Corpus/chemprot_development_new/chemprot_development_relations.tsv'
        else:
            abdir = '../../ChemProt_Corpus/chemprot_test_gs/chemprot_test_abstracts_gs.tsv'
            edir = '../../ChemProt_Corpus/chemprot_test_gs/chemprot_test_entities_gs.tsv'
            rdir = '../../ChemProt_Corpus/chemprot_test_gs/chemprot_test_relations_gs.tsv'
        fp = cs.open(abdir,'r','utf-8')
        text = fp.read().split('\n')[:-1]
        fp.close()
        #读入所有实体，dic, key为文章号，值为list，列表的元素为实体信息list [实体序号，实体类型，左边界，右边界，实体名]
        fp1 = cs.open(edir,'r','utf-8')
        entitys = fp1.read().split('\n')[:-1]
        fp1.close()
        edic = {}
        for line in entitys:
            #
            elements = line.split('\t')
            id = elements[0]
            if id in edic:
                edic[id].append([elements[1],elements[2],int(elements[3]),int(elements[4]),elements[5]])
            else:
                edic[id] = []
                edic[id].append([elements[1],elements[2],int(elements[3]),int(elements[4]),elements[5]])
        
        #读入所有关系，dic, key为文章号，值为list，列表的元素为实体信息list [关系group，是否正例，细分类别，实体1序号，实体2序号]
        fp2 = cs.open(rdir,'r','utf-8')
        relations = fp2.read().split('\n')[:-1]
        fp2.close()
        rdic = {}
        for line in relations:
            elements = line.split('\t')
            id = elements[0]
            if id in rdic:
                rdic[id].append([elements[1],elements[2],elements[3],elements[4][5:],elements[5][5:]])
            else:
                rdic[id] = []
                rdic[id].append([elements[1],elements[2],elements[3],elements[4][5:],elements[5][5:]])
        max_len = 0
        all_len = 0
        Senslist = []
        for line in text:
            article_id = line.split('\t')[0]
            abstract = line.split('\t')[1]
            text = line.split('\t')[2]
            #切分句子
            sentences = []
            for s in text.split('. '):
                if s[0] in small and len(sentences)>0:#如果该句的开头是小写字母，且前面有句子，则拼接到上一个句子中
                    sentences[-1] = sentences[-1] + s + '. '
                else:#否则该句单独作为一个句子
                    sentences.append(s + '. ')
            sentences[-1] = sentences[-1][:-2]#最后一句话无需加上". "
            sens = []
            sens.append(abstract+' ')
            sens.extend(sentences)
            #统计句子最大长度
            for each in sens:
                all_len += len(each)
                if len(each)>max_len:
                    #print each
                    max_len = len(each)
            #得到每个句子的起止位置
            sens_len = []
            begin = 0
            end = 0
            for i in range(len(sens)):
                end = begin + len(sens[i])
                sens_len.append([begin,end])
                begin = end
            #对于文章里每个句子的起始位置，得到属于该句子的所有实体和关系
            rnum = 0
            for i in range(len(sens_len)):
                sendic = {}#存放一个句子的文本、实体、关系
                sendic['text'] = sens[i]
                sendic['entity'] = []
                sendic['pair'] = []
                begin = sens_len[i][0]
                end = sens_len[i][1]
        
                sen_e_index = []#该句子的所有实体在文章中的序号
                #取出属于该句子的实体
                for e in edic[article_id]:
                    if e[2] >= begin and e[3] <= end:
                        sendic['entity'].append([e[2]-begin,e[3]-begin-1,e[1]])#实体位置改为含头含尾
                        sen_e_index.append(e[0])
                #得到该句子的关系
                if article_id in rdic:#如果该文章是包含关系的
                    for r in rdic[article_id]:
                        if r[1] == u'Y ':#Y后有空格
                            e1 = r[3]
                            e2 = r[4]
                            if e1 in sen_e_index and e2 in sen_e_index:
                                e1_index = sen_e_index.index(e1)
                                e2_index = sen_e_index.index(e2)
                                sendic['pair'].append([e1_index,e2_index,r[0]])
                                rnum +=1
                
                Senslist.append(sendic)
        #将实体位置替换为忽略空格的模式
        for sentence in Senslist:#每个句子：字典
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
        return Senslist

    def GetGoldAnwer(self,SentencesList):#保存标注的实体位置和关系位置/类别
        gold_entity = []#所有实体的位置
        gold_relation = []#所有关系的实体位置及类别
        for sentence in SentencesList:#每句话
            text = sentence['text']
            text = sample_token4(text)
            text = nltk.tokenize.word_tokenize(text)
    
            entity_s = []#一句话中的实体
            relation_s = []#一句话中的关系
            for entity in sentence['entity']:#将这句话中实体们的位置保存
                entity_s.append([entity[0],entity[1],eAbbr[entity[2]]])
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
        return gold_entity,gold_relation
    
    def GetMap(self):
        '''
            生成每个句子的：
            tokens: [[token11,token12...]]
            eindex2tindex = [{0:[2,3,4],1:[7]...},...]
            rindex2eindex = [[[0,1,'C1'],...],...]
        '''
        alltokens = []#所有句子的tokens 二维列表
        eindex2tindex_C = []#句子的实体index to tokenindex的映射字典 只包含药物实体
        eindex2tindex_G = []#句子的实体index to tokenindex的映射字典 只包含基因实体
        rindex2eindex = []#所有句子的关系index to 实体index的映射字典
        for i in range(len(self.SentencesList)):
            #提取每句话的tokens
            text = self.SentencesList[i]['text']#取出文本
            text = sample_token4(text)
            text = nltk.tokenize.word_tokenize(text)
            tokens = []
            for token in text:
                tokens.append(token.lower())
            alltokens.append(tokens)
            
            #提取entity2token的映射
            entitys = self.SentencesList[i]['entity']
            e2t_c = {}#句子级别的实体到token的映射
            e2t_g = {}#句子级别的实体到token的映射
            for j in range(len(entitys)):#每个实体至少对应一个token，不然无法生成位置索引！
                if entitys[j][2] == 'CHEMICAL':
                    e2t_c[j] = []
                    entity = entitys[j]
                    left = -1
                    right = -1
                    for k in range(len(text)):
                        token = text[k]
                        left = right + 1#当前token的左右边界（含头含尾）
                        right = right + len(token)
                        if left == entity[0] or left > entity[0] and left <= entity[1] or left < entity[0] and right >= entity[0]:
                            e2t_c[j].append(k)
                    if len(e2t_c[j]) == 0:
                        print (text)
                        print (u'this entity cannot find tokens %s'%entitys[j])
                else:
                    e2t_g[j] = []
                    entity = entitys[j]
                    left = -1
                    right = -1
                    for k in range(len(text)):
                        token = text[k]
                        left = right + 1#当前token的左右边界（含头含尾）
                        right = right + len(token)
                        if left == entity[0] or left > entity[0] and left <= entity[1] or left < entity[0] and right >= entity[0]:
                            e2t_g[j].append(k)
                    if len(e2t_g[j]) == 0:
                        print (text)
                        print (u'this entity cannot find tokens %s'%entitys[j])
            eindex2tindex_C.append(e2t_c)
            eindex2tindex_G.append(e2t_g)
            #提取关系到实体的映射
            relations = self.SentencesList[i]['pair']
            rindex2eindex.append(relations)
        return alltokens,eindex2tindex_C,eindex2tindex_G,rindex2eindex

    def GenerateSample(self):
        Samples = []
        sampleindex2sentence = []#每个样本到句子索引的映射
        ENTITYPOS = []#每个样本的实体信息

        for k in range(len(self.eindex2tindex_C)):#遍历每句话中的实体
            num_c = len(self.eindex2tindex_C[k])
            num_g = len(self.eindex2tindex_G[k])
            if num_c == 0 or num_g == 0 or len(self.tokens[k]) > self.len_sentence:
                continue
            for chem_index in self.eindex2tindex_C[k]:#药物实体的索引
                for gene_index in self.eindex2tindex_G[k]:#基因实体的索引
                    #根据这两个实体的对应的token索引生成样本，在该环节剔除存在嵌套的实体对
                    chem = self.eindex2tindex_C[k][chem_index]
                    gene = self.eindex2tindex_G[k][gene_index]
                    jiaoji = [val for val in chem if val in gene]#如果两个实体存在重合则跳过
                    if len(jiaoji) > 0:
                        continue
                    sample = {}
                    sample['tokens'] = self.tokens[k]
                    sample['e1b'] = chem[0]
                    sample['e1e'] = chem[-1]#实体1的首个token的索引
                    sample['e2b'] = gene[0]
                    sample['e2e'] = gene[-1]

                    if_rela = 0
                    for rela in self.rindex2eindex[k]:
                        if chem_index == rela[0] and gene_index ==rela[1]:
                            sample['label'] = self.l2i_dic[RelationAbbr[rela[2]]]
                            if_rela = 1
                            break
                    if if_rela == 0:
                        sample['label'] = 0
                    #生成最后用于结合结果的entitypos
                    entitypos = []
                    entitypos.extend(self.gold_e[k][chem_index][0:2])
                    entitypos.extend(self.gold_e[k][gene_index][0:2])
                    ENTITYPOS.append(entitypos)
                    #将该样本对应的句子序号append进映射中
                    sampleindex2sentence.append(k)
                    Samples.append(sample)
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
            if len(distance_e1_indexs) >self.len_sentence:
                distance_e1_indexs = distance_e1_indexs[0:self.len_sentence]
                distance_e2_indexs = distance_e2_indexs[0:self.len_sentence]
                word_indexs = word_indexs[0:self.len_sentence]
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

        tokens = ''
        tokens = sample['tokens']
        if sample['e1b'] < sample['e2b']:
            e1_b,e1_e,e2_b,e2_e = sample['e1b'],sample['e1e'],sample['e2b'],sample['e2e']
        else:
            e1_b,e1_e,e2_b,e2_e = sample['e2b'],sample['e2e'],sample['e1b'],sample['e1e']
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
            if word.lower() in self.word_2_index:
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
            if distance == 0:#如果原本是pad 则继续使用0
                distance_indexs.append(0)
                continue
            if distance in self.distance_2_index:#否则使用dis字典替换，其中实体相对于自己的位置由-150映射为0
                distance_indexs.append(self.distance_2_index[distance])
            else:
                print (distance)
                print ('Impossible! This program will stop!')
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
            distance_e1.append(-self.len_sentence)
            distance_e2.append(-len_middle)
        ### middle part
        for i in range(len_middle):
            distance_e1.append(i + 1)
            distance_e2.append(i - len_middle)
        ### entry2 part
        for e in e2:
            distance_e1.append(len_middle)
            distance_e2.append(-self.len_sentence)
        ### right part
        for i in range(len_right):
            if right_part[i] == PADDING:
                distance_e1.append(0)
                distance_e2.append(0)
            else:
                distance_e1.append(len_middle + 1 + i + 1)
                distance_e2.append(i + 1)
        return distance_e1, distance_e2
    
    def load_wordvecs(self, wordvec_file):
        
        file = cs.open(wordvec_file,'r','utf-8')
        first_line = file.readline()
        word_count = int(first_line.split()[0])
        dimension = int(first_line.split()[1])
        vec_table = np.zeros((word_count, dimension))
        word_2_index = {PADDING:0}
        index_2_word = {0:PADDING}
        padding_vector = np.zeros(dimension)
        for col in range(dimension):
            vec_table[0][col] = padding_vector[col]

        row = 1
        for line in file:
            if row < self.frequency:
                line_split = line[:-1].split()
                word_2_index[line_split[0]] = row
                index_2_word[row] = line_split[0]
                for col in range(dimension):
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
            for i in range(dimension):
                sparse_vectors[i] += float(line_split[i])

        sparse_vectors /= (word_count - self.frequency)

        for col in range(dimension):
            vec_table[row][col] = sparse_vectors[col]


        
        file.close()

        return vec_table, word_2_index, index_2_word, dimension
    
rep = RepresentationLayer(wordvec_file = '../../token2vec/chemdner_pubmed_biov5_drug.token4_d100_CPR', pattern ='test',frequency=20000)

dis2index,gold_e,gold_r,tokens,e2t_c,e2t_g,r2e = rep.distance_2_index,rep.gold_e,rep.gold_r,rep.tokens,rep.eindex2tindex_C,rep.eindex2tindex_G,rep.rindex2eindex

label, word_index_list, distance_e1_index_list, distance_e2_index_list= rep.represent_instances()

EntityPos,indexmap= rep.EntityPos,rep.indexmap