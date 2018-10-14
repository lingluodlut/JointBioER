# -*- coding: utf-8 -*-

import numpy as np
#from gensim.models import word2vec
import codecs as cs
from utils import GetToken2Index,GetCharMap
from constants import l2i_dic,len_sentence,len_word
#将label变成对应的序号
#def label2index(labels,num):
#    for j in range(len(labels)):#每句话
#        for k in range(len(labels[j])): #每个标签
#            labels[j][k] = l2i_dic[labels[j][k]]
#            if labels[j][k]>4:
#                num+=1
#    return labels,num

def GenerateChars(word,CharMap):
    '''
    为一个字符串生成它的字符索引列表
    eg: HCC to [8,3,3]
    '''
    word_char = []
    for c in word:
        try:
            word_char.append(CharMap[c])
        except:
            word_char.append(0)
    for i in range(len_word - len(word)):
        word_char.append(0)
    return word_char

def GetXY(path,mask):
    f = cs.open(path,'r','utf-8')
    text = f.read()
    f.close()
    Token2index = GetToken2Index( u'../../token2vec/medline_chemdner_pubmed_biov5_drug.token4_d50',mask)
    CharMap = GetCharMap(u'../../token2vec/char_dic.txt')
    labels = []#整个训练集的标签，二维list
    tokens = []#整个训练集的token，二维list
    chars = []#三维list
    sentences = text.split(u'\n\n')[:-1]
    
    for i in range(len(sentences)):#读取预料中的每句话、每个词及标签
        sentence_token = []
        sentence_label = []
        sentence_chars = []
        sentence = sentences[i]
        sentence = sentence.split('\n')
        for k in range(len(sentence)):
            label1 = sentence[k].split('\t')[1]
            sentence_label.append(l2i_dic[label1])
            word1 = sentence[k].split('\t')[0]
            try:
                sentence_token.append(Token2index[word1.lower()])
            except:
                sentence_token.append(0)
            word_char = GenerateChars(word1,CharMap)
            sentence_chars.append(word_char)
        labels.append(sentence_label)
        tokens.append(sentence_token)
        chars.append(sentence_chars)
    #将每个句子变为等长        
    for i in range(len(labels)):
        if len(labels[i]) >= len_sentence:
            labels[i] = labels[i][0:len_sentence]
            tokens[i] = tokens[i][0:len_sentence]
            chars[i] = chars[i][0:len_sentence]
        else:
            senlen = len(labels[i])
            for j in range(len_sentence-senlen):
                labels[i].append(0)
                tokens[i].append(0)
                chars[i].append([0 for k in range(len_word)])

    return np.array(tokens),np.array(labels),np.array(chars)
