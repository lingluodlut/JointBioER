# -*- coding: utf-8 -*-

import numpy as np
import codecs as cs
from utils import GetToken2Index,GetCharMap
from constants import l2i_dic,len_sentence,len_word,num_class
from keras.utils import np_utils

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

def GetXY(path,mask=0,if_lemma=0,if_pos=0,if_chunk=0,if_ner=0):
    Data = {}
    #导入各特征的映射
    Token2index = GetToken2Index(u'../../token2vec/chemdner_pubmed_biov5_drug.token4_d100',mask)
    #Token2index = GetToken2Index(u'../../token2vec/medline_chemdner_pubmed_biov5_drug.token4_d50',mask)#从词向量模型里得到token到index的映射
    CharMap = GetCharMap(u'../data/dic/char_dic.txt')
    posMap = GetCharMap(u'../data/dic/pos_dic.txt')
    chunkMap = GetCharMap(u'../data/dic/chunk_dic.txt')
    nerMap = GetCharMap(u'../data/dic/ner_dic.txt')
    f = cs.open(path,'r','utf-8')
    text = f.read()
    f.close()
    
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
            label1 = sentence[k].split('\t')[-1]
            sentence_label.append(l2i_dic[label1])
            word1 = sentence[k].split('\t')[0]
            try:
                sentence_token.append(Token2index[word1.lower()])
            except:
                sentence_token.append(0)
            word_char = GenerateChars(word1,CharMap)
            sentence_chars.append(word_char)
        if len(sentence_label) >= len_sentence:
            sentence_token = sentence_token[:len_sentence]
            sentence_label = sentence_label[:len_sentence]
            sentence_chars = sentence_chars[:len_sentence]
        else:
            for x in range(len_sentence - len(sentence_label)):
                sentence_chars.append([0 for y in range(len_word)])
            sentence_token = sentence_token + [0 for x in range(len_sentence - len(sentence_label))]
            sentence_label = sentence_label + [0 for x in range(len_sentence - len(sentence_label))]
            
        labels.append(sentence_label)
        tokens.append(sentence_token)
        chars.append(sentence_chars)
    tokens,labels,chars = np.array(tokens), np.array(labels), np.array(chars)
    Data['word'] = tokens
    Data['char'] = chars
    labels = np_utils.to_categorical(labels, num_class)
    labels = np.reshape(labels, (len(tokens),len_sentence,num_class))
    Data['label'] = labels
    
    
    if if_lemma:
        lemmas = []
        for i in range(len(sentences)):#读取预料中的每句话、每个词及标签
            lemma = []
            sentence = sentences[i].split('\n')
            for k in range(len(sentence)):
                word1 = sentence[k].split('\t')[1]
                try:
                    lemma.append(Token2index[word1.lower()])
                except:
                    lemma.append(0)
            if len(lemma) >= len_sentence:
                lemma = lemma[:len_sentence]
            else:
                lemma = lemma + [0 for x in range(len_sentence - len(lemma))]
            lemmas.append(lemma)
        Data['lemma'] = np.array(lemmas)
        
    if if_pos:
        poss = []
        for i in range(len(sentences)):#读取预料中的每句话、每个词及标签
            pos = []
            sentence = sentences[i].split('\n')
            for k in range(len(sentence)):
                p = sentence[k].split('\t')[2]
                try:
                    pos.append(posMap[p])
                except:
                    pos.append(posMap['PADDING'])
            if len(pos) >= len_sentence:
                pos = pos[:len_sentence]
            else:
                pos = pos + [0 for x in range(len_sentence - len(pos))]
            poss.append(pos)
        Data['pos'] = np.array(poss)
        
    if if_chunk:
        chunks = []
        for i in range(len(sentences)):#读取预料中的每句话、每个词及标签
            chunk = []
            sentence = sentences[i].split('\n')
            for k in range(len(sentence)):
                p = sentence[k].split('\t')[2]
                try:
                    chunk.append(chunkMap[p])
                except:
                    chunk.append(chunkMap['PADDING'])
            if len(chunk) >= len_sentence:
                chunk = chunk[:len_sentence]
            else:
                chunk = chunk + [0 for x in range(len_sentence - len(chunk))]
            chunks.append(chunk)
        Data['chunk'] = np.array(chunks)
        
    if if_ner:
        ners = []
        for i in range(len(sentences)):#读取预料中的每句话、每个词及标签
            ner = []
            sentence = sentences[i].split('\n')
            for k in range(len(sentence)):
                p = sentence[k].split('\t')[2]
                try:
                    ner.append(nerMap[p])
                except:
                    ner.append(nerMap['PADDING'])
            if len(ner) >= len_sentence:
                ner = ner[:len_sentence]
            else:
                ner = ner + [0 for x in range(len_sentence - len(ner))]
            ners.append(ner)
        Data['ner'] = np.array(ners)

    return Data