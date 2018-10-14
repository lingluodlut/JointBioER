# -*- coding: utf-8 -*-

import os
import codecs as cs
pattern = 'train'
if pattern == 'train':
    dir1 = '../../DDI/DDICorpus/Train/DrugBank'
    dir2 = '../../DDI/DDICorpus/Train/MedLine'
else:
    dir1 = '../../DDI/DDICorpus/Test/Test for DDI Extraction task/DrugBank'
    dir2 = '../../DDI/DDICorpus/Test/Test for DDI Extraction task/MedLine'
#生成所有文件的路径
files = []#训练集所有文件的路径
for each in os.listdir(dir1):
    files.append(dir1 +'/'+each)
for each in os.listdir(dir2):
    files.append(dir2 +'/'+each)
fp = cs.open('../data/trainfiles.txt','w','utf-8')
for each in files:
    fp.write(each+'\n')
fp.close()