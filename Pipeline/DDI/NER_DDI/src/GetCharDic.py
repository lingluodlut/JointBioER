# -*- coding: utf-8 -*-

import codecs as cs

fp = cs.open('../data/DDI_train_BIOES.txt','r','utf-8')
text = fp.read().split('\n\n')[:-1]
fp.close()
maxlen = 0
dic = {}
index = 1
for sen in text:
    words = sen.split('\n')
    for w in words:
        token = w.split('\t')[0]
        if len(token) > maxlen:
            maxlen = len(token)
        for c in token:
            if c not in dic:
                dic[c] = index
                index +=1
print maxlen
fp = cs.open('../data/char_dic.txt','w','utf-8')
for c in dic:
    fp.write('%s\t%s\n'%(c,dic[c]))
fp.close()