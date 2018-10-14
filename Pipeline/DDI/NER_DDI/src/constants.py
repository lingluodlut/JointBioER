# -*- coding: utf-8 -*-

#一些超参数

bils = 250
ls = 100
wv = 50
label_mode = 'BIOES'
len_sentence = 117
len_word = 25
l2i_dic = i2l_dic = B_label = I_label = E_label = S_label = O_label = {}

if label_mode == 'BIO':
    l2i_dic = {'B': 1, 'I': 2, 'O': 0}
    i2l_dic = {0: 'O', 1: 'B', 2: 'I'}
    B_label = {1}
    I_label = {2}
    O_label = {0}
    S_label = {}
else:
    l2i_dic = { 'B': 1, 'E': 3, 'I': 2, 'O': 0, 'S': 4}
    i2l_dic = {0: 'O', 1: 'B', 2: 'I', 3: 'E', 4: 'S',}
    B_label = {1}
    I_label = {2}
    E_label = {3}
    S_label = {4}
    O_label = {0}

num_class = len(l2i_dic)