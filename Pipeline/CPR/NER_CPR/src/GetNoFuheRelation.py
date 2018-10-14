 # -*- coding: utf-8 -*-

import codecs as cs

def GetNoRelation(pattern):
    if pattern == 'train':
        edir = '../../ChemProt_Corpus/chemprot_train_new/chemprot_training_entities.tsv'
        rdir = '../../ChemProt_Corpus/chemprot_train_new/chemprot_training_relations.tsv'
    elif pattern == 'vaild':
        edir = '../../ChemProt_Corpus/chemprot_development_new/chemprot_development_entities.tsv'
        rdir = '../../ChemProt_Corpus/chemprot_development_new/chemprot_development_relations.tsv'
    elif pattern == 'test':
        edir = '../../ChemProt_Corpus/chemprot_test_gs/chemprot_test_entities_gs.tsv'
        rdir = '../../ChemProt_Corpus/chemprot_test_gs/chemprot_test_relations_gs.tsv'
        
    #读入所有实体，dic, key为文章号，值为list，列表的元素为实体信息list [实体序号，实体类型，左边界，右边界，实体名]
    fp1 = cs.open(edir,'r','utf-8')
    entitys = fp1.read().split('\n')[:-1]
    fp1.close()
    edic = {}
    num_chem = 0
    num_gene = 0
    for line in entitys:
        #
        elements = line.split('\t')
        id = elements[0]
        if elements[2] == 'CHEMICAL':
            num_chem +=1
        elif elements[2] == 'GENE-Y' or elements[2] == 'GENE-N':
            num_gene +=1
        if id in edic:
            edic[id].append([elements[1],elements[2],int(elements[3]),int(elements[4]),elements[5]])
        else:
            edic[id] = []
            edic[id].append([elements[1],elements[2],int(elements[3]),int(elements[4]),elements[5]])
    print 'numchem= %s numgene= %s'%(num_chem,num_gene)
    #读入所有关系，dic, key为文章号，值为list，列表的元素为实体信息list [关系group，是否正例，细分类别，实体1序号，实体2序号]
    fp2 = cs.open(rdir,'r','utf-8')
    relations = fp2.read().split('\n')[:-1]
    fp2.close()
    rdic = {}
    num_r = 0
    for line in relations:
        #
        elements = line.split('\t')
        id = elements[0]
        if elements[2] == 'Y ':
            num_r +=1
            if id in rdic:
                rdic[id].append([elements[1],elements[2],elements[3],elements[4][5:],elements[5][5:]])
            else:
                rdic[id] = []
                rdic[id].append([elements[1],elements[2],elements[3],elements[4][5:],elements[5][5:]])
    print 'num_r =%s'%num_r
    
    #先统计每个文章里的嵌套实体数量
    fuhe_chemical = {}
    fuhe_gene = {}
    num1 = 0#左边界重叠
    num2 = 0#右边界重叠
    num3 = 0#完全嵌套
    num4 = 0#交叉重叠
    for id in edic:
        fuhe_chemical[id] = []
        fuhe_gene[id] = []
        for ec in edic[id]:
            if ec[1] == 'CHEMICAL':
                for eg in edic[id]:
                    if eg[1] == 'GENE-Y' or eg[1] == 'GENE-N':
                        if ec[2] == eg[2] and ec[3] <= eg[3]:
                            if ec[0] not in fuhe_chemical[id]:
                                fuhe_chemical[id].append(ec[0])
                            else:
                                print ec[4]+'|'+eg[4]
                            if eg[0] not in fuhe_gene[id]:
                                fuhe_gene[id].append(eg[0])
                            else:
                                print ec[4]+'|'+eg[4]
                            num1 += 1
                            #print ec[4]+'|'+eg[4]
                        elif ec[2] >= eg[2] and ec[3] == eg[3]:
                            fuhe_chemical[id].append(ec[0])
                            fuhe_gene[id].append(eg[0])
                            num2 += 1
                            #print ec[4]+'|'+eg[4]
                        elif ec[2] > eg[2] and ec[3] < eg[3]:
                            if ec[0] not in fuhe_chemical[id]:
                                fuhe_chemical[id].append(ec[0])
                            else:
                                print ec[4]+'|'+eg[4]
                            if eg[0] not in fuhe_gene[id]:
                                fuhe_gene[id].append(eg[0])
                            else:
                                print ec[4]+'|'+eg[4]
                            num3 += 1
                            #print ec[4]+'|'+eg[4]
                        elif ec[3]>=eg[2] and ec[3]<eg[3] or eg[3]>=ec[2] and eg[3]<ec[3]:
                            fuhe_chemical[id].append(ec[0])
                            fuhe_gene[id].append(eg[0])
                            num4 += 1
                            #print ec[4]+'|'+eg[4]
    print u'there is %s left overlap entity in %s corpus' %(num1,pattern)
    print 'there is %s right overlap entity in %s corpus' %(num2,pattern)
    print u'there is %s complete overlap entity in %s corpus' %(num3,pattern)
    print 'there is %s jiaocha overlap entity in %s corpus' %(num4,pattern)
    
    #统计复合的化学物有多少是参与关系的
    num1 = 0
    for id in fuhe_chemical:
        if id in rdic:
            for eid in fuhe_chemical[id]:
                for r in rdic[id]:
                    if r[3] == eid or r[4] == eid:
                        num1 += 1
    print num1
    
    #统计复合的基因有多少是参与关系的
    num2 = 0
    for id in fuhe_gene:
        if id in rdic:
            for eid in fuhe_gene[id]:
                for r in rdic[id]:
                    if r[3] == eid or r[4] == eid:
                        num2 +=1
    print num2
    
    #得到剔除过复合实体的关系
    new_rdic = {}
    
    for id in rdic:
        new_rdic[id] = []
        for relation in rdic[id]:#对于每个关系
            #只剔除嵌套化学物参与的关系
            #if relation[3] not in fuhe_chemical[id] and relation[4] not in fuhe_gene[id]:
            if relation[3] not in fuhe_chemical[id]:
                new_rdic[id].append(relation)
    return new_rdic

if __name__ == '__main__':
    x = GetNoRelation('test')

                    
            
        