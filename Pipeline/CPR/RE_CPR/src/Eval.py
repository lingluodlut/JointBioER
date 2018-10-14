
import numpy as np



positive = 1
negtive = 0

def change_real2class(real_res_matrix):
    res_matrix = np.zeros_like(real_res_matrix, dtype=int)
    max_indexs = np.argmax(real_res_matrix, 1)
    for i in xrange(len(max_indexs)):
        res_matrix[i][max_indexs[i]] = 1
        
    return res_matrix



def eval_mulclass(ans_matrix, res_matrix, real=True):
    confuse_matrixs = np.zeros((ans_matrix.shape[1], 4))
    
    if real == True:
        res_matrix = change_real2class(res_matrix)
    
    class_indexs = np.argmax(ans_matrix, 1)
    for class_index in range(confuse_matrixs.shape[0]):
        for i in range(ans_matrix.shape[0]):
            if class_index == class_indexs[i]: #positive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][0] += 1 #TP
                else:
                    confuse_matrixs[class_index][1] += 1 #FN
            else: #negtive entry
                if res_matrix[i][class_index] == positive:
                    confuse_matrixs[class_index][2] += 1 #FP
                else:
                    confuse_matrixs[class_index][3] += 1 #TN

    
    P, R = .0, .0    
    for i in range(confuse_matrixs.shape[0]-1):
#         print confuse_matrixs[i]
        p = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][2])
        r = confuse_matrixs[i][0]/(confuse_matrixs[i][0] + confuse_matrixs[i][1])
        P += p
        R += r
#         print 'Evaluation for the ' + str(i + 1) + 'th class'
#         print 'P:    ', p
#         print 'R:    ', r
#         print 'F1:    ', 2*p*r/(p+r)
#         print        
    P /= (confuse_matrixs.shape[0]-1)
    R /= (confuse_matrixs.shape[0]-1)
    F1 = 2*P*R/(P+R)
    print 'Evaluation for all the class'
    print 'P:    ', P
    print 'R:    ', R
    print 'F1:    ', F1
    print
    
    return F1


def eval_mulclass2(res_matrix, word_array, index_2_word, word_array_weight, instances_visual, real=True):

    # if real == True:
    #     res_matrix = change_real2class(res_matrix)

    list = []
    for i in range(res_matrix.shape[0]):
        if res_matrix[i][0] > 0.95:
            max1_index = np.argmax(word_array_weight[i])
            word_array_weight[i][max1_index] = 0
            max2_index = np.argmax(word_array_weight[i])
            word_array_weight[i][max2_index] = 0
            max3_index = np.argmax(word_array_weight[i])
            e1_begin, e1_end, e2_begin, e2_end, e1_type, e2_type, sent = instances_visual[i]
            new_sent = []
            for index in range(len(sent)):
                if index == max1_index or index == max2_index or index == max3_index:
                    new_sent.append('<font style="color:#FF0000">'
                                    + sent[index] + '</font>')
                else:
                    new_sent.append(sent[index])

            new_new_sent = ['<br>']
            for index in range(len(new_sent)):
                if index == e1_begin:
                    if e1_type == 'chemical':
                        new_new_sent.append('<font style="background-color: rgb(171,221,164)">')
                        new_new_sent.append(new_sent[index])
                    elif e1_type == 'disease':
                        new_new_sent.append('<font style = "background-color: rgb(0,208,255)" >')
                        new_new_sent.append(new_sent[index])
                    else:
                        new_new_sent.append('<font style = "background-color: rgb(215,25,28)" >')
                        new_new_sent.append(new_sent[index])
                elif index == e1_end:
                    new_new_sent.append('</font>')
                    new_new_sent.append(new_sent[index])
                elif index == e2_begin:
                    if e2_type == 'chemical':
                        new_new_sent.append('<font style="background-color: rgb(171,221,164)">')
                        new_new_sent.append(new_sent[index])
                    elif e2_type == 'disease':
                        new_new_sent.append('<font style = "background-color: rgb(0,208,255)" >')
                        new_new_sent.append(new_sent[index])
                    else:
                        new_new_sent.append('<font style = "background-color: rgb(215,25,28)" >')
                        new_new_sent.append(new_sent[index])
                elif index == e2_end:
                    new_new_sent.append('</font>')
                    new_new_sent.append(new_sent[index])
                else:
                    new_new_sent.append(new_sent[index])
            list.append(' '.join(new_new_sent))

    return list




if __name__ == '__main__':
    
    pass