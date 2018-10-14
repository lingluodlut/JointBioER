
# merge map1 and map2 to map1
# where each map's value is list type
def combine_map_file(map1, map2):

    for key in map2:
        if map1.has_key(key):
            map1[key].extend(map2[key])
        else:
            map1[key] = map2[key]
    return map1
            

def filter_overlaps_of_list(listfile):
    
    tempt_set = set()
    for elem in listfile:
        tempt_set.add(elem.lower())
    
    new_list = []
    for elem in tempt_set:
        new_list.append(elem)
    
    return new_list

def write_map_file(fileName, map):
    
    buffer = []
    buffer_size = 10000
    count = 0
    
    writeObj = open(fileName, 'w')
    for key in map.keys():
        buffer.append(key + '\t' + '|'.join(filter_overlaps_of_list(map[key])))
        count += 1
        if count == buffer_size:
            writeObj.write('\n'.join(buffer))
            count = 0
            buffer = []
    writeObj.write('\n'.join(buffer))
    writeObj.close()
    

def writeStrLines(fileName, strlist):
    writeObj = open(fileName, 'w')
    writeObj.write('\n'.join(strlist))
    writeObj.close()

def writeListLines(fileName, listOfList, itemSep):
    writeObj = open(fileName, 'w')
    for entry in listOfList:
        writeObj.write(itemSep.join(entry) + '\n')
    writeObj.close()
    
def writeFloatListLines(fileName, listOfList, itemSep):
    
    strListOfList = []
    
    for list in listOfList:
        strList = []
        for item in list:
            strList.append(str(item))
        strListOfList.append(strList)
    
    writeListLines(fileName, strListOfList, itemSep)
    
def writeFloatMatrix(A, output):
    entryList = []
    for row in A:
        entry = []
        for elem in row:
            entry.append(str(round(elem,5)))
        entryList.append(entry)
    writeListLines(output, entryList, ' ')

def writeFloatBias(B, output):
    biases = []
    for bias in B:
        biases.append(str(bias))
    writeStrLines(output, biases)

if __name__ == '__main__':
    map1 = {}
    map1[2] = [2,3,4]
    map1[3] = [4,5,6]

    map2 = {}
    map2[2] = [8,3,9]
    map2[1] = [5,6,7]

    print combine_map_file(map1, map2)
    
    