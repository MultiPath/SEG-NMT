from nltk import metrics
import numpy
import time
import editdistance

fr_list = []
en_list = []

def cos (a,b):
    tmp = [val for val in a if val in b]
    length_common = len(tmp)
    length = len(a) * len(b)
    length_sqrt = numpy.sqrt(length)

    cos_value = length_common/length_sqrt
    return cos_value


with open('/root/workspace/dl4mt-tutorial2/data/train.fr.tok.shuf', 'rb') as f:
    while True:        
        ss = f.readline()
        if ss=="":
            break
        fr_list.append(ss)       
    
with open('/root/workspace/dl4mt-tutorial2/data/train.en.tok.shuf', 'rb') as f:
    while True:
        ss = f.readline()
        if ss=="":
            break
        en_list.append(ss)   
        
print len(fr_list)
print len(en_list)

match_sentences_en = []
match_sentences_fr = []


for i in range(len(en_list)):
    edit_dis = []
    start = time.time()
    for j in range(len(en_list)):
        edit_dis.append(cos(en_list[i].split(),en_list[j].split()))
#         if j%10000==0:
#             end = time.time()
#             time_ = end-start
#             print time_
#         print j
    same_sentence = numpy.argmax(edit_dis)
    edit_dis[same_sentence] = 0
    matched_sentence = numpy.argmax(edit_dis)
    match_sentences_en.append(en_list[matched_sentence])
    print en_list[i]
    print match_sentences_en[i]
    match_sentences_fr.append(fr_list[matched_sentence])
    end = time.time()
    time_ = end-start
    print time_    
    
    print i
    
with open('/root/workspace/dl4mt-tutorial2/data/train.fr.tok.shuf.matched', 'w') as f:
    for i in match_sentences_en:
        f.write(i)
        f.write("\n")
    
with open('/root/workspace/dl4mt-tutorial2/data/train.en.tok.shuf.matched', 'w') as f:
    for i in match_sentences_fr:
        f.write(i)
        f.write("\n")
 
        
            
        
