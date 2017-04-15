from __future__ import division
import numpy
import os
import  cPickle as pkl
import matplotlib.pylab as plt

# file_list =[[]]*len(range(2500, 10000, 2500))
# file_list = []
# for j, i in enumerate(range(2500, 7500, 2500)):
#     file = open('/root/workspace/TMNMT/.translate/TM2.B7.bpe.dev.translate.iter='+str(i)+'.pkl', 'r')
#     file_list.append(pkl.load(file))    
#     print i
# 
# with open('/root/workspace/TMNMT/dl4mt-tm2/dl4mt-tm2.txt', 'w') as f:
#     f.write(str(file_list[0]))
# 
# with open('/root/workspace/TMNMT/dl4mt-tm2/dl4mt-tm2_2.txt', 'w') as f:
#     f.write(str(file_list[1]))
action = [[]]* 2

file_list = [[[1,2][2,3]], [[2,3,4],[3,4]]]
for i in range(len(file_list)):
    for j in range(len(file_list[i])):
#         print len(file_list[i][0])
        action[i].append(file_list[i][j][0])
print action

