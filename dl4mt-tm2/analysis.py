from __future__ import division
import numpy
import os
import  cPickle as pkl
import matplotlib.pylab as plt

# file_list =[[]]*len(range(2500, 10000, 2500))
file_list = []
for j, i in enumerate(range(2500, 190000, 2500)):
    file = open('/root/workspace/TMNMT/.translate/TM2.B7.bpe.dev.translate.iter='+str(i)+'.pkl', 'r')
    file_list.append(pkl.load(file))    
    print i
    
action = [[] for _ in range(len(file_list))]
gating = [[] for _ in range(len(file_list))]
# print action
# print len(file_list)
# print len(file_list[0])
# print len(file_list[0][0])
# print len(file_list[0][0][0])


for i in range(len(file_list)):
    for j in range(len(file_list[i])):
#         print len(file_list[i][0])
        action[i].append(file_list[i][j][2])
        gating[i].append(file_list[i][j][3])
        
# print action
aver_action = []
aver_gating = []
# print len(action[0])
# print len(action[1])
# print len(action[0][0])
# print len(action[0][1])
# print len(action[0][2])
# print len(action[1][0])
# print len(action[1][1])
# print len(action[1][2])
# print action[0][0][0]
# print action[1][0][0]
# print len(action[0][1])
for i in range(len(file_list)):
    action_init = []
    gating_init = []
    for j in range(len(action[i])):
        
        action_init.append(numpy.asarray(action[i][j]).mean())
        gating_init.append(numpy.asarray(gating[i][j]).mean())
    aver_action.append(numpy.asarray(action_init[i]).mean())
    aver_gating.append(numpy.asarray(gating_init[i]).mean())

print aver_action
print aver_gating
plt.plot(aver_action, '--*b')
plt.plot(aver_gating, '--*r')
plt.savefig('/root/workspace/TMNMT/dl4mt-tm2/Myfig1.jpg') 
        
    