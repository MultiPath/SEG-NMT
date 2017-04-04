import os
import numpy
hyp = []
ref = []
for i in numpy.arange(10):
    hyp.append('/root/workspace/TMNMT/translate/en_sort_'+str(i))
    ref.append('/root/workspace/TMNMT/translate/en_trans_sort_'+str(i))

for i in numpy.arange(10):
    os.system('perl ./data/multi-bleu.perl {0} < {1} | tee {1}.score'.format(ref[i], hyp[i]))
print 'done'
