import numpy
import cPickle as pickle
with open('/root/disk/dl4mt-tutorial2/data_preprocess_enes_top5000_bpe/train.en.top5.shuf.tok.bpe.pkl','rb') as f:
    c = pickle.load(f)
    print len(c)
    
with open('/root/disk/dl4mt-tutorial2/data_preprocess_enes_top5000_bpe/train.es.top5.shuf.tok.bpe.pkl','rb') as f:
    c = pickle.load(f)
    print len(c)