import argparse
import os

from setup import setup
parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, default='fren')
args = parser.parse_args()

config = setup(args.m)
hyp = config['trans_to']
ref = config['trans_ref']

print 'compute BLEU score for {}'.format(hyp)
# os.system("ed -i 's/@@ //g' {}".format(hyp))
os.system('perl ./data/multi-bleu.perl {0} < {1} | tee {1}.score'.format(ref, hyp))
print 'done'
