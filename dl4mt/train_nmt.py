from nmt import train
from pprint import pprint
from setup import setup

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, default='fren')
args = parser.parse_args()

config = setup(args.m)
pprint(config)

validerr = train(**config)
print 'done'
