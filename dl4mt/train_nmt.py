from nmt import train
from pprint import pprint
from setup import setup
import sys


config = setup(sys.argv[1])
pprint(config)

validerr = train(**config)

print 'done'
