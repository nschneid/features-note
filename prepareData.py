#!/usr/bin/env python2.7
from __future__ import print_function
import itertools, random, codecs, json

data = []

for inFP in ['cummings.txt','dickinson.txt','williams.txt']:
    with codecs.open(inFP, 'r', 'utf-8') as inF:
        titleLn = next(inF)
        author = inFP.replace('.txt','')
        verse = ''
        for ln in itertools.chain(inF,'\n'):
            if not ln.strip():
                if verse:
                    data.append([{"words": verse.strip().split()},author])
                    verse = ''
                continue
            verse += ln
assert not verse
random.shuffle(data)

train = data[:-30]
test = data[-30:]

with codecs.open('train.json', 'w', 'utf-8') as trainF:
    for instance in train:
        trainF.write(json.dumps(instance)+'\n')
with codecs.open('test.json', 'w', 'utf-8') as testF:
    for instance in test:
        testF.write(json.dumps(instance)+'\n')
