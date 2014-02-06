#!/usr/bin/env python2.7
'''
Toy implementation of a feature-rich classifier 
trained with the perceptron algorithm.

Strings and feature template names are indexed to integers. 
Complex percepts are named with tuples of numbers, and 
all percept names are indexed to integers.
The cross-product parametrization is used, 
treating the last label as the background label.
By default, the training instances are cached in memory as 
active percept maps and labels.
The default percept cutoff is 2.

This code could be made more scalable by using high-performance 
data structures, implementing critical operations as Cython functions, etc.

@author: Nathan Schneider (nschneid@cs.cmu.edu)
@since: 2014-02-05
'''
from __future__ import print_function, division
from collections import Counter
import codecs, json

labelNames = ['cummings','dickinson','williams']
labels = list(range(len(labelNames)))

# string -> int
s2i = {}

percepts = []
p2i = {}    # percept name tuple -> int
percept_cutoff = 2
pCounts = Counter()

# templates
BIAS, INPUT_LEN, CONTAINS_WORD, HAS_CAPITAL_LETTER = range(4)


def index_string(s):
    return s2i.setdefault(s,len(s2i))

def fire(template, template_vars, val, instantiation):
    perceptName = (template,)+template_vars
    if instantiation:
        p = p2i.setdefault(perceptName, len(percepts))
        if len(percepts)==p: percepts.append(perceptName)
        pCounts[p] += 1
    elif val==0: p = None
    else: p = p2i.get(perceptName)
    
    if p is not None: return {p: val}
    return {}    # percept not in model, so ignore it

def extract(x, instantiation):
    active_percepts = {BIAS: 1}
    active_percepts.update(fire(INPUT_LEN, (), len(x['words']), instantiation))
    for w in x['words']:
        active_percepts.update(fire(CONTAINS_WORD, (index_string(w),), 1, instantiation))
    active_percepts.update(fire(HAS_CAPITAL_LETTER, (), 
                                int(any(c.isupper() for c in ''.join(x['words']))), 
                                instantiation))
    return active_percepts

def feature_index(p, l):
    assert isinstance(p,int)
    assert isinstance(l,int)
    return l*len(percepts)+p

def classify(active_percepts, weights):    
    score,l = max((sum(v*weights[feature_index(p,l)] for p,v in active_percepts.items()),
                   l) for l in labels[:-1])
    return labels[-1] if score<0 else l

def load_instance(json_line):
    instance = json.loads(json_line)
    instance[1] = labelNames.index(instance[1])
    return instance

def instantiate(training_data, cache=None, percept_cutoff=1):
    for instance in training_data:
        x, y = load_instance(instance)
        active_percepts = extract(x, instantiation=True)
        if cache is not None:
            cache.append((active_percepts, y))
    train = cache if cache else training_data
    if percept_cutoff>1:
        deleted = set()
        for p,n in pCounts.items():
            if n<percept_cutoff:
                del p2i[percepts[p]]
                deleted.add(p)
        for active_percepts,y in train:
            if p in active_percepts:
                del active_percepts[p]
    return train

EPOCHS = 25

def learn(training):
    weights = [0]*len(percepts)*(len(labels)-1)
    for e in range(EPOCHS):
        incorrect = total = 0
        for active_percepts,y in training:
            y_pred = classify(active_percepts, weights)
            if y_pred!=y:
                incorrect += 1
                # perceptron update
                for p,v in active_percepts.items():
                    if y!=labels[-1]: weights[feature_index(p, y)] += v
                    if y_pred!=labels[-1]: weights[feature_index(p, y_pred)] -= v
            total += 1
        print('Train accuracy (epoch {}): {}/{} = {:%}'.format(e, 
                                                               (total-incorrect), 
                                                               total, 
                                                               (total-incorrect)/total))
    return weights


with codecs.open('train.json','r','utf-8') as training_data: 
    training = instantiate(training_data, [], percept_cutoff)
    weights = learn(training)

with codecs.open('test.json','r','utf-8') as test_data:
    correct = total = 0
    for instance in test_data:
        x,y = load_instance(instance)
        active_percepts = extract(x, instantiation=False)
        y_pred = classify(active_percepts, weights)
        if y==y_pred: correct += 1
        total += 1
    print('Test accuracy: {}/{} = {:%}'.format(correct, total, correct/total))
