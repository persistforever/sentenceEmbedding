# coding=utf-8
# -*- coding: utf-8 -*-
__keywordList = None

def __init():
    import codecs
    globals()['__keywordList'] = list()
    with codecs.open("util/rubbish.dat", "r", "utf-8", "ignore") as f:
        for line in f:
            line = line.strip()
            globals()['__keywordList'].append(line)

def isRubbishSentence(sentence):
    if globals()['__keywordList'] == None:
        __init()
    flag = False
    for s in globals()['__keywordList']:
        if s in sentence:
            flag = True
            break
    return flag

