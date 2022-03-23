# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 02:03:47 2019

@author: arimeeeing
"""

#%% packages

import nltk
nltk.download('stopwords')

from nltk.tokenize import sent_tokenize  #텍스트를 문장으로 토큰화
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.tokenize import word_tokenize  #문장을 단어로 토큰화
from replacers import RegexpReplacer # 축약어
from nltk.corpus import stopwords # 불용어사전

import re
#%% review file

f = open('english_review.txt','r',encoding = 'UTF8')
en_reviews = f.readlines()
#%% 1. 2차원 리스트로 토큰화

def get_tokenized(tok):
    new = []
    for review in tok:
        for line in sent_tokenize(review):
            new.append(word_tokenize(line))
    return new
#%% 2. 영어만 남기기

def only_english(tok):
    new = []
    for sent in tok:
        eng = []
        for word in sent:
            #print(word)
            eng.append(re.sub('[^a-zA-Z]', '', word))
        new.append(eng)
    return new
#%% 3. 소문자로 변경

def upper_to_lower(tok):
    new = []
    for sent in tok:
        small = []
        for word in sent:
            small.append(word.lower())
        new.append(small)
    return new
#%% 4. 축약어 대체

replacer = RegexpReplacer()
def replace(tok):
    new = []
    for sent in tok:
        rep = []
        for word in sent:
            rep.append(replacer.replace(word))
        new.append(rep)
    return new
#%% 5. 불용어 제거
def remove_stopword(tok):
    new = []
    for sent in tok:
        no_stop = [ w for w in sent if not w in stopwords.words('english')]
        new.append(no_stop)
    return new
#%% 6. 짧은단어 제거
def remove_short(tok):
    new = []
    for sent in tok:
        long = []
        for word in sent:
            if len(word)>=3:
                long.append(word)
        new.append(long)
    return new
#%% 7. 원형복원
from nltk.stem import WordNetLemmatizer
def lemmazation(tok):
    n = WordNetLemmatizer()
    new = []
    for sent in tok:
        tokens = [ n.lemmatize(word) for word in sent ]
        new.append(tokens)
    return new

#%%
tokenized_list1 = get_tokenized(en_reviews)
tokenized_list2 = only_english(tokenized_list1)
tokenized_list3 = upper_to_lower(tokenized_list2)
tokenized_list4 = replace(tokenized_list3)
tokenized_list5 = remove_stopword(tokenized_list4)
tokenized_list6 = remove_short(tokenized_list5)
tokenized_list = lemmazation(tokenized_list6)
