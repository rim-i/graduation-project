# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 03:42:41 2019

@author: arimeeeing
"""

#%%
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
#%% 품사 태깅
def tagging(tok):
    new = []
    for sentence in tok:
        new.append(nltk.pos_tag(sentence))
    return new
#%% 명사 추출
def get_noun(tok):
    pos = []
    for sent in tok:
        pos.extend(nltk.pos_tag(sent))
    
    nouns = []
    for keyword, type in pos:
        if type == 'NN':
            nouns.append(keyword)
    
    return nouns
#%%
tokenized_list = tokenized_list7
tag = tagging(tokenized_list)
noun = get_noun(tokenized_list)
#%%
text = nltk.Text(noun, name='NMSC')
# 전체 토큰의 개수
print(len(text.tokens))
# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))            
# 출현 빈도가 높은 상위 토큰 10개
print(text.vocab().most_common(40))
#%%
tag[9]
print(' '.join(tokenized_list1[9]))
print(tokenized_list7[9])
#%%
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
#%%
# Counter: 단어수 세기, 가장 많이 등장한 단어(명사) 40개
count = Counter(noun)
tags = count.most_common(45)

# WordCloud, matplotlib: 단어 구름 그리기
font_path = '/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf'
wc = WordCloud(background_color='white', width=1000, height=800)
cloud = wc.generate_from_frequencies(dict(tags))
plt.figure(figsize=(10,8))
plt.axis('off')
plt.imshow(cloud)
#%%
### topic modeling
#%% 리스트의 요소가 하나의 리뷰 단어토큰화
tokenized_doc = []
n = WordNetLemmatizer()
for one in en_reviews:
    new= []
    for sent in sent_tokenize(one):
        for word in word_tokenize(sent):
            eng = re.sub('[^a-zA-Z]', '', word).lower()
            rep = replacer.replace(eng)
            if rep and len(rep)>=3 and not rep in stopwords.words('english'):
                new.append(n.lemmatize(rep))
    tokenized_doc.append(new)
#%%
a = sent_tokenize(en_reviews[0])
print(word_tokenize(a[0]))
#%%
import gensim
from gensim import corpora, models, similarities
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import time
import pandas as pd
import numpy as np
import re
import warnings
from pymongo import MongoClient
from pprint import pprint
#%% 사전과 코퍼스 생성
#noun2 = [[i for i in noun]]
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
#%%
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 8, id2word=dictionary, passes=20) #passes:알고리즘 동작 횟수
topics = ldamodel.print_topics(num_words=6)  #num_words:단어출력개
for topic in topics:
    print(topic)
#%%최적 eoches
coherences=[]
perplexities=[]
passes=[]
warnings.filterwarnings('ignore')

for i in range(10):
    
    ntopics, nwords = 200, 100
    if i==0:
        p=1
    else:
        p=i*5
    tic = time.time()
    lda4 = LdaModel(corpus, id2word=dictionary, num_topics=ntopics, iterations=400, passes=p)
    print('epoch',p,time.time() - tic)
    # tfidf, corpus 무슨 차이?
    # lda = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=ntopics, iterations=200000)

    cm = CoherenceModel(model=lda4, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print("Cpherence",coherence)
    coherences.append(coherence)
    print('Perplexity: ', lda4.log_perplexity(corpus),'\n\n')
    perplexities.append(lda4.log_perplexity(corpus))
#%% 최적 coherence 찾기
coherencesT=[]
perplexitiesT=[]
passes=[]
warnings.filterwarnings('ignore')

for i in range(2,11):
    ntopics = i*5
    nwords = 100
    tic = time.time()
    lda4 = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=ntopics, iterations=400, passes=30)
    print('ntopics',ntopics,time.time() - tic)

    cm = CoherenceModel(model=lda4, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print("Cpherence",coherence)
    coherencesT.append(coherence)
    print('Perplexity: ', lda4.log_perplexity(corpus),'\n\n')
    perplexitiesT.append(lda4.log_perplexity(corpus))
#%% 시각화
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)
vis
sort=True
#%% 토픽 일관성 점수
coherence_model_lda = CoherenceModel(model=ldamodel, corpus=corpus, texts=tokenized_doc, dictionary=dictionary, coherence='u_mass')
coherence_lda=coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)
print('Perplexity: ', ldamodel.log_perplexity(corpus))
#%%
### word2vec
#%% 유사한 단어 찾기
from gensim.models.word2vec import Word2Vec
set.seed=100
embedding_model = Word2Vec(tokenized_list,size=100,min_count=50,sg=1,window=2,workers=4,iter=100)
embedding_model.init_sims(replace=True)   #필요없는 메모리 unload

print(embedding_model.most_similar(positive=["location"], topn=20))
print(embedding_model.most_similar(positive=["area"], topn=20))
print(embedding_model.most_similar(positive=["station"], topn=20))
print(embedding_model.most_similar(positive=["airport"], topn=20))
print(embedding_model.most_similar(positive=["restaurant"], topn=20))
print(embedding_model.most_similar(positive=["convenient"], topn=20))
print(embedding_model.most_similar(positive=["hongdae"], topn=20))

print(embedding_model.most_similar(positive=["room"], topn=20))
print(embedding_model.most_similar(positive=["bed"], topn=20))
print(embedding_model.most_similar(positive=["bathroom"], topn=20))
print(embedding_model.most_similar(positive=["floor"], topn=20))
print(embedding_model.most_similar(positive=["clean"], topn=20))

print(embedding_model.most_similar(positive=["subway"], topn=20))
print(embedding_model.most_similar(positive=["bus"], topn=20))
print(embedding_model.most_similar(positive=["line"], topn=20))
print(embedding_model.most_similar(positive=["train"], topn=20))

print(embedding_model.most_similar(positive=["staff"], topn=20))
print(embedding_model.most_similar(positive=["service"], topn=20))
print(embedding_model.most_similar(positive=["breakfast"], topn=20))