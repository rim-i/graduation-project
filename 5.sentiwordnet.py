# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:37:24 2019

@author: arimeeeing
"""

#%%
f = open('english_review.txt','r',encoding = 'UTF8')
en_reviews = f.readlines()
#%%
sentence = []
for review in en_reviews:
    for line in sent_tokenize(review):
        sentence.append(line)
#%%
tokenized_list = []
for line in sentence:
    tokenized_list.append(word_tokenize(line))
#%% 데이터 전처리
tokenized_review =[]
for one in tokenized_list:
    new = []
    for i in one:
        eng = re.sub('[^a-zA-Z]', '', i).lower()
        if eng and len(eng)>=3 and eng not in stopwords.words('english'):
            new.append(eng)
    tokenized_review.append(new)
#%%
from nltk.stem import WordNetLemmatizer
n = WordNetLemmatizer()
tokenized_review1 = []
for i in tokenized_review:
   token = [ n.lemmatize(word) for word in i ]
   tokenized_review1.append(token)
#%%
def tagging(tok):
    new = []
    for sentence in tok:
        new.append(nltk.pos_tag(sentence))
    return new
#%%
def review_sentiment(tok):
    #리뷰와 감성점수를 컬럼으로 하는 데이터프레임 생성
    review_data = pd.DataFrame(np.zeros((len(tok),2)),columns=['review','senti'])
    
    #컬럼0에 리뷰 넣기
    for i in range(len(tok)):
        review_data['review'][i] = tok[i]
    
    #컬럼1에 감성점수 계산해서 넣기
    tagged = tagging(tok)
    
    for i in range(len(tok)):
        n = 0
        pscore = 0
        nscore = 0

        for word, tag in tagged[i]:
            if tag.startswith('J') and len(list(swn.senti_synsets(word,'a')))>0:
                pscore += (list(swn.senti_synsets(word,'a'))[0]).pos_score()
                nscore += (list(swn.senti_synsets(word,'a'))[0]).neg_score()
                n += 1
            elif tag.startswith('N') and len(list(swn.senti_synsets(word,'n')))>0:
                pscore += (list(swn.senti_synsets(word,'n'))[0]).pos_score()
                nscore += (list(swn.senti_synsets(word,'n'))[0]).neg_score()
                n += 1
            elif tag.startswith('R') and len(list(swn.senti_synsets(word,'r')))>0:
                pscore += (list(swn.senti_synsets(word,'r'))[0]).pos_score()
                nscore += (list(swn.senti_synsets(word,'r'))[0]).neg_score()
                n += 1
            elif tag.startswith('V') and len(list(swn.senti_synsets(word,'v')))>0:
                pscore += (list(swn.senti_synsets(word,'v'))[0]).pos_score()
                nscore += (list(swn.senti_synsets(word,'v'))[0]).neg_score()
                n += 1
            else:
                pass
            
        total_senti = pscore-nscore
        
        if n==0:
            review_data['senti'][i] = 0
        else:
            review_data['senti'][i] = total_senti/n
    
    return review_data

#%%
room_tag = tagging(room_reviews)
room_tag[6]
nltk.pos_tag(room_review[2])
room_sentiment = review_sentiment(room_reviews)

#%% 부정리뷰 개수 세기
for i in range(len(tokenized_review)):
    if review_data['senti'][i]<0:
        print(review_data['review'][i])
        
#%%
print(list(swn.senti_synsets('clean','a')))
print((list(swn.senti_synsets('view')))[1])
a.pos_score()
a = list(swn.senti_synsets(tokenized_review[0][0]))[0]
print(a)
print(list(swn.senti_synsets('clean','a'))[0])
