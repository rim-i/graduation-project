# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:35:09 2019

@author: arimeeeing
"""
#%%
import math
#%%
class ordering:
    def __init__(self,tok_list,word_list):
        self.token = tok_list
        self.keywords = word_list     #속성 단어
        self.nkeys = len(word_list)   #속성 개수
        self.embmodel = Word2Vec(self.token,size=100,min_count=1,sg=1,window=2,workers=4,iter=100)
    
    #리뷰 분류
    def classification(self):
        new = []
        for n in range(len(self.token)):
            x = 0
            for i in self.token[n]:
                if i in self.keywords:
                    x += 1
            if x:
                new.append(self.token[n])
        self.reviews = new           #속성 리뷰
        
#        from gensim.models.word2vec import Word2Vec
#        self.embmodel = Word2Vec(self.reviews,size=100,min_count=50,sg=1)
        
        return new

    #속성 관련 단어
    def similar_words(self):
        self.similar = []
        self.similar.extend(self.keywords)
        for keyword in self.keywords:
            emb = self.embmodel.wv.most_similar(positive=[keyword], topn=30)
            for word, per in emb:
                if word not in self.similar:
                    self.similar.append(word)
        self.nsimilar = len(self.similar)  #관련 단어 개수
        return self.similar
    
    #속성 리뷰에 포함된 단어
    def words_group(self):
        self.words = []
        self.words.extend(self.keywords)  #속성 키워드 먼저 추가
        for sent in self.reviews:
            for word in sent:
                if word not in self.words:  #리뷰 내 단어 중복 제거하여 추가
                    self.words.append(word)
        self.dim = len(self.words)   #리뷰 내 단어 수 : 행렬 차원
        return self.words
        

    #단어 벡터 행렬 
    def get_matrix(self):
        mat = np.zeros((self.dim,100))
        for i in range(self.dim):
            mat[i] = self.embmodel.wv[self.words[i]]
        self.matrix = mat
        return mat

    #거리행렬    
    def distance_matrix(self):
        self.distance = np.zeros((self.dim,self.dim))
        for i in range(self.dim):
            for k in range(self.dim):
                self.distance[i][k] = math.sqrt(sum([(xi-yi)**2 for xi,yi in zip(self.matrix[i], self.matrix[k])]))
        return self.distance
    
    #가중치행렬(속성keyword에 대하여 나머지 단어들의 거리에 따른 가중치 점수를 구한 것)
    def weight_matrix(self):
        weight = []
        weight = np.zeros((self.nkeys,self.dim))  #속성(행)*단어(열)
        for i in range(self.nkeys):
            for k in range(self.dim):
                weight[i][k] = 1/float((self.distance[i][k])**2)  #가중치=거리 제곱의 역수
        self.weight = weight     #단어(행)*속성(열)
        return self.weight   

    def weight_matrix2(self):
        weight = []
        weight = np.zeros((self.nkeys,self.dim))  #속성(행)*단어(열)
        for i in range(self.nkeys):
            for k in range(self.dim):
                weight[i][k] = 1/float(self.distance[i][k])  #가중치=거리의 역수
        self.weight = weight     #단어(행)*속성(열)
        return self.weight 
 
    def weight_matrix3(self,sigma):
        weight = []
        weight = np.zeros((self.nkeys,self.dim))  #속성(행)*단어(열)
        for i in range(self.nkeys):
            for k in range(self.dim):
                weight[i][k] = math.exp(-float((self.distance[i][k])**2)/float(2*(sigma)**2))  #가중치=거리 제곱의 역수
        self.weight = weight     #단어(행)*속성(열)
        return self.weight 
    
    #정수인코딩 : 리뷰 내 단어
    def word2index(self):
        self.index = {word : index+1 for index, word in enumerate(self.words)}
        return self.index  #단어별 index : 1부터 시작
    
    # dtm
    def document_term_matrix(self):
        dtm = []
        for sent in self.reviews:
            one_hot_vector = [0]*self.dim  #관련 단어 개수 길이의 벡터 : 0부터시작
            for word in sent:
                index = self.index[word]      #해당 단어의 index를 찾아
                one_hot_vector[index-1] += 1  #one-hot-vector의 index 위치에 1을 더해준다
            dtm.append(one_hot_vector)   #리뷰*단어 형태 list
        self.dtm = np.array(dtm)        #리뷰(행)*단어(열)
        return self.dtm
    
    # dtm * 가중치행렬
    def score_matrix(self):
        self.scorematrix = np.dot(self.dtm,self.weight.T)  #리뷰(행)*속성(열)
        return self.scorematrix
    
    def review_score(self):        
        self.reviewscore = np.zeros(len(self.reviews))
        for i in range(len(self.reviews)):
            self.reviewscore[i] = sum(self.scorematrix[i])
        return self.reviewscore
    
#%%
room = ordering(tokenized_list,['room','bed','bathroom','clean', 'floor', 'size', 'shower', 'design'])
room_reviews = room.classification()
room_words = room.words_group()
room_matrix = room.get_matrix()
room_distance = room.distance_matrix()
room_weight = room.weight_matrix()
room_weight2 = room.weight_matrix2()
room_weight3 = room.weight_matrix3(1.5)
room_index = room.word2index()
room_tdm = room.document_term_matrix()
room_score = room.score_matrix()
room_rs = room.review_score()

np.std(room_distance)
math.exp(-float((room_distance[0][1])**2)/float(2*(50)**2))

#%%
room = ordering(tokenized_list,['area','location','located','hongdae'])
area_reviews = area.classification()
area_words = area.words()
area_matrix = area.get_matrix()
area_distance = area.distance_matrix()
area_weight = area.weight_matrix()
area_index = area.word2index()
area_tdm = area.term_document_matrix()
area_score = area.score_matrix()
area_review = area.review_score()

#%%
area = classification(tokenized_list,['area','location','hongdae','restaurant, convenient','shopping','supermarket','store'])
service = classification(tokenized_list,['service','staff','breakfast','english','desk','pool','check'])
transportation = classification(tokenized_list,['subway','line','train','bus','airport','station','exit','metro'])
room = classification(tokenized_list,['room','bed','bathroom','clean', 'floor', 'size', 'shower', 'design'])

