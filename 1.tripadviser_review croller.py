# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:36:57 2019

@author: arimeeeing
"""
#%% packages
import requests
import bs4
from urllib.request import urlopen
from bs4 import BeautifulSoup

#%% max page 구하기 (안씀)
def count_max_page(url):
    html = requests.get(url)
    bs_obj = bs4.BeautifulSoup(html.content, 'html.parser')
    a = bs_obj.find('div',{'class','pageNumbers'})
    b = a.findAll('a')
    for a in b:
        maxpage = int(a.text)
#%%
def get_hotel_name(url):
    html = requests.get(url)
    bs_obj = bs4.BeautifulSoup(html.content, 'html.parser')
    a = bs_obj.find('div',{'class','ui_column is-12-tablet is-10-mobile hotels-hotel-review-atf-info-parts-ATFInfo__description--1njly'})
    b = a.find('div')
    return b.text
#%% 크롤링할 url 리스트
def make_urls(url):
    urlist = url.split('-',4)
    front = urlist[0]+'-'+urlist[1]+'-'+urlist[2]+'-'+urlist[3]+'-'
    
    html = requests.get(url)
    bs_obj = bs4.BeautifulSoup(html.content, 'html.parser')
    a = bs_obj.find('div',{'class','pageNumbers'})
    b = a.findAll('a')
    for a in b:
        maxpage = int(a.text)
    
    urls = []
    for i in range(maxpage):
        n = 5
        pages = 'or%d-' % (n*i)
        url = front + pages + urlist[4]
        urls.append(url)
    
    return urls
#%% url에서 review 데이터 추출
def get_review(url):
    html = requests.get(url)
    bs_obj = bs4.BeautifulSoup(html.content,'html.parser')
    qs = bs_obj.findAll("q",{'class','hotels-review-list-parts-ExpandableReview__reviewText--3oMkH'})
    
    review=[]
    for q in qs:
        span = q.find('span')
        review.append(span.text)
    
    return review
#%% review 저장할 list
en_reviews = []
#%% crolling
url = 'https://www.tripadvisor.com/Hotel_Review-g294197-d4134800-Reviews-Hotel_The_Designers_Hongdae-Seoul.html'

urls = make_urls(url)

for i in range(len(urls)):
    review = get_review(urls[i])
    en_reviews.extend(review)
#%% 텍스트 파일로 저장
file = open('english_review.txt','w',encoding='UTF8')

for a in en_reviews:
    file.write(a +'\n')

file.close()


