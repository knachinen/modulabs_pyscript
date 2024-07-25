from numpy import dot
from numpy.linalg import norm
from math import log

from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
from os import path
import numpy as np
import pandas as pd

import urllib.request

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def get_abs_path():
    abs_path = path.dirname(
    path.dirname(
        path.abspath('.')))
    return abs_path

def bag_of_words1(text:list) -> None:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence) # 단어장 생성
    bow = dict(tokenizer.word_counts) # 각 단어와 각 단어의 빈도를 bow에 저장
    
    print("Bag of Words :", bow) # bow 출력
    print('단어장(Vocabulary)의 크기 :', len(tokenizer.word_counts)) # 중복을 제거한 단어들의 개수

def bag_of_words2(text:list) -> None:
    vector = CountVectorizer()
    bow = vector.fit_transform(sentence).toarray()
    
    print('Bag of Words : ', bow) # 코퍼스로부터 각 단어의 빈도수를 기록한다.
    print('각 단어의 인덱스 :', vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def get_rand():
    return np.random.randint(low=0, high=3, size=4, dtype='int64')

def make_cos_sim():
    docs = [get_rand() for i in range(3)]

    for index, item in enumerate(docs):
        print(f"doc {index+1}: {item}")
    
    print('doc 1,2 : {:.2f}'.format(cos_sim(docs[0], docs[1]))) #문서1과 문서2의 코사인 유사도
    print('doc 1,3 : {:.2f}'.format(cos_sim(docs[0], docs[2]))) #문서1과 문서3의 코사인 유사도
    print('doc 2,3 : {:.2f}'.format(cos_sim(docs[1], docs[2]))) #문서2과 문서3의 코사인 유사도

def dtm(corpus):
    vector = CountVectorizer()
    
    print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도수를 기록.
    print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.

def get_tf(t, d):
    return d.count(t)
 
def get_idf(t):
    df = 0
    for doc in docs:
        df += t in doc    
    return log(N/(df + 1)) + 1
 
def get_tfidf(t, d):
    return get_tf(t,d) * get_idf(t)

def process_tf(vocab, N):
    result = []
    for i in range(N): # 각 문서에 대해서 아래 명령을 수행
        result.append([])
        d = docs[i]
        for j in range(len(vocab)):
            t = vocab[j]
            
            result[-1].append(get_tf(t, d))
            
    tf_ = pd.DataFrame(result, columns = vocab)
    return tf_

def process_idf(vocab):
    result = []
    for j in range(len(vocab)):
        t = vocab[j]
        result.append(get_idf(t))
    
    idf_ = pd.DataFrame(result, index = vocab, columns=["IDF"])
    return idf_


def process_tfidf(vocab, N):
    result = []
    for i in range(N):
        result.append([])
        d = docs[i]
        for j in range(len(vocab)):
            t = vocab[j]
            
            result[-1].append(get_tfidf(t,d))
    
    tfidf_ = pd.DataFrame(result, columns = vocab)
    return tfidf_

from sklearn.feature_extraction.text import TfidfVectorizer

def process_tfidf_sklearn(corpus):
    tfidfv = TfidfVectorizer().fit(corpus)
    vocab = list(tfidfv.vocabulary_.keys()) # 단어장을 리스트로 저장
    vocab.sort() # 단어장을 알파벳 순으로 정렬
    
    # TF-IDF 행렬에 단어장을 데이터프레임의 열로 지정하여 데이터프레임 생성
    tfidf_ = pd.DataFrame(tfidfv.transform(corpus).toarray(), columns = vocab)
    return tfidf_
