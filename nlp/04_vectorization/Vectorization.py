from numpy import dot
from numpy.linalg import norm
from math import log

from tensorflow.keras.preprocessing.text import Tokenizer

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
    tokenizer.fit_on_texts(text) # 단어장 생성
    bow = dict(tokenizer.word_counts) # 각 단어와 각 단어의 빈도를 bow에 저장
    
    print("Bag of Words :", bow) # bow 출력
    print('단어장(Vocabulary)의 크기 :', len(tokenizer.word_counts)) # 중복을 제거한 단어들의 개수


def bag_of_words2(text:list) -> None:
    vector = CountVectorizer()
    bow = vector.fit_transform(text).toarray()
    
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


class TFIDF:

    def __init__(self, documents):
        self.docs = documents
        self.make_vocab()


    def make_vocab(self):
        self.vocab = list(set(w for doc in self.docs for w in doc.split()))
        self.vocab.sort()
        self.docs_n = len(self.docs)
        self.vocab_n = len(self.vocab)
        
    
    def get_tf(self, t, d):
        return d.count(t)
    
    def process_tf(self):
        result = []
        for i in range(self.docs_n): # 각 문서에 대해서 아래 명령을 수행
            result.append([])
            d = self.docs[i]
            for j in range(self.vocab_n):
                t = self.vocab[j]
                
                result[-1].append(self.get_tf(t, d))
                
        self.tf = pd.DataFrame(result, columns=self.vocab)
        return self.tf
    
    
    def get_idf(self, t):
        df = 0
        for doc in self.docs:
            df += t in doc    
        return log(self.docs_n / (df + 1)) + 1
    
    def process_idf(self):
        result = []
        for j in range(self.vocab_n):
            t = self.vocab[j]
            result.append(self.get_idf(t))
        
        self.idf = pd.DataFrame(result, index = self.vocab, columns=["IDF"])
        return self.idf
    
    
    def get_tfidf(self, t, d):
        return self.get_tf(t, d) * self.get_idf(t)
    
    def process_tfidf(self):
        result = []
        for i in range(self.docs_n):
            result.append([])
            d = self.docs[i]
            for j in range(self.vocab_n):
                t = self.vocab[j]
                
                result[-1].append(self.get_tfidf(t, d))
        
        self.tfidf = pd.DataFrame(result, columns=self.vocab)
        return self.tfidf
    
    
    def process_tfidf_sklearn(self):
        tfidfv = TfidfVectorizer().fit(self.docs)
        self.vocab_skl = list(tfidfv.vocabulary_.keys()) # 단어장을 리스트로 저장
        self.vocab_skl.sort() # 단어장을 알파벳 순으로 정렬
        
        # TF-IDF 행렬에 단어장을 데이터프레임의 열로 지정하여 데이터프레임 생성
        self.tfidf_skl = pd.DataFrame(tfidfv.transform(self.docs).toarray(), columns = self.vocab_skl)
        return self.tfidf_skl
    


