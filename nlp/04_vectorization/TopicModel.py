from utils.file_util import get_abs_path, check_file_exists, check_directory

import pandas as pd
import numpy as np
import urllib

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class BaseTopicModel:

    def __init__(self, data):
        self.data = data
    
    
    def init_vectorizer(self, max_features=5000):
        pass
        

    def vectorize(self):
        self.doc_term_matrix = self.vectorizer.fit_transform(self.data)


    def set_model(self):
        pass
        

    def fit(self):
        self.model.fit_transform(self.doc_term_matrix)
        

    def make_terms(self):
        self.feature_names = self.vectorizer.get_feature_names_out() # 단어 집합. 5,000개의 단어가 저장됨.


    def make_top_topics(self, n=5):
        self.top_topics = []
        for idx, topic in enumerate(self.model.components_):
            self.top_topics.append(
                [(self.feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])

    
    def print_topics(self):
        for idx, topic in enumerate(self.top_topics):
            print("Topic %d:" % (idx+1), topic)


    def process(self):
        self.init_vectorizer()
        self.vectorize()
        self.set_model()
        self.fit()
        self.make_terms()
        self.make_top_topics()
        self.print_topics()


class LSA(BaseTopicModel):

    def init_vectorizer(self, max_features=5000):
        # 상위 5000개의 단어만 사용
        self.vectorizer = CountVectorizer(
            stop_words='english', 
            max_features=max_features)


    def set_model(self, n_topics=10):
        self.model = TruncatedSVD(n_components=n_topics)


class LDA(BaseTopicModel):

    def init_vectorizer(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_features)


    def set_model(self, n_topics=10):
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            learning_method='online',
            random_state=777,
            max_iter=1)


def load_abcnews():
    abs_path = get_abs_path()
    data_path = f'{abs_path}\\data'
    csv_filename = f'{data_path}\\abcnews-date-text.csv'
    if check_file_exists(csv_filename) is False:
        check_directory(data_path)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv", 
            filename=csv_filename)
    data = pd.read_csv(csv_filename, on_bad_lines='skip')
    return data


def text_preprocess(data):
    text = data[['headline_text']].copy()
    text.nunique() # 중복을 제외하고 유일한 시퀀스를 가지는 샘플의 개수를 출력
    text.drop_duplicates(inplace=True) # 중복 샘플 제거
    text.reset_index(drop=True, inplace=True)
    
    # NLTK 토크나이저를 이용해서 토큰화
    text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1)
    
    # 불용어 제거
    stop_words = stopwords.words('english')
    text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop_words)])

    # 단어 정규화. 3인칭 단수 표현 -> 1인칭 변환, 과거형 동사 -> 현재형 동사 등을 수행한다.
    text['headline_text'] = text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
    
    # 길이가 1 ~ 2인 단어는 제거.
    text = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 2])
    return text


def detokenize(text):
    # 역토큰화 (토큰화 작업을 역으로 수행)
    detokenized_doc = []
    for i in range(len(text)):
        t = ' '.join(text[i])
        detokenized_doc.append(t)
    
    train_data = detokenized_doc
    return train_data


