import os
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import seaborn as sns

def cos_sim(i, j):
    return dot(i, j.T)/(norm(i)*norm(j))

def s(w, A, B):
    c_a = cos_sim(w, A)
    c_b = cos_sim(w, B)
    mean_A = np.mean(c_a, axis=-1)
    mean_B = np.mean(c_b, axis=-1)
    return mean_A - mean_B #, c_a, c_b

def weat_score(X, Y, A, B):
    
    s_X = s(X, A, B)
    s_Y = s(Y, A, B)

    mean_X = np.mean(s_X)
    mean_Y = np.mean(s_Y)
    
    std_dev = np.std(np.concatenate([s_X, s_Y], axis=0))
    
    return  (mean_X-mean_Y)/std_dev

def read_token(file_name):
    okt = Okt()
    result = []
    with open(abs_path + '//data//synopsis//' + file_name, 'r', encoding='utf-8') as fread: 
        print(file_name, '파일을 읽고 있습니다.')
        while True:
            line = fread.readline() 
            if not line: break 
            tokenlist = okt.pos(line, stem=True, norm=True) 
            for word in tokenlist:
                if word[1] in ["Noun"]:#, "Adjective", "Verb"]:
                    result.append((word[0])) 
    return ' '.join(result)


def plot_heatmap(matrix, labels):
    np.random.seed(0)

    plt.figure(figsize=[10,10])
    # 한글 지원 폰트
    # font_name = 'NanumGothic'
    font_name = 'Malgun Gothic'
    # sns.set(font='NanumGothic')
    plt.rcParams['font.family'] = font_name
    
    # 마이너스 부호 
    
    plt.rcParams['axes.unicode_minus'] = False
    
    ax = sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True,  cmap='RdYlGn_r')
    return ax


class SynopsisWEAT:

    def set_model(self, model):
        self.model = model

    def make_target(self, target):
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(target)
    
        self.m1 = self.X[0].tocoo()   # art를 TF-IDF로 표현한 sparse matrix를 가져옵니다. 
        self.m2 = self.X[1].tocoo()   # gen을 TF-IDF로 표현한 sparse matrix를 가져옵니다. 
        
        self.w1 = [[i, j] for i, j in zip(self.m1.col, self.m1.data)]
        self.w2 = [[i, j] for i, j in zip(self.m2.col, self.m2.data)]
        
        self.w1.sort(key=lambda x: x[1], reverse=True)   #art를 구성하는 단어들을 TF-IDF가 높은 순으로 정렬합니다. 
        self.w2.sort(key=lambda x: x[1], reverse=True)   #gen을 구성하는 단어들을 TF-IDF가 높은 순으로 정렬합니다. 

    def print_top_words(self):
        print('Target X:')  # 예술영화
        for i in range(100):
            print(self.vectorizer.get_feature_names_out()[self.w1[i][0]], end=', ')
        
        print('\n')
            
        print('Target Y:')  # 일반영화
        for i in range(100):
            print(self.vectorizer.get_feature_names_out()[self.w2[i][0]], end=', ')

    def make_excluded_words(self):
        
        n = 15
        self.w1_, self.w2_ = [], []
        for i in range(100):
            self.w1_.append(self.vectorizer.get_feature_names_out()[self.w1[i][0]])
            self.w2_.append(self.vectorizer.get_feature_names_out()[self.w2[i][0]])
        
        # w1에만 있고 w2에는 없는, 예술영화를 잘 대표하는 단어를 15개 추출한다.
        self.target_x, self.target_y = [], []
        for i in range(100):
            if (self.w1_[i] not in self.w2_) and (self.w1_[i] in self.model.wv):
                self.target_x.append(self.w1_[i])
            if len(self.target_x) == n: break 
        
        # w2에만 있고 w1에는 없는, 일반영화를 잘 대표하는 단어를 15개 추출한다.
        for i in range(100):
            if (self.w2_[i] not in self.w1_) and (self.w2_[i] in self.model.wv):
                self.target_y.append(self.w2_[i])
            if len(self.target_y) == n: break

    def make_attributes(self, genre, genre_name):
        self.genre = genre
        self.genre_name = genre_name
        
        self.X_genre = self.vectorizer.fit_transform(genre)
        self.m = [self.X_genre[i].tocoo() for i in range(self.X_genre.shape[0])]
        self.w = [[[i, j] for i, j in zip(mm.col, mm.data)] for mm in self.m]
        
        for i in range(len(self.w)):
            self.w[i].sort(key=lambda x: x[1], reverse=True)
            
        self.attributes = []
        for i in range(len(self.w)):
            # print(genre_name[i], end=': ')
            attr = []
            j = 0
            while (len(attr) < 15):
                if self.vectorizer.get_feature_names_out()[self.w[i][j][0]] in self.model.wv:
                    attr.append(self.vectorizer.get_feature_names_out()[self.w[i][j][0]])
                    # print(self.vectorizer.get_feature_names_out()[self.w[i][j][0]], end=', ')
                j += 1
            self.attributes.append(attr)

    def make_matrix(self):
        self.n_genre_name = len(self.genre_name)
        self.matrix = [[0 for _ in range(self.n_genre_name)] for _ in range(self.n_genre_name)]

        self.X = np.array([self.model.wv[word] for word in self.target_x])
        self.Y = np.array([self.model.wv[word] for word in self.target_y])
        
        for i in range(self.n_genre_name-1):
            for j in range(i+1, self.n_genre_name):
                self.A = np.array([self.model.wv[word] for word in self.attributes[i]])
                self.B = np.array([self.model.wv[word] for word in self.attributes[j]])
                self.matrix[i][j] = weat_score(self.X, self.Y, self.A, self.B)

    def print_attributes(self):
        for i in range(self.n_genre_name-1):
            for j in range(i+1, self.n_genre_name):
                print(self.genre_name[i], self.genre_name[j], self.matrix[i][j])



