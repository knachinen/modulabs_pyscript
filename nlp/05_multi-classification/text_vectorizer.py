from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class TextVectorizer():

    def __init__(self) -> None:
        pass


    def set_data(self, data_samples):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data_samples
        

    def load_reuters(self, num_words=10000, test_split=0.2):
        return reuters.load_data(
            num_words=num_words,
            test_split=test_split
            )
        

    def make_word_index(self):
        self.word_index = reuters.get_word_index(path="reuters_word_index.json")
        self.index_to_word = { index+3 : word for word, index in self.word_index.items() }

        # index_to_word에 숫자 0은 <pad>, 숫자 1은 <sos>, 숫자 2는 <unk>를 넣어줍니다.
        for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
            self.index_to_word[index]=token


    def get_decoded(self, sample):
        decoded = []
        for i in range(len(sample)):
            t = ' '.join([self.index_to_word[index] for index in sample[i]])
            decoded.append(t)
        return decoded
    

    def make_decoded(self):
        self.x_train_decoded = self.get_decoded(self.x_train)
        self.x_test_decoded = self.get_decoded(self.x_test)


    def vectorize(self):
        self.dtmvector = CountVectorizer()
        self.x_train_dtm = self.dtmvector.fit_transform(self.x_train_decoded)
        self.tfidf_transformer = TfidfTransformer()
        self.tfidfv_train = self.tfidf_transformer.fit_transform(self.x_train_dtm)
        self.x_test_dtm = self.dtmvector.transform(self.x_test_decoded) #테스트 데이터를 DTM으로 변환
        self.tfidfv_test = self.tfidf_transformer.transform(self.x_test_dtm) #DTM을 TF-IDF 행렬로 변환


    def process(self, data_samples=None):
        if isinstance(data_samples, tuple):
            self.set_data(data_samples)
        else:
            self.set_data(self.load_reuters())
        self.make_word_index()
        self.make_decoded()
        self.vectorize()


    def print_class_numbers(self):
        num_classes = max(self.y_train) + 1
        print('클래스의 수 : {}'.format(num_classes))


    def print_sample_length(self):
        print('훈련용 뉴스의 최대 길이 :{}'.format(max(len(l) for l in self.x_train)))
        print('훈련용 뉴스의 평균 길이 :{}'.format(sum(map(len, self.x_train))/len(self.x_train)))


    def print_elements(self):
        unique_elements, counts_elements = np.unique(self.y_train, return_counts=True)
        print("각 클래스 빈도수:")
        print(np.asarray((unique_elements, counts_elements)))


    def plot_samples_histogram(self, image_filename='sample_histogram.png'):
        plt.hist([len(s) for s in self.x_train], bins=50)
        plt.xlabel('length of samples')
        plt.ylabel('number of samples')
        plt.savefig(image_filename)


    def plot_samples_countplot(self, image_filename='sample_countplot.png'):
        fig, axe = plt.subplots(ncols=1)
        fig.set_size_inches(11,5)
        sns.countplot(x=self.y_train)
        plt.savefig(image_filename)


