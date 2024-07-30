# from gensim.models import Word2Vec
# from nltk.corpus import abc

# corpus = abc.sents()
# model = Word2Vec(sentences = corpus, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)


import re
import nltk

from konlpy.tag import Okt
from collections import Counter
from nltk.corpus import abc

import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.models import FastText

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

# def one_hot_encoding(word, word2index):
#     one_hot_vector = [0]*(len(word2index))
#     index = word2index[word]
#     one_hot_vector[index-1] = 1
#     return one_hot_vector


# text = "임금님 귀는 당나귀 귀! 임금님 귀는 당나귀 귀! 실컷~ 소리치고 나니 속이 확 뚫려 살 것 같았어."

# # re
# reg = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]")
# text = reg.sub('', text)
# print("정규식표현 처리 후 결과:\n", text)

# # okt
# okt=Okt()
# tokens = okt.morphs(text)
# print("\nOkt 토큰화:\n", tokens)

# vocab = Counter(tokens)
# print("\n단어장:\n", vocab)
# print("'임금님' 빈도수: ", vocab['임금님'])

# vocab_size = 5
# vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
# print("\n빈도수 상위 5개: ", vocab)

# word2idx={word[0] : index+1 for index, word in enumerate(vocab)}
# print("\n인덱스 처리: ", word2idx)

# one_hot = one_hot_encoding("임금님", word2idx)
# print("\n원-핫 벡터: ", one_hot)

# # keras tokenizer

# text = [['강아지', '고양이', '강아지'],['애교', '고양이'], ['컴퓨터', '노트북']]
# print("\n3개 문서:\n", text)

# t = Tokenizer()
# t.fit_on_texts(text)
# print("\n인코딩 결과:\n", t.word_index) # 각 단어에 대한 인코딩 결과 출력.

# vocab_size = len(t.word_index) + 1

# sub_text = ['강아지', '고양이', '강아지', '컴퓨터']
# encoded = t.texts_to_sequences([sub_text])
# print("\nkeras 토큰화: \n", encoded)

# one_hot = to_categorical(encoded, num_classes = vocab_size)
# print("\nkeras 원-핫 벡터: \n", one_hot)

# Word Embedding

# nltk.download('abc')
# nltk.download('punkt')

def print_items(items:list):
    for item in items:
        print(f"{item[0]:>20}: {item[1]:.3f}")

def most_similar(model, model_name, word):
    result = model.most_similar(word)
    print(f"\n{model_name} - '{word}' 유사한 단어:")
    print_items(result)

corpus = abc.sents()
print("nltk 'abc' 코퍼스의 크기 :", len(corpus))

model = Word2Vec(sentences = corpus, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
most_similar(model.wv, 'Word2Vec', 'man')

file_path = './w2v'
model.wv.save_word2vec_format(file_path) 
loaded_model = KeyedVectors.load_word2vec_format(file_path)
most_similar(loaded_model, 'Word2Vec_Saved', 'memory')

# FastText

fasttext_model = FastText(corpus, window=5, min_count=5, workers=4, sg=1)
most_similar(fasttext_model.wv, 'FastText', 'memory')
most_similar(fasttext_model.wv, 'FastText', 'memoryy')

# GloVe

glove_model = api.load("glove-wiki-gigaword-50")  # glove vectors 다운로드
most_similar(glove_model, 'GloVe', 'dog')
most_similar(glove_model, 'GloVe', 'overacting')
