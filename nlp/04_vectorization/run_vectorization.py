import Vectorization as vec

def run_bag_of_words(sentence):
    print("Bag of Words:\n")
    vec.bag_of_words1(sentence)
    vec.bag_of_words2(sentence)


def run_dtm(corpus):
    print("\nDTM:\n")
    vec.dtm(corpus)


def run_tfidf(corpus):
    print("\nTF-IDF:\n")
    tfidf = vec.TFIDF(corpus)

    print('\n- 문서 :\n', corpus)
    print('\n- 총 문서의 수 :', tfidf.docs_n)
    print('\n- 단어장 :\n', tfidf.vocab)
    print('\n- 단어장의 크기 :', tfidf.vocab_n)

    print('\nTF :\n', tfidf.process_tf())
    print('\nIDF :\n', tfidf.process_idf())
    print('\nTF-IDF :\n', tfidf.process_tfidf())
    print('\nTF-IDF (sklearn) :\n', tfidf.process_tfidf_sklearn())

def run_process():
    sentence = ["John likes to watch movies. Mary likes movies too! Mary also likes to watch football games."]
    corpus = [
        'John likes to watch movies',
        'Mary likes movies too',
        'Mary also likes to watch football games',    
    ]

    run_bag_of_words(sentence)
    print("\ncosine similarity:\n")
    vec.make_cos_sim()
    run_dtm(corpus)
    run_tfidf(corpus)

run_process()