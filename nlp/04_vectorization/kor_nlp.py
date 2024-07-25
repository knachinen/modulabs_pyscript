from konlpy.tag import Okt
from soynlp import DoublespaceLineCorpus
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from soynlp.tokenizer import MaxScoreTokenizer

def okt_tokenize(kor_text):
    tokenizer = Okt()
    return tokenizer.morphs(kor_text)


def load_soynlp_tutorials():
    abs_path = get_abs_path()
    data_path = f'{abs_path}\\data'
    file_name = f'{data_path}\\soynlp_tutorials_2016-10-20.txt'
    if check_file_exists(file_name) is False:
        check_directory(data_path)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", 
            filename=file_name)
    # 말뭉치에 대해서 다수의 문서로 분리
    corpus = DoublespaceLineCorpus(file_name)
    return corpus


def check_corpus(corpus, n=3):
    for index, document in enumerate(corpus):
        if len(document) > 0:
            print(f"[{index}]:\n {document}\n")
            n -= 1
        if n <= 0:
            break


def cohesion_probability(word_score_table, word):
    word_len = len(word)
    for i in range(word_len):
        text = word[:i+1]
        try:
            score = word_score_table[text].cohesion_forward
            print(f"{text:{word_len}}\t : {score}")
        except:
            print(f"{text:{word_len}}\t : KeyError")


def branching_entropy(word_score_table, word):
    word_len = len(word)
    for i in range(word_len):
        text = word[:i+1]
        fill = '_'
        try:
            score = word_score_table[text].right_branching_entropy
            print(f"{text:{fill}<{word_len+2}}\t : {score}")
        except:
            print(f"{text:{fill}<{word_len+2}}\t : KeyError")


def get_scores(word_score_table):
    scores = {word:score.cohesion_forward for word, score in word_score_table.items()}
    return scores


def l_tokenize(scores, text):    
    l_tokenizer = LTokenizer(scores=scores)
    return l_tokenizer.tokenize(text, flatten=False)


def max_tokenize(scores, text):
    maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
    return maxscore_tokenizer.tokenize(text)