import sys
import os 

if __name__ == '__main__':
	if __package__ is None:
		# upper_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		abs_path = os.path.dirname(os.path.dirname(os.path.abspath('.')))
		sys.path.append(abs_path)

import kor_nlp as kn

def run_okt():
    print("okt tokenizer:\n")
    text = '모두의연구소에서 자연어 처리를 공부하는 건 정말 즐거워'
    print(kn.okt_tokenize(text))
	
def run_soynlp():
    print("\nsoynlp:\n")
    print("\nloading data...\n")
    corpus = kn.load_soynlp_tutorials()
    kn.check_corpus(corpus)

    print("\nmaking word score table...\n")
    word_score_table = kn.get_word_score_table(corpus)

    text = "반포한강공원에"
    print("\ncohesion probability:\n")
    kn.cohesion_probability(word_score_table, text)

    text = "디스플레이"
    print("\nbranching entropy:\n")
    kn.branching_entropy(word_score_table, text)

    scores = kn.get_scores(word_score_table)
    text = "국제사회와 우리의 노력들로 범죄를 척결하자"

    print("\nLTokenizer:\n")
    print(kn.l_tokenize(scores, text))

    print("\nMaxScoreTokenizer:\n")
    print(kn.max_tokenize(scores, text))

run_okt()
run_soynlp()