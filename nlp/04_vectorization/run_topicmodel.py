import sys
import os 

if __name__ == '__main__':
	if __package__ is None:
		# upper_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		abs_path = os.path.dirname(os.path.dirname(os.path.abspath('.')))
		sys.path.append(abs_path)
		
import TopicModel as tm

def run_topicmodel():

    print("loading abc news dataset...")
    df_abc = tm.load_abcnews()
    print("text preprocessing...")
    text = tm.text_preprocess(df_abc)
    print("detokenizing...")
    train_data = tm.detokenize(text)

    print("LSA processing...")
    lsa = tm.LSA(train_data)
    lsa.process()

    print("LDA processing...")
    lsa = tm.LDA(train_data)
    lsa.process()

run_topicmodel()