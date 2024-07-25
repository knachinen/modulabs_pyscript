## TopicModel.py

- load_abcnews() : loading absnews dataset
- text_preprocess() : preprocessing
- detokenize() : detokenizing
- LSA : LSA class
- LDA : LDA class


## Vectorization.py

- Bag of Words
	- bag_of_words1() : keras tokenizer
	- bag_of_words2() : sklearn CountVectorizer
- DTM & cosine similarity
	- dtm(corpus) : DTM
	- cos_sim(A, B)
	- get_rand()
	- make_cos_sim() : cos similarity betwen documents
- TF-IDF
	- get_tf(t, d)
	- get_idf(t)
	- get_tfidf(vocab, N)
	- process_tf(vocab, N) : TF processing
	- process_idf(vocab) : IDF processing
	- process_tfidf(vocab, N) : TF-IDF processing
	- process_tfidf_sklearn(corpus) : sklearn TF-IDF vectorizer