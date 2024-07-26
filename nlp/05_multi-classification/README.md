## NLP 05. Multi-Classification

### modules

- **lstm.py**
	- loading reuters dataset
	- making lstm model and fitting
	- evaluating and plotting the result
- **ml_metrics.py**
	- accuracy
	- f1-score
	- confusion matrix
- **ml_models.py**: Classifiers
	- MultinomialNB
	- ComplementNB
	- LogisticRegression
	- LinearSVC
	- DecisionTree
	- RandomForest
	- GradientBoosting
	- Voting
- **save_file.py**
	- saving a result to a text file
- **text_vectorizer.py**
	- vectorize samples
	- loading reuters dataset

### run

- **run_process.py**
	- Processing the classification
	- three cases: num_words=[5000, 10000, None]
	- each 8 classifiers
- **run_text_vectorizer.py**
	- testing a classification (MultinomialNB)
- **run_lstm.py**
	- LSTM model for comparing with ML models