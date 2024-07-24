from text_vectorizer import TextVectorizer
from ml_models import MLModels
from ml_metrics import MLMetrics

tv = TextVectorizer()
tv.process()

tv.print_class_numbers()
tv.print_elements()
tv.print_sample_length()
tv.plot_samples_histogram()
tv.plot_samples_countplot()

mlm = MLModels()
mlm.set_model(mlm.get_MultinomialNB())
mlm.fit(tv.tfidfv_train, tv.y_train)
mlm.predict(tv.tfidfv_test)

mlmet = MLMetrics()
score = mlmet.get_accuracy_score(tv.y_test, mlm.pred)
print(score)

report = mlmet.get_report(tv.y_test, mlm.pred)
print(report)

mlmet.get_probability(mlm.model, tv.tfidfv_train[3])
mlmet.graph_confusion_matrix(tv.y_test, mlm.pred)
