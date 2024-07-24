from text_vectorizer import TextVectorizer
from ml_models import MLModels
from ml_metrics import MLMetrics
from save_file import string_to_file, check_directory

tv = TextVectorizer()
mlm = MLModels()
mlmet = MLMetrics()

num_samples = [5000, 10000, None]

for n in num_samples:
    print(f"- num_words: {n}")
    samples = tv.load_reuters(num_words=n, test_split=0.2)

    tv.process(data_samples=samples)

    models = []

    models.append({'model': mlm.get_MultinomialNB(), 'name': 'MultinomialNB'})
    models.append({'model': mlm.get_ComplementNB(), 'name': 'ComplementNB'})
    models.append({'model': mlm.get_LogisticRegression(), 'name': 'LogisticRegression'})
    models.append({'model': mlm.get_LinearSVC(), 'name': 'LinearSVC'})
    models.append({'model': mlm.get_DecisionTreeClassifier(), 'name': 'DecisionTree'})
    models.append({'model': mlm.get_RandomForestClassifier(), 'name': 'RandomForest'})
    models.append({'model': mlm.get_GradientBoostingClassifier(), 'name': 'GradientBoosting'})
    
    estimators = [
        ('LogisticRegression', mlm.get_LogisticRegression()),
        ('LinearSVC', mlm.get_LinearSVC())
    ]
    models.append({'model': mlm.get_VotingClassifier(estimators=estimators), 'name': 'Voting'})

    for item in models:
        print(f"  - model: {item['name']}", end=' ')
        
        mlm.set_model(item['model'])
        mlm.fit(tv.tfidfv_train, tv.y_train)
        mlm.predict(tv.tfidfv_test)
        
        score = mlmet.get_accuracy_score(tv.y_test, mlm.pred)
        item['score'] = score
        print(f"\t accuracy: {score:.3f}")
        
        report = mlmet.get_report(tv.y_test, mlm.pred)
        item['report'] = report
        
        dir_name = 'result'
        check_directory(dir_name)
        string_to_file(report, file_path=f"{dir_name}//n{n}_{item['name']}_report.txt")
    
        mlmet.get_probability(mlm.model, tv.tfidfv_train[3], image_filename=f"{dir_name}//n{n}_{item['name']}_probability")
        mlmet.graph_confusion_matrix(tv.y_test, mlm.pred, image_filename=f"{dir_name}//n{n}_{item['name']}_confusion_matrix")

    print()