from sklearn.naive_bayes import MultinomialNB #다항분포 나이브 베이즈 모델
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC



class MLModels:

    def __init__(self):
        pass

    def set_model(self, model):
        self.model = model

    
    def fit(self, x, y):
        self.model.fit(x, y)


    def predict(self, x):
        self.pred = self.model.predict(x)


    def process(self, x_train, y_train, x_test, model=None):
        self.set_model(model if model != None else self.get_MultinomialNB())
        self.fit(x_train, y_train)
        self.predict(x_test)
        

    def get_MultinomialNB(self):
        return MultinomialNB()
    

    def get_ComplementNB(self):
        return ComplementNB()


    def get_LogisticRegression(self):
        return LogisticRegression(C=10000, penalty='l2', max_iter=3000)
    

    def get_LinearSVC(self):
        return LinearSVC(C=1000, penalty='l1', max_iter=3000, dual=False)
    

    def get_DecisionTreeClassifier(self):
        return DecisionTreeClassifier(max_depth=10, random_state=0)
    

    def get_RandomForestClassifier(self):
        return RandomForestClassifier(max_depth=2, random_state=0)
    

    def get_GradientBoostingClassifier(self):
        return GradientBoostingClassifier(random_state=0) # verbose=3
    

    def get_VotingClassifier(self, estimators=None):
        if estimators == None:
            estimators = [
                ('MultinomialNB', self.get_MultinomialNB()),
                ('ComplimentNB', self.get_ComplementNB())
            ]
        return VotingClassifier(
            estimators=estimators,
            voting='hard'
        )
