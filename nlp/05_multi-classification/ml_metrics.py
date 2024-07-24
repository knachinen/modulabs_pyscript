from sklearn.metrics import accuracy_score #정확도 계산
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class MLMetrics:

    def __init__(self) -> None:
        pass

    def get_accuracy_score(self, true_val, pred_val):
        self.score = accuracy_score(true_val, pred_val)
        return self.score
    

    def get_probability(self, model, sample, image_filename="probability.png"):
        probability = model.predict_proba(sample)[0]

        plt.rcParams["figure.figsize"] = (11,5)
        plt.bar(model.classes_, probability)
        plt.xlim(-1, 21)
        plt.xticks(model.classes_)
        plt.xlabel("Class")
        plt.ylabel("Probability")
        # plt.show()
        plt.savefig(image_filename)


    def get_report(self, y_test, y_pred, zero_division=0):
        self.report = classification_report(y_test, y_pred, zero_division=zero_division)
        return self.report
    
    
    def get_confusion_matrix(self, true_val, pred_val):
        self.confusion_matrix = confusion_matrix(true_val, pred_val)
        return self.confusion_matrix

    
    def graph_confusion_matrix(self, true_val, pred_val, image_filename='confusion_matrix.png'):#, classes_name):
        df_cm = pd.DataFrame(self.get_confusion_matrix(true_val, pred_val))#, index=classes_name, columns=classes_name)
        fig = plt.figure(figsize=(16,8))
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
        plt.ylabel('label')
        plt.xlabel('predicted value')
        plt.savefig(image_filename)
        

