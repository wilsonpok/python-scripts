import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import metrics

clf = svm.SVC(kernel='linear', C=1)

iris = load_iris()

scores = cross_val_score(clf, iris.data, iris.target, cv=5)

scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')

scores                                              
