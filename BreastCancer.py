# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:55:18 2024

@author: pchri
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn as skl


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

breast = pd.read_csv(r'C:\Users\pchri\Downloads\Cancer_Data.csv')
print(breast)
del breast["id"]
print(breast)

print(breast.shape)

del breast["Unnamed: 32"]

print(breast.shape)
print(breast)

#descriptions
print(breast.describe())

#class distribution
print(breast.groupby('diagnosis').size())

breast = breast.replace({"M": 1, "B": 0})
breast

#Split-out validation dataset
array = breast.values
X = array[:,0:30]
Y = array[:,0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=(seed))

#Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed, shuffle=(True))
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#CART prediction
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions1 = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictions1))
print(confusion_matrix(Y_validation, predictions1))
print(classification_report(Y_validation, predictions1))

#NB prediction
NB = GaussianNB()
NB.fit(X_train, Y_train)
predictions2 = NB.predict(X_validation)
print(accuracy_score(Y_validation, predictions2))
print(confusion_matrix(Y_validation, predictions2))
print(classification_report(Y_validation, predictions2))
