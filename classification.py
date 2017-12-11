#coding=utf8
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split as split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import tree

featurepath = 'output.csv'
labelpath = 'labels.csv'

'''
load and format feature and label matrices
'''
print '\nloading data...'

X = pd.read_csv(featurepath, header = None, skiprows = 1, usecols = range(0,69))
del X[0] # remove indices column
del X[2] # remove citation count column since we are testing the classes

y = pd.read_csv(labelpath, header = None, skiprows = 1) 
del y[0] # remove indices column

'''
random train and test set division
'''
print '\nsplitting data into training and test sets...'

Xtrain, Xtest, ytrain, ytest = split(X, y, test_size = 0.1) # 90/10 split
del X
del y

'''
fit the models
'''
print '\ntraining models...'

dtree = tree.DecisionTreeClassifier()
dtree.fit(Xtrain, ytrain.values.ravel()) # change shape of y to (nsample,)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(Xtrain, ytrain.values.ravel()) # change shape of y to (nsample,)

lr = linear_model.LogisticRegression() # default C is 1
lr.fit(Xtrain, ytrain.values.ravel()) # change shape of y to (nsample,)

'''
test the models
'''
print '\ntesting models...'

ypredicted_dtree = dtree.predict(Xtest)
ypredicted_knn = knn.predict(Xtest)
ypredicted_lr = lr.predict(Xtest)

'''
calculate diagnostics
'''
print '\ncalculating diagnostics...'

accuracy = sklearn.metrics.accuracy_score(ytest, ypredicted_dtree)
precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(ytest, ypredicted_dtree)

print '\nAccuracy = ', accuracy
print 'Precision = ', precision
print 'Recall = ', recall
print 'F1 Score = ', f1

accuracy = sklearn.metrics.accuracy_score(ytest, ypredicted_knn)
precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(ytest, ypredicted_knn)

print '\nAccuracy = ', accuracy
print 'Precision = ', precision
print 'Recall = ', recall
print 'F1 Score = ', f1

accuracy = sklearn.metrics.accuracy_score(ytest, ypredicted_lr)
precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(ytest, ypredicted_lr)

print '\nAccuracy = ', accuracy
print 'Precision = ', precision
print 'Recall = ', recall
print 'F1 Score = ', f1
