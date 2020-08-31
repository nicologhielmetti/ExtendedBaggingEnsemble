from functools import reduce

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ModelSelection.CustomBaggingClassifier import CustomBaggingClassifier

df = load_breast_cancer(as_frame=True)
# df = load_wine(as_frame=True)

features_range = range(30)
# features_range = range(13)

scaler = StandardScaler()
df.data.iloc[:, features_range] = scaler.fit_transform(df.data.iloc[:, features_range])

dfc = df.frame.copy()
train = dfc.sample(frac=0.8)
test  = dfc.drop(train.index)
X_train = train.drop('target', axis=1)
y_train = train.target
X_test  = test.drop('target', axis=1)
y_test  = test.target
print("Test size: ", test.shape[0])
print("Train size: ", train.shape[0])

custom_bagging = CustomBaggingClassifier(verbose=True, parallel=True)

params = {
    'C': np.linspace(0.01, 4, num=10),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
lr = custom_bagging.add_models(LogisticRegression, params)
print("Number of LR models: %i" % lr)

params = {
    'max_depth': list(range(1, 20, 1)),
    'criterion': ['gini', 'entropy'],
}

dt = custom_bagging.add_models(DecisionTreeClassifier, params) - lr
print("Number of DT models: %i" % dt)

nb = custom_bagging.add_model(GaussianNB()) - dt - lr
print("Number of NB models: %i" % nb)

params = {
    'C': np.linspace(0.01, 4, num=5),
    'kernel': ["linear", "poly", "rbf", "sigmoid"]
}

svm = custom_bagging.add_models(SVC, params) - nb - dt - lr
print("Number of SVM models: %i" % svm)
print("Total models: %i" % (svm + nb + dt + lr))
'''
custom_bagging.fit(X_train, y_train)
performances = custom_bagging.models_oob_score(X_train, y_train)
performances_val = []
for performance in performances:
    print("Model %s OOB accuracy: %.10f" % (performance[0], performance[1]))
    performances_val.append(performance[1])
best_model = custom_bagging.best_model(X_train, y_train)
print("Best model %s with %.10f OOB accuracy" % (best_model[0], best_model[1]))
print("Average performance of the models over the OOB set: %.10f" % (float(reduce(lambda a, b: a + b, performances_val))
                                                                     / len(performances_val)))
print("Ensemble test accuracy: %.10f" % custom_bagging.score(X_test, y_test))
'''
