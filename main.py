from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from ModelSelection.CustomBaggingClassifier import CustomBaggingClassifier

df = load_breast_cancer(as_frame=True)
dfc = df.frame.copy()
train = dfc.sample(frac=0.8)
test  = dfc.drop(train.index)
X_test = test.iloc[:, 0:30]
y_test = test.target
print("Test size: ", test.shape[0])
print("Train size: ", train.shape[0])

# scale is not working, still in development
custom_bagging = CustomBaggingClassifier(train, verbose=False, scale=False)
custom_bagging.add_model('dt5', DecisionTreeClassifier(max_depth=5))
custom_bagging.add_model('dt10', DecisionTreeClassifier(max_depth=10))
custom_bagging.add_model('dt15', DecisionTreeClassifier(max_depth=15))
custom_bagging.add_model('dt30', DecisionTreeClassifier(max_depth=30))
custom_bagging.add_model('dt50', DecisionTreeClassifier(max_depth=50))
custom_bagging.add_model('dt75', DecisionTreeClassifier(max_depth=75))
custom_bagging.add_model('dt100', DecisionTreeClassifier(max_depth=100))
custom_bagging.add_model('nb', GaussianNB())
custom_bagging.add_model('lr1e5', LogisticRegression(C=1e5))
custom_bagging.add_model('lr1e3', LogisticRegression(C=1e3))
custom_bagging.add_model('lr1', LogisticRegression(C=1))
custom_bagging.add_model('lr10', LogisticRegression(C=10))
custom_bagging.train_models()
performances = custom_bagging.get_models_oob_score()
for performance in performances:
    print("Model %s OOB accuracy: %.10f" % (performance[0], performance[1]))
best_model = custom_bagging.get_best_model()
print("Best model %s with %.10f OOB accuracy" % (best_model[0], best_model[1]))
# print("Ensemble model OOB accuracy: %.10f" % custom_bagging.get_ensemble_oob_score())
print("Ensemble test accuracy: %.10f" % np.mean(y_test == custom_bagging.predict(X_test)))

