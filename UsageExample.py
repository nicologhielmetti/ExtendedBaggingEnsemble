from functools import reduce

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from ModelSelection.CustomBaggingClassifier import CustomBaggingClassifier

# df = load_breast_cancer(as_frame=True)
df = load_wine(as_frame=True)
dfc = df.frame.copy()
train = dfc.sample(frac=0.8)
test  = dfc.drop(train.index)
# features_range = range(30)
features_range = range(13)
X_test = test.iloc[:, features_range]
y_test = test.target
print("Test size: ", test.shape[0])
print("Train size: ", train.shape[0])

custom_bagging = CustomBaggingClassifier(verbose=True, parallel=True)
custom_bagging.add_model(DecisionTreeClassifier(max_depth=5))
custom_bagging.add_model(DecisionTreeClassifier(max_depth=10))
custom_bagging.add_model(DecisionTreeClassifier(max_depth=15))
custom_bagging.add_model(DecisionTreeClassifier(max_depth=30))
custom_bagging.add_model(DecisionTreeClassifier(max_depth=50))
custom_bagging.add_model(DecisionTreeClassifier(max_depth=75))
custom_bagging.add_model(DecisionTreeClassifier(max_depth=100))
custom_bagging.add_model(GaussianNB())
custom_bagging.add_model(LogisticRegression(C=1e5, solver="liblinear"))
custom_bagging.add_model(LogisticRegression(C=1e3, solver="liblinear"))
custom_bagging.add_model(LogisticRegression(C=1, solver="liblinear"))
custom_bagging.add_model(LogisticRegression(C=10, solver="liblinear"))
custom_bagging.commit_models()
custom_bagging.train_models()
performances = custom_bagging.get_models_oob_score()
performances_val = []
for performance in performances:
    print("Model %s OOB accuracy: %.10f" % (performance[0], performance[1]))
    performances_val.append(performance[1])
best_model = custom_bagging.get_best_model()
print("Best model %s with %.10f OOB accuracy" % (best_model[0], best_model[1]))
print("Average performance of the models over the OOB set: %.10f" % (float(reduce(lambda a, b: a + b, performances_val)) / len(performances_val)))
print("Ensemble test accuracy: %.10f" % np.mean(y_test == custom_bagging.predict(X_test)))

