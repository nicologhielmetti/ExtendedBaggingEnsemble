# ExtendedBaggingEnsemble
This library is an extension of BaggingClassifier and BaggingRegressor of Scikit-learn that allows the creation of ensembles composed by different kind of models.
Up to now, ExtendedBaggingRegressor is not implemented but it should be easy to adapt the code from classifier to obtain the regressor.
## Example
```python
custom_bagging = CustomBaggingClassifier(verbose=True, parallel=True)

    params = {
        'C': np.linspace(0.01, 4, num=50),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    custom_bagging.add_models(LogisticRegression, params)

    params = {
        'max_depth': list(range(1, 200, 1)),
        'criterion': ['gini', 'entropy'],
    }

    custom_bagging.add_models(DecisionTreeClassifier, params)

    custom_bagging.add_model(GaussianNB())

    params = {
        'C': np.linspace(0.01, 4, num=100),
        'kernel': ["linear", "poly", "rbf", "sigmoid"]
    }

    n_models = custom_bagging.add_models(SVC, params)
```
With the code above you create approximately 1000 total models taken from LogisticRegression, DecisionTreeClassifier, GaussianNB, SVC.
Varing the `params` dict you can decide how many models use in the ensemble and which kind to use.
Once you added all the models you want just follow the same flow as another Scikit-learn model:
```python
  custom_bagging.fit(X, y)
```
`X` and `y` have to be pandas dataframes.
To get the score just use:
```python
custom_bagging.score(X, y)
```
Please, check the implementations for more details and create an issue to discuss about problem or possible improvements.
