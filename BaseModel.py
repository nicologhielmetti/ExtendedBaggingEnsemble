class BaseModel:
    def __init__(self, name, model, trainset, oobset, features_range):
        self.name  = name
        self.model = model
        self.trainset = trainset
        self.oobset = oobset
        self.features_range = features_range

    def get_oob_set_X_y(self):
        X = self.oobset.iloc[:, self.features_range]
        y = self.oobset.target
        return X, y

    def get_trainset_X_y(self):
        X = self.trainset.iloc[:, self.features_range]
        y = self.trainset.target
        return X, y

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
