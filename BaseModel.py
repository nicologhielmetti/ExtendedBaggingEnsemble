class BaseModel:
    def __init__(self, name, model, trainset, oobset, target_name='target'):
        self.name  = name
        self.model = model
        self.trainset = trainset
        self.oobset = oobset
        self.target_name = target_name

    def get_oob_set_X_y(self):
        X = self.oobset.drop(self.target_name, axis=1)
        y = self.oobset.target
        return X, y

    def get_trainset_X_y(self):
        X = self.trainset.drop(self.target_name, axis=1)
        y = self.trainset.target
        return X, y

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
