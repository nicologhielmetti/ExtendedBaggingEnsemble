class BaseModelIdx:
    def __init__(self, name, model, train_idxs, oob_idxs, target_name='target'):
        self.name = name
        self.model = model
        self.train_idxs = train_idxs
        self.oob_idxs = oob_idxs
        self.target_name = target_name

    def get_oob_idxs(self):
        return self.oob_idxs

    def get_train_idxs(self):
        return self.train_idxs

    def fit(self, X, y):
        self.model.fit(X.iloc[self.train_idxs, :], y.iloc[self.train_idxs])
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X.iloc[self.oob_idxs, :], y.iloc[self.oob_idxs])
