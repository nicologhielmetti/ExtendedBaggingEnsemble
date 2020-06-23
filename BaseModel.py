class BaseModel:
    def __init__(self, name, model, trainset, oobset):
        self.name  = name
        self.model = model
        self.trainset = trainset
        self.oobset = oobset

    def get_oob_set_X_y(self):
        X = self.oobset.iloc[:, 0:30]
        y = self.oobset.target
        return X, y

    def get_trainset_X_y(self):
        X = self.trainset.iloc[:, 0:30]
        y = self.trainset.target
        return X, y
