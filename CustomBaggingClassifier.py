from mlxtend.classifier import EnsembleVoteClassifier

from ModelSelection.BaseModel import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class CustomBaggingClassifier:
    def __init__(self, df, voting="hard", verbose=False, scale=False):
        self.models = []
        self.df = df
        self.voting = voting
        self.predictions = []
        self.votingClassifier = None
        self.verbose = verbose
        if scale:
            self.scale()

    def scale(self):
        '''
        Scale input data by using StandardScaler
        :return: ---
        '''
        scaler = StandardScaler()
        self.df.iloc[:, 0:30] = scaler.fit_transform(self.df.iloc[:, 0:30])

    def add_model(self, name, model):
        '''
        Add a model to the ensemble
        :param name: name of the model
        :param model: instance of the model
        :return: ---
        '''
        if self.votingClassifier is not None:
            self.votingClassifier = None
        train_set, oob_set = self.__generate_bootstrap_sample()
        self.models.append(BaseModel(name, model, train_set, oob_set))

    def train_models(self):
        '''
        Train all the models in the ensemble
        :return: ---
        '''
        for model in self.models:
            X, y = model.get_trainset_X_y()
            model.model.fit(X, y)

    def predict_each_model(self, X):
        '''
        Perform a prediction for each model in the ensemble. NOTE! train_models() must be called before getting the
        predictions
        :param X: input values to be used for predictions
        :return: list of predictions with model name associated
        '''
        predictions = []
        for model in self.models:
            predictions.append((model.name, model.model.predict(X)))
        return predictions

    def predict(self, X):
        '''
        Perform a prediction considering the models as an ensemble. NOTE! train_models() must be called before getting the
        predictions
        :param X: input values to be used for predictions
        :return: list of predictions with model name associated
        '''
        if self.votingClassifier is not None:
            return self.votingClassifier.predict(X)
        base_models = []
        for model in self.models:
            base_models.append(model.model)
        self.votingClassifier = EnsembleVoteClassifier(clfs=base_models, voting=self.voting, refit=False)
        self.votingClassifier.fit(self.df.iloc[:, 0:30], self.df.target)
        return self.votingClassifier.predict(X)

    def get_ensemble_oob_score(self):
        '''
        !!!---EXPERIMENTAL---!!! Get the OOB score of the ensemble.
        :return: the accuracy over the OOB set of the ensemble
        '''
        predictions = []
        ys = []
        for i, row in self.df.iterrows():
            clfs = []
            for model in self.models:
                check_oob = model.oobset.append(row)
                if model.oobset.shape == check_oob.drop_duplicates(keep="first").shape:
                    clfs.append(model.model)
            if self.verbose is True:
                print("Usable models for OOB score ensemble: ", len(clfs))
            if len(clfs) == 0:
                continue
            voting_ensemble = EnsembleVoteClassifier(clfs=clfs, voting=self.voting, refit=False)
            voting_ensemble.fit(self.df.iloc[:, 0:30], self.df.target)
            predictions.append(voting_ensemble.predict(row.to_frame().T.iloc[:, 0:30])[0])
            ys.append(int(row.target))
        return 1 - np.mean(ys == predictions)

    def get_models_oob_score(self):
        '''
        Computes the OOB score for each model in the ensemble
        :return: list of OOB scores, one for each model in the ensemble
        '''
        oob_scores = []
        for model in self.models:
            X_oob, y_oob = model.get_oob_set_X_y()
            oob_scores.append((model.name, np.mean(y_oob == model.model.predict(X_oob))))
        return oob_scores

    def __ret_accuracy(self, array):
        return array[1]

    def get_best_model(self):
        '''
        Find the best model comparing performances over OOB set
        :return: the model with the best OOB score
        '''
        performances = self.get_models_oob_score()
        performances.sort(key=self.__ret_accuracy, reverse=True)
        return performances[0]

    def __generate_bootstrap_sample(self):
        df_boot = pd.DataFrame.sample(self=self.df, n=self.df.shape[0], replace=True)
        oob = pd.concat([df_boot, self.df]).drop_duplicates(keep=False)
        if self.verbose is True:
            print("OOB set size: %.2f" % float(oob.shape[0]/df_boot.shape[0]*100), "%")
            print("OOB set abs.:   %i" % oob.shape[0])
        return df_boot, oob