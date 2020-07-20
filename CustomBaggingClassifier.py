from functools import partial
from random import randrange, randint

from mlxtend.classifier import EnsembleVoteClassifier

from ModelSelection.BaseModel import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import multiprocessing


class CustomBaggingClassifier:
    def __init__(self, df, features_range, voting="hard", verbose=False, scale=False, parallel=True):
        self.models = []
        self.temporary_models = []
        self.df = df
        self.features_range = features_range
        self.voting = voting
        self.predictions = []
        self.votingClassifier = None
        self.verbose = verbose
        self.parallel = parallel
        self.scale = scale
        if scale:
            self.rescale()

    def __get_models(self):
        base_models = []
        for model in self.models:
            base_models.append(model.model)
        return base_models

    def rescale(self):
        """
        Scale input data by using StandardScaler
        :return: ---
        """
        self.scaler = StandardScaler()
        self.df.iloc[:, self.features_range] = self.scaler.fit_transform(self.df.iloc[:, self.features_range])

    def add_model(self, name, model):
        """
        Add a model to the ensemble
        :param name: name of the model
        :param model: instance of the model
        :return: ---
        """
        if self.votingClassifier is not None:
            self.votingClassifier = None
        self.temporary_models.append((name, model))

    def add_single_model(self, temp_model):
        train_set, oob_set = self.__generate_bootstrap_sample()
        return BaseModel(temp_model[0], temp_model[1], train_set, oob_set, self.features_range)

    def commit_models(self):
        """
        Create the OOB set for each added model
        :return: ---
        """
        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            self.models = pool.map(self.add_single_model, self.temporary_models)
            pool.close()
            pool.join()
        else:
            for temp_model in self.temporary_models:
                train_set, oob_set = self.__generate_bootstrap_sample()
                self.models.append(BaseModel(temp_model[0], temp_model[1], train_set, oob_set, self.features_range))

    def train_single(self, single_model):
        X, y = single_model.get_trainset_X_y()
        return single_model.fit(X, y)

    def train_models(self):
        """
        Train all the models in the ensemble. It requires commit_models()
        :return: ---
        """

        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            self.models = pool.map(self.train_single, self.models)
            pool.close()
            pool.join()
        else:
            for model in self.models:
                self.train_single(model)

    def predict_single_model(self, X, model):
        return model.name, model.predict(X)

    def predict_each_model(self, X):
        """
        Perform a prediction for each model in the ensemble. NOTE! train_models() must be called before getting the
        predictions
        :param X: input values to be used for predictions
        :return: list of predictions with model name associated
        """

        predictions = []
        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            f = partial(self.predict_single_model, X)
            predictions = pool.map(f, self.models)
            pool.close()
            pool.join()
        else:
            for model in self.models:
                predictions.append(self.predict_single_model(model, X))
        return predictions

    def score(self, X, y):
        if self.votingClassifier is not None:
            if self.scale:
                X = self.scaler.fit_transform(X)
            return self.votingClassifier.score(X, y)

        self.votingClassifier = EnsembleVoteClassifier(clfs=self.__get_models(), voting=self.voting, refit=False)
        self.votingClassifier.fit(self.df.iloc[:, self.features_range], self.df.target)
        if self.scale:
            X = self.scaler.fit_transform(X)
        return self.votingClassifier.score(X, y)

    def predict(self, X):
        """
        Perform a prediction considering the models as an ensemble. NOTE! train_models() must be called before getting the
        predictions
        :param X: input values to be used for predictions
        :return: list of predictions with model name associated
        """
        if self.votingClassifier is not None:
            if self.scale:
                X = self.scaler.fit_transform(X)
            return self.votingClassifier.predict(X)

        self.votingClassifier = EnsembleVoteClassifier(clfs=self.__get_models(), voting=self.voting, refit=False)
        self.votingClassifier.fit(self.df.iloc[:, self.features_range], self.df.target)
        if self.scale:
            X = self.scaler.fit_transform(X)
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
            voting_ensemble.fit(self.df.iloc[:, self.features_range], self.df.target)
            predictions.append(voting_ensemble.predict(row.to_frame().T.iloc[:, self.features_range])[0])
            ys.append(int(row.target))
        return 1 - np.mean(ys == predictions)

    def get_single_oob(self, model):
        X_oob, y_oob = model.get_oob_set_X_y()
        return model.name, model.model.score(X_oob, y_oob)

    def get_models_oob_score(self):
        '''
        Computes the OOB score for each model in the ensemble
        :return: list of OOB scores, one for each model in the ensemble
        '''

        oob_scores = []
        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            oob_scores = pool.map(self.get_single_oob, self.models)
            pool.close()
            pool.join()
        else:
            for model in self.models:
                X_oob, y_oob = model.get_oob_set_X_y()
                oob_scores.append((model.name, model.model.score(X_oob, y_oob)))
        return oob_scores

    def __ret_accuracy(self, array):
        return array[1]

    def get_best_model(self):
        '''
        Find the best model comparing performances over OOB set
        :return: the model with the best OOB score
        '''
        performances = self.get_models_oob_score()
        performances.sort(key=self.__ret_accuracy, reverse=False)
        return performances.pop()

    def __generate_bootstrap_sample(self):
        df_boot = self.df.sample(n=self.df.shape[0], replace=True, random_state=randint(0, 10000))
        oob = pd.concat([df_boot, self.df]).drop_duplicates(keep=False)
        if self.verbose is True:
            print("OOB set size: %.2f" % float(oob.shape[0] / df_boot.shape[0] * 100), "%")
            print("OOB set abs.:   %i" % oob.shape[0])
        return df_boot, oob
