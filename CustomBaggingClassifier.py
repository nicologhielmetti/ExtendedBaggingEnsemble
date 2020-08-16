import multiprocessing
import numbers
from functools import partial
from random import randint
import itertools

import numpy as np
import pandas as pd
from mlxtend.classifier import EnsembleVoteClassifier

from ModelSelection.BaseModel import BaseModel
from ModelSelection.BaseModelIdx import BaseModelIdx


class CustomBaggingClassifier:
    def __init__(self, voting="hard", verbose=False, parallel=True, target_name='target'):
        self.models = []
        self.temporary_models = []
        self.voting = voting
        self.predictions = []
        self.votingClassifier = None
        self.verbose = verbose
        self.parallel = parallel
        self.target_name = target_name

    def _get_models(self):
        base_models = []
        for model in self.models:
            base_models.append(model.model)
        return base_models

    def add_models(self, model, params):
        """
        Create all the possible combinations of the model with given parameters.
        Usage example:
            params = {
                'C': np.logspace(0, 4, num=10),
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }

            custom_bagging = CustomBaggingClassifier(verbose=True, parallel=True)
            custom_bagging.add_models(LogisticRegression, params)

        :param model: The name of the model (passed without calling the constructor) that is intended to be used
        :param params: key-value pairs of hyperparameters that will be used to generate all the possible models
        :return: the number of models of the ensemble
        """
        if self.votingClassifier is not None:
            self.votingClassifier = None
        keys = list(params)
        for values in itertools.product(*map(params.get, keys)):
            model_instance = model(**dict(zip(keys, values)))
            self.temporary_models.append((str(model_instance), model_instance))
        return len(self.temporary_models)

    def add_model(self, model):
        """
        Add a model to the ensemble
        :param model: instance of the model
        :return: ---
        """
        if self.votingClassifier is not None:
            self.votingClassifier = None
        self.temporary_models.append((str(model), model))

    def _commit_single_model(self, Xy, temp_model):
        """
        train_set, oob_set = self._generate_bootstrap_sample(Xy)
        return BaseModel(temp_model[0], temp_model[1], train_set, oob_set, self.target_name)
        """
        sampled_idx, unsampled_idx = self._generate_indexes(len(self.temporary_models), Xy.shape[0])
        return BaseModelIdx(temp_model[0], temp_model[1], sampled_idx, unsampled_idx, self.target_name)

    def _commit_models(self, X, y):
        """
        Create the OOB set for each added model
        """
        Xy = pd.concat([X, y], axis=1)
        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            f = partial(self._commit_single_model, Xy)
            self.models = pool.map(f, self.temporary_models)
            pool.close()
            pool.join()
        else:
            for temp_model in self.temporary_models:
                self.models.append(self._commit_single_model(Xy, temp_model))

    def _fit_single_model(self, X, y, single_model):
        return single_model.fit(X, y)

    def fit(self, X, y):
        """
        Train all the models in the ensemble.
        :return: ---
        """
        self._commit_models(X, y)
        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            f = partial(self._fit_single_model, X, y)
            self.models = pool.map(f, self.models)
            pool.close()
            pool.join()
        else:
            for model in self.models:
                self._fit_single_model(X, y, model)
        self.votingClassifier = EnsembleVoteClassifier(clfs=self._get_models(), voting=self.voting, refit=False)
        self.votingClassifier.fit(X, y)

    def _predict_single_model(self, X, model):
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
            f = partial(self._predict_single_model, X)
            predictions = pool.map(f, self.models)
            pool.close()
            pool.join()
        else:
            for model in self.models:
                predictions.append(self._predict_single_model(model, X))
        return predictions

    def score(self, X, y):
        return self.votingClassifier.score(X, y)

    def predict(self, X):
        """
        Perform a prediction considering the models as an ensemble. NOTE! train_models() must be called before getting the
        predictions
        :param X: input values to be used for predictions
        :return: list of predictions with model name associated
        """
        return self.votingClassifier.predict(X)

    def _get_single_oob(self, X, y, model):
        return model.name, model.score(X, y)

    def models_oob_score(self, X, y):
        '''
        Computes the OOB score for each model in the ensemble
        :return: list of OOB scores, one for each model in the ensemble
        '''

        oob_scores = []
        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            f = partial(self._get_single_oob, X, y)
            oob_scores = pool.map(f, self.models)
            pool.close()
            pool.join()
        else:
            for model in self.models:
                oob_scores.append((self._get_single_oob(X, y, model)))
        return oob_scores

    def _ret_accuracy(self, array):
        return array[1]

    def best_model(self, X, y):
        '''
        Find the best model comparing performances over OOB set
        :return: the model with the best OOB score
        '''
        performances = self.models_oob_score(X, y)
        performances.sort(key=self._ret_accuracy, reverse=False)
        return performances.pop()

    def _generate_bootstrap_sample(self, X):
        df_boot = X.sample(n=X.shape[0], replace=True, random_state=randint(0, 10000))
        oob = pd.concat([df_boot, X]).drop_duplicates(keep=False)
        if self.verbose is True:
            print("OOB set size: %.2f" % float(oob.shape[0] / df_boot.shape[0] * 100), "%")
            print("OOB set abs.:   %i" % oob.shape[0])
        return df_boot, oob

    def _generate_indexes(self, num_models, n_samples):
        rand_state = randint(0, num_models)
        sampled_idxs   = self._generate_sample_indices(rand_state, n_samples)
        unsampled_idxs = self._generate_unsampled_indices(rand_state, n_samples)
        return sampled_idxs, unsampled_idxs

    def _generate_unsampled_indices(self, random_state, n_samples):
        sample_indices = self._generate_sample_indices(random_state, n_samples)
        sample_counts = np.bincount(sample_indices, minlength=n_samples)
        unsampled_mask = sample_counts == 0
        indices_range = np.arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]
        return unsampled_indices

    def _generate_sample_indices(self, random_state, n_samples):
        random_instance = self._check_random_state(random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples)

        return sample_indices

    def _check_random_state(self, seed):
        if isinstance(seed, numbers.Integral):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
        raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                         ' instance' % seed)
