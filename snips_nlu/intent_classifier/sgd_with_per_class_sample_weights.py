# coding=utf-8
from __future__ import unicode_literals

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.stochastic_gradient import fit_binary
from sklearn.utils import check_random_state
from sklearn.utils.fixes import _joblib_parallel_args

# In Python 2, sklearn does not contain the fix making the SGD deterministic
# and MAX_INT is not defined
try:
    from sklearn.linear_model.stochastic_gradient import MAX_INT
except ImportError:
    MAX_INT = np.iinfo(np.int32).max


class SGDClassifierWithSampleWeightsPerClass(SGDClassifier):

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        num_classes = self._expanded_class_weight.shape[0]
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(
                (num_classes, n_samples), dtype=np.float64, order="C")
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape != (num_classes, n_samples):
            raise ValueError(
                "Expected sample_weight to be of shape %s, found %s" %
                ((num_classes, n_samples), sample_weight.shape))
        return sample_weight

    def _fit_multiclass(self, X, y, alpha, C, learning_rate,
                        sample_weight, max_iter):
        """Fit a multi-class classifier by combining binary classifiers

        Each binary classifier predicts one class versus all others. This
        strategy is called OvA (One versus All) or OvR (One versus Rest).

        Here the we set the class weights so that
        """
        # Precompute the validation split using the multiclass labels
        # to ensure proper balancing of the classes.
        validation_mask = self._make_validation_split(y)

        # Use joblib to fit OvA in parallel.
        # Pick the random seed for each job outside of fit_binary to avoid
        # sharing the estimator random state between threads which could lead
        # to non-deterministic behavior
        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(MAX_INT, size=len(self.classes_))
        result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                          **_joblib_parallel_args(require="sharedmem"))(
            delayed(fit_binary)(self, i, X, y, alpha, C, learning_rate,
                                max_iter, self._expanded_class_weight[i],
                                1., sample_weight[i],
                                validation_mask=validation_mask,
                                random_state=seed)
            for i, seed in enumerate(seeds))

        # take the maximum of n_iter_ over every binary fit
        n_iter_ = 0.
        for i, (_, intercept, n_iter_i) in enumerate(result):
            self.intercept_[i] = intercept
            n_iter_ = max(n_iter_, n_iter_i)

        self.t_ += n_iter_ * X.shape[0]
        self.n_iter_ = n_iter_

        if self.average > 0:
            if self.average <= self.t_ - 1.0:
                self.coef_ = self.average_coef_
                self.intercept_ = self.average_intercept_
            else:
                self.coef_ = self.standard_coef_
                self.standard_intercept_ = np.atleast_1d(self.intercept_)
                self.intercept_ = self.standard_intercept_
