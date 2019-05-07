from sklearn.linear_model import LogisticRegression
import numpy as np
import logging

__author__ = "Mengqiao Yu"
__email__ = "mengqiao.yu@berkeley.edu"

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

class TransitionModel():
    """Variant of Multinomial Logit Model"""

    def __init__(self, intercept_fit=True, std_fit=True, num_classes=2, num_covariates=0):
        """
        :param X_data: (num of obs, num of covariates)
        :param y_data: (num of obs, num of classes) in prob but not discrete choice
        """
        self.intercept_fit = intercept_fit
        self.std_fit = std_fit
        self.num_classes = num_classes
        self.num_covariates = num_covariates
        self.model = LogisticRegression(solver='lbfgs',
                                        fit_intercept=self.intercept_fit,
                                        warm_start=True)

    def _data_formatter(self, X, y):
        """
        reformat/augment data matrix to fit multinomial logit model with prob output.
        Parameters
        ----------
        X: np array (num of obs , num of covariates)
        y: np array (num of obs , num of classes)
        Returns
        ----------
        X_augmented: (num of obs * num of classes, num of covariates)
        y_augmented: (num of obs * num of classes, )
        sample_weight: (num of obs * num of classes, )
        """

        num_obs, num_classes = X.shape[0], y.shape[1]
        X_augmented = np.repeat(X, num_classes, axis=0)
        y_augmented = np.tile(np.arange(num_classes), num_obs)
        sample_weight = y.reshape(-1, )

        return X_augmented, y_augmented, sample_weight

    def fit(self, X, y):
        """estimate covariates in transition model"""
        num_classes = y.shape[1]
        multi_class = 'multinomial' if num_classes >= 3 else 'ovr'
        self.model.multi_class = multi_class
        X_new, y_new, sample_weight = self._data_formatter(X, y)
        self.model.fit(X_new, y_new, sample_weight=sample_weight)

    def predict_log_proba(self, X):
        return self.model.predict_log_proba(X)

    def get_params(self):
        return np.concatenate((
            self.model.intercept_.reshape(self.model.intercept_.shape[0],1),
            self.model.coef_))







