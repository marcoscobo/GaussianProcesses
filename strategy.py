import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor


class Strategy:

    def __init__(self, kernel, X, y, year_pred, years_before, days_before, n_restarts_optimizer=1, verbose=False):

        self.kernel = kernel
        self.X = X
        self.y = y
        self.year_pred = year_pred
        self.years_before = years_before
        self.days_before = days_before
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose

        self.mask_train = None
        self.mask_test = None
        self.sc = None
        self.X_train = None
        self.y_train = None
        self.X_pred = None
        self.y_test = None
        self.gp = None
        self.y_pred = None
        self.y_std = None
        self.action = None

    def get_train_pred(self):

        self.mask_train = ((self.X['year'] >= self.year_pred - self.years_before) & (
                            self.X['year'] < self.year_pred)) | (
                            (self.X['year'] == self.year_pred) & (self.X['day'] < self.days_before))
        self.mask_test = (self.X['year'] == self.year_pred) & (self.X['day'] >= self.days_before)

        self.sc = StandardScaler()
        self.sc.fit(self.X)

        self.X_train = self.X.loc[self.mask_train]
        self.X_train = pd.DataFrame(self.sc.transform(self.X_train), columns=['year', 'day'])
        self.y_train = self.y[self.mask_train]

        self.X_pred = self.X.loc[self.mask_test]
        self.X_pred = pd.DataFrame(self.sc.transform(self.X_pred), columns=['year', 'day'])
        self.y_test = self.y[self.mask_test]

    def get_prediction(self):

        if self.verbose:
            print('Predicting year {} with data from {} to {}'.format(self.year_pred,
                                                                      self.year_pred - self.years_before,
                                                                      self.year_pred - 1))

        self.gp = GaussianProcessRegressor(kernel=self.kernel,
                                           normalize_y=True,
                                           n_restarts_optimizer=self.n_restarts_optimizer,
                                           alpha=0.01,
                                           random_state=0)
        t0 = time()
        self.gp.fit(self.X_train, self.y_train)

        if self.verbose:
            print("Elapsed time: %0.3fs" % (time() - t0))
            print("Log-marginal-likelihood: %.3f" % self.gp.log_marginal_likelihood(self.gp.kernel_.theta))

        self.y_pred, self.y_std = self.gp.predict(self.X_pred, return_std=True)

    def buy_sell(self):

        self.action = [0] * len(self.y_pred)
        top, bottom = np.argmax(self.y_pred), np.argmin(self.y_pred)
        if top > bottom:
            self.action[bottom:top] = [1] * (top - bottom)
        else:
            self.action[top:bottom] = [-1] * (bottom - top)

        self.action = [0] * self.days_before + self.action

    def execute(self):

        self.get_train_pred(self)
        self.get_prediction(self)
        self.buy_sell(self)
