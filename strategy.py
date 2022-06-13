import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import random

import warnings

warnings.filterwarnings("ignore")


class Strategy:

    def __init__(self, df, kernel, year_pred, years_before, days_before, alg='GP', column='Adj Close',
                 n_restarts_optimizer=1, verbose=False):

        self.df = df
        self.kernel = kernel
        self.year_pred = year_pred
        self.years_before = years_before
        self.days_before = days_before
        self.alg = alg
        self.column = column
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose

        self.starts = None
        self.ends = None
        self.sd = None
        self.df_years = None
        self.X = None
        self.y = None
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
        self.mask_last = None
        self.action = None

    def get_df_years(self):

        self.starts, self.ends = [], []
        for i in range(self.df.index[0].year, self.df.index[-1].year):
            self.starts.append('{}-01-01'.format(i))
            self.ends.append('{}-12-31'.format(i))

        self.sd = np.std(self.df[self.column].values)
        self.df_years = pd.DataFrame()
        for start, end in zip(self.starts, self.ends):
            s = [None] * 253
            v = self.df[self.column][start:end].values
            v = (v - v[0]) / self.sd
            s[:len(v)] = v
            self.df_years[end[:4]] = s

    def get_X_y(self):

        days = self.df_years.index.values
        years = [int(year) for year in self.df_years.columns.values]

        self.X = pd.DataFrame(columns=['year', 'day'])
        self.y, i = np.array([None] * len(years) * len(days)), 0
        for year in years:
            for day in days:
                self.X.loc[i, 'year'], self.X.loc[i, 'day'] = year, day
                self.y[i] = self.df_years[str(year)].values[day]
                i += 1

        self.X = self.X.loc[~np.isnan(list(self.y)), :].reset_index(drop=True)
        self.y = self.y[~np.isnan(list(self.y))]

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

        if self.alg == 'GP':
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

        elif self.alg == 'Simple':
            self.mask_last = self.X['year'] == self.year_pred - 1

            self.y_pred = self.y[self.mask_last][-len(self.X_pred):]
            self.y_std = [0] * len(self.y_pred)

        elif self.alg == 'Random':
            self.y_pred = self.y_test
            self.y_std = [0] * len(self.y_pred)
        else:
            raise Exception('Algoritmo de predicción no válido.')

    def buy_sell(self):

        if self.alg != 'Random':
            self.action = [0] * len(self.y_pred)
            top, bottom = np.argmax(self.y_pred), np.argmin(self.y_pred)
            if top > bottom:
                self.action[bottom:top] = [1] * (top - bottom)
            else:
                self.action[top:bottom] = [-1] * (bottom - top)

            self.action = [0] * self.days_before + self.action

        else:
            self.action = random.choices([-1, 0, 1], k=len(self.y_pred))
            self.action = [0] * self.days_before + self.action

    def execute(self):

        self.get_df_years()
        self.get_X_y()
        self.get_train_pred()
        self.get_prediction()
        self.buy_sell()
