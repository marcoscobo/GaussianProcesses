import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import GPy
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class Strategy:

    def __init__(self, df, df_s1, df_s2, df_s3, kernel, year_pred, years_before, days_before, days_between_rows=7,
                 days_between_deltas=31, scale_X=False, column='Adj Close', lower_bound=1e-4, upper_bound=1e+4,
                 max_iter=50, time_pen=1.001, min_IR=1, min_days=10, verbose=False, verbose_kernel=False, plot_strategy=False,
                 plot_prediction=False, save=None):

        self.df = df
        self.df_s1 = df_s1
        self.df_s2 = df_s2
        self.df_s3 = df_s3
        self.kernel = kernel
        self.year_pred = year_pred
        self.years_before = years_before
        self.days_before = days_before
        self.days_between_rows = days_between_rows
        self.days_between_deltas = days_between_deltas
        self.scale_X = scale_X
        self.column = column
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.time_pen = time_pen
        self.min_IR = min_IR
        self.min_days = min_days
        self.verbose = verbose
        self.verbose_kernel = verbose_kernel
        self.plot_strategy = plot_strategy
        self.plot_prediction = plot_prediction
        self.save = save

        self.df_years = None
        self.df_years_s1 = None
        self.df_years_s2 = None
        self.df_years_s3 = None
        self.X = None
        self.y = None
        self.sc = None
        self.X_train = None
        self.y_train = None
        self.X_pred = None
        self.y_test = None
        self.gp = None
        self.y_pred = None
        self.y_std = None
        self.y_cov = None
        self.Z_grid_buy = None
        self.Z_grid_sell = None
        self.op = None
        self.buy = None
        self.sell = None
        self.action = None

    @staticmethod
    def get_df_years(df):

        starts, ends = [], []
        for i in range(df.index[0].year, df.index[-1].year):
            starts.append('{}-01-01'.format(i))
            ends.append('{}-12-31'.format(i))

        sd = np.std(df['Adj Close'].values)
        df_years = pd.DataFrame()
        for start, end in zip(starts, ends):
            s = [None] * 253
            v = df['Adj Close'][start:end].values
            v = (v - v[0]) / sd
            s[:len(v)] = v
            df_years[end[:4]] = s

        df_years = df_years.fillna(-1e+6)

        return df_years

    def get_X_y(self):

        self.df_years = self.get_df_years(self.df)
        self.df_years_s1 = self.get_df_years(self.df_s1)
        self.df_years_s2 = self.get_df_years(self.df_s2)
        self.df_years_s3 = self.get_df_years(self.df_s3)

        years = [int(year) for year in self.df_years.columns.values]
        days = np.arange(0, len(self.df_years), self.days_between_rows)
        deltas = np.arange(0, len(self.df_years), self.days_between_deltas)

        self.X, self.y, i = pd.DataFrame(columns=['year', 'day', 'delta', 's1', 's2', 's3']), [], 0
        for year in years:
            for day in days:
                for delta in deltas:
                    if (day + delta) < len(self.df_years):
                        self.X.loc[i, 'year'], self.X.loc[i, 'day'], self.X.loc[i, 'delta'] = year, day, delta
                        self.X.loc[i, 's1'] = self.df_years_s1[str(year)].values[day]
                        self.X.loc[i, 's2'] = self.df_years_s2[str(year)].values[day]
                        self.X.loc[i, 's3'] = self.df_years_s3[str(year)].values[day]
                        self.y.append(self.df_years[str(year)].values[day + delta])
                        i += 1
        self.y = np.array(self.y)

        self.X = self.X.loc[~(self.y == -1e+6), :].reset_index(drop=True)
        self.y = self.y[~(self.y == -1e+6)]

    def get_train_pred(self):

        mask_train = ((self.X['year'] >= self.year_pred - self.years_before) & (
                self.X['year'] < self.year_pred)) | (
                             (self.X['year'] == self.year_pred) & (self.X['day'] < self.days_before))
        mask_test = (self.X['year'] == self.year_pred) & (self.X['day'] >= self.days_before)
        len_year = len(self.df.loc[str(self.year_pred)])

        if self.scale_X:
            self.sc = StandardScaler()
            self.sc.fit(self.X)

        self.X_train = self.X.loc[mask_train]
        if self.scale_X:
            self.X_train = pd.DataFrame(self.sc.transform(self.X_train),
                                        columns=['year', 'day', 'delta', 's1', 's2', 's3'])
        self.y_train = self.y[mask_train]

        self.X_pred = pd.DataFrame()
        for i in range(len_year - self.days_before):
            self.X_pred = self.X_pred.append(self.X_train.iloc[-1:])
        self.X_pred['delta'] = range(len_year - self.days_before)
        self.X_pred = self.X_pred.reset_index(drop=True)
        if self.scale_X:
            self.X_pred = pd.DataFrame(self.sc.transform(self.X_pred),
                                       columns=['year', 'day', 'delta', 's1', 's2', 's3'])
        y_year = self.df_years[str(self.year_pred)]
        self.y_test = y_year[y_year != -1e+6][self.days_before:].reset_index(drop=True)

    def get_prediction(self):

        if self.verbose:
            print('Predicting year {} with data from {} to {}'.format(self.year_pred,
                                                                      self.year_pred - self.years_before,
                                                                      self.year_pred - 1))

        self.gp = GPy.models.GPRegression(np.array(self.X_train), np.array(self.y_train)[:, None], self.kernel)
        # self.gp['.*lengthscale'].constrain_bounded(self.lower_bound, self.upper_bound)
        self.gp.optimize(messages=self.verbose_kernel, ipython_notebook=False, max_iters=self.max_iter)

        self.y_pred, self.y_cov = self.gp.predict(np.array(self.X_pred), full_cov=True)
        self.y_pred = self.y_pred.reshape(self.y_pred.shape[0])
        self.y_std = np.sqrt(np.diagonal(self.y_cov))

    def buy_sell(self):

        if self.verbose:
            print('Calculating buy sell signals for {}'.format(self.year_pred))

        self.action = [0] * (len(self.y_pred) + self.days_before)

        def cost_fun_buy(i, j):
            x0, x1 = self.y_pred[i], self.y_pred[j]
            var0, var1, cov01 = self.y_cov[i, i], self.y_cov[j, j], self.y_cov[i, j]
            return ((1 + x1) / (1 + x0)) / (np.sqrt(var0 + var1 - 2 * cov01)) * self.time_pen ** (i - j)

        def cost_fun_sell(i, j):
            x0, x1 = self.y_pred[i], self.y_pred[j]
            var0, var1, cov01 = self.y_cov[i, i], self.y_cov[j, j], self.y_cov[i, j]
            return ((1 + x0) / (1 + x1)) / (np.sqrt(var0 + var1 - 2 * cov01)) * self.time_pen ** (i - j)

        X_grid, Y_grid = np.meshgrid(range(len(self.y_pred)), range(len(self.y_pred)))
        self.Z_grid_buy = pd.DataFrame(cost_fun_buy(X_grid, Y_grid))
        self.Z_grid_buy = (self.Z_grid_buy * np.tri(*self.Z_grid_buy.shape, k=-self.min_days)).fillna(1).replace(0, np.NAN)
        max_buy = self.Z_grid_buy.max().max()

        self.Z_grid_sell = pd.DataFrame(cost_fun_sell(X_grid, Y_grid))
        self.Z_grid_sell = (self.Z_grid_sell * np.tri(*self.Z_grid_sell.shape, k=-self.min_days)).fillna(1).replace(0, np.NAN)
        max_sell = self.Z_grid_sell.max().max()

        if max_buy >= max_sell:
            self.op = 1
            for i in range(len(self.y_pred)):
                for j in range(len(self.y_pred)):
                    if self.Z_grid_buy.iloc[i, j] == max_buy:
                        self.buy, self.sell = j, i
        else:
            self.op = -1
            for i in range(len(self.y_pred)):
                for j in range(len(self.y_pred)):
                    if self.Z_grid_sell.iloc[i, j] == max_sell:
                        self.sell, self.buy = j, i

        if max(max_buy, max_sell) > self.min_IR:
            if self.op == 1:
                self.action[self.days_before + self.buy: self.days_before + self.sell] = [1] * (self.sell - self.buy)
            elif self.op == -1:
                self.action[self.days_before + self.sell: self.days_before + self.buy] = [-1] * (self.buy - self.sell)

        if self.plot_strategy:
            fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 6))
            for ax, Z_grid, t in zip(axs.flat, [self.Z_grid_buy, self.Z_grid_sell], ['Buy surface', 'Sell surface']):
                cs = ax.contourf(range(len(self.y_pred)), range(len(self.y_pred)), Z_grid, 20, cmap='brg')
                fig.colorbar(cs, ax=ax)
                ax.set_title(t)

            if self.save is not None:
                plt.savefig('Actions/Figures/{}_surface.png'.format(self.save))
            plt.show()

        if self.plot_prediction:
            df_y = pd.DataFrame({'y_pred': 1 + self.y_pred, 'y_std': self.y_std})
            predp = df_y.plot(y=['y_pred'], legend=False, figsize=(16, 8))
            p = predp.fill_between(range(len(df_y['y_pred'])), df_y['y_pred'] - df_y['y_std'], df_y['y_pred'] + df_y['y_std'],
                                   alpha=0.25, color='k')
            p = predp.plot(1 + self.y_test, linestyle='--')
            if max(max_buy, max_sell) > self.min_IR and self.op == 1:
                p = predp.axvspan(self.buy, self.sell, alpha=0.15, color='green')
            elif max(max_buy, max_sell) > self.min_IR and self.op == -1:
                p = predp.axvspan(self.sell, self.buy, alpha=0.15, color='red')
            if self.save is not None:
                f = predp.get_figure()
                f.savefig('Actions/Figures/{}_curve.png'.format(self.save))
            plt.show()

    def execute(self):

        self.get_X_y()
        self.get_train_pred()
        self.get_prediction()
        self.buy_sell()
