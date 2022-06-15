import numpy as np
from datetime import datetime as dt
from tabulate import tabulate
from sklearn.linear_model import LinearRegression


class BackTesting:

    def __init__(self, df_market, trade_col, action_col):
        self.df_market = df_market
        self.action_col = action_col
        self.trade_col = trade_col

        self.buys_limits = None
        self.sells_limits = None
        self.count_buys_wins = None
        self.count_buys_loses = None
        self.count_sells_wins = None
        self.count_sells_loses = None
        self.start = None
        self.end = None
        self.duration = None
        self.exposure_time = None
        self.buy_ops = None
        self.sell_ops = None
        self.total_ops = None
        self.win_buy = None
        self.win_sell = None
        self.profit = None
        self.profit_factor = None
        self.max_drawdown = None
        self.average_drawdown = None
        self.recovery_factor = None
        self.r2 = None

    def define_growth(self):
        self.df_market['Growth'] = ((self.df_market[self.trade_col].shift(-1) - self.df_market[self.trade_col]) /
                                    self.df_market[self.trade_col]).shift(1).fillna(0)
        self.df_market['Cumulative_Growth'] = np.cumprod(1 + self.df_market['Growth'])

    def compute_return_trad(self):
        self.df_market['Profit_Growth'] = self.df_market['Growth'] * self.df_market[self.action_col].shift(1).fillna(0)
        self.df_market['Cumulative_Profit_Growth'] = np.cumprod(1 + self.df_market['Profit_Growth'])

    def get_buy_sell_limits(self):
        buys = self.df_market[self.action_col][self.df_market[self.action_col] == 1].index
        if len(buys) > 0:
            self.buys_limits = [buys[0]]
            for i in range(len(buys) - 1):
                if (buys[i] - buys[i - 1] > 1) | (buys[i + 1] - buys[i] > 1):
                    self.buys_limits.append(buys[i])
            self.buys_limits.append(buys[-1])

        sells = self.df_market[self.action_col][self.df_market[self.action_col] == -1].index
        if len(sells) > 0:
            self.sells_limits = [sells[0]]
            for i in range(len(sells) - 1):
                if (sells[i] - sells[i - 1] > 1) | (sells[i + 1] - sells[i] > 1):
                    self.sells_limits.append(sells[i])
            self.sells_limits.append(sells[-1])

    def calculate_metrics(self, verbose, save):
        self.start = str(self.df_market['Date'][0])
        self.end = str(self.df_market['Date'].iloc[-1])
        self.duration = (dt.strptime(self.end, '%Y-%m-%d %H:%M:%S') - dt.strptime(self.start, '%Y-%m-%d %H:%M:%S')).days
        self.exposure_time = np.sum(np.abs(self.df_market[self.action_col])) / len(self.df_market) * 100
        self.profit = self.df_market['Cumulative_Profit_Growth'].iloc[-1]

        self.calculate_counts()
        self.buy_ops = self.count_buys_wins + self.count_buys_loses
        self.sell_ops = self.count_sells_wins + self.count_sells_loses
        self.total_ops = self.buy_ops + self.sell_ops
        self.win_buy = self.count_buys_wins / self.buy_ops
        self.win_sell = self.count_sells_wins / self.sell_ops

        self.calculate_profit_factor()
        self.profit_factor = self.df_market['Profit_Factor'].iloc[-1]

        self.calculate_drawdown()
        self.calculate_r2()

        if verbose:
            with open('Backtesting/Results/{}.txt'.format(save), 'w') as external_file:
                print('-' * 15, 'BACKTESTING', '-' * 15)
                print('-' * 15, 'BACKTESTING', '-' * 15, file=external_file)

                tab = tabulate([['Start', self.start],
                                ['End', self.end],
                                ['Duration', '{} days'.format(self.duration)],
                                ['Exposure time [%]', '{:.2f}'.format(self.exposure_time)],

                                ['Total ops.', self.total_ops],
                                ['Buy ops.', self.buy_ops],
                                ['Win rate Buy ops. [%]', '{:.2f}'.format(100 * self.win_buy)],
                                ['Sell ops.', self.sell_ops],
                                ['Win rate Sell ops. [%]', '{:.2f}'.format(100 * self.win_sell)],
                                ['Profit factor', '{:.2f}'.format(self.profit_factor)],

                                ['Max. drawdown [%]', '{:.2f}'.format(self.max_drawdown * 100)],
                                ['Avg. drawdown [%]', '{:.2f}'.format(self.average_drawdown * 100)],
                                ['Recovery factor', '{:.2f}'.format(self.recovery_factor)],
                                ['Determination coef. R2', '{:.2f}'.format(self.r2)]],
                               tablefmt='plain', colalign=('left', 'right'), floatfmt=".4f")
                print(tab)
                print(tab, file=external_file)

                print('-' * 43)
                print('-' * 43, file=external_file)
                external_file.close()

    def execute(self, metrics=False, verbose=False, plot=False, title='', save=None):
        self.define_growth()
        self.compute_return_trad()
        self.get_buy_sell_limits()
        if metrics:
            self.calculate_metrics(verbose, save)
        if plot:
            self.plot_backtest(title, save)

        return self.df_market

    def calculate_counts(self):
        self.count_buys_wins, self.count_buys_loses = 0, 0
        if len(self.buys_limits) > 0:
            for i in range(len(self.buys_limits) // 2):
                ini = self.df_market[self.trade_col][self.buys_limits[2 * i]]
                fin = self.df_market[self.trade_col][self.buys_limits[2 * i + 1]]
                if ini < fin:
                    self.count_buys_wins += 1
                else:
                    self.count_buys_loses += 1
        self.count_sells_wins, self.count_sells_loses = 0, 0
        if len(self.sells_limits) > 0:
            for i in range(len(self.sells_limits) // 2):
                ini = self.df_market[self.trade_col][self.sells_limits[2 * i]]
                fin = self.df_market[self.trade_col][self.sells_limits[2 * i + 1]]
                if ini > fin:
                    self.count_sells_wins += 1
                else:
                    self.count_sells_loses += 1

    def calculate_profit_factor(self):
        self.df_market['Profit_Factor'] = 0
        wins_returns = 1e-4
        loses_returns = 1e-4
        for i, trade_row in self.df_market.iterrows():
            result = trade_row.Profit_Growth
            if result < 0:
                loses_returns += -result
                pf = wins_returns / loses_returns
            elif result > 0:
                wins_returns += result
                pf = wins_returns / loses_returns
            else:
                pf = wins_returns / loses_returns

            self.df_market.loc[self.df_market['Date'] == trade_row.Date, 'Profit_Factor'] = pf

    def calculate_drawdown(self):
        self.df_market['Drawdown'] = self.df_market['Cumulative_Profit_Growth'] - \
                                     self.df_market['Cumulative_Profit_Growth'].cummax()
        self.max_drawdown = - self.df_market['Drawdown'].min()
        self.average_drawdown = - self.df_market['Drawdown'].mean()
        self.recovery_factor = self.df_market['Cumulative_Profit_Growth'].iloc[-1] / (1 + self.max_drawdown)
        self.df_market['Drawdown'] = 1 + self.df_market['Drawdown']

    def calculate_r2(self):
        index = np.arange(0, len(self.df_market), 1).reshape(-1, 1)
        linear_regression = LinearRegression()
        linear_regression.fit(index, self.df_market['Cumulative_Profit_Growth'])
        self.r2 = linear_regression.score(index, self.df_market['Cumulative_Profit_Growth'])

    def plot_backtest(self, title='', save=None):
        pbt = self.df_market[['Date', 'Cumulative_Growth', 'Cumulative_Profit_Growth', 'Drawdown']].plot(x='Date', y=[
            'Cumulative_Growth', 'Cumulative_Profit_Growth', 'Drawdown'], title=title, figsize=(16, 8))

        if len(self.buys_limits) > 0:
            for i in range(len(self.buys_limits) // 2):
                p = pbt.axvspan(self.df_market['Date'][self.buys_limits[2 * i]],
                                self.df_market['Date'][self.buys_limits[2 * i + 1]], alpha=0.15, color='green')

        if len(self.sells_limits) > 0:
            for i in range(len(self.sells_limits) // 2):
                p = pbt.axvspan(self.df_market['Date'][self.sells_limits[2 * i]],
                                self.df_market['Date'][self.sells_limits[2 * i + 1]], alpha=0.15, color='red')

        if save is not None:
            f = pbt.get_figure()
            f.savefig('Backtesting/Figures/{}.png'.format(save))

    def plot_profit_factor(self, title='', save=None):
        ppf = self.df_market['Profit_Factor'].plot(ylim=(np.min(self.df_market['Profit_Factor']) - 0.1,
                                                         np.mean(self.df_market['Profit_Factor']) * 1.5),
                                                   title=title, figsize=(16, 8))
        if save is not None:
            f = ppf.get_figure()
            f.savefig('Backtesting/Figures/{}.png'.format(save))
