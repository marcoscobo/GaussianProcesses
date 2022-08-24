import numpy as np
import pandas as pd
from datetime import datetime as dt
from tabulate import tabulate
from sklearn.linear_model import LinearRegression


class BackTesting:

    def __init__(self, df_market, trade_col, action_col, direction=1):
        self.df_market = df_market
        self.action_col = action_col
        self.trade_col = trade_col
        self.direction = direction

        self.buys = None
        self.sells = None
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
        self.annual_yield = None
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
        self.df_market['Profit_Growth'] = self.df_market['Growth'] * self.df_market[
            self.action_col].shift(1).fillna(0)
        self.df_market['Cumulative_Profit_Growth'] = np.cumprod(1 + self.df_market['Profit_Growth'])

    def get_buy_sell_limits(self):
        self.buys = self.df_market[self.action_col][self.df_market[self.action_col] == 1].index
        if len(self.buys) > 0:
            self.buys_limits = [self.buys[0]]
            for i in range(len(self.buys) - 1):
                if (self.buys[i] - self.buys[i - 1] > 1) | (self.buys[i + 1] - self.buys[i] > 1):
                    self.buys_limits.append(self.buys[i])
            self.buys_limits.append(self.buys[-1])

        self.sells = self.df_market[self.action_col][self.df_market[self.action_col] == -1].index
        if len(self.sells) > 0:
            self.sells_limits = [self.sells[0]]
            for i in range(len(self.sells) - 1):
                if (self.sells[i] - self.sells[i - 1] > 1) | (self.sells[i + 1] - self.sells[i] > 1):
                    self.sells_limits.append(self.sells[i])
            self.sells_limits.append(self.sells[-1])

    def calculate_metrics(self, verbose, save):
        self.start = str(self.df_market['Date'][0])
        self.end = str(self.df_market['Date'].iloc[-1])
        self.duration = (dt.strptime(self.end, '%Y-%m-%d %H:%M:%S') - dt.strptime(self.start, '%Y-%m-%d %H:%M:%S')).days
        self.exposure_time = np.sum(np.abs(self.df_market[self.action_col])) / len(self.df_market) * 100
        self.profit = self.df_market['Cumulative_Profit_Growth'].iloc[-1]
        self.annual_yield = (self.profit ** (365 / self.duration))

        self.calculate_counts()
        self.buy_ops = self.count_buys_wins + self.count_buys_loses
        self.sell_ops = self.count_sells_wins + self.count_sells_loses
        self.total_ops = self.buy_ops + self.sell_ops
        self.win_buy = self.count_buys_wins / (self.buy_ops + 1e-4)
        self.win_sell = self.count_sells_wins / (self.sell_ops + 1e-4)

        self.calculate_profit_factor()
        self.profit_factor = self.df_market['Profit_Factor'].iloc[-1]

        self.calculate_drawdown()
        self.calculate_r2()

        with open('Backtesting/Results/{}.txt'.format(save), 'w') as external_file:
            if verbose:
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
                            ['Profit [%]', '{:.2f}'.format(100 * (self.profit - 1))],
                            ['Annual yield [%]', '{:.2f}'.format(100 * (self.annual_yield - 1))],
                            ['Profit factor', '{:.2f}'.format(self.profit_factor)],

                            ['Max. drawdown [%]', '{:.2f}'.format(self.max_drawdown * 100)],
                            ['Avg. drawdown [%]', '{:.2f}'.format(self.average_drawdown * 100)],
                            ['Recovery factor', '{:.2f}'.format(self.recovery_factor)],
                            ['Determination coef. R2', '{:.2f}'.format(self.r2)]],
                           tablefmt='plain', colalign=('left', 'right'), floatfmt=".4f")
            if verbose:
                print(tab)
                print('-' * 43)
            print(tab, file=external_file)
            print('-' * 43, file=external_file)
            external_file.close()

    def execute(self, metrics=False, verbose=False, plot=False, title='', save=None):
        self.df_market[self.action_col] = self.df_market[self.action_col] * self.direction
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
        if len(self.buys) > 0:
            for i in range(len(self.buys_limits) // 2):
                ini = self.df_market[self.trade_col][self.buys_limits[2 * i]]
                fin = self.df_market[self.trade_col][self.buys_limits[2 * i + 1]]
                if ini < fin:
                    self.count_buys_wins += 1
                else:
                    self.count_buys_loses += 1
        self.count_sells_wins, self.count_sells_loses = 0, 0
        if len(self.sells) > 0:
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

        if len(self.buys) > 0:
            for i in range(len(self.buys_limits) // 2):
                p = pbt.axvspan(self.df_market['Date'][self.buys_limits[2 * i]],
                                self.df_market['Date'][self.buys_limits[2 * i + 1]], alpha=0.15, color='green')

        if len(self.sells) > 0:
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


def total_backtesting(backtests, symbols, verbose=False, plot=False, save=None):
    l_index = np.argmax([len(backtests[name].df_market) for name in backtests.keys()])
    total_backtest = pd.DataFrame({'Date': backtests[list(backtests.keys())[l_index]].df_market['Date']})
    l = len(total_backtest)
    total_backtest['price'], total_backtest['growth'] = np.ones(l), np.ones(l)
    total_ops, buy_ops, count_buys_wins, sell_ops, count_sells_wins, exposure_time, profit_factor = 0, 0, 0, 0, 0, 0, 0

    for name in backtests.keys():
        df_ = total_backtest.copy()
        df_ = pd.merge(df_, backtests[name].df_market, on='Date', how='left')
        df_ = df_.fillna(1)

        weight = symbols['Weight'][symbols['Name'] == name].values[0]
        total_backtest['price'] = total_backtest['price'] + weight * (df_['Cumulative_Growth'] - 1)
        total_backtest['growth'] = total_backtest['growth'] + weight * (df_['Cumulative_Profit_Growth'] - 1)

        total_ops += backtests[name].total_ops
        buy_ops += backtests[name].buy_ops
        count_buys_wins += backtests[name].count_buys_wins
        sell_ops += backtests[name].sell_ops
        count_sells_wins += backtests[name].count_sells_wins
        exposure_time += weight * backtests[name].exposure_time
        profit_factor += weight * backtests[name].profit_factor

    total_backtest.rename(columns={'price': 'Cumulative_Growth', 'growth': 'Cumulative_Profit_Growth'}, inplace=True)
    total_backtest['Drawdown'] = total_backtest['Cumulative_Profit_Growth'] - total_backtest[
        'Cumulative_Profit_Growth'].cummax()

    start = str(total_backtest['Date'][0])
    end = str(total_backtest['Date'].iloc[-1])
    duration = (dt.strptime(end, '%Y-%m-%d %H:%M:%S') - dt.strptime(start, '%Y-%m-%d %H:%M:%S')).days

    win_buy = count_buys_wins / (buy_ops + 1e-4)
    win_sell = count_sells_wins / (sell_ops + 1e-4)
    profit = total_backtest['Cumulative_Profit_Growth'].iloc[-1]
    annual_yield = (profit ** (365 / duration))

    max_drawdown = - total_backtest['Drawdown'].min()
    average_drawdown = - total_backtest['Drawdown'].mean()
    recovery_factor = total_backtest['Cumulative_Profit_Growth'].iloc[-1] / (1 + max_drawdown)
    total_backtest['Drawdown'] = 1 + total_backtest['Drawdown']

    index = np.arange(0, len(total_backtest), 1).reshape(-1, 1)
    linear_regression = LinearRegression()
    linear_regression.fit(index, total_backtest['Cumulative_Profit_Growth'])
    r2 = linear_regression.score(index, total_backtest['Cumulative_Profit_Growth'])

    metrics = {'Start': start,
               'End': end,
               'Duration': '{} days'.format(duration),
               'Exposure time [%]': '{:.2f}'.format(exposure_time),
               'Total ops.': total_ops,
               'Buy ops.': buy_ops,
               'Win rate Buy ops. [%]': '{:.2f}'.format(100 * win_buy),
               'Sell ops.': sell_ops,
               'Win rate Sell ops. [%]': '{:.2f}'.format(100 * win_sell),
               'Profit [%]' : '{:.2f}'.format(100 * (profit - 1)),
               'Annual yield [%]' : '{:.2f}'.format(100 * (annual_yield - 1)),
               'Profit factor': '{:.2f}'.format(profit_factor),
               'Max. drawdown [%]': '{:.2f}'.format(max_drawdown * 100),
               'Avg. drawdown [%]': '{:.2f}'.format(average_drawdown * 100),
               'Recovery factor': '{:.2f}'.format(recovery_factor),
               'Determination coef. R2': '{:.2f}'.format(r2)}

    with open(save + '/Results/Total_Backtesting.txt', 'w') as external_file:
        if verbose:
            print('-' * 15, 'BACKTESTING', '-' * 15)
        print('-' * 15, 'BACKTESTING', '-' * 15, file=external_file)

        tab = tabulate([['Start', start],
                        ['End', end],
                        ['Duration', '{} days'.format(duration)],
                        ['Exposure time [%]', '{:.2f}'.format(exposure_time)],

                        ['Total ops.', total_ops],
                        ['Buy ops.', buy_ops],
                        ['Win rate Buy ops. [%]', '{:.2f}'.format(100 * win_buy)],
                        ['Sell ops.', sell_ops],
                        ['Win rate Sell ops. [%]', '{:.2f}'.format(100 * win_sell)],
                        ['Profit [%]', '{:.2f}'.format(100 * (profit - 1))],
                        ['Annual yield [%]', '{:.2f}'.format(100 * (annual_yield - 1))],
                        ['Profit factor', '{:.2f}'.format(profit_factor)],

                        ['Max. drawdown [%]', '{:.2f}'.format(max_drawdown * 100)],
                        ['Avg. drawdown [%]', '{:.2f}'.format(average_drawdown * 100)],
                        ['Recovery factor', '{:.2f}'.format(recovery_factor)],
                        ['Determination coef. R2', '{:.2f}'.format(r2)]],
                       tablefmt='plain', colalign=('left', 'right'), floatfmt=".4f")
        if verbose:
            print(tab)
            print('-' * 43)
        print(tab, file=external_file)
        print('-' * 43, file=external_file)
        external_file.close()

    if plot:
        pbt = total_backtest.plot(x='Date', y=['Cumulative_Growth', 'Cumulative_Profit_Growth', 'Drawdown'],
                                  title='Total Backtesting', figsize=(16, 8))
        if save is not None:
            f = pbt.get_figure()
            f.savefig(save + '/Figures/Total_Backtesting.png')

    return total_backtest, metrics
