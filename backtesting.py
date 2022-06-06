import numpy as np
from datetime import datetime
from tabulate import tabulate
from sklearn.linear_model import LinearRegression

class BackTesting:

    def __init__(self, SL, TP, df_market, pip_value, points_spread, action_column='trade_action',result_column='Trade_Result', just_one=False, save_df=None):

        self.SL = SL
        self.TP = TP
        self.df_market = df_market
        self.pip_value = pip_value
        self.points_spread = points_spread
        self.action_column = action_column
        self.result_column = result_column
        self.just_one = just_one
        self.save_df = save_df

    def define_SL_and_TP(self):

        if self.SL is not None and self.TP is not None:
            if self.SL <= 0 or self.TP <= 0:
                raise Exception('TP y SL must be greater than 0')
            self.df_market['SL'] = self.SL
            self.df_market['TP'] = self.TP

        self.df_market['SL_value_buy'] = self.df_market['open'] - self.df_market['SL'] * self.pip_value + (self.points_spread * self.pip_value / 10)
        self.df_market['SL_value_sell'] = self.df_market['open'] + self.df_market['SL'] * self.pip_value - (self.points_spread * self.pip_value / 10)
        self.df_market['TP_value_buy'] = self.df_market['open'] + self.df_market['TP'] * self.pip_value + (self.points_spread * self.pip_value / 10)
        self.df_market['TP_value_sell'] = self.df_market['open'] - self.df_market['TP'] * self.pip_value - (self.points_spread * self.pip_value / 10)

    def compute_return_trad(self, trade_row, future_df):

        index_win = 1000000
        index_lost = 1000000

        if trade_row[self.action_column] > 0:
            if len(np.where(future_df['high'] > trade_row['TP_value_buy'])[0]) > 0:
                index_win = np.where(future_df['high'] > trade_row['TP_value_buy'])[0][0]
            if len(np.where(future_df['low'] < trade_row['SL_value_buy'])[0]) > 0:
                index_lost = np.where(future_df['low'] < trade_row['SL_value_buy'])[0][0]

        elif trade_row[self.action_column] < 0:
            if len(np.where(future_df['low'] < trade_row['TP_value_sell'])[0]) > 0:
                index_win = np.where(future_df['low'] < trade_row['TP_value_sell'])[0][0]
            if len(np.where(future_df['high'] > trade_row['SL_value_sell'])[0]) > 0:
                index_lost = np.where(future_df['high'] > trade_row['SL_value_sell'])[0][0]

        if index_win < index_lost:
            trade_result = self.TP
            trade_duration = index_win

        elif index_win > index_lost:
            trade_result = -self.SL
            trade_duration = index_lost

        else:
            trade_result = -self.points_spread
            trade_duration = 0

        if self.just_one:
            index = np.min([index_win,index_lost])
            if index < 1000000:
                if ~np.isnan(index):
                    self.df_market.loc[(self.df_market.time > trade_row.time) & (self.df_market.time <= future_df.iloc[np.min([index_win,index_lost])]['time']), self.action_column] = 0
                else:
                    self.df_market.loc[(self.df_market.time == trade_row.time), self.action_column] = 0
            elif index == 1000000:
                self.df_market.loc[(self.df_market.time > trade_row.time), self.action_column] = 0

        return trade_result, trade_duration

    @staticmethod
    def net_std_pip(df_market_action, column):

        metric_pip_net_profit = df_market_action[column].sum()
        metric_pip_std_strategy = df_market_action[column].std()

        return metric_pip_net_profit, metric_pip_std_strategy

    @staticmethod
    def sharpe_ratio(df_market_action, action_column, metric_pip_net_profit, metric_pip_std_strategy):

        sharpe_ratio = np.sqrt(len(df_market_action[df_market_action[action_column].abs() > 0])) * metric_pip_net_profit / metric_pip_std_strategy

        return sharpe_ratio

    @staticmethod
    def trades_returns(df_market_action, action_column, direction, column='Trade_Result'):

        trades_wins_returns = df_market_action.loc[(df_market_action[action_column] == direction) & (df_market_action[column] > 0), column].sum()
        trades_loses_returns = df_market_action.loc[(df_market_action[action_column] == direction) & (df_market_action[column] < 0), column].sum()

        return trades_wins_returns, trades_loses_returns

    @staticmethod
    def calculate_profit(self):

        self.df_market['Profit'] = 0
        profit = 0
        for i, trade_row in self.df_market.iterrows():
            profit += trade_row.Trade_Result

            self.df_market.loc[self.df_market['time'] == trade_row.time, 'Profit'] = profit

    @staticmethod
    def calculate_profit_factor(self):

        self.df_market['Profit_Factor'] = 0
        wins_returns = 1e-4
        loses_returns = 1e-4
        for i, trade_row in self.df_market.iterrows():
            result = trade_row.Trade_Result
            if result < 0:
                loses_returns += -result
                pf = wins_returns / loses_returns
            elif result > 0:
                wins_returns += result
                pf = wins_returns / loses_returns
            else:
                pf = wins_returns / loses_returns

            self.df_market.loc[self.df_market['time'] == trade_row.time, 'Profit_Factor'] = pf

    @staticmethod
    def calculate_drawdown(self):

        self.df_market['Drawdown'] = self.df_market['Profit'] - self.df_market['Profit'].cummax()

    @staticmethod
    def drawdown_metrics(self):

        max_drawdown = - self.df_market['Drawdown'].min()
        average_drawdown = - self.df_market['Drawdown'].mean()
        recovery_factor = self.df_market['Profit'].iloc[-1] / max_drawdown

        return max_drawdown, average_drawdown, recovery_factor

    @staticmethod
    def regression_metrics(self):

        index = np.arange(0, len(self.df_market), 1).reshape(-1, 1)
        regresion_lineal = LinearRegression()
        regresion_lineal.fit(index, self.df_market['Profit'])
        r2 = regresion_lineal.score(index, self.df_market['Profit'])

        return r2

    def calculate_metrics(self, df_market_action, action_column, verbose=False):

        df_market_action['benchmark'] = 0
        df_market_action.loc[df_market_action[action_column].abs() > 0, 'benchmark'] = np.random.randint(1, 3, size=len(df_market_action.loc[df_market_action[action_column].abs() > 0, 'benchmark']))
        df_market_action.loc[df_market_action['benchmark'].abs() == 2, 'benchmark'] = -1
        df_market_action = BackTesting(pip_value=self.pip_value, points_spread=self.points_spread, action_column='benchmark', SL=self.SL, TP=self.TP, df_market=df_market_action, result_column='Trade_Result_benchmark', just_one=False).execute(metrics=False)

        start = str(self.df_market['time'][0])
        end = str(self.df_market['time'].iloc[-1])
        duration = (datetime.strptime(end, '%Y-%m-%d %H:%M:%S%z') - datetime.strptime(start, '%Y-%m-%d %H:%M:%S%z')).days
        exposure_time = sum(self.df_market['Trade_duration']) / len(self.df_market) * 100
        average_exposure_time = self.df_market['Trade_duration'].loc[self.df_market['Trade_duration'] > 0].mean()

        count_buy_wins = len(df_market_action[(df_market_action[action_column] == 1) & (df_market_action['Trade_Result'] > 0)])
        count_sell_wins = len(df_market_action[(df_market_action[action_column] == -1) & (df_market_action['Trade_Result'] > 0)])
        count_buy_lose = len(df_market_action[(df_market_action[action_column] == 1) & (df_market_action['Trade_Result'] < 0)])
        count_sell_lose = len(df_market_action[(df_market_action[action_column] == -1) & (df_market_action['Trade_Result'] < 0)])
        count_buy_wins_bench = len(df_market_action[(df_market_action[action_column] == 1) & (df_market_action['Trade_Result_benchmark'] > 0)])
        count_sell_wins_bench = len(df_market_action[(df_market_action[action_column] == -1) & (df_market_action['Trade_Result_benchmark'] > 0)])
        count_buy_lose_bench = len(df_market_action[(df_market_action[action_column] == 1) & (df_market_action['Trade_Result_benchmark'] < 0)])
        count_sell_lose_bench = len(df_market_action[(df_market_action[action_column] == -1) & (df_market_action['Trade_Result_benchmark'] < 0)])

        buy_wins_returns, buy_loses_returns = self.trades_returns(df_market_action, action_column, direction=1, column='Trade_Result')
        sell_wins_returns, sell_loses_returns = self.trades_returns(df_market_action, action_column, direction=-1, column='Trade_Result')
        buy_wins_returns_bench, buy_loses_returns_bench = self.trades_returns(df_market_action, action_column, direction=1, column='Trade_Result_benchmark')
        sell_wins_returns_bench, sell_loses_returns_bench = self.trades_returns(df_market_action, action_column, direction=-1, column='Trade_Result_benchmark')

        win_rate = (count_buy_wins + count_sell_wins) / (count_buy_wins + count_sell_wins + count_buy_lose + count_sell_lose) * 100
        win_rate_bench = (count_buy_wins_bench + count_sell_wins_bench) / (count_buy_wins_bench + count_sell_wins_bench + count_buy_lose_bench + count_sell_lose_bench) * 100
        profit_factor = (buy_wins_returns + sell_wins_returns) / -(buy_loses_returns + sell_loses_returns)
        profit_factor_benchmark = (buy_wins_returns_bench + sell_wins_returns_bench) / -(buy_loses_returns_bench + sell_loses_returns_bench)

        average_pip_win = (buy_wins_returns + sell_wins_returns) / (count_buy_wins + count_sell_wins)
        average_pip_lose = (buy_loses_returns + sell_loses_returns) / (count_buy_lose + count_sell_lose)

        metric_pip_net_profit, metric_pip_std_strategy = self.net_std_pip(df_market_action, 'Trade_Result')
        sharpe_ratio = self.sharpe_ratio(df_market_action, action_column, metric_pip_net_profit, metric_pip_std_strategy)
        # metric_pip_net_profit_bench, metric_pip_std_strategy_bench = self.net_std_pip(df_market_action, 'Trade_Result_benchmark')
        # sharpe_ratio_benchmark = self.sharpe_ratio(df_market_action, action_column, metric_pip_net_profit_bench, metric_pip_std_strategy_bench)

        max_drawdown, average_drawdown, recovery_factor = self.drawdown_metrics(self)

        r2 = self.regression_metrics(self)

        if verbose:
            print('-' * 18, 'BACKTESTING', '-' * 19)
            print(tabulate([['Start', start],
                            ['End', end],
                            ['Duration', '{} days'.format(duration)],
                            ['Exposure time [%]', '{:.2f}'.format(exposure_time)],
                            ['Avg. exposure time [cd]', '{:.2f}'.format(average_exposure_time)],
                            ['Total ops.', count_buy_wins + count_sell_wins + count_buy_lose + count_sell_lose],
                            ['Total buy ops.', count_buy_lose + count_buy_wins],
                            ['Total sell ops.', count_sell_lose + count_sell_wins],
                            ['Total buy ops. winned', count_buy_wins],
                            ['Total sell ops. winned', count_sell_wins],
                            ['Winned pips avg.', '{:.2f}'.format(average_pip_win)],
                            ['Losed pips avg.', '{:.2f}'.format(average_pip_lose)],
                            ['Net profit [pip]', '{:.2f}'.format(metric_pip_net_profit)],
                            ['Volatility [pip]', '{:.2f}'.format(metric_pip_std_strategy)],
                            ['Sharpe ratio', '{:.2f}'.format(sharpe_ratio)],
                            ['Win rate [%]', '{:.2f}'.format(win_rate)],
                            ['Win rate benchmark [%]', '{:.2f}'.format(win_rate_bench)],
                            ['Profit factor', '{:.2f}'.format(profit_factor)],
                            ['Profit factor benchmark', '{:.2f}'.format(profit_factor_benchmark)],
                            ['Max. drawdown [pip]', '{:.2f}'.format(max_drawdown)],
                            ['Avg. drawdown [pip]', '{:.2f}'.format(average_drawdown)],
                            ['Recovery factor', '{:.2f}'.format(recovery_factor)],
                            ['Determination coef. R2', '{:.2f}'.format(r2)]],
                           tablefmt='plain', colalign=('left', 'right'), floatfmt=".4f"))
            print('-' * 50)

    def execute(self, metrics=False, verbose=False):

        self.df_market[self.result_column] = 0
        self.define_SL_and_TP()
        self.df_market.reset_index(inplace=True, drop=True)

        self.df_market['Trade_duration'] = 0
        df_trades = self.df_market[(self.df_market[self.action_column].abs() > 0)]
        for i, trade_row in df_trades.iterrows():
            future_df = self.df_market[self.df_market.time >= trade_row.time]
            future_df.reset_index(inplace=True, drop=True)
            if self.df_market.loc[self.df_market.time == trade_row.time,self.action_column].abs().iloc[0] == 1:
                trade_result, trade_duration = self.compute_return_trad(trade_row,future_df)
                self.df_market.loc[self.df_market['time'] == trade_row.time, self.result_column] = trade_result
                self.df_market.loc[self.df_market['time'] == trade_row.time, 'Trade_duration'] = trade_duration

        if self.just_one:
            self.df_market.loc[self.df_market[self.action_column] == 0, self.result_column] = 0

        self.calculate_profit(self)
        self.calculate_profit_factor(self)
        self.calculate_drawdown(self)

        if metrics:
            self.calculate_metrics(df_market_action=self.df_market, action_column=self.action_column, verbose=verbose)

        return self.df_market