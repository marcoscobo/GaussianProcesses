import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from tabulate import tabulate


def get_df_symbols(symbols, verbose=False):
    datasets = {}
    for symbol, name in zip(symbols['symbol'].values, symbols['Name'].values):

        if verbose:
            print('Getting ' + name + ' dataset')
        df = wb.DataReader(symbol, data_source='yahoo', start='2000-1-1')
        datasets[name] = df

    return datasets


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

    total_backtest.rename(columns={'price':'Cumulative_Growth', 'growth':'Cumulative_Profit_Growth'}, inplace=True)
    total_backtest['Drawdown'] = total_backtest['Cumulative_Profit_Growth'] - total_backtest[
        'Cumulative_Profit_Growth'].cummax()

    start = str(total_backtest['Date'][0])
    end = str(total_backtest['Date'].iloc[-1])
    duration = (dt.strptime(end, '%Y-%m-%d %H:%M:%S') - dt.strptime(start, '%Y-%m-%d %H:%M:%S')).days

    win_buy = count_buys_wins / buy_ops
    win_sell = count_sells_wins / sell_ops
    profit = total_backtest['Cumulative_Profit_Growth'].iloc[-1]

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
               'Profit': profit,
               'Profit factor': '{:.2f}'.format(profit_factor),
               'Max. drawdown [%]': '{:.2f}'.format(max_drawdown * 100),
               'Avg. drawdown [%]': '{:.2f}'.format(average_drawdown * 100),
               'Recovery factor': '{:.2f}'.format(recovery_factor),
               'Determination coef. R2': '{:.2f}'.format(r2)}

    if verbose:
        with open('Backtesting/Results/Total_Backtesting.txt', 'w') as external_file:
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
                            ['Profit factor', '{:.2f}'.format(profit_factor)],

                            ['Max. drawdown [%]', '{:.2f}'.format(max_drawdown * 100)],
                            ['Avg. drawdown [%]', '{:.2f}'.format(average_drawdown * 100)],
                            ['Recovery factor', '{:.2f}'.format(recovery_factor)],
                            ['Determination coef. R2', '{:.2f}'.format(r2)]],
                           tablefmt='plain', colalign=('left', 'right'), floatfmt=".4f")
            print(tab)
            print(tab, file=external_file)

            print('-' * 43)
            print('-' * 43, file=external_file)
            external_file.close()

    if plot:
        pbt = total_backtest.plot(x='Date', y=['Cumulative_Growth', 'Cumulative_Profit_Growth', 'Drawdown'],
                                  title='Total Backtesting', figsize=(16, 8))
        if save is not None:
            f = pbt.get_figure()
            f.savefig('Backtesting/Figures/' + save + '.png')

    return total_backtest, metrics
