import numpy as np
import pandas as pd
from pandas_datareader import data as wb
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def get_df_symbols(symbols, verbose=False):
    datasets = {}
    for symbol, name in zip(symbols['symbol'].values, symbols['Name'].values):

        if verbose:
            print('Getting ' + name + ' dataset')
        df = wb.DataReader(symbol, data_source='yahoo', start='2001-1-1')
        datasets[name] = df

    return datasets


def get_corr(datasets, n=3, year=None, method='pearson', alpha=0.05, notrend=False, plot=False, save=None):
    df_closes = pd.DataFrame(columns=list(datasets.keys()))
    for name in list(datasets.keys()):
        df_closes[name] = datasets[name]['Adj Close']
        df_closes[name] = (df_closes[name] - df_closes[name][0]) / df_closes[name].std()
        if notrend:
            df_closes[name] = df_closes[name] - df_closes[name].rolling(5, center=True).mean().fillna(0)
    if year is not None:
        df_closes = df_closes[:str(year)]

    df_corr = pd.DataFrame(columns=list(datasets.keys()), index=list(datasets.keys()))
    for name1 in df_corr.columns:
        for name2 in df_corr.index:
            if method == 'pearson':
                corr = pearsonr(df_closes[name1].fillna(0), df_closes[name2].fillna(0))
            elif method == 'spearman':
                corr = spearmanr(df_closes[name1].fillna(0), df_closes[name2].fillna(0))
            df_corr.at[name1, name2] = 0 if corr[1] >= alpha else float(corr[0])
    df_corr = df_corr.astype('float64')

    if plot:
        f, ax = plt.subplots(figsize=(15, 10))
        ax = sns.heatmap(df_corr, xticklabels=df_corr.columns, yticklabels=df_corr.columns, annot=True, ax=ax)
        if save is not None:
            plt.savefig(save + '.png')

    top_n_corr = {}
    for name in df_corr.columns:
        top_n_corr[name] = list(df_corr[name][df_corr[name] < 1].abs().nlargest(n).index)

    return top_n_corr


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

    if verbose:
        with open(save + '/Results/Total_Backtesting.txt', 'w') as external_file:
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
            f.savefig(save + '/Figures/Total_Backtesting.png')

    return total_backtest, metrics
