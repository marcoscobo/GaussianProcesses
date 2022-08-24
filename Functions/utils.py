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
