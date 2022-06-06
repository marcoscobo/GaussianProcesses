import pandas as pd
import numpy as np
from pandas_datareader import data as wb
from sklearn.preprocessing import StandardScaler

symbols = pd.read_csv('futures.txt', sep=';', header=0)


def get_df_symbols(symbols, verbose=False):
    datasets, datasets_years = {}, {}
    for symbol, name in zip(symbols['symbol'].values, symbols['Name'].values):

        if verbose:
            print('Getting ' + name + ' dataset')
        df = wb.DataReader(symbol, data_source='yahoo', start='2000-1-1')
        datasets[name] = df

        starts, ends = [], []
        for i in range(1, 22):
            starts.append('20{:02d}-01-01'.format(i))
            ends.append('20{:02d}-12-31'.format(i))

        sd = np.std(df['Adj Close'].values)
        df_years = pd.DataFrame()
        for start, end in zip(starts, ends):
            s = [None] * 253
            v = df['Adj Close'][start:end].values
            v = (v - v[0]) / sd
            s[:len(v)] = v
            df_years[end[:4]] = s

        datasets_years[name] = df_years

    return datasets_years, datasets


def get_Xs_ys(datasets):
    Xs, ys = {}, {}
    for name in list(datasets.keys()):

        df_years = datasets[name]

        days = df_years.index.values
        years = [int(year) for year in df_years.columns.values]

        X = pd.DataFrame(columns=['year', 'day'])
        y, i = np.array([None] * len(years) * len(days)), 0
        for year in years:
            for day in days:
                X.loc[i, 'year'], X.loc[i, 'day'] = year, day
                y[i] = df_years[str(year)].values[day]
                i += 1

        X = X.loc[~np.isnan(list(y)), :].reset_index(drop=True)
        y = y[~np.isnan(list(y))]

        Xs[name], ys[name] = X, y

    return Xs, ys
