from pandas_datareader import data as wb


def get_df_symbols(symbols, verbose=False):
    datasets = {}
    for symbol, name in zip(symbols['symbol'].values, symbols['Name'].values):

        if verbose:
            print('Getting ' + name + ' dataset')
        df = wb.DataReader(symbol, data_source='yahoo', start='2000-1-1')
        datasets[name] = df

    return datasets
