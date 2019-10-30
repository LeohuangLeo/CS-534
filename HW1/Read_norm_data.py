import pandas as pd
import numpy as np

#Read and normalize data
def read_norm_data(filename):
    df =pd.read_csv(filename)
    del df['id']
    df['day'] = pd.DatetimeIndex(df['date']).day
    df['month'] = pd.DatetimeIndex(df['date']).month
    df['year'] = pd.DatetimeIndex(df['date']).year
    del df['date']
    try:
        y_train = df['price'].values.reshape(-1, 1)
        X_train = df.drop(columns=['price', 'dummy'])
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min()) #Normalization
        X_1 = np.ones(len(X_train)).reshape(-1, 1)
        X_train = np.concatenate((X_1, X_train), axis=1)

        return X_train, y_train
    except:
        X_train = df.drop(columns=['dummy'])
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min()) #Normalization
        X_1 = np.ones(len(X_train)).reshape(-1, 1)
        X_train = np.concatenate((X_1, X_train), axis=1)

        return X_train
