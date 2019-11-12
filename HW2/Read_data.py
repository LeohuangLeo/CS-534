import numpy as np
import pandas as pd
#Read training and validation data
def Read_data(filename):
    df = pd.read_csv(filename,header=None)
    df[0].replace(5, -1,inplace=True)
    df[0].replace(3, 1,inplace=True)
    X_train = df.drop([0], axis=1).values
    X_1 = np.ones(X_train.shape[0]).reshape(-1, 1)
    X_train = np.concatenate([X_1, X_train], axis=1)
    y_train = df[0].values.reshape(-1, 1)
    return X_train, y_train