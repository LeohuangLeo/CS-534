import numpy as np
import pandas as pd
# Read testing data
def Read_test_data(filename):
    df = pd.read_csv('pa2_test_no_label.csv',header=None).values
    X_1 = np.ones(df.shape[0]).reshape(-1, 1)
    X_test = np.concatenate([X_1, df], axis=1)
    return X_test